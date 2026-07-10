"""
Dataset loading and sampling (code in dataset/, HDF5 data files in data/).

Uses stable_worldmodel-style flat HDF5 layout:
    rgb      (Total, 224, 224, 3)  uint8
    depth    (Total, 224, 224)     uint8
    states   (Total, 16)           float64
    actions  (Total, 8)            float32
    events   (Total, 2)            float32  (optional)
    ep_len   (N,)                  int32
    ep_offset(N,)                  int64

Read strategy:
  - Small columns (states / actions / events) are cached in memory at startup (~240 MB)
  - Large columns (rgb / depth) are sliced on demand via per-worker SWMR file handles
  - 256 MB HDF5 raw-data chunk cache for hot data
"""

import random

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, Sampler


def compute_self_occlusion_mask(h5_path, n_sample=200):
    """Detect spacecraft self-occlusion regions from depth maps.

    Sample n_sample frames and find fixed pixel==255 locations across all frames.
    Returns (224, 224) bool mask where True = self-occluded pixel (should be excluded).
    """
    with h5py.File(str(h5_path), "r") as f:
        depth_ds = f["depth"]
        total = depth_ds.shape[0]
        indices = np.linspace(0, total - 1, min(n_sample, total), dtype=int)
        mask = np.ones((depth_ds.shape[1], depth_ds.shape[2]), dtype=bool)
        for idx in indices:
            frame = depth_ds[int(idx)]
            mask &= (frame == 255)
    n_masked = mask.sum()
    if n_masked > 0:
        ys, xs = np.where(mask)
        print(f"Self-occlusion mask: {n_masked} pixels "
              f"at y=[{ys.min()}-{ys.max()}], x=[{xs.min()}-{xs.max()}]")
    else:
        print("Self-occlusion mask: no fixed-255 pixels found")
    return mask


def depth_to_nearest_distance(depth_uint8, occlusion_mask=None):
    """Convert raw uint8 depth to normalized nearest-obstacle distance (excluding self-occlusion and background).

    Args:
        depth_uint8: (..., H, W) uint8 depth frames
        occlusion_mask: (H, W) bool, True = self-occluded pixel

    Returns:
        Normalized distance (...,) in [0, 1]; 0 ≈ very close, 1 ≈ no obstacle
    """
    original_shape = depth_uint8.shape[:-2]
    frames = depth_uint8.reshape(-1, depth_uint8.shape[-2], depth_uint8.shape[-1])
    n = frames.shape[0]

    result = np.empty(n, dtype=np.float32)
    for i in range(n):
        frame = frames[i].copy()
        if occlusion_mask is not None:
            frame[occlusion_mask] = 0
        nonzero = frame[frame > 0]
        if len(nonzero) > 0:
            max_pixel = float(nonzero.max())
        else:
            max_pixel = 0.0
        nearest_m = ((255.0 - max_pixel) / 255.0) * 50.0 + 0.1
        result[i] = nearest_m / 50.1

    return result.reshape(original_shape)


class OrbitPlannerDataset(Dataset):

    def __init__(self, h5_path, window_size=20, window_stride=1,
                 traj_ids=None, augment=False, load_events=False,
                 load_depth=True, rgb_load_frames=None,
                 load_min_depth=False):
        self.h5_path = str(h5_path)
        self.window_size = window_size
        self.window_stride = window_stride
        self.augment = augment
        self.load_events = load_events
        self.load_depth = load_depth
        self.load_min_depth = load_min_depth
        self.rgb_load_frames = rgb_load_frames or window_size

        with h5py.File(self.h5_path, "r") as f:
            all_ep_len = f["ep_len"][:]
            all_ep_offset = f["ep_offset"][:]

        if traj_ids is None:
            traj_ids = list(range(len(all_ep_len)))

        # Build clip index: (global_start, global_end)
        self.clip_indices = []
        for i in traj_ids:
            length = int(all_ep_len[i])
            offset = int(all_ep_offset[i])
            if length < window_size:
                continue
            n_win = (length - window_size) // window_stride + 1
            for w in range(n_win):
                s = offset + w * window_stride
                self.clip_indices.append((s, s + window_size))

        # Cache small columns (load entire column into memory, ~240 MB)
        with h5py.File(self.h5_path, "r") as f:
            self._states = f["states"][:]
            self._actions = f["actions"][:]
            self._events = f["events"][:] if (load_events and "events" in f) else None

        # Self-occlusion mask (computed only when load_min_depth is enabled)
        if self.load_min_depth:
            self._occlusion_mask = compute_self_occlusion_mask(h5_path)
        else:
            self._occlusion_mask = None

        self._h5 = None

        mem_mb = (self._states.nbytes + self._actions.nbytes) / (1024**2)
        if self._events is not None:
            mem_mb += self._events.nbytes / (1024**2)
        print(f"Dataset ready: {len(traj_ids)} episodes, "
              f"{len(self.clip_indices)} clips, "
              f"cached {mem_mb:.0f} MB (states+actions"
              f"{'+events' if self._events is not None else ''}), "
              f"rgb={self.rgb_load_frames}f"
              f"{'' if self.load_depth else ', no depth'}"
              f"{', +min_depth' if self.load_min_depth and not self.load_depth else ''}")

    def _open_h5(self):
        """Lazily open SWMR handle with 256 MB chunk cache."""
        if self._h5 is None:
            self._h5 = h5py.File(
                self.h5_path, "r",
                swmr=True,
                rdcc_nbytes=256 * 1024 * 1024,
            )
        return self._h5

    def __getstate__(self):
        """Drop file handle on pickle (DataLoader fork safety)."""
        state = self.__dict__.copy()
        state["_h5"] = None
        return state

    def __del__(self):
        if getattr(self, "_h5", None) is not None:
            try:
                self._h5.close()
            except Exception:
                pass

    def __len__(self):
        return len(self.clip_indices)

    def __getitem__(self, idx):
        g_start, g_end = self.clip_indices[idx]
        rgb_end = g_start + self.rgb_load_frames

        f = self._open_h5()
        rgb = torch.from_numpy(
            f["rgb"][g_start:rgb_end].astype(np.float32) / 255.0
        ).permute(0, 3, 1, 2)  # (T, C, H, W)

        states = torch.from_numpy(self._states[g_start:g_end].astype(np.float32))
        actions = torch.from_numpy(self._actions[g_start:g_end].astype(np.float32))

        if self.augment:
            rgb = self._augment_rgb(rgb)

        sample = {"rgb": rgb, "states": states, "actions": actions}
        if self.load_depth:
            depth = torch.from_numpy(
                f["depth"][g_start:g_end].astype(np.float32) / 255.0
            ).unsqueeze(1)
            sample["depth"] = depth
        if self.load_min_depth:
            depth_raw = f["depth"][g_start:g_end]   # (W, 224, 224) uint8
            gt_dist = depth_to_nearest_distance(depth_raw, self._occlusion_mask)
            sample["min_depth"] = torch.from_numpy(gt_dist)
        if self._events is not None:
            sample["events"] = torch.from_numpy(
                self._events[g_start:g_end].astype(np.float32)
            )
        return sample

    @staticmethod
    def _augment_rgb(rgb):
        if torch.rand(1).item() > 0.5:
            rgb = rgb * (0.8 + 0.4 * torch.rand(1).item())
        if torch.rand(1).item() > 0.5:
            mean = rgb.mean(dim=(-3, -2, -1), keepdim=True)
            rgb = (rgb - mean) * (0.8 + 0.4 * torch.rand(1).item()) + mean
        return rgb.clamp(0, 1)


class ChunkedRandomSampler(Sampler):
    """Chunked random sampler: shuffle between chunks + shuffle within chunks.

    clip_indices are ordered by episode; adjacent indices map to adjacent disk regions.
    Split indices into chunks of chunk_size, shuffle chunks, then shuffle within each chunk.
    Samples in the same batch are likely from the same file region → sequential IO.
    """

    def __init__(self, data_source, chunk_size=5000, seed=None):
        self.n = len(data_source)
        self.chunk_size = chunk_size
        self.seed = seed
        self._epoch = 0

    def set_epoch(self, epoch):
        self._epoch = epoch

    def __iter__(self):
        rng = random.Random(self.seed + self._epoch if self.seed is not None else None)
        indices = list(range(self.n))
        chunks = [indices[i:i + self.chunk_size]
                  for i in range(0, self.n, self.chunk_size)]
        rng.shuffle(chunks)
        for chunk in chunks:
            rng.shuffle(chunk)
        return iter([idx for chunk in chunks for idx in chunk])

    def __len__(self):
        return self.n


# ---- Shared helpers ----

def is_flat_h5(f):
    """Return True if HDF5 is flat layout (has ep_len + ep_offset)."""
    return "ep_len" in f and "ep_offset" in f


def list_trajectory_keys(h5_path):
    """Return episode index list. flat → [0,1,...]; nested → ['traj_0000', ...]."""
    with h5py.File(str(h5_path), "r") as f:
        if is_flat_h5(f):
            return list(range(len(f["ep_len"])))
        return sorted(k for k in f.keys() if isinstance(f[k], h5py.Group))


def load_episode(h5_path, episode_idx):
    """Load one trajectory for evaluation; returns (rgb, depth, states, actions)."""
    with h5py.File(str(h5_path), "r") as f:
        if is_flat_h5(f):
            idx = int(episode_idx)
            offset = int(f["ep_offset"][idx])
            length = int(f["ep_len"][idx])
            s, e = offset, offset + length
            rgb_raw = f["rgb"][s:e]
            depth_raw = f["depth"][s:e]
            states_raw = f["states"][s:e]
            actions_raw = f["actions"][s:e]
        else:
            key = (episode_idx if isinstance(episode_idx, str)
                   else f"traj_{int(episode_idx):04d}")
            if key not in f:
                raise KeyError(f"Trajectory '{key}' not found in {h5_path}")
            traj = f[key]
            rgb_raw = traj["rgb"][:]
            depth_raw = traj["depth"][:]
            states_raw = traj["states"][:]
            actions_raw = traj["actions"][:]

    rgb = torch.from_numpy(rgb_raw.astype(np.float32) / 255.0).permute(0, 3, 1, 2)
    depth = torch.from_numpy(depth_raw.astype(np.float32) / 255.0).unsqueeze(1)
    states = torch.from_numpy(states_raw.astype(np.float32))
    actions = torch.from_numpy(actions_raw.astype(np.float32))
    return rgb, depth, states, actions


def worker_init_fn(worker_id):
    """Reopen an independent HDF5 handle per DataLoader worker to avoid cross-process sharing."""
    info = torch.utils.data.get_worker_info()
    if info is not None:
        info.dataset._h5 = None
