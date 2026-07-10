"""
Convert nested HDF5 (traj_XXXX/{rgb,depth,...}) to flat HDF5 (one array per column + ep_len/ep_offset).

Flat layout enables global contiguous slicing and is much faster than nested group random access.

Usage:
    python dataset/convert_hdf5_to_flat.py
    python dataset/convert_hdf5_to_flat.py --src data/dataset.h5 --dst data/dataset_flat.h5

Output (default: data/):
    data/dataset_flat.h5
    ├── rgb        (Total, 224, 224, 3)  uint8
    ├── depth      (Total, 224, 224)     uint8
    ├── states     (Total, 16)           float64
    ├── actions    (Total, 8)            float32
    ├── events     (Total, 2)            float32   (if present)
    ├── ep_len     (N,)                  int32
    └── ep_offset  (N,)                  int64
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import h5py
import numpy as np
from tqdm import tqdm

CHUNK_TIME = 32


def main():
    parser = argparse.ArgumentParser(description="Nested HDF5 → flat HDF5")
    parser.add_argument("--src", type=str, default="data/dataset.h5")
    parser.add_argument("--dst", type=str, default="data/dataset_flat.h5")
    args = parser.parse_args()

    src_path = Path(args.src)
    if not src_path.is_absolute():
        src_path = ROOT / src_path
    dst_path = Path(args.dst)
    if not dst_path.is_absolute():
        dst_path = ROOT / dst_path

    if not src_path.exists():
        print(f"Error: source {src_path} not found")
        sys.exit(1)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(src_path), "r") as src:
        keys = sorted(src.keys())
        print(f"Source: {src_path}")
        print(f"  {len(keys)} trajectories")

        # ---- Phase 1: scan metadata ----
        ep_lengths = []
        has_events = False
        for key in tqdm(keys, desc="Scanning"):
            traj = src[key]
            ep_lengths.append(traj["states"].shape[0])
            if "events" in traj:
                has_events = True

        total_steps = sum(ep_lengths)
        offsets = np.cumsum([0] + ep_lengths[:-1]).astype(np.int64)
        lengths = np.array(ep_lengths, dtype=np.int32)

        first = src[keys[0]]
        meta = {
            "rgb": (first["rgb"].shape[1:], first["rgb"].dtype),
            "depth": (first["depth"].shape[1:], first["depth"].dtype),
            "states": (first["states"].shape[1:], first["states"].dtype),
            "actions": (first["actions"].shape[1:], first["actions"].dtype),
        }
        if has_events:
            meta["events"] = (first["events"].shape[1:], first["events"].dtype)

        print(f"\n  Total steps: {total_steps:,}")
        for col, (shp, dt) in meta.items():
            nbytes = total_steps * int(np.prod(shp)) * dt.itemsize
            print(f"  {col:>8s}: per_step={shp}  dtype={dt}  total={nbytes / 1e9:.1f} GB")

        # ---- Phase 2: create flat datasets & write ----
        print(f"\nWriting: {dst_path}")
        with h5py.File(str(dst_path), "w") as dst:
            datasets = {}
            for col, (shp, dt) in meta.items():
                ct = min(CHUNK_TIME, total_steps)
                datasets[col] = dst.create_dataset(
                    col,
                    shape=(total_steps, *shp),
                    dtype=dt,
                    chunks=(ct, *shp),
                )

            ptr = 0
            for key in tqdm(keys, desc="Converting"):
                traj = src[key]
                L = int(traj["states"].shape[0])
                for col in meta:
                    if col in traj:
                        datasets[col][ptr : ptr + L] = traj[col][:]
                ptr += L

            dst.create_dataset("ep_len", data=lengths)
            dst.create_dataset("ep_offset", data=offsets)

    print(f"\nDone! {len(keys)} episodes, {total_steps:,} steps → {dst_path}")
    fsize = dst_path.stat().st_size / (1024**3)
    print(f"  File size: {fsize:.1f} GB")


if __name__ == "__main__":
    main()
