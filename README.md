<p align="center">
  <img src="assets/orbit_planner_icon.png" alt="Orbit-Planner" width="100"/>
</p>

<h1 align="center"><strong>Orbit-Planner: Towards Latent World Models for On-Orbit Obstacle Avoidance of Satellite Agent</strong></h1>

<p align="center">
  <strong><a href="https://zhijianli2003.github.io/">Zhijian Li</a><sup>1,2</sup></strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Chao Ren<sup>2,*</sup></strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Peijin Wang<sup>2</sup></strong>
  &nbsp;&nbsp;&nbsp;
  <strong>Xian Sun<sup>2</sup></strong><br/>
  <sup>1</sup> University of Chinese Academy of Sciences<br/>
  <sup>2</sup> Key Laboratory of Target Cognition and Application Technology (TCAT),<br/>
  Aerospace Information Research Institute, Chinese Academy of Sciences
</p>

<p align="center">
  <a href="https://arxiv.org/abs/2608.16651"><img src="https://img.shields.io/badge/Paper-Arxiv-red?style=flat-square&labelColor=555555" alt="Arxiv"></a>
  &nbsp;
  <a href="https://huggingface.co/datasets/warriorLZJ/Orbit_Planner/tree/main"><img src="https://img.shields.io/badge/🤗%20Hugging%20Face-Dataset-blue?style=flat-square&labelColor=555555" alt="Hugging Face Dataset&Model"></a>
  &nbsp;
  <a href="https://zhijianli2003.github.io/Orbit_Planner/"><img src="https://img.shields.io/badge/Project-Homepage-brightgreen?style=flat-square&labelColor=555555&logo=googlechrome&logoColor=white" alt="Project Homepage"></a>
</p>


## 📢 News

- [2026/07/11] 🔥 We release **Orbit-Planner** training & evaluation code, along with Stage 1 & Stage 2 pre-trained checkpoints. [Code](https://github.com/ZhijianLi2003/Orbit_Planner) [Checkpoints](https://huggingface.co/datasets/warriorLZJ/Orbit_Planner/tree/main)
- [2026/08/07] 🔥 Our Paper was accepted by IEEE AP-GARSS 2026.
- [2026/08/14] 🔥 The complete Orbit-Planner Orbital Evasion Dataset has been released on [Huggingface](https://huggingface.co/datasets/warriorLZJ/Orbit_Planner).
- [Coming Soon] **Orbit-Planner** arXiv preprint will be available soon. [Paper](https://arxiv.org/abs/0000.00000)

## 🎬 Demo

<p align="center">
  <img src="assets/OrbitPlanner_Demo.gif" width="98%" alt="Orbit-Planner demo"/>
</p>

## 🏗️ Framework

<p align="center">
  <img src="assets/framework.png" width="98%" alt="Orbit-Planner framework"/>
</p>

Orbit-Planner is a two-stage latent world model for on-orbit obstacle avoidance:

- **Stage 1**: RGB observations and proprioceptive states are encoded into latent dynamics; an AdaLN Transformer predicts future latents with auxiliary depth supervision.
- **Stage 2**: A physics prober maps frozen latent rollouts to interpretable spacecraft states for planning and control.

### Project Structure

```text
Orbit_Planner/
├── configs/
│   ├── stage1.yaml                 # Stage 1 configuration
│   └── stage2.yaml                 # Stage 2 configuration
├── dataset/
│   ├── dataset.py                  # Dataset loading and preprocessing
│   └── convert_hdf5_to_flat.py     # HDF5 format conversion
├── scripts/
│   ├── train_stage1.py             # Stage 1 training
│   ├── eval_stage1.py              # Stage 1 evaluation
│   ├── train_stage2.py             # Stage 2 training
│   └── eval_stage2.py              # Stage 2 evaluation
├── world_model/
│   ├── encoder.py                  # Visual and state encoders
│   ├── predictor.py                # Latent dynamics predictor
│   ├── depth_head.py               # Auxiliary depth prediction head
│   ├── prober.py                   # Physics prober
│   ├── losses.py                   # Training objectives
│   └── world_model.py              # Orbit-Planner world model
├── data/                            # Dataset files
├── checkpoints/                     # Pre-trained and trained weights
└── requirements.txt                 # Python dependencies
```

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/ZhijianLi2003/Orbit_Planner.git
cd Orbit_Planner

conda create -n orbitplanner python=3.10 -y
conda activate orbitplanner

pip install -r requirements.txt
```

### Dataset and Checkpoints

Download the Orbit-Planner dataset and pre-trained checkpoints from [Hugging Face](https://huggingface.co/datasets/warriorLZJ/Orbit_Planner/tree/main). From the project root, run:

```bash
pip install -U "huggingface_hub[cli]"

hf download warriorLZJ/Orbit_Planner \
  data/dataset.h5 data/meta.json \
  --repo-type dataset \
  --local-dir .

hf download warriorLZJ/Orbit_Planner \
  stage1_world_model.pt stage2_physical_prober.ptt \
  --repo-type dataset \
  --local-dir checkpoints

mv checkpoints/stage2_physical_prober.ptt \
  checkpoints/stage2_physical_prober.pt
```

The downloaded files should be organized as follows:

```text
Orbit_Planner/
├── data/
│   ├── dataset.h5
│   └── meta.json
└── checkpoints/
    ├── stage1_world_model.pt
    └── stage2_physical_prober.pt
```

Convert the downloaded dataset to the flat HDF5 format used by the default configurations:

```bash
python dataset/convert_hdf5_to_flat.py \
  --src data/dataset.h5 \
  --dst data/dataset_flat.h5
```

### Evaluation

Evaluate the Stage 1 world model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python scripts/eval_stage1.py \
  --ckpt checkpoints/stage1_world_model.pt \
  --num_trajs 100 \
  > logs/eval_stage1_world_model.log 2>&1 &
```

Evaluate the Stage 2 physics prober:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python scripts/eval_stage2.py \
  --stage2_ckpt checkpoints/stage2_physical_prober.pt \
  --stage1_ckpt checkpoints/stage1_world_model.pt \
  --pred_horizon 25 \
  --num_trajs 100 \
  > logs/eval_stage2_physical_prober.log 2>&1 &
```

Evaluation results are saved under `eval_output/` by default.


### Training

If you want to retrain Orbit-Planner from scratch, first train the Stage 1 latent world model:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python scripts/train_stage1.py \
  --config configs/stage1.yaml \
  > logs/train_stage1_world_model.log 2>&1 &
```

After Stage 1 training, select the checkpoint you want to use and name it `stage1_best.pt`. For example, with the default 50-epoch configuration:

```bash
cp checkpoints/stage1_epoch050.pt checkpoints/stage1_best.pt
```

Use the selected Stage 1 checkpoint to train the Stage 2 physics prober:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONUNBUFFERED=1 nohup python scripts/train_stage2.py \
  --config configs/stage2.yaml \
  --stage1_ckpt checkpoints/stage1_best.pt \
  > logs/train_stage2_physical_prober.log 2>&1 &
```

The best Stage 2 checkpoint is saved to `checkpoints/stage2_best.pt` with the default configuration.

### Closed-Loop Obstacle-Avoidance Navigation

Our closed-loop obstacle-avoidance navigation experiments are built upon [Space Robotics Bench](https://github.com/AndrejOrsula/space_robotics_bench). As our implementation involves substantial modifications to the original framework, the related code is currently being organized and will be released soon. If you are interested in this feature, please ⭐ this repository to keep up with the latest updates and future releases.

## 📝 Citation

If you find Orbit-Planner useful in your research, please consider citing our work:

```bibtex
@article{li2026orbitplanner,
  title   = {Orbit-Planner: Towards Latent World Models for On-Orbit Obstacle Avoidance of Satellite Agent},
  author  = {Li, Zhijian and Ren, Chao and Wang, Peijin and Sun, Xian},
  journal = {arXiv preprint arXiv:XXXX.XXXXX},
  year    = {2026}
}
```

## 🔭 Future Work

Our future work will focus on on-orbit perception, scene understanding, and autonomous navigation for spacecraft, as well as broader applications of distributed foundation models in the aerospace domain. 
