<h1 align="center">PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur</h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://gopirajumatta.github.io/PRISM3D/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](https://arxiv.org/pdf/2607.03855v1)
[![ECCV](https://img.shields.io/badge/ECCV-2026-blue.svg)](#)

**[Gopi Raju Matta](https://www.linkedin.com/in/gopi-raju-matta-1b5347105)** · **[Trisha Reddypalli](https://www.linkedin.com/in/reddypalli-trisha)** · **[Divya Madhuri Vemunuri](https://www.linkedin.com/in/divya-madhuri-4641b7132)** · **[Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra)**

**Computational Imaging Lab, Indian Institute of Technology Madras**

</div>

---

## 📰 News
* **[July 2026]** Project page and GitHub repository launched!
* **[July 2026]** Our pre-print is now available on [arXiv](https://arxiv.org/pdf/2607.03855v1).
* **[June 2026]** PRISM3D has been accepted to **ECCV 2026**!

---

## 🚀 Teaser: The PRISM3D Pipeline

<p align="center">
  <img src="https://gopirajumatta.github.io/PRISM3D/static/images/pipeline.png" alt="PRISM3D Pipeline Overview" width="100%">
</p>
<p align="center">
  <em><strong>Overview of PRISM3D and PRISM3D-E.</strong> Our framework jointly estimates camera motion and a sharp Gaussian representation directly from severely motion-blurred images, utilizing deep SfM for robust initialization and MCMC for probabilistic geometric refinement.</em>
</p>

---

## 📝 Abstract

<div align="justify">

We address the inverse problem of **blind 3D scene reconstruction from extremely motion-blurred images**, a scenario where traditional Structure-from-Motion pipelines fail. Unlike traditional methods that require sharp images for pose estimation, **PRISM3D** directly processes blurred images, making it practical for real-world scenarios.

Our approach utilizes a **Robust Initialization strategy** via deep dense tracking (VGGSfM) to recover global topology where feature matching fails. To robustly populate these sparse priors, we adopt a probabilistic formulation for geometric densification via **Markov Chain Monte Carlo (MCMC)**, while simultaneously modeling physical image formation via continuous Bézier Trajectories.

Furthermore, we introduce **PRISM3D-E**, a multi-modal (RGB + Events) extension that seamlessly integrates high-temporal-resolution events (EDI deblurring) as structural priors to maximize geometric recovery. To facilitate future research, we concurrently contribute the **PRISM3D-E Benchmark dataset** specifically curated for extreme blur scenarios.

</div>

---

## Quickstart

### 1. Installation

This repo assumes PyTorch (with CUDA) is already installed. Then install the remaining dependencies:

```bash
# (Optional) create a fresh conda env
conda create --name prism3d -y "python<3.11"
conda activate prism3d

# install dependencies
pip install --upgrade pip setuptools
pip install -r requirements.txt
```

`requirements.txt` pulls in `gsplat` and `fused-ssim` directly from their git repos (pinned commits), along with
`viser`/`nerfview` for the interactive viewer, `tyro` for CLI configs, and `pypose` for pose/spline optimization.

Then install this repo's own package (camera optimizer, spline utilities, MCMC strategy) in editable mode:

```bash
pip install -e .
```

### 2. Prepare the dataset

Each scene lives in its own folder (`data/<dataset>/<scene>/`) in COLMAP format:

```
<scene>/
├── images/            # training images (blurry), full resolution
├── images_<factor>/    # optional pre-downsampled variant, e.g. images_2/ for --data_factor 2
├── images_test/        # held-out GT sharp images (deblurring + novel-view-synthesis eval)
├── images_test_<factor>/
├── sparse/0/           # COLMAP-format sparse reconstruction: cameras.bin, images.bin, points3D.bin
├── hold=<n>             # empty marker file — sets this scene's eval interval (see below)
└── events_bins_<k>.pt   # PRISM3D-E only: event tensor, shape [num_frames, k, H, W]
```

- **`sparse/0/`** is generated with **VGGSfM** (deep dense tracking), not classic COLMAP feature matching — this is
  what lets PRISM3D initialize from severely blurred images where feature matching fails.
- **`hold=<n>`** is read directly from the scene folder at load time (see `datasets/deblur_nerf.py`), so each scene
  can use a different eval interval `n` rather than one fixed value for the whole dataset.
- **`events_bins_<k>.pt`** holds `k` event bins per consecutive blurry-frame transition; `k` must match
  `camera_optimizer.num_virtual_views - 1`.

**PRISM3D** (RGB-only) uses the existing public datasets directly, no complementary files needed:
- Synthetic: [ExBluRF dataset](https://drive.google.com/drive/folders/1kd061Ip9l9RUrze_6MOPAiz-Mcw_bwux?usp=drive_link)
- Real: [E2NeRF dataset](https://drive.google.com/drive/folders/1XhOEp4UdLL7EnDNyWdxxX8aRvzF53fWo?usp=sharing)

**PRISM3D-E** (RGB+Events) reuses the *same* RGB images from ExBlur-f/E2NeRF, but pairs them with our own
VGGSfM-based `sparse/0/` and per-scene `hold=<n>`, since these differ from what ships with the original datasets —
plus new `events_bins_13.pt` event tensors for the synthetic scenes. To avoid redistributing data we don't own, we
release only this complementary layer as the **PRISM3D-E Benchmark dataset**:

📦 **[Download PRISM3D-E Benchmark (Google Drive)](https://drive.google.com/drive/folders/1xzMbtLh5cck_C9et-MGIcoNsL2MoWofY?usp=sharing)**

```
PRISM3D-E/
├── synthetic/<scene>/   # sparse/0/, hold=<n>, events_bins_13.pt   (8 scenes)
└── real/<scene>/        # sparse/0/, hold=<n>                      (5 scenes)
```

Note: **real** scenes ship without event tensors here — E2NeRF's own release already provides the corresponding
real-world event data, so use that alongside this archive's `sparse/0/` and `hold=<n>`.

To set up a scene: download the original ExBlur-f (synthetic) / E2NeRF (real) images, then copy this archive's
`sparse/0/`, `hold=<n>`, and (for synthetic) `events_bins_13.pt` into the same per-scene folder, alongside the
original `images/`.

`--scale_factor 0.25` is recommended for forward-facing/LLFF-style captures; use `1.0` for object-centric scenes.

### 3. Training

Training entrypoints follow the `gsplat` examples pattern: `python <script>.py <default|mcmc> [options]`, where
`default` uses the original 3DGS densification heuristics and `mcmc` uses the MCMC-based (probabilistic geometric
refinement) densification strategy described in the paper.

**PRISM3D** and **PRISM3D-E** use the same training code (`simple_trainer_deblur.py`) — the only difference is which
dataset you point at. `train.sh` trains each scene in `SCENE_LIST`, then evaluates + renders every saved checkpoint,
then prints the deblurring/NVS/train stats for all scenes:

```bash
bash train.sh
```

Edit `SCENE_DIR` / `RESULT_DIR` / `SCENE_LIST` / `CAP_MAX` at the top of the script for your data layout and scenes.

Key flags:
- `--camera-optimizer.mode {off,linear,cubic,bezier}`: trajectory model for the bundle-adjusted camera optimizer
  (`bezier` is the continuous Bézier trajectory model described in the paper).
- `--strategy.cap-max`: max Gaussian count for the MCMC strategy (`mcmc` subcommand only).
- `--data_factor` / `--scale_factor`: image downscale factor and camera-origin scale (see dataset prep above).
- `--disable_viewer`: disable the interactive `viser`/`nerfview` viewer (recommended for batch/headless runs).

### 4. Outputs

Each run writes to `--result_dir`:
- `ckpts/` — model checkpoints (`ckpt_<step>_rank<r>.pt`)
- `stats/` — per-step JSON metrics (`train*`, `deblur*`, `nvs*`)
- `renders/` — rendered images/trajectories
- `tb/` — TensorBoard logs
- `cfg.yml` — the resolved run config

### 5. Viewer / pose inspection

- `simple_viewer.py` — standalone gsplat viewer for a trained checkpoint.
- `pose_viewer.py` — visualize optimized camera trajectories/poses.

---

## 📖 Citation

If you find our work useful in your research, please consider citing:

```bibtex
@inproceedings{matta2026prism3d,
  author    = {Matta, Gopi Raju and Reddypalli, Trisha and Vemunuri, Divya Madhuri and Mitra, Kaushik},
  title     = {{PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur}},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

This project builds directly on BAD-Gaussians; if you use this code, please also cite:

```bibtex
@inproceedings{zhao2024badgaussians,
    author = {Zhao, Lingzhe and Wang, Peng and Liu, Peidong},
    title = {Bad-gaussians: Bundle adjusted deblur gaussian splatting},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year = {2024}
}
```

## Acknowledgments

- Thanks to the **VGGSfM** authors for their deep dense-tracking SfM, which we use for robust initialization from severely blurred images.
- Thanks to the **ExBluRF** authors for their synthetic motion-blur dataset.
- Thanks to the [BAD-Gaussians](https://github.com/WU-CVGL/BAD-Gaussians) authors — this codebase builds on their bundle-adjusted deblur Gaussian splatting formulation (see Citation above).
