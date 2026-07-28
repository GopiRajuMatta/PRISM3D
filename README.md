<h1 align="center">PRISM3D: Probabilistic Refinement and Robust Initialization for Physically Consistent Scene Modeling under Extreme Motion Blur</h1>

<div align="center">

[![Project Page](https://img.shields.io/badge/Project-Page-green.svg)](https://gopirajumatta.github.io/PRISM3D/)
[![Paper](https://img.shields.io/badge/Paper-ArXiv-red.svg)](#)
[![ECCV](https://img.shields.io/badge/ECCV-2026-blue.svg)](#)

**[Gopi Raju Matta](https://www.linkedin.com/in/gopi-raju-matta-1b5347105)** · **[Trisha Reddypalli](https://www.linkedin.com/in/reddypalli-trisha)** · **[Divya Madhuri Vemunuri](https://www.linkedin.com/in/divya-madhuri-4641b7132)** · **[Kaushik Mitra](https://www.ee.iitm.ac.in/kmitra)**

**Computational Imaging Lab, Indian Institute of Technology Madras**

</div>

---

## 🚀 Teaser

<div align="center">
  <video src="https://gopirajumatta.github.io/PRISM3D/static/videos/camellia.mp4" autoplay muted loop playsinline width="80%"></video>
</div>
<br>
<p align="center">
  <em>PRISM3D framework learns a sharp 3D Gaussian representation of the scene along with its camera motion trajectories directly from extreme motion-blurred images, enabling state-of-the-art deblurring, high-quality novel view synthesis, and real-time rendering.</em>
</p>

---

## 📝 Abstract

We address the inverse problem of **blind 3D scene reconstruction from extremely motion-blurred images**, a scenario where traditional Structure-from-Motion pipelines fail. Unlike traditional methods that require sharp images for pose estimation, **PRISM3D** directly processes blurred images, making it practical for real-world scenarios.

Our approach utilizes a **Robust Initialization strategy** via deep dense tracking (VGGSfM) to recover global topology where feature matching fails. To robustly populate these sparse priors, we adopt a probabilistic formulation for geometric densification via **Markov Chain Monte Carlo (MCMC)**, while simultaneously modeling physical image formation via continuous Bézier Trajectories.

Furthermore, we introduce **PRISM3D-E**, a multi-modal (RGB + Events) extension that seamlessly integrates high-temporal-resolution events (EDI deblurring) as structural priors to maximize geometric recovery. To facilitate future research, we concurrently contribute the **PRISM3D-E Benchmark dataset** specifically curated for extreme blur scenarios.

---

## ⚙️ Method: The PRISM3D Pipeline

<p align="center">
  <img src="https://gopirajumatta.github.io/PRISM3D/static/images/pipeline.png" alt="PRISM3D Pipeline Overview" width="100%">
</p>
<p align="center">
  <em><strong>Overview of PRISM3D and PRISM3D-E.</strong> Our framework jointly estimates camera motion and a sharp Gaussian representation directly from severely motion-blurred images, utilizing deep SfM for robust initialization and MCMC for probabilistic geometric refinement.</em>
</p>

---

## 📅 Release Roadmap

We are currently preparing the codebase and datasets for public release. 

- [x] Project Page & Teaser Released
- [x] ArXiv Pre-print Available
- [ ] **Coming Soon:** Full Source Code (Training & Evaluation)
- [ ] **Coming Soon:** PRISM3D-E Benchmark Dataset
- [ ] **Coming Soon:** Pre-trained Models

*Star ⭐ this repository to get notified when the code drops!*

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
