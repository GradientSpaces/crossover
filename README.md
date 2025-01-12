<p align="center">
  <h2 align="center"> CrossOver: 3D Scene Cross-Modal Alignment </h2>

  <p align="center">
    <a href="https://sayands.github.io/">Sayan Deb Sarkar</a><sup>1</sup>
    .
    <a href="https://miksik.co.uk/">Ondrej Miksik</a><sup>2</sup>
    .
    <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>2, 3</sup>
    .
    <a href="https://www.linkedin.com/in/d%C3%A1niel-bar%C3%A1th-3a489092/">Dániel Béla Baráth</a><sup>3</sup>
    .
    <a href="https://ir0.github.io/">Iro Armeni</a><sup>1</sup>
  </p>

  <p align="center">
    <sup>1</sup>Stanford University · <sup>2</sup>Microsoft Spatial AI Lab · <sup>3</sup>ETH Zürich
  </p>
  <h3 align="center">

 [![arXiv](https://img.shields.io/badge/arXiv-blue?logo=arxiv&color=%23B31B1B)]() 
 [![ProjectPage](https://img.shields.io/badge/Project_Page-CrossOver-blue)](https://sayands.github.io/crossover)
 [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
 <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://github.com/sayands/crossover/blob/main/static/videos/teaser.gif" width="100%">
  </a>
</p>

## 📃 Abstract

Multi-modal 3D object understanding has gained significant attention, yet current approaches often rely on rigid object-level modality alignment or 
assume complete data availability across all modalities. We present **CrossOver**, a novel framework for cross-modal 3D scene understanding via flexible, scene-level modality alignment. Unlike traditional methods that require paired data for every object instance, CrossOver learns a unified, modality-agnostic embedding space for scenes by aligning modalities - RGB images, point clouds, CAD models, floorplans, and text descriptions - without explicit object semantics. Leveraging dimensionality-specific encoders, a multi-stage training pipeline, and emergent cross-modal behaviors, CrossOver supports robust scene retrieval and object localization, even with missing modalities. Evaluations on ScanNet and 3RScan datasets show its superior performance across diverse metrics, highlighting CrossOver’s adaptability for real-world applications in 3D scene understanding.

<!-- TABLE OF CONTENTS -->
<details open="open" style='padding: 10px; border-radius:5px 30px 30px 5px; border-style: solid; border-width: 1px;'>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#arrow_down-data-download">Data Download</a>
    </li>
    <li>
      <a href="#hammer_and_wrench-installation">Installation</a>
    </li>
    <li>
      <a href="#wrench-data-preprocessing">Data Preprocessing</a>
    </li>
    <li>
      <a href="#film_projector-demo">Demo</a>
    </li>
    <li>
      <a href="#weight_lifting-training">Training</a>
    </li>
    <li>
      <a href="#bar_chart-evaluation">Evaluation</a>
    </li>
    <li>
      <a href="#pray-acknowledgements">Acknowledgements</a>
    </li>
    <li>
      <a href="#page_facing_up-citation">Citation</a>
    </li>
  </ol>
</details>

# :arrow_down: Data Download
We currently host the required preprocessed data + meta-data on Redivis and request all applicants to fill out the form [here](). We also release instructions for data download & preparation with scripts for ScanNet + 3RScan. For detailed instructions, please look at `README.MD` in `prepare_data/` directory/.

> Note: You agree to the terms of ScanNet, 3RScan, ShapeNet, Scan2CAD and SceneVerse datasets by downloading our hosted data.

# :hammer_and_wrench: Installation
The code has been tested on: 
```yaml
Ubuntu: 22.04 LTS
Python: 3.9.20
CUDA: 12.1
GPU: GeForce RTX 4090/RTX 3090
```

## 📦 Setup

Clone the repo and setup as follows:

```
git clone git@github.com:GradientSpaces/CrossOver.git
cd CrossOver
conda env create -f req.yml
```

# :wrench: Data Preprocessing
In order to process data faster during training + inference, we preprocess 1D (referral), 2D (RGB + floorplan) & 3D (Point Cloud + CAD) for both object instances and scenes. Note that, since for 3RScan dataset, they do not provide frame-wise RGB segmentations, we project the 3D data to 2D and store it in `.pt` format for every scan. We provide the scripts for projection and release the data.

Please refer to `PREPROCESS.MD` for details. 

# :film_projector: Demo

# :weight_lifting: Training 

# :bar_chart: Evaluation

# :pray: Acknowledgements

# :page_facing_up: Citation
