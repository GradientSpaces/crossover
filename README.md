<div align='center'>
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

   [![arXiv](https://img.shields.io/badge/arXiv-blue?logo=arxiv&color=%23B31B1B)]() [![ProjectPage](https://img.shields.io/badge/Project_Page-LoopSplat-blue)]([https://loopsplat.github.io/](https://sayands.github.io/crossover/)) [![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
  <div align="center"></div>
</p>

<p align="center">
  <a href="">
    <img src="https://sayands.github.io/crossover/static/videos/teaser.gif" width="100%">
  </a>
</p>

## 📃 Abstract

Multi-modal 3D object understanding has gained significant attention, yet current approaches often rely on rigid object-level modality alignment or 
assume complete data availability across all modalities. We present **CrossOver**, a novel framework for cross-modal 3D scene understanding via flexible, scene-level modality alignment. 
Unlike traditional methods that require paired data for every object instance, CrossOver learns a unified, modality-agnostic embedding space for scenes by aligning modalities - 
RGB images, point clouds, CAD models, floorplans, and text descriptions - without explicit object semantics. Leveraging dimensionality-specific encoders, a multi-stage 
training pipeline, and emergent cross-modal behaviors, CrossOver supports robust scene retrieval and object localization, even with missing modalities. Evaluations on ScanNet and 
3RScan datasets show its superior performance across diverse metrics, highlighting CrossOver’s adaptability for real-world applications in 3D scene understanding.
