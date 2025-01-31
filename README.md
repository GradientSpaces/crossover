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

### 🚀 Features

- Flexible Scene-Level Alignment 🌐 - Aligns RGB, point clouds, CAD, floorplans, and text at the scene level— no perfect data needed!
- Emergent Cross-Modal Behaviors 🤯 - Learns unseen modality pairs (e.g., floorplan ↔ text) without explicit pairwise training.
- Real-World Applications 🌍 AR/VR, robotics, construction—handles temporal changes (e.g., object rearrangement) effortlessly.

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


# :newspaper: News
- ![](https://img.shields.io/badge/New!-8A2BE2) [2025-02] We release the preprocessed + generated embedding data. Fill out the [form]() for the download link!
- ![](https://img.shields.io/badge/New!-8A2BE2) [2025-02] We release CrossOver on [arXiv](). Checkout our [paper]() and [website](https://sayands.github.io/crossover/).


# :arrow_down: Data Download

## Preprocessed Data
We release required preprocessed data + meta-data and provide instructions for data download & preparation with scripts for ScanNet + 3RScan. 

- For dataset download (single inference setup), please look at `README.MD` in `data_prepare/` directory.
- For preprocessed data download (training + evaluation only), please refer to [Data Preprocessing](#wrench-data-preprocessing).

> You agree to the terms of ScanNet, 3RScan, ShapeNet, Scan2CAD and SceneVerse datasets by downloading our hosted data.

## Generated Embedding Data
We release the embeddings created with CrossOver on the datasets used (`embed_data`/ in GDrive), which can be used for cross-modal retrieval with a custom dataset.

- `embed_scannet.pt`: Scene Embeddings For All Modalities (Point Cloud, RGB, Floorplan, Referral) in ScanNet
- `embed_scan3r.pt` : Scene Embeddings For All Modalities (Point Cloud, RGB, Referral) in ScanNet

File structure below:

```json
{
  "scene": [{
    "scan_id": "the ID of the scan",
    "scene_embeds": {
        "modality_name"     : "modality_embedding"
      }
    "mask" : "modality_name" : "True/False whether modality was present in the scan"
    },
    {
      ...
    },...
  ]
}
```

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

```bash
git clone git@github.com:GradientSpaces/CrossOver.git
cd CrossOver
conda env create -f req.yml
conda activate crossover
```

Further installation for `MinkowskiEngine` and `Pointnet2_PyTorch`. Setup as follows:

```bash
git clone --recursive "https://github.com/EthenJ/MinkowskiEngine"
conda install openblas-devel -c anaconda
cd MinkowskiEngine/
python setup.py install --blas_include_dirs=${CONDA_PREFIX}/include --force_cuda --blas=openblas
cd ..
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git
pip install pointnet2_ops_lib/.
```

> Since we use CUDA 12.1, we use the above `MinkowskiEngine` fork; for other CUDA drivers, please refer to the official [repo](https://github.com/NVIDIA/MinkowskiEngine).

# :film_projector: Demo
This demo script allows users to process a custom scene and retrieve the closest match from ScanNet/3RScan using different modalities. Detailed usage can be found inside the script. Example usage below:

```bash
python demo/demo_scene_retrieval.py
```

Various configurable parameters:

- `--query_path`: Path to the query scene file (eg: `./example_data/dining_room/scene_cropped.ply`).
- `--database_path`: Path to the precomputed embeddings of the database scenes downloaded before (eg: `./release_data/embed_scannet.pt`).
- `--query_modality`: Modality of the query scene, Options: `point`, `rgb`, `floorplan`, `referral`
- `--database_modality`: Modality used for retrieval.same options as above
- `--ckpt`: Path to the pre-trained scene crossover model checkpoint (details [here](#checkpoints)), example_path: `./checkpoints/scene_crossover_scannet+scan3r.pth/`).

For pre-trained model download, refer to data download and checkpoints sections.

> We also provide scripts for inference on a single scan Scannet/3RScan data. Details in [Single Inference](#shield-single-inference) section.

# :wrench: Data Preprocessing
In order to process data faster during training + inference, we preprocess 1D (referral), 2D (RGB + floorplan) & 3D (Point Cloud + CAD) for both object instances and scenes. Note that, since for 3RScan dataset, they do not provide frame-wise RGB segmentations, we project the 3D data to 2D and store it in `.pt` format for every scan. We provide the scripts for projection and release the data.

Please refer to `PREPROCESS.MD` for details. 

# :weight_lifting: Training 

#### Train Instance Baseline
Adjust path parameters in `configs/train/train_instance_baseline.yaml` and run the following:

```bash
bash scripts/train/train_instance_baseline.sh
```

#### Train Instance Retrieval Pipeline
Adjust path parameters in `configs/train/train_instance_crossover.yaml` and run the following:

```bash
bash scripts/train/train_instance_crossover.sh
```

#### Train Scene Retrieval Pipeline
Adjust path/configuration parameters in `configs/train/train_scene_crossover.yaml`. You can also add your customised dataset or choose to train on Scannet & 3RScan or either. Run the following:

```bash
bash scripts/train/train_scene_crossover.sh
```

> The scene retrieval pipeline uses the trained weights from instance retrieval pipeline (for object feature calculation), please ensure to update `task:UnifiedTrain:object_enc_ckpt` in the config file.

#### Checkpoints
We provide all available checkpoints on G-Drive [here](https://drive.google.com/drive/folders/1iGhLQY86RTfc87qArOvUtXAhpbFSFq6w?usp=sharing). Detailed descriptions in the table below:

| Model Type             | Description                                    | Checkpoint                                                                                              |
| ---------------------- | ---------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| ``instance_baseline``  | Instance Baseline trained on 3RScan            |  [3RScan](https://drive.google.com/drive/folders/1X_gHGLM-MssNrFu8vIMzt1sUtD3qM0ub?usp=sharing)         |
| ``instance_baseline``  | Instance Baseline trained on ScanNet           |  [ScanNet](https://drive.google.com/drive/folders/1iNWVK-r89vOIkr3GR-XfI1rItrZ0EECJ?usp=sharing)        | 
| ``instance_baseline``  | Instance Baseline trained on ScanNet + 3RScan  |  [ScanNet+3RScan](https://drive.google.com/drive/folders/1gRjrYmo4lxbLHGLBnYojB5aXM6iPf4D5?usp=sharing) |
| ``instance_crossover`` | Instance CrossOver trained on 3RScan           |  [3RScan](https://drive.google.com/drive/folders/1oPwYpn4yLExxcoLOqpoLJVTfWJPGvrTc?usp=sharing)         |
| ``instance_crossover`` | Instance CrossOver trained on ScanNet          |  [ScanNet](https://drive.google.com/drive/folders/1iIwjxKD8fBGo4eINBle78liCyLT8dK8y?usp=sharing)        | 
| ``instance_crossover`` | Instance CrossOver trained on ScanNet + 3RScan |  [ScanNet+3RScan](https://drive.google.com/drive/folders/1B_DqBY47SDQ5YmjDFAyHu59Oi7RzY3w5?usp=sharing) |
| ``scene_crossover``    | Unified CrossOver trained on ScanNet + 3RScan  |  [ScanNet+3RScan](https://drive.google.com/drive/folders/1TMlMTbBRkTHc0AU5SbpKHLCRFUh_mMn5?usp=sharing) |

# :shield: Single Inference
We release script to perform inference (generate scene-level embeddings) on a single scan of 3RScan/Scannet. Detailed usage in the file. Quick instructions below:

```bash
python single_inference/scene_inference.py
```

Various configurable parameters:

- `--dataset`: dataset name, Scannet/Scan3R
- `--data_dir`: data directory (eg: `./datasets/Scannet`, assumes similar structure as in `preprocess.md`).
- `--floorplan_dir`: directory consisting of the rasterized floorplans (this can point to the downloaded preprocessed directory), only for Scannet
- `--ckpt`: Path to the pre-trained scene crossover model checkpoint (details [here](#checkpoints)), example_path: `./checkpoints/scene_crossover_scannet+scan3r.pth/`).
- `--scan_id`: the scan id from the dataset you'd like to calculate embeddings for (if not provided, embeddings for all scans are calculated).

The script will output embeddings in the same format as provided [here](#generated-embedding-data).

# :bar_chart: Evaluation
#### Cross-Modal Object Retrieval
Run the following script (refer to the script to run instance baseline/instance crossover). Detailed usage inside the script.

```bash
bash scripts/evaluation/eval_instance_retrieval.sh
```

> This will also show you scene retrieval results using the instance based methods.

#### Cross-Modal Scene Retrieval
Run the following script (for scene crossover). Detailed usage inside the script.

```bash
bash scripts/evaluation/eval_instance_retrieval.sh
```

## 🚧 TODO List
- [ ] Release evaluation on temporal instance matching
- [ ] Release inference code on single image-based scene retrieval
- [ ] Release inference on single scan cross-modal object retrieval
- [ ]  Release inference using baselines

# :pray: Acknowledgements
We thank the authors from [3D-VisTa](https://github.com/3d-vista/3D-VisTA), [SceneVerse](https://github.com/scene-verse/sceneverse) and [SceneGraphLoc](https://github.com/y9miao/VLSG) for open-sourcing their codebases.

# :page_facing_up: Citation

```bibtex
@article{

}
```
