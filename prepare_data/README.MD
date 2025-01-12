# Dataset Preparation

## Overview

This document provides instructions for pre-processing different datasets, including 
- ScanNet
- 3RScan

## Prerequisites

### Environment
Before you begin, simply activate the `crossover` conda environment.

### Download the Original Datasets
- **ScanNet**: Download ScanNet v2 data from the [official website](https://github.com/ScanNet/ScanNet).

- **3RScan**: Download 3RScan dataset from the [official website](https://github.com/WaldJohannaU/3RScan).

- **ShapeNet**: Download Shapenet dataset from the [official website](https://shapenet.org/).

> You only need to download ShapeNet dataset if you are running our preprocessing script(s), training + evaluation directly uses our provided preprocessed data.

### Download Referral and CAD annotations
We use [SceneVerse](https://scene-verse.github.io/) for instance referrals (ScanNet & 3RScan) and [Scan2CAD](https://github.com/skanti/Scan2CAD) for CAD annotations (ScanNet). We currently host all data on Redivis (refer to README.md in the root directory of the repo for access). Exact instructions for data setup below.

#### ScanNet
1. Run the following to extract ScanNet data 
```bash
cd scannet
python preprocess_2d_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
python unzip_scannet.py --scannet_path PATH_TO_SCANNET --output_path PATH_TO_SCANNET
```

2. Download `files/` under `meta_data/ScanNet/` from Redivis and place under `PATH_TO_SCANNET/`.

Once completed, the data structure would look like the following:

```
ScanNet/
├── scans/
│   ├── scene0000_00/
│   │   ├── data/
│   │   │    ├── color/
│   │   |    ├── depth/
|   |   |    ├── instance-filt/
│   │   |    └── pose/
|   |   ├── intrinsics.txt
│   │   ├── scene0000_00_vh_clean_2.ply 
|   |   ├── scene0000_00_vh_clean_2.labels.ply
|   |   ├── scene0000_00_vh_clean_2.0.010000.segs.json
|   |   ├── scene0000_00_vh_clean.aggregation.json
|   |   └── scene0000_00_2d-instance-filt.zip
|   └── ...
└── files
    ├── scannetv2_val.txt
    ├── scannetv2_train.txt
    ├── scannetv2-labels.combined.tsv
    ├── scan2cad_full_annotations.json
    ├── objects.json
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```

For clarity, We provide the script `scannet_objectdata.py` to generate `files/objects.json` for ScanNet dataset. The purpose is to have a unified structure of objects like provided in `3RScan`.

#### 3RScan

1. Download `files/` under `meta_data/Scan3R/` from Redivis and place under `PATH_TO_SCAN3R/`.

2. Run the following to align the re-scans and reference scans in the same coordinate system & unzip `sequence.zip` for every scan:

```bash
cd scan3r
python align_scan.py  (change `root_scan3r_dir` to `PATH_TO_SCAN3R`)
python unzip_scan3r.py --scan3r_path PATH_TO_SCAN3R --output_path PATH_TO_SCAN3R
```

Once completed, the data structure would look like the following:

```
Scan3R/
├── scans/
│   ├── 20c993b5-698f-29c5-85a5-12b8deae78fb/
│   │   ├── sequence/ (folder containing frame-wise color + depth + pose information)
|   |   ├── labels.instances.align.annotated.v2.ply
│   │   └── labels.instances.annotated.v2.ply
|   └── ...
└── files
    ├── 3RScan.json
    ├── 3RScan.v2 Semantic Classes - Mapping.csv
    ├── objects.json
    ├── train_scans.txt
    ├── test_scans.txt
    └── sceneverse  
        └── ssg_ref_rel2_template.json
```