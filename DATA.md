# :arrow_down: Data

## Data Download

### Data Preparation + Processing

We list the available data used in the current version of CrossOver in the table below:

| Dataset Name | Object Modality               | Scene Modality                      | Object Temporal Information | Scene Temporal Information
| ------------ | ----------------------------- | ----------------------------------- |  -------------------------- | -------------------------- |
| Scannet      | `[point, rgb, cad, referral]` | `[point, rgb, floorplan, referral]` |    ❌                       |          ✅                |
| 3RScan       | `[point, rgb, referral]`      | `[point, rgb, referral]`            |    ✅                       |          ✅                |


We detail data download + release required preprocessed data + meta-data and provide instructions for data download & preparation with scripts for ScanNet + 3RScan. 

- For dataset download (data preparation, needed for single inference setup), please look at [README.MD](prepare_data/README.MD) in `data_prepare/` directory.
- For preprocessed data download (training + evaluation only), download our released data by filling the form. details later in [Data Preprocessing](#wrench-data-preprocessing) Section.

> You agree to the terms of ScanNet, 3RScan, ShapeNet, Scan2CAD and SceneVerse datasets by downloading our hosted data.

### Generated Embedding Data
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

## :wrench: Data Preprocessing
In order to process data faster during training + inference, we preprocess 1D (referral), 2D (RGB + floorplan) & 3D (Point Cloud + CAD) for both object instances and scenes. Note that, since for 3RScan dataset, they do not provide frame-wise RGB segmentations, we project the 3D data to 2D and store it in `.pt` format for every scan. We provide the scripts for projection and release the data. Here's an overview which data features are precomputed:

- Object Instance: Referral, Multi-view RGB images, Point Cloud & CAD (only for ScanNet)
- Scene: Referral, Multi-view RGB images, Floorplan (only for ScanNet) Point Cloud 

We release the preprocessed & generated data on GDrive. We also provide the scripts which should be easily cusotmizable for new datasets. Further instructions below.

### ScanNet
Here we refer to the contents of the folder `processed_data/Scannet` on GDrive. The data structure is the following:

```
Scannet/
├── objects_chunked/ (object data chunked into hdf5 format for instance baseline training)
|   ├── train_objects.h5
|   └── val_objects.h5
├── scans/
|   ├── scene0000_00/
|   │   ├── data1D.pt -> all 1D data + encoded (object referrals + BLIP features) 
|   │   ├── data2D.pt -> all 2D data + encoded (RGB + floorplan + DinoV2 features)
|   │   ├── data3D.pt -> all 3D data + encoded (Point Cloud + I2PMAE features - object only)
|   │   ├── object_id_to_label_id_map.pt -> Instance ID to NYU40 Label mapped
|   │   ├── objectsDataMultimodal.pt -> object data combined from data1D.pt + data2D.pt + data3D.pt (for easier loading)
|   │   ├── sel_cams_on_mesh.png (visualisation of the cameras selected for computing RGB features per scan)
|   │   ├── floor+obj.png -> rasterized floorplan (top-down projection of the floor+obj.ply)
|   |   └── floor+obj.ply -> floorplan + CAD mesh
|   └── ...
```

#### Running preprocessing scripts
Adjust the path parameters of `Scannet` in the config files under `configs/preprocess` (remember to adjust the path of `Scannet:shape_dir` to ShapeNet directory). Run the following (after changing the `--config-path` in the bash file):

```bash
$ bash scripts/preprocess/process_scannet.sh
```

### 3RScan
Here we refer to the contents of the folder `processed_data/Scan3R` on GDrive. The data structure is the following:

```
Scan3R/
├── objects_chunked/ (object data chunked into hdf5 format for instance baseline training)
|   ├── train_objects.h5
|   └── val_objects.h5
├── scans/
|   ├── 7f30f36c-42f9-27ed-87c6-23ceb65f1f9b/
|   │   ├── gt-projection-seg.pt -> 3D-to-2D projected data  consisting of framewise 2D instance segmentation
|   │   ├── data1D.pt -> all 1D data + encoded (object referrals + BLIP features) 
|   │   ├── data2D.pt -> all 2D data + encoded (RGB + floorplan + DinoV2 features)
|   │   ├── data2D_all_images.pt (RGB features of every image of every scan -- for comparison with SceneGraphLoc)
|   │   ├── data3D.pt -> all 3D data + encoded (Point Cloud + I2PMAE features - object only)
|   │   ├── object_id_to_label_id_map.pt -> Instance ID to NYU40 Label mapped
|   │   ├── objectsDataMultimodal.pt -> object data combined from data1D.pt + data2D.pt + data3D.pt (for easier loading)
|   │   └── sel_cams_on_mesh.png (visualisation of the cameras selected for computing RGB features per scan)
|   └── ...
```

#### Running preprocessing scripts
Adjust the path parameters of `Scan3R` in the config files under `configs/preprocess`. Run the following (after changing the `--config-path` in the bash file):

```bash
$ bash scripts/preprocess/process_scan3r.sh
```

Our script for 3RScan dataset performs the following additional processing:

- 3D-to-2D projection for 2D segmentation and stores as `gt-projection-seg.pt` for each scan.
- DinoV2 feature computation for every image in every scan in the `val` split. (used only for comparison with SceneGraphLoc) - `computeAllImageFeaturesEachScan()` in [preprocess/feat2D/scan3r.py](preprocess/feat2D/scan3r.py).