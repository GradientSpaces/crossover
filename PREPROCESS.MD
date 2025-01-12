# Dataset Preprocessing
For faster processing, we pre-compute the 1D, 2D & 3D encoder features. Here's an overview which data features are precomputed:

- Object Instance: Referral, Multi-view RGB images, Point Cloud & CAD (only for ScanNet)
- Scene: Referral, Multi-view RGB images, Floorplan (only for ScanNet) Point Cloud 

We release the preprocssed & generated data on our Redivis. Aside from releasing the data, we also provide the scripts which should be easily cusotmizable for new datasets. Further instructions below.

## ScanNet
Here we refer to the contents of the folder `preprocessed_data/Scannet` on our Redivis. The data structure is the following:

```
Scannet/
├── objects_chunked/ (object data chunked into hdf5 format for instance level baseline training)
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

### Running preprocessing scripts
Adjust the path parameters of `Scannet` in the config files under `configs/preprocess` (remember to adjust the path of `Scannet:shape_dir` to ShapeNet directory). Run the following (after changing the `--config-path` in the bash file):

```bash
bash scripts/preprocess/process_scannet.sh
```

## Scan3R
Here we refer to the contents of the folder `preprocessed_data/Scan3R` on our Redivis. The data structure is the following:

```
Scan3R/
├── objects_chunked/ (object data chunked into hdf5 format for instance level baseline training)
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

### Running preprocessing scripts


Adjust the path parameters of `Scan3R` in the config files under `configs/preprocess`. Run the following (after changing the `--config-path` in the bash file):

```bash
bash scripts/preprocess/process_scan3r.sh
```

Our script for Scan3R dataset performs the following additional processing:

- 3D-to-2D projection for 2D segmentation and stores as `gt-projection-seg.pt` for each scan.
- DinoV2 feature computation for every image in every scan in the `val` split. (used for comparison with SceneGraphLoc) - `computeAllImageFeaturesEachScan()` in `preprocess/feat2d/scan3r.py` 