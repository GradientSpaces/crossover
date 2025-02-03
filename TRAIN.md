# :weight_lifting: Training and Inference

#### Environment Setup
Follow setup instructions from README. 
```bash
$ conda activate crossover
```

#### Train Instance Baseline
Adjust path parameters in `configs/train/train_instance_baseline.yaml` and run the following:

```bash
$ bash scripts/train/train_instance_baseline.sh
```

#### Train Instance Retrieval Pipeline
Adjust path parameters in `configs/train/train_instance_crossover.yaml` and run the following:

```bash
$ bash scripts/train/train_instance_crossover.sh
```

#### Train Scene Retrieval Pipeline
Adjust path/configuration parameters in `configs/train/train_scene_crossover.yaml`. You can also add your customised dataset or choose to train on Scannet & 3RScan or either. Run the following:

```bash
$ bash scripts/train/train_scene_crossover.sh
```

> The scene retrieval pipeline uses the trained weights from instance retrieval pipeline (for object feature calculation), please ensure to update `task:UnifiedTrain:object_enc_ckpt` in the config file.

#### Checkpoint Inventory
We provide all an inventory of all checkpoints on G-Drive [here](https://drive.google.com/drive/folders/1iGhLQY86RTfc87qArOvUtXAhpbFSFq6w?usp=sharing). Detailed descriptions in the table below:

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
$ python single_inference/scene_inference.py
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
$ bash scripts/evaluation/eval_instance_retrieval.sh
```

> This will also show you scene retrieval results using the instance based methods.

#### Cross-Modal Scene Retrieval
Run the following script (for scene crossover). Detailed usage inside the script.

```bash
$ bash scripts/evaluation/eval_instance_retrieval.sh
```