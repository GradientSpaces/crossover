export PYTHONWARNINGS="ignore"

# Scene CrossOver
python run.py --config-path "$(pwd)/configs/train" --config-name train_scene_crossover.yaml hydra.run.dir=. hydra.output_subdir=null