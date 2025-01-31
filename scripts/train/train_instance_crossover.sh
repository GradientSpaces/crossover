export PYTHONWARNINGS="ignore"

# Instance CrossOver
python run.py --config-path "$(pwd)/configs/train" --config-name train_instance_crossover.yaml hydra.run.dir=. hydra.output_subdir=null 