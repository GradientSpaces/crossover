export PYTHONWARNINGS="ignore"

# Instance Baseline
python run.py --config-path "$(pwd)/configs/train" --config-name train_instance_baseline.yaml hydra.run.dir=. hydra.output_subdir=null 