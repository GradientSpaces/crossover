export PYTHONWARNINGS="ignore"

# Instance Baseline
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" \
--config-name eval_instance.yaml \
task.InferenceObjectRetrieval.val = ['Scannet'] \ # Change to Scan3R for evaluation on scan3r
task.InferenceObjectRetrieval.ckpt_path = ./checkpoints/instance_baseline_scannet+scan3r.pth \
model.name=ObjectLevelEncoder \
hydra.run.dir=. hydra.output_subdir=null 

# Instance CrossOver
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" \
--config-name eval_instance.yaml \
task.InferenceObjectRetrieval.val=['Scannet'] \ # Change to Scan3R for evaluation on scan3r
task.InferenceObjectRetrieval.ckpt_path=./checkpoints/instance_crossover_scannet+scan3r.pth \
model.name=SceneLevelEncoder \
hydra.run.dir=. hydra.output_subdir=null 