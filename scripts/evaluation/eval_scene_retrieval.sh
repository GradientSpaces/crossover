export PYTHONWARNINGS="ignore"

# Scene Retrieval Inference
python run_evaluation.py --config-path "$(pwd)/configs/evaluation" --config-name eval_scene.yaml \
task.InferenceSceneRetrieval.val=['Scannet'] \ # Change to Scan3R for evaluation on scan3r
task.InferenceSceneRetrieval.ckpt_path=./checkpoints/instance_crossover_scannet+scan3r.pth \
hydra.run.dir=. hydra.output_subdir=null 