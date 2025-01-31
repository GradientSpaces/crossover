from pathlib import Path
import hydra
from datetime import datetime
from omegaconf import OmegaConf

from common import load_utils
from common.misc import rgetattr
from retrieval.build import build_evaluation_module

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg):
    naming_keys = []
    for name in cfg.get('naming_keywords', []):
        if name == 'time':
           continue
        if name == 'task':
            task_name = cfg.task.name
            naming_keys.append(task_name)
            
            datasets = rgetattr(cfg, f"task.{task_name}.val")
            dataset_names = "+".join([str(x) for x in datasets])
            naming_keys.append(dataset_names)

    exp_name = "_".join(naming_keys)    
    if not cfg.exp_dir:
        cfg.exp_dir = Path(cfg.base_dir) / 'runs' / exp_name / f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')}" 
    
    load_utils.make_dir(cfg.exp_dir)
    OmegaConf.save(cfg, cfg.exp_dir / "config.yaml")
    
    inference_module = build_evaluation_module(cfg)
    inference_module.run()

if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()     