from pathlib import Path
import hydra
from datetime import datetime
from omegaconf import OmegaConf, open_dict
import wandb

from common import load_utils
from common.misc import rgetattr
from trainer.build import build_trainer

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: OmegaConf) -> None:
    if cfg.resume:
        assert Path(cfg.exp_dir).exists(), f"Resuming failed: {cfg.exp_dir} does not exist."
        print(f"Resuming from {cfg.exp_dir}")
        cfg = OmegaConf.load(Path(cfg.exp_dir) / 'config.yaml')
        cfg.resume = True
    else:
        run_id = wandb.util.generate_id()
        with open_dict(cfg):
            cfg.logger.run_id = run_id
    
    naming_keys = []
    for name in cfg.get('naming_keywords', []):
        if name == 'time':
           continue
        if name == 'task':
            task_name = cfg.task.name
            naming_keys.append(task_name)
            
            datasets = rgetattr(cfg, f"task.{task_name}.train")
            dataset_names = "+".join([str(x) for x in datasets])
            naming_keys.append(dataset_names)

    exp_name = "_".join(naming_keys)    
    if not cfg.exp_dir:
        cfg.exp_dir = Path(cfg.base_dir) / 'runs' / exp_name / f"{datetime.now().strftime('%Y-%m-%d-%H:%M:%S.%f')}" 
    
    load_utils.make_dir(cfg.exp_dir)
    OmegaConf.save(cfg, cfg.exp_dir / "config.yaml")
    
    trainer = build_trainer(cfg)
    trainer.run()

if __name__ == "__main__":
    import torch
    torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    main()     