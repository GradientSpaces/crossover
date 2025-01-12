import json
import numpy as np
from itertools import combinations
from accelerate import Accelerator
import json
import numpy as np
import torch
from omegaconf import DictConfig
from typing import Dict, Tuple, Any
from fvcore.common.registry import Registry

from common.misc import gather_dict

EVALUATOR_REGISTRY = Registry("EVALUATOR")

def get_eval(name, cfg, accelerator, **kwargs):
    """Get an evaluator or a list of evaluators."""
    if isinstance(name, str):
        eval = EVALUATOR_REGISTRY.get(name)(cfg, accelerator, **kwargs)
    else:
        eval = [EVALUATOR_REGISTRY.get(i)(cfg, accelerator, **kwargs) for i in name]
    return eval

def build_eval(cfg: DictConfig, accelerator: Accelerator, **kwargs) -> Dict[str, Any]:
    """Build train and validation evaluators based on configuration."""
    if cfg.eval.get("train", None) is not None:
        train_eval = get_eval(cfg.eval.train.name, cfg, accelerator, **kwargs)
        val_eval = get_eval(cfg.eval.val.name, cfg, accelerator, **kwargs)
        return {"train": train_eval, "val": val_eval}

class BaseEvaluator:
    def __init__(self, cfg: DictConfig, accelerator: Accelerator) -> None:
        """Initialize the base evaluator with configuration and accelerator."""
        self.accelerator = accelerator
        self.best_result = np.inf
        self.save = cfg.eval.save
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.cfg = cfg
        
        self.modalities = self.cfg.task.get(self.cfg.task.name).modalities
        self.modality_combinations = list(combinations(list(self.modalities), 2))
        self.eval_dict = {}
        
        self.reset() 

    def reset(self) -> None:
        """Reset the evaluation dictionary by clearing all metrics."""
        for key in self.eval_dict.keys():
            self.eval_dict[key] = []
        
    def batch_metrics(self, data_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Calculate metrics for a single batch of data."""
        raise NotImplementedError("Per batch metrics calculation is required for evaluation")

    def update(self, data_dict: Dict[str, torch.Tensor]) -> None:
        """Update evaluation metrics with new batch results."""
        metrics = self.batch_metrics(data_dict)
        for key in metrics.keys():
            if key not in self.eval_dict:
                self.eval_dict[key] = []
            self.eval_dict[key].append(metrics[key])

    def record(self) -> Tuple[bool, Dict[str, float]]:
        """Record and save evaluation results, returning whether current result is best and metrics dict."""
        self.eval_dict = gather_dict(self.accelerator, self.eval_dict)
        for k, metrics in self.eval_dict.items():
            if not isinstance(metrics, list):
                continue
            total_value = sum(metrics)
            total_count = len(metrics)
            self.eval_dict[k] = total_value / max(total_count, 1)

        if self.save and self.accelerator.is_main_process:
            self.best_result = self.eval_dict["target_metric"]
            with (self.save_dir / "results.json").open("w") as f:
                json.dump(self.eval_results, f)
        
        if self.eval_dict["target_metric"] < self.best_result:
            is_best = True
            self.best_result = self.eval_dict["target_metric"]
        else:
            is_best = False
        return is_best, self.eval_dict