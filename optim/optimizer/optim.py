from omegaconf import DictConfig
from common.type_utils import cfg2dict

from typing import List
import torch
import torch.optim as optim
from fvcore.common.registry import Registry

OPTIM_REGISTRY = Registry("loss")

def get_optimizer(cfg: DictConfig, params: List[torch.tensor]):
  if getattr(optim, cfg.solver.optim.name, None) is not None:
    optimizer = getattr(optim, cfg.solver.optim.name)(params, **cfg2dict(cfg.solver.optim.args))
  else:
      raise NotImplementedError
  return optimizer