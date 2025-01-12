import math

import torch
from torch.optim.lr_scheduler import LambdaLR
from typing import Any


def warmup_cosine(step: int, warmup_step: int, total_step: int, minimum_ratio: float = 1e-5) -> float:
    if step <= warmup_step and warmup_step > 0:
        return step / warmup_step
    return max(
        0.5 * (1 + math.cos((step - warmup_step) / (total_step - warmup_step) * math.pi)),
        minimum_ratio
    )


def warmup_exp(step: int, warmup_step: int, total_step: int, **kwargs: Any) -> float:
    if step <= warmup_step and warmup_step > 0:
        return step / warmup_step
    return kwargs["gamma"] ** (step * 1. / (total_step - warmup_step))


def get_scheduler(cfg: Any, optimizer: torch.optim, total_steps: int) -> LambdaLR:
    warmup_steps = cfg.solver.sched.args.warmup_steps * cfg.num_gpu
    
    def lambda_func(step: int) -> float:
        return warmup_cosine(step, warmup_steps, total_steps)
    
    return LambdaLR(optimizer=optimizer, lr_lambda=lambda_func)