import torch
from torch import Tensor

def pad_tensors(tensors: Tensor, lens: int = 10, pad: int = 0) -> Tensor:
    """Pads a tensor with given value to reach specified length along first dimension."""
    assert tensors.shape[0] <= lens
    if tensors.shape[0] == lens:
        return tensors
    shape = list(tensors.shape)
    shape[0] = lens - shape[0]
    res = torch.ones(shape, dtype=tensors.dtype) * pad
    res = torch.cat((tensors, res), dim=0)
    return res