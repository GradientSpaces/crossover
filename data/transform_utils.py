
import torch

def inverse(g: torch.Tensor) -> torch.Tensor:
    """Returns the inverse of an SE3 transform matrix.
    
    Args:
        g: SE3 transform of shape (B, 3/4, 4)
    Returns:
        Inverse transform of shape (B, 3, 4)
    """
    # Compute inverse
    rot = g[..., 0:3, 0:3]
    trans = g[..., 0:3, 3]
    inverse_transform = torch.cat([rot.transpose(-1, -2), rot.transpose(-1, -2) @ -trans[..., None]], dim=-1)

    return inverse_transform

def rotation_error(R1: torch.Tensor, R2: torch.Tensor) -> torch.Tensor:
    """Calculates angular error between estimated and ground truth rotation matrices.
    
    Args:
        R1: Estimated rotation matrices of shape (B, 3, 3)
        R2: Ground truth rotation matrices of shape (B, 3, 3)
    Returns:
        Angular error in degrees of shape (B, 1)
    """
    R_ = torch.matmul(R1.transpose(1,2), R2)
    e = torch.stack([(torch.trace(R_[_, :, :]) - 1) / 2 for _ in range(R_.shape[0])], dim=0).unsqueeze(1)

    # Clamp the errors to the valid range (otherwise torch.acos() is nan)
    e = torch.clamp(e, -1, 1, out=None)

    ae = torch.acos(e)

    ae = 180. * ae / torch.pi

    return ae

def translation_error(t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Calculates L2 error between estimated and ground truth translation vectors.
    
    Args:
        t1: Estimated translation vectors of shape (B, 3, 1)
        t2: Ground truth translation vectors of shape (B, 3, 1)
    Returns:
        Translation error of shape (B, 1)
    """
    return torch.norm(t1-t2, dim=(-2,-1))