"""Utilities for handling SE(3) transformations, quaternions, and bounding boxes."""
from typing import Tuple, Dict, Union, Optional
import numpy as np
import quaternion

def convert_quat_to_rot_mat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion to rotation matrix."""
    q = np.quaternion(q[0], q[1], q[2], q[3])
    rot_mat = quaternion.as_rotation_matrix(q)
    return rot_mat

def make_M_from_tqs(t: np.ndarray, q: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Create transformation matrix from translation, quaternion rotation, and scale."""
    q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)
    M = T.dot(R).dot(S)
    
    return M 

def calc_bbox(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate axis-aligned bounding box from points."""
    min_coords = np.min(points, axis = 0)
    max_coords = np.max(points, axis = 0)
    return min_coords, max_coords

def calc_Mbbox(model: Dict[str, Union[Dict, np.ndarray]]) -> np.ndarray:
    """Calculate transformation matrix for model's bounding box."""
    trs_obj = model["trs"]
    bbox_obj = np.asarray(model["bbox"], dtype=np.float64)
    center_obj = np.asarray(model["center"], dtype=np.float64)
    trans_obj = np.asarray(trs_obj["translation"], dtype=np.float64)
    rot_obj = np.asarray(trs_obj["rotation"], dtype=np.float64)
    q_obj = np.quaternion(rot_obj[0], rot_obj[1], rot_obj[2], rot_obj[3])
    scale_obj = np.asarray(trs_obj["scale"], dtype=np.float64)

    tcenter1 = np.eye(4)
    tcenter1[0:3, 3] = center_obj
    trans1 = np.eye(4)
    trans1[0:3, 3] = trans_obj
    rot1 = np.eye(4)
    rot1[0:3, 0:3] = quaternion.as_rotation_matrix(q_obj)
    scale1 = np.eye(4)
    scale1[0:3, 0:3] = np.diag(scale_obj)
    bbox1 = np.eye(4)
    bbox1[0:3, 0:3] = np.diag(bbox_obj)
    M = trans1.dot(rot1).dot(scale1).dot(tcenter1).dot(bbox1)
    return M

def compose_mat4(
    t: np.ndarray, 
    q: Union[np.ndarray, np.quaternion], 
    s: np.ndarray, 
    center: Optional[np.ndarray] = None
) -> np.ndarray:
    """Compose 4x4 transformation matrix from translation, rotation, scale, and optional center."""
    if not isinstance(q, np.quaternion):
        q = np.quaternion(q[0], q[1], q[2], q[3])
    T = np.eye(4)
    T[0:3, 3] = t
    R = np.eye(4)
    R[0:3, 0:3] = quaternion.as_rotation_matrix(q)
    S = np.eye(4)
    S[0:3, 0:3] = np.diag(s)

    C = np.eye(4)
    if center is not None:
        C[0:3, 3] = center

    M = T.dot(R).dot(S).dot(C)
    return M 

def decompose_mat4(M: np.ndarray) -> Tuple[np.ndarray, np.quaternion, np.ndarray]:
    """Decompose 4x4 transformation matrix into translation, quaternion rotation, and scale."""
    R = M[0:3, 0:3].copy()
    sx = np.linalg.norm(R[0:3, 0])
    sy = np.linalg.norm(R[0:3, 1])
    sz = np.linalg.norm(R[0:3, 2])

    s = np.array([sx, sy, sz])

    R[:,0] /= sx
    R[:,1] /= sy
    R[:,2] /= sz

    q = quaternion.from_rotation_matrix(R[0:3, 0:3])

    t = M[0:3, 3]
    return t, q, s