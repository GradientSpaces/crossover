import numpy as np
import open3d as o3d
from sklearn.neighbors import NearestNeighbors
from shapely.geometry.polygon import Polygon
from typing import Tuple, List

def load_obj(filename: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load vertices and faces from an OBJ file."""
    with open(filename, 'r') as f:
        vertices = []
        faces = []
        for line in f:
            if line.startswith('v '):
                vertex = list(map(float, line.strip().split()[1:]))
                vertices.append(vertex)
            elif line.startswith('f '):
                face = [int(vertex.split('/')[0]) - 1 for vertex in line.strip().split()[1:]]
                faces.append(face)

    v = np.asarray(vertices)
    f = np.asarray(faces)
    assert v.shape[1] == f.shape[1]
    return v, f


def sample_and_normalize_pcl(pcl: np.ndarray, npoint: int = 1024) -> np.ndarray:
    """Sample and normalize a point cloud."""
    pcl_idxs = np.random.choice(len(pcl), size=npoint, replace=(len(pcl) < npoint))
    pcl = pcl[pcl_idxs]
    pcl[:, :3] = pcl[:, :3] - pcl[:, :3].mean(0)
    max_dist = np.max(np.sqrt(np.sum(pcl[:, :3] ** 2, 1)))
    if max_dist < 1e-6:
        max_dist = 1
    pcl[:, :3] = pcl[:, :3] / max_dist

    return pcl


def farthest_sample(point: np.ndarray, npoint: int) -> Tuple[np.ndarray, np.ndarray]:
    """Perform farthest point sampling on a point cloud."""
    N, D = point.shape
    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint)
        point = point[indices]
        return point, indices

    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    point = point[centroids.astype(np.int32)]

    return point, centroids.astype(np.int32)


def get_object_loc_box(object_pcd: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Get the location and bounding box of an object point cloud."""
    object_center = object_pcd[:, :3].mean(0)
    object_size = object_pcd[:, :3].max(0) - object_pcd[:, :3].min(0)
    object_loc = np.concatenate([object_center, object_size], axis=0)

    object_box_center = (object_pcd[:, :3].max(0) + object_pcd[:, :3].min(0)) / 2
    object_box_size = object_pcd[:, :3].max(0) - object_pcd[:, :3].min(0)
    object_box = np.concatenate([object_box_center, object_box_size], axis=0)

    return object_loc, object_box


def normalize(a: np.ndarray, axis: int = -1, order: int = 2) -> np.ndarray:
    """Normalize a tensor along a specific axis."""
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1

    if len(a.shape) == 1:
        return a / l2
    else:
        return a / np.expand_dims(l2, axis)


def get_box_corners(center: np.ndarray, vectors: np.ndarray) -> List[Tuple[float, float, float]]:
    """Convert box center and vectors to corner points."""
    corner_pnts = [None] * 8
    corner_pnts[0] = tuple(center - vectors[0] - vectors[1] - vectors[2])
    corner_pnts[1] = tuple(center + vectors[0] - vectors[1] - vectors[2])
    corner_pnts[2] = tuple(center + vectors[0] + vectors[1] - vectors[2])
    corner_pnts[3] = tuple(center - vectors[0] + vectors[1] - vectors[2])

    corner_pnts[4] = tuple(center - vectors[0] - vectors[1] + vectors[2])
    corner_pnts[5] = tuple(center + vectors[0] - vectors[1] + vectors[2])
    corner_pnts[6] = tuple(center + vectors[0] + vectors[1] + vectors[2])
    corner_pnts[7] = tuple(center - vectors[0] + vectors[1] + vectors[2])

    return corner_pnts


def get_iou_cuboid(cu1: np.ndarray, cu2: np.ndarray) -> float:
    """Calculate the Intersection over Union (IoU) of two 3D cuboids."""
    # 2D projection on the horizontal plane (x-y plane)
    polygon2D_1 = Polygon(
        [(cu1[0][0], cu1[0][1]), (cu1[1][0], cu1[1][1]), (cu1[2][0], cu1[2][1]), (cu1[3][0], cu1[3][1])])

    polygon2D_2 = Polygon(
        [(cu2[0][0], cu2[0][1]), (cu2[1][0], cu2[1][1]), (cu2[2][0], cu2[2][1]), (cu2[3][0], cu2[3][1])])

    # 2D intersection area of the two projections.
    intersect_2D = polygon2D_1.intersection(polygon2D_2).area

    # the volume of the intersection part of cu1 and cu2
    inter_vol = intersect_2D * max(0.0, min(cu1[4][2], cu2[4][2]) - max(cu1[0][2], cu2[0][2]))

    # the volume of cu1 and cu2
    vol1 = polygon2D_1.area * (cu1[4][2] - cu1[0][2])
    vol2 = polygon2D_2.area * (cu2[4][2] - cu2[0][2])

    # return 3D IoU
    return inter_vol / (vol1 + vol2 - inter_vol)


def sample_faces(vertices: np.ndarray, faces: np.ndarray, n_samples: int = 10**4) -> np.ndarray: 
    """
    Sample point cloud on the surface of a mesh.
    https://chrischoy.github.io/research/barycentric-coordinate-for-mesh-sampling/
    """
    vec_cross = np.cross(vertices[faces[:, 0], :] - vertices[faces[:, 2], :],
                         vertices[faces[:, 1], :] - vertices[faces[:, 2], :])
    face_areas = np.sqrt(np.sum(vec_cross ** 2, 1))
    face_areas = face_areas / np.sum(face_areas)

    # Sample exactly n_samples. First, oversample points and remove redundant
    # Error fix by Yangyan (yangyan.lee@gmail.com) 2017-Aug-7
    n_samples_per_face = np.ceil(n_samples * face_areas).astype(int)
    floor_num = np.sum(n_samples_per_face) - n_samples
    if floor_num > 0:
        indices = np.where(n_samples_per_face > 0)[0]
        floor_indices = np.random.choice(indices, floor_num, replace=True)
        n_samples_per_face[floor_indices] -= 1

    n_samples = np.sum(n_samples_per_face)

    # Create a vector that contains the face indices
    sample_face_idx = np.zeros((n_samples,), dtype=int)
    acc = 0
    for face_idx, _n_sample in enumerate(n_samples_per_face):
        sample_face_idx[acc: acc + _n_sample] = face_idx
        acc += _n_sample

    r = np.random.rand(n_samples, 2)
    A = vertices[faces[sample_face_idx, 0], :]
    B = vertices[faces[sample_face_idx, 1], :]
    C = vertices[faces[sample_face_idx, 2], :]
    P = (1 - np.sqrt(r[:, 0:1])) * A + np.sqrt(r[:, 0:1]) * (1 - r[:, 1:]) * B + \
        np.sqrt(r[:, 0:1]) * r[:, 1:] * C
    return P


def transform(points: np.ndarray, align_mat: np.ndarray) -> np.ndarray:
    """Apply a transformation matrix to a set of points."""
    tmp = np.ones((points.shape[0], 4))
    tmp[:, 0:3] = points[:, 0:3]
    tmp = np.dot(tmp, align_mat.transpose())  # Nx4

    points[:, 0:3] = tmp[:, 0:3]
    return points


def is_clockwise(points: List[Tuple[float, float]]) -> bool:
    """Check if a list of 2D points is in clockwise order."""
    assert len(points) > 0
    s = 0.0
    for p1, p2 in zip(points, points[1:] + [points[0]]):
        s += (p2[0] - p1[0]) * (p2[1] + p1[1])
    return s > 0.0


def random_sample(point: np.ndarray, npoint: int) -> np.ndarray:
    """Randomly sample points from a point cloud."""
    N, D = point.shape

    if N < npoint:
        indices = np.random.choice(point.shape[0], npoint, replace=True)
    else:
        indices = np.random.choice(point.shape[0], npoint, replace=False)

    point = point[indices]
    return point


def downsample_with_voxel_grid(points: np.ndarray, colors: np.ndarray, instance_ids: np.ndarray,
                               voxel_size: float = 0.05) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray, np.ndarray]:
    """Downsample a point cloud using a voxel grid."""
    points = points.reshape(-1, 3)
    colors = colors.reshape(-1, 3) / 255.
    instance_ids = instance_ids.reshape(-1, )

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    # Perform voxel downsampling
    voxel_size = 0.05
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    # Extract downsampled points
    downsampled_points = np.asarray(downsampled_point_cloud.points)
    downsampled_colors = np.asarray(downsampled_point_cloud.colors)

    # Assign instance IDs to downsampled points
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(points)
    distances, indices = nbrs.kneighbors(downsampled_points)
    downsampled_instance_ids = instance_ids[indices.flatten()]

    return downsampled_point_cloud, downsampled_points, downsampled_colors, downsampled_instance_ids