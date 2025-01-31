import torch
import torch.nn.functional as F
import numpy as np
from scipy.spatial import distance
from PIL import Image
from typing import Union

# openmask3d multi-level functions
def mask2box(mask: torch.Tensor) -> Union[tuple[int, int, int, int], None]:
    """Computes the bounding box of a binary mask."""
    row = torch.nonzero(mask.sum(axis=0))[:, 0]
    if len(row) == 0:
        return None
    x1 = row.min().item()
    x2 = row.max().item()
    col = np.nonzero(mask.sum(axis=1))[:, 0]
    y1 = col.min().item()
    y2 = col.max().item()
    return x1, y1, x2 + 1, y2 + 1

def mask2box_multi_level(mask: torch.Tensor, level: int, expansion_ratio: float = 0.2) -> tuple[int, int, int, int]:
    """Computes a multi-level bounding box of a binary mask with an expansion ratio."""
    x1, y1, x2 , y2  = mask2box(mask)
    if level == 0:
        return x1, y1, x2 , y2
    shape = mask.shape
    x_exp = int(abs(x2- x1)*expansion_ratio) * level
    y_exp = int(abs(y2-y1)*expansion_ratio) * level
    return max(0, x1 - x_exp), max(0, y1 - y_exp), min(shape[1], x2 + x_exp), min(shape[0], y2 + y_exp)

def generate_uniform_grid(num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Generates a uniform grid of points on a sphere."""
    indices = np.arange(0, num_points, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/num_points)
    theta = np.pi * (1 + 5**0.5) * indices
    return theta, phi

def spherical_distance(theta1: float, phi1: float, theta2: float, phi2: float) -> float:
    """Computes the spherical distance between two points on the sphere."""
    cos_dist = (np.sin(phi1) * np.sin(phi2) +
                np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2))
    return np.arccos(np.clip(cos_dist, -1.0, 1.0))

def quaternion_to_spherical(quaternion: list[float]) -> tuple[float, float]:
    """Converts a quaternion to spherical coordinates (azimuth, elevation)."""
    x, y, z, w = quaternion
    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x**2 + y**2))
    return azimuth, elevation

def sample_camera_pos_on_grid(pose_data: np.ndarray, num_to_sample: int = 20) -> list[int]:
    """Samples camera positions on a grid based on pose data."""
    min_vals = np.min(pose_data, axis=0)
    max_vals = np.max(pose_data, axis=0)
    pose_data_normalised = (pose_data - min_vals) / (max_vals - min_vals)
    
    initial_index = np.random.randint(len(pose_data_normalised))
    sampled_indices = [initial_index]
    distances = distance.cdist([pose_data_normalised[initial_index]], pose_data_normalised, 'euclidean').flatten()
    
    for _ in range(num_to_sample - 1):
        # Find the index of the point that is farthest from the currently sampled points
        farthest_index = np.argmax(distances)
        sampled_indices.append(farthest_index)
        
        # Update the distances array with the minimum distance to any sampled point
        new_distances = distance.cdist([pose_data_normalised[farthest_index]], pose_data_normalised, 'euclidean').flatten()
        distances = np.minimum(distances, new_distances)
    
    return sampled_indices

def crop_image(img_np: np.ndarray, out_path: str, pad: int = 50) -> Image.Image:
    """Crops an image to the bounding box of non-white pixels and saves it."""
    # Mask of non-white pixels (we assume white to be [255, 255, 255])
    mask = np.all(img_np != [255, 255, 255], axis=-1)
    
    # Find the bounding box of the non-white pixels
    coords = np.argwhere(mask)
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0) + 1  # slices are exclusive at the top

    # Crop the image to the bounding box
    image = Image.fromarray(img_np)
    cropped_image = image.crop((x_min - pad, y_min - pad, x_max + pad, y_max + pad))
    cropped_image.save(out_path)
    
    return cropped_image

def process_image(image: Image.Image, stride: int, transforms: callable, device: str = 'cuda') -> torch.Tensor:
    """Processes an image by resizing it to the nearest multiple of the stride."""
    image_tensor = transforms(image).float().to(device)
    h, w = image_tensor.shape[1:]

    height_int = (h // stride)*stride
    width_int = (w // stride)*stride
    
    image_resized = F.interpolate(image_tensor.unsqueeze(0), size=(height_int, width_int), mode='bicubic')
    return image_resized