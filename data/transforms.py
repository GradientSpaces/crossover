import torch
import numpy as np
import random
from typing import Tuple, Optional, List

def angle_axis(angle: float, axis: np.ndarray) -> torch.Tensor:
    """Compute rotation matrix from angle and axis."""
    u = axis / np.linalg.norm(axis)
    cosval, sinval = np.cos(angle), np.sin(angle)

    # yapf: disable
    cross_prod_mat = np.array([[0.0, -u[2], u[1]],
                                [u[2], 0.0, -u[0]],
                                [-u[1], u[0], 0.0]])

    R = torch.from_numpy(
        cosval * np.eye(3)
        + sinval * cross_prod_mat
        + (1.0 - cosval) * np.outer(u, u)
    )
    # yapf: enable
    return R.float()


class PointcloudScale:
    """Scale pointcloud coordinates by a random factor."""
    def __init__(self, lo: float = 0.8, hi: float = 1.25) -> None:
        self.lo, self.hi = lo, hi

    def __call__(self, points: torch.Tensor, feats: torch.Tensor, 
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        scaler = np.random.uniform(self.lo, self.hi)
        points[:, 0:3] *= scaler
        return points, feats, labels


class PointcloudRotate:
    """Rotate pointcloud around a specified axis by a random angle."""
    def __init__(self, axis: np.ndarray = np.array([0.0, 0.0, 1.0])) -> None:
        self.axis = axis

    def __call__(self, points: torch.Tensor, feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        rotation_angle = np.random.uniform() * 2 * np.pi
        rotation_matrix = angle_axis(rotation_angle, self.axis)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t()), feats, labels
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points, feats, labels


class PointcloudRotatePerturbation:
    """Apply small random rotations to pointcloud."""
    def __init__(self, angle_sigma: float = 0.06, angle_clip: float = 0.18) -> None:
        self.angle_sigma, self.angle_clip = angle_sigma, angle_clip

    def _get_angles(self) -> np.ndarray:
        angles = np.clip(
            self.angle_sigma * np.random.randn(3), -self.angle_clip, self.angle_clip
        )

        return angles

    def __call__(self, points: torch.Tensor, feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        angles = self._get_angles()
        Rx = angle_axis(angles[0], np.array([1.0, 0.0, 0.0]))
        Ry = angle_axis(angles[1], np.array([0.0, 1.0, 0.0]))
        Rz = angle_axis(angles[2], np.array([0.0, 0.0, 1.0]))

        rotation_matrix = torch.matmul(torch.matmul(Rz, Ry), Rx)

        normals = points.size(1) > 3
        if not normals:
            return torch.matmul(points, rotation_matrix.t()), feats, labels
        else:
            pc_xyz = points[:, 0:3]
            pc_normals = points[:, 3:]
            points[:, 0:3] = torch.matmul(pc_xyz, rotation_matrix.t())
            points[:, 3:] = torch.matmul(pc_normals, rotation_matrix.t())

            return points, feats, labels


class PointcloudJitter:
    """Add random jitter to pointcloud coordinates."""
    def __init__(self, std: float = 0.01, clip: float = 0.05) -> None:
        self.std, self.clip = std, clip

    def __call__(self, points: torch.Tensor, feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        jittered_data = (
            points.new(points.size(0), 3)
            .normal_(mean=0.0, std=self.std)
            .clamp_(-self.clip, self.clip)
        )
        points[:, 0:3] += jittered_data
        return points, feats, labels


class PointcloudTranslate:
    """Translate pointcloud by a random offset."""
    def __init__(self, translate_range: float = 0.1) -> None:
        self.translate_range = translate_range

    def __call__(self, points: torch.Tensor, feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        translation = np.random.uniform(-self.translate_range, self.translate_range)
        points[:, 0:3] += translation
        return points, feats, labels


class PointcloudToTensor:
    """Convert numpy arrays to torch tensors."""
    def __call__(self, points: np.ndarray, feats: np.ndarray,
                 labels: Optional[np.ndarray] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        return torch.from_numpy(points).float(), torch.from_numpy(feats).float(), labels


class PointcloudRandomInputDropout:
    """Randomly drop points from pointcloud."""
    def __init__(self, max_dropout_ratio: int = 15) -> None:
        self.max_dropout_ratio = max_dropout_ratio

    def __call__(self, points: torch.Tensor, feats: torch.Tensor,
                 labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        pc = points.numpy()
        feats = feats.numpy()
        dropout_ratio = np.random.randint(5, self.max_dropout_ratio) / 100  # random number between 0.05~0.15
        drop_idx = np.where(np.random.random((pc.shape[0])) <= dropout_ratio)[0]
        if len(drop_idx) > 0:
            pc[drop_idx] = pc[0]  # set to the first point
            feats[drop_idx] = feats[0]
        if labels is not None:
            labels[drop_idx] = labels[0]

        return torch.from_numpy(pc).float(), torch.from_numpy(feats).float(), labels

##########################
# features augmentation 
##########################

class ChromaticTranslation:
    """Add random color translation to features."""
    def __init__(self, trans_range_ratio: float = 1e-1) -> None:
        self.trans_range_ratio = trans_range_ratio

    def __call__(self, coords: np.ndarray, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.95:
            tr = (np.random.rand(1, 3) - 0.5) * 255 * 2 * self.trans_range_ratio
            feats[:, :3] = np.clip(tr + feats[:, :3], 0, 255)
        return coords, feats

class ChromaticAutoContrast:
    """Apply auto contrast to color features."""
    def __init__(self, randomize_blend_factor: bool = True, blend_factor: float = 0.5) -> None:
        self.randomize_blend_factor = randomize_blend_factor
        self.blend_factor = blend_factor

    def __call__(self, coords: np.ndarray, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if random.random() < 0.2:
            lo = np.min(feats, 0, keepdims=True)
            hi = np.max(feats, 0, keepdims=True)

            scale = 255 / (hi - lo)

            contrast_feats = (feats - lo) * scale

            blend_factor = random.random() if self.randomize_blend_factor else self.blend_factor
            feats = (1 - blend_factor) * feats + blend_factor * contrast_feats
        return coords, feats

class HueSaturationTranslation:
    """Modify hue and saturation of color features."""
    def __init__(self, hue_max: float, saturation_max: float) -> None:
        self.hue_max = hue_max
        self.saturation_max = saturation_max

    @staticmethod
    def rgb_to_hsv(rgb: np.ndarray) -> np.ndarray:
        """Convert RGB to HSV color space."""
        rgb = rgb.astype('float')
        hsv = np.zeros_like(rgb)
        hsv[..., 3:] = rgb[..., 3:]
        r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
        maxc = np.max(rgb[..., :3], axis=-1)
        minc = np.min(rgb[..., :3], axis=-1)
        hsv[..., 2] = maxc
        mask = maxc != minc
        hsv[mask, 1] = (maxc - minc)[mask] / maxc[mask]
        rc = np.zeros_like(r)
        gc = np.zeros_like(g)
        bc = np.zeros_like(b)
        rc[mask] = (maxc - r)[mask] / (maxc - minc)[mask]
        gc[mask] = (maxc - g)[mask] / (maxc - minc)[mask]
        bc[mask] = (maxc - b)[mask] / (maxc - minc)[mask]
        hsv[..., 0] = np.select([r == maxc, g == maxc], [bc - gc, 2.0 + rc - bc], default=4.0 + gc - rc)
        hsv[..., 0] = (hsv[..., 0] / 6.0) % 1.0
        return hsv

    @staticmethod
    def hsv_to_rgb(hsv: np.ndarray) -> np.ndarray:
        """Convert HSV to RGB color space."""
        rgb = np.empty_like(hsv)
        rgb[..., 3:] = hsv[..., 3:]
        h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
        i = (h * 6.0).astype('uint8')
        f = (h * 6.0) - i
        p = v * (1.0 - s)
        q = v * (1.0 - s * f)
        t = v * (1.0 - s * (1.0 - f))
        i = i % 6
        conditions = [s == 0.0, i == 1, i == 2, i == 3, i == 4, i == 5]
        rgb[..., 0] = np.select(conditions, [v, q, p, p, t, v], default=v)
        rgb[..., 1] = np.select(conditions, [v, v, v, q, p, p], default=t)
        rgb[..., 2] = np.select(conditions, [v, p, t, v, v, q], default=p)
        return rgb.astype('uint8')

    def __call__(self, coords: np.ndarray, feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        hsv = HueSaturationTranslation.rgb_to_hsv(feats[:, :3])
        hue_val = (random.random() - 0.5) * 2 * self.hue_max
        sat_ratio = 1 + (random.random() - 0.5) * 2 * self.saturation_max
        hsv[..., 0] = np.remainder(hue_val + hsv[..., 0] + 1, 1)
        hsv[..., 1] = np.clip(sat_ratio * hsv[..., 1], 0, 1)
        feats[:, :3] = np.clip(HueSaturationTranslation.hsv_to_rgb(hsv), 0, 255)

        return coords, feats


class Compose:
    """Compose multiple transforms together."""
    def __init__(self, transforms: List) -> None:
        self.transforms = transforms

    def __call__(self, *args) -> Tuple:
        for t in self.transforms:
            args = t(*args)
        return args

def get_transform() -> Compose:
    """Create a default transform pipeline for pointcloud augmentation."""
    tsfm = Compose(
        [
            PointcloudToTensor(),
            PointcloudScale(),
            PointcloudRotate(),
            PointcloudRotatePerturbation(),
            PointcloudTranslate(),
            PointcloudJitter(),
            PointcloudRandomInputDropout()
        ]
    )
    return tsfm
