import trimesh
import pyrender
import json
import os.path as osp
import subprocess
import numpy as np
import open3d as o3d

def load_and_center_mesh(filename: str) -> trimesh.Trimesh:
    """Load a mesh file and center it at the origin."""
    mesh = trimesh.load(filename)  # load file with trimesh
    center = np.min(mesh.vertices, axis=0)+(np.max(mesh.vertices, axis=0)-np.min(mesh.vertices, axis=0))/2
    mesh.vertices -= center
    
    return mesh

def get_camera_zoom(mesh: trimesh.Trimesh) -> tuple[float, float]:
    """Calculate camera zoom parameters based on mesh dimensions."""
    min_bounds = mesh.vertices.min(axis=0)
    max_bounds = mesh.vertices.max(axis=0)

    # Compute the extents
    x_extent = max_bounds[0] - min_bounds[0]
    y_extent = max_bounds[1] - min_bounds[1]

    # Adjust the extents based on the aspect ratio of the rendering window
    xmag = x_extent * 0.8
    ymag = y_extent * 0.8
    
    return xmag, ymag

def get_camera(xmag: float, ymag: float) -> tuple[pyrender.OrthographicCamera, np.ndarray]:
    """Create a camera with orthographic projection."""
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)

    camera_pose = np.zeros((4,4), dtype=np.float32)
    theta=-90
    Rx = np.identity(3)
    Rx[1,1] = np.cos(np.radians(theta))
    Rx[1,2] = -1*np.sin(np.radians(theta))
    Rx[2,1] = np.sin(np.radians(theta))
    Rx[2,2] = np.cos(np.radians(theta))
    camera_pose[3,3] = 1

    camera_pose[0:3, 0:3] = np.dot(Rx, np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]))
    
    return camera, camera_pose

def get_light() -> tuple[pyrender.DirectionalLight, np.ndarray]:
    """Create a directional light source."""
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2) #intensity is important for the final colors
    # set the pose of the directional light
    light_pose = np.zeros((4,4), dtype=np.float32)
    theta=-90
    Rx = np.identity(3)
    Rx[1,1] = np.cos(np.radians(theta))
    Rx[1,2] = -1*np.sin(np.radians(theta))
    Rx[2,1] = np.sin(np.radians(theta))
    Rx[2,2] = np.cos(np.radians(theta))
    light_pose[3,3] = 1

    # # Correct the camera rotation to look down
    light_pose[0:3, 0:3] = np.dot(Rx, np.array([
        [1, 0, 0],
        [0, 0, -1],
        [0, 1, 0]
    ]))
    
    return light, light_pose

def render_scene(mesh_filename: str) -> np.ndarray:
    """Render a 3D mesh into a 2D image."""
    scene = pyrender.Scene()
    
    mesh = load_and_center_mesh(mesh_filename)
    xmag, ymag = get_camera_zoom(mesh)
    
    mesh = pyrender.Mesh.from_trimesh(mesh, smooth = False) 
    scene.add(mesh) #adding the mesh
    
    camera, camera_pose = get_camera(xmag, ymag)
    scene.add(camera, pose=camera_pose) # adding the camera with the computed camera pose
    
    light, light_pose = get_light()
    scene.add(light, pose=light_pose) # adding the directional light with the computed pose       

    r = pyrender.OffscreenRenderer(800, 400, 224)
    color, depth = r.render(scene) # it gives a tuple as output, where one is an RGB image and one is a depth image
    
    return color

def project_mesh3DTo2D_with_objectseg(
    scene: o3d.t.geometry.RaycastingScene,
    intrinsics: np.ndarray,
    extrinsics: np.ndarray,
    width: int,
    height: int,
    mesh_triangles: np.ndarray,
    num_triangles: int,
    obj_ids: np.ndarray
) -> np.ndarray:
    """Project 3D mesh onto 2D plane with object segmentation."""
    rays = o3d.t.geometry.RaycastingScene.create_rays_pinhole(
            intrinsic_matrix = intrinsics.astype(np.float64),
            extrinsic_matrix = extrinsics.astype(np.float64),
            width_px = width, height_px = height
        )
    
    ans = scene.cast_rays(rays)
    hit_triangles_ids = ans['primitive_ids'].numpy()
    hit_triangles_ids_valid_masks = (hit_triangles_ids<num_triangles)
    hit_triangles_ids_valid = hit_triangles_ids[hit_triangles_ids_valid_masks]
    hit_triangles_valid = mesh_triangles[hit_triangles_ids_valid]
    hit_points_ids_valid = hit_triangles_valid[:, 0]
    
    obj_id_map = np.zeros((height,width), dtype=np.uint16)
    obj_id_map[hit_triangles_ids_valid_masks] = obj_ids[hit_points_ids_valid]
     
    return obj_id_map