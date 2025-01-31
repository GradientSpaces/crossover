"""
Visualization utilities for 3D camera poses and point clouds using Open3D.
Provides functions to create and manipulate camera visualization elements.
"""

import numpy as np
import os
from typing import List, Dict, Optional
from scipy.spatial.transform import Rotation as R
import open3d as o3d
from PIL import Image

CAM_POINTS = np.array(
    [
        [0, 0, 0],
        [-1, -1, 1.5],
        [1, -1, 1.5],
        [1, 1, 1.5],
        [-1, 1, 1.5],
        [-0.5, 1, 1.5],
        [0.5, 1, 1.5],
        [0, 1.2, 1.5],
    ]
)

CAM_LINES = np.array(
    [[1, 2], [2, 3], [3, 4], [4, 1], [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6]]
)


def create_origin_and_axis() -> o3d.geometry.LineSet:
    """Creates coordinate axes visualization in red (X), green (Y), and blue (Z)."""
    x_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[0, 0, 0], [1, 0, 0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    color = (1.0, 0.0, 0.0)
    x_axis.paint_uniform_color(color)

    y_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[0, 0, 0], [0, 1, 0]]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    color = (0.0, 1.0, 0.0)
    y_axis.paint_uniform_color(color)

    z_axis = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 1]]),
        lines=o3d.utility.Vector2iVector([[0, 1]]),
    )
    color = (0.0, 0.0, 1.0)
    z_axis.paint_uniform_color(color)
    return x_axis + y_axis + z_axis


def create_camera_actor(g: bool, scale: float = 0.05) -> o3d.geometry.LineSet:
    """Creates a camera wireframe visualization with specified scale."""
    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(scale * CAM_POINTS),
        lines=o3d.utility.Vector2iVector(CAM_LINES),
    )

    color = [1, 0, 0] #(g * 1.0, 0.5 * (1 - g), 0.9 * (1 - g))
    camera_actor.paint_uniform_color(color)
    return camera_actor

def visualise_camera_on_mesh(
    mesh: o3d.geometry.TriangleMesh,
    pose_data: List[List[float]],
    intrinsics: Dict[str, float],
    stride: int = 5,
    depth_scale: float = 0.15
) -> np.ndarray:
    """Visualizes camera poses on a mesh and returns the rendered image."""
    img_stride = 16
    geometries = []
    for frame_idx in range(0, len(pose_data), stride):
        u, v = np.meshgrid(
                range(intrinsics["w"]),
                range(intrinsics["h"]),
            )
        u = u[::img_stride, ::img_stride].reshape(-1)
        v = v[::img_stride, ::img_stride].reshape(-1)
        cx = intrinsics["cx"]
        cy = intrinsics["cy"]
        fx = intrinsics["f"]
        fy = intrinsics["f"]
        
        depth = np.ones((intrinsics['h'], intrinsics['w'])) * depth_scale
        depth = depth[::img_stride,::img_stride].reshape(-1)
        
        points_c = np.vstack(((u-cx)*depth/fx, (v-cy)*depth/fy, depth, np.ones(depth.shape)))
        points_c = points_c
        points_c = points_c[:,depth>0]
        
        # load poses
        line = pose_data[frame_idx]
        quat_xyzw = []
        for i in range(3, 7):
            quat_xyzw.append(float(line[i]))

        position = []
        for i in range(0, 3):
            position.append(float(line[i]))

        cam_actor = create_camera_actor(True,scale = 0.1)
        pose_test = np.eye(4)
        pose_test[:3, 3] = np.array(position)
        pose_test[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
        
        # Create point cloud for the frame
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector((pose_test @ points_c)[:3].transpose())
        pcd.paint_uniform_color([0, 0, 1])
        
        cam_actor.transform(pose_test)

        geometries.append(pcd)
        geometries.append(cam_actor)
    
    geometries.append(mesh)
    
    # Initialize Open3D visualizer and add all geometries
    vis = o3d.visualization.Visualizer()
    vis.create_window(height=540, width=960)
    
    for geom in geometries:
        vis.add_geometry(geom)

    image_np_array = np.asarray(vis.capture_screen_float_buffer(do_render=True))
    vis.destroy_window()
    
    return image_np_array

def o3d_visualization(
    pose_data: List[List[float]],
    intrinsics: Dict[str, float],
    stride: int = 5,
    pc: Optional[o3d.geometry.TriangleMesh] = None,
    focus_frame_idxs: Optional[List[int]] = None
) -> None:
    """Creates an interactive visualization of camera poses with keyboard controls."""
    o3d_visualization.pose_data = pose_data
    o3d_visualization.intrinsics = intrinsics
    o3d_visualization.depth_scale = 0.15
    o3d_visualization.current_frame = 0
    o3d_visualization.stride = stride
    o3d_visualization.last_added_frame = -1
    o3d_visualization.last_pose = None
    o3d_visualization.pc_mesh = pc
    o3d_visualization.last_pcd = None
    o3d_visualization.last_camera = None

    def next_frame(vis: o3d.visualization.Visualizer) -> bool:
        """Advances to the next frame when 'A' key is pressed."""
        o3d_visualization.current_frame += o3d_visualization.stride

    def animation_callback(vis: o3d.visualization.Visualizer) -> bool:
        """Updates the visualization for each animation frame."""
        cam = vis.get_view_control().convert_to_pinhole_camera_parameters()
        if o3d_visualization.current_frame > o3d_visualization.last_added_frame:
            img_stride = 16
            u, v = np.meshgrid(
                range(o3d_visualization.intrinsics["w"]),
                range(o3d_visualization.intrinsics["h"]),
            )
            u = u[::img_stride, ::img_stride].reshape(-1)
            v = v[::img_stride, ::img_stride].reshape(-1)
            cx = o3d_visualization.intrinsics["cx"]
            cy = o3d_visualization.intrinsics["cy"]
            fx = o3d_visualization.intrinsics["f"]
            fy = o3d_visualization.intrinsics["f"]

            
            depth = np.ones((intrinsics['h'], intrinsics['w']))*o3d_visualization.depth_scale
            depth = depth[::img_stride,::img_stride].reshape(-1)
            
            points_c = np.vstack(((u-cx)*depth/fx, (v-cy)*depth/fy, depth, np.ones(depth.shape)))
            points_c = points_c
            points_c = points_c[:,depth>0]
            
            # load poses
            line = o3d_visualization.pose_data[o3d_visualization.current_frame]
            quat_xyzw = []
            for i in range(3, 7):
                quat_xyzw.append(float(line[i]))

            position = []
            for i in range(0, 3):
                position.append(float(line[i]))

            cam_actor = create_camera_actor(True,scale = 0.1)
            pose_test = np.eye(4)
            pose_test[:3, 3] = np.array(position)
            pose_test[:3, :3] = R.from_quat(quat_xyzw).as_matrix()
            

            if o3d_visualization.last_pose is not None:
                if np.sum(o3d_visualization.last_pose==np.array(position)) == 3:
                    return
            o3d_visualization.last_pose = np.array(position)

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector((pose_test@points_c)[:3].transpose())
            
            if o3d_visualization.current_frame in focus_frame_idxs:
                pcd.paint_uniform_color([0, 0, 1])

            
            if o3d_visualization.last_pcd is not None:
                vis.remove_geometry(o3d_visualization.last_pcd)
            if o3d_visualization.last_camera is not None:
                vis.remove_geometry(o3d_visualization.last_camera)

            cam_actor.transform(pose_test)
            if o3d_visualization.current_frame > -1:
                vis.add_geometry(cam_actor)
                o3d_visualization.last_camera = cam_actor

            if o3d_visualization.current_frame > -1:
                vis.add_geometry(pcd)
                o3d_visualization.last_pcd = pcd

            if o3d_visualization.current_frame > 0:
                cam = vis.get_view_control().convert_from_pinhole_camera_parameters(cam)
            else:
                if o3d_visualization.pc_mesh is not None:
                    vis.add_geometry(o3d_visualization.pc_mesh)
            
            
            # Capture the current frame and save it as an image
            image_path = os.path.join('test', f"frame_{o3d_visualization.current_frame:03}.png")
            image_np_array = np.asarray(vis.capture_screen_float_buffer(do_render=True))
            Image.fromarray((image_np_array * 255).astype(np.uint8)).save(image_path)

            vis.poll_events()
            vis.update_renderer()

            o3d_visualization.last_added_frame = o3d_visualization.current_frame

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)
    vis.register_key_callback(ord("A"), next_frame)

    vis.create_window(height=540, width=960)
    vis.run()
    vis.destroy_window()