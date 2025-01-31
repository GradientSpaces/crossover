import numpy as np
import json
import os.path as osp
from glob import glob
import csv
import open3d as o3d
from typing import Tuple, Dict, List, Union
from plyfile import PlyData

from common import load_utils, misc
from util import se3, point_cloud
from util.geometry import line_mesh


def read_label_map(metadata_dir: str, label_from: str = 'raw_category', label_to: str = 'nyu40class') -> Dict[Union[str, int], str]:
    """Load the label map from metadata."""
    LABEL_MAP_FILE = osp.join(metadata_dir, 'scannetv2-labels.combined.tsv')
    assert osp.exists(LABEL_MAP_FILE)
    
    raw_label_map = read_label_mapping(LABEL_MAP_FILE, label_from=label_from, label_to=label_to)
    return raw_label_map

def read_label_mapping(filename: str, label_from: str = 'raw_category', label_to: str = 'nyu40class') -> Dict[Union[str, int], str]:
    """Parse label mappings from a file."""
    assert osp.isfile(filename)
    mapping = dict()
    with open(filename) as csvfile:
        reader = csv.DictReader(csvfile, delimiter='\t')
        for row in reader:
            mapping[row[label_from]] = row[label_to]
    
    if misc.represents_int(list(mapping.keys())[0]):
        mapping = {int(k):v for k,v in mapping.items()}
    
    return mapping

def read_mesh_vertices_rgb(filename: str) -> np.ndarray:
    """Read XYZ and RGB data for each mesh vertex.
    Note: RGB values are in 0-255
    """
    assert osp.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']
    return vertices

def read_aggregation(filename: str) -> Tuple[Dict[int, List[int]], Dict[str, List[int]]]:
    """Read aggregation data from a file."""
    assert osp.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1  # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    
    return object_id_to_segs, label_to_segs

def read_segmentation(filename: str) -> Tuple[Dict[int, List[int]], int]:
    """Parse segmentation data from a file."""
    assert osp.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts

def export(mesh_file: str, agg_file: str, seg_file: str, meta_file: str, label_map: Dict[str, int], axis_alignment: bool = True, output_file: str = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, int], np.ndarray]:
    """Export processed mesh and label data.
    points are XYZ RGB (RGB in 0-255),
    semantic label as nyu40 ids,
    instance label as 1-#instance,
    box as (cx,cy,cz,dx,dy,dz,semantic_label)
    """
    mesh_vertices = read_mesh_vertices_rgb(mesh_file)

    # Load scene axis alignment matrix
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    pts = np.ones((mesh_vertices.shape[0], 4))
    pts[:, 0:3] = mesh_vertices[:, 0:3]
    if axis_alignment : 
        pts = np.dot(pts, axis_align_matrix.transpose())  # Nx4
    mesh_vertices[:, 0:3] = pts[:, 0:3]

    # Load semantic and instance labels
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    object_id_to_label_id = {}

    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
    num_instances = len(np.unique(list(object_id_to_segs.keys())))
    for object_id, segs in object_id_to_segs.items():
        for seg in segs:
            verts = seg_to_verts[seg]
            instance_ids[verts] = object_id
            if object_id not in object_id_to_label_id:
                object_id_to_label_id[object_id] = label_ids[verts][0]
    instance_bboxes = np.zeros((num_instances, 7))
    
    for obj_id in object_id_to_segs:
        label_id = object_id_to_label_id[obj_id]
        obj_pc = mesh_vertices[instance_ids == obj_id, 0:3]
        if len(obj_pc) == 0: 
            continue
        # Compute axis aligned box -- an axis aligned bounding box is parameterized by (cx,cy,cz) and (dx,dy,dz) and label id
        # where (cx,cy,cz) is the center point of the box, and dx is the x-axis length of the box.
        xmin = np.min(obj_pc[:, 0])
        ymin = np.min(obj_pc[:, 1])
        zmin = np.min(obj_pc[:, 2])
        xmax = np.max(obj_pc[:, 0])
        ymax = np.max(obj_pc[:, 1])
        zmax = np.max(obj_pc[:, 2])
        bbox = np.array([(xmin + xmax) / 2, (ymin + ymax) / 2, (zmin + zmax) / 2,
                         xmax - xmin, ymax - ymin, zmax - zmin, label_id])
        # NOTE: this assumes obj_id is in 1,2,3,.,,,.NUM_INSTANCES
        instance_bboxes[obj_id - 1, :] = bbox

    if output_file is not None:
        np.save(output_file + '_vert.npy', mesh_vertices)
        np.save(output_file + '_sem_label.npy', label_ids)
        np.save(output_file + '_ins_label.npy', instance_ids)
        np.save(output_file + '_bbox.npy', instance_bboxes)

    return mesh_vertices, label_ids, instance_ids, \
           instance_bboxes, object_id_to_label_id, axis_align_matrix

def load_intrinsics(data_dir: str, scan_id: str) -> Dict[str, Union[float, np.ndarray]]:
    """Load camera intrinsics for a scan."""
    intrinsic_width, intrinsic_height = 640, 480
    color_intrinsic_path = osp.join(data_dir, scan_id, 'intrinsics.txt')
    intrinsic_mat = np.genfromtxt(color_intrinsic_path)[:3, :3]
    
    intrinsics = {'width' : float(intrinsic_width), 'height' : float(intrinsic_height), 
                  'intrinsic_mat' : intrinsic_mat}
    
    return intrinsics

def load_poses(data_dir: str, scan_id: str, skip: int) -> Dict[int, np.ndarray]:
    """Load camera poses from files."""
    num_frames = len(glob(osp.join(data_dir, scan_id, 'data/pose', '*.txt')))
    
    poses = {}
    for index in range(0, num_frames, skip):
        pose_path = osp.join(data_dir, scan_id, 'data/pose', '{}.txt'.format(index))
        pose = np.genfromtxt(pose_path)
        
        if np.any(np.isnan(pose)) or np.any(np.isinf(pose)) or np.any(np.isneginf(pose)):
            continue
        
        poses[index] = pose
        
    return poses

def get_scan_ids(dirname: str, split: str) -> np.ndarray:
    """Retrieve scan IDs for a given split."""
    filepath = osp.join(dirname, 'scannetv2_{}.txt'.format(split))
    scan_ids = np.genfromtxt(filepath, dtype = str)
    return scan_ids

def get_cad_model_to_instance_mapping(instance_bboxes: np.ndarray, scan2cad_annotation: Dict, meta_file: str, shapenet_dir: str) -> Dict[int, Dict]:
    """
    Map CAD models to their corresponding instances.
    From: https://github.com/GAP-LAB-CUHK-SZ/RfDNet/blob/main/utils/scannet/gen_scannet_w_orientation.py
    """
    
    lines = open(meta_file).readlines()
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
            break
    axis_align_matrix = np.array(axis_align_matrix).reshape((4, 4))
    Mscan = se3.make_M_from_tqs(scan2cad_annotation["trs"]["translation"], scan2cad_annotation["trs"]["rotation"], scan2cad_annotation["trs"]["scale"])
    R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))
    
    '''preprocess boxes'''
    shapenet_instances = dict()
    for model in scan2cad_annotation['aligned_models']:
        # read corresponding shapenet scanned points
        catid_cad = model["catid_cad"]
        id_cad = model["id_cad"]
        obj_path = osp.join(shapenet_dir, catid_cad, id_cad + '/models/model_normalized.obj')
        assert osp.exists(obj_path)
        
        obj_verts, obj_faces = point_cloud.load_obj(obj_path)
        '''transform shapenet obj to scannet'''
        
        t = model["trs"]["translation"]
        q =  model["trs"]["rotation"]
        s = model["trs"]["scale"]
        sym = model['sym']
        
        Mcad = se3.make_M_from_tqs(t, q, s)
        transform_shape = R_transform.dot(Mcad)
        
        '''get transformed axes'''
        center = (obj_verts.max(0) + obj_verts.min(0)) / 2.
        axis_points = np.array([center,
                                center - np.array([0, 0, 1]),
                                center - np.array([1, 0, 0]),
                                center + np.array([0, 1, 0])])

        axis_points_transformed = np.hstack([axis_points, np.ones((axis_points.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        center_transformed = axis_points_transformed[0]
        forward_transformed = axis_points_transformed[1] - axis_points_transformed[0]
        left_transformed = axis_points_transformed[2] - axis_points_transformed[0]
        up_transformed = axis_points_transformed[3] - axis_points_transformed[0]
        forward_transformed = point_cloud.normalize(forward_transformed)
        left_transformed = point_cloud.normalize(left_transformed)
        up_transformed = point_cloud.normalize(up_transformed)
        axis_transformed = np.array([forward_transformed, left_transformed, up_transformed])
        '''get rectified axis'''
        axis_rectified = np.zeros_like(axis_transformed)
        up_rectified_id = np.argmax(axis_transformed[:, 2])
        forward_rectified_id = 0 if up_rectified_id != 0 else (up_rectified_id + 1) % 3
        left_rectified_id = np.setdiff1d([0, 1, 2], [up_rectified_id, forward_rectified_id])[0]
        up_rectified = np.array([0, 0, 1])
        forward_rectified = axis_transformed[forward_rectified_id]
        forward_rectified = np.array([*forward_rectified[:2], 0.])
        forward_rectified = point_cloud.normalize(forward_rectified)
        left_rectified = np.cross(up_rectified, forward_rectified)
        axis_rectified[forward_rectified_id] = forward_rectified
        axis_rectified[left_rectified_id] = left_rectified
        axis_rectified[up_rectified_id] = up_rectified
        if np.linalg.det(axis_rectified) < 0: 
            axis_rectified[left_rectified_id] *= -1
        '''deploy points'''
        obj_verts_aligned = np.hstack([obj_verts, np.ones((obj_verts.shape[0], 1))]).dot(transform_shape.T)[..., :3]
        coordinates = (obj_verts_aligned - center_transformed).dot(axis_transformed.T)
        # obj_points = coordinates.dot(axis_rectified) + center_transformed
        '''define bounding boxes'''
        # [center, edge size, orientation]
        sizes = (coordinates.max(0) - coordinates.min(0))
        box3D = np.hstack([center_transformed, sizes[[forward_rectified_id, left_rectified_id, up_rectified_id]],
                             np.array([np.arctan2(forward_rectified[1], forward_rectified[0])])])
        
        '''to get instance id'''
        axis_rectified = np.array([[np.cos(box3D[6]), np.sin(box3D[6]), 0], [-np.sin(box3D[6]), np.cos(box3D[6]), 0], [0, 0, 1]])
        vectors = np.diag(box3D[3:6]/2.).dot(axis_rectified)
        scan2cad_corners = np.array(point_cloud.get_box_corners(box3D[:3], vectors))

        best_iou_score = 0.
        best_instance_id = 0 # means background points
        
        for inst_id, instance_bbox in enumerate(instance_bboxes):
            center = instance_bbox[:3]
            vectors = np.diag(instance_bbox[3:6]) / 2.
            scannet_corners = np.array(point_cloud.get_box_corners(center, vectors))
            iou_score = point_cloud.get_iou_cuboid(scan2cad_corners, scannet_corners)

            if iou_score > best_iou_score:
                best_iou_score = iou_score
                best_instance_id = inst_id + 1
        
        if best_instance_id != 0:
            obj_points = point_cloud.sample_faces(obj_verts, obj_faces, n_samples=5000)
            shapenet_instances[best_instance_id] = { 't' : t, 'q' : q, 's' : s, 'Mscan' : Mscan,
                                                    'verts': obj_verts, 'faces': obj_faces,
                                                    'shapenet_catid': catid_cad, 'shapenet_id': id_cad, 'points' : obj_points, 
                                                    'sym' : sym,
                                                    }

            R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))
            Mcad = se3.make_M_from_tqs(t, q, s)
            transform_shape = R_transform.dot(Mcad)
            
            shapenet_instances[best_instance_id]['transform_shape'] = transform_shape
            
    return shapenet_instances

def visualiseShapeAnnotation(shape_dir: str, scene_mesh: o3d.geometry.TriangleMesh, shape_annot_to_instance_map: Dict[int, Dict], Mscan: np.ndarray, axis_align_matrix: np.ndarray) -> None:
    """Visualize shape annotations within a scene."""
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1280, height=840, left=0, top=0, visible=True)
    vis.get_render_option().light_on = False
    vis.get_render_option().line_width = 100.0 
    
    scan_layout_and_object_mesh = o3d.geometry.TriangleMesh()
    
    scene_mesh.transform(axis_align_matrix)
    vis.add_geometry(scene_mesh.sample_points_uniformly(number_of_points=1000000))
    
    scan_layout_and_object_mesh += scene_mesh
    
    R_transform = np.array(axis_align_matrix).reshape((4, 4)).dot(np.linalg.inv(Mscan))
    
    for instance_id in shape_annot_to_instance_map:
        shape_annot_instance = shape_annot_to_instance_map[instance_id]
        
        id_cad = shape_annot_instance["shapenet_id"]
        catid_cad = shape_annot_instance["shapenet_catid"] 
        
        obj_mesh = o3d.io.read_triangle_mesh(osp.join(shape_dir, catid_cad, id_cad, "models", "model_normalized.obj"))
        verts = np.asarray(obj_mesh.vertices)
        
        scan_layout_and_object_mesh += obj_mesh
            
        t, q, s = shape_annot_instance["t"], shape_annot_instance["q"], shape_annot_instance["s"]
        Mcad = se3.make_M_from_tqs(t, q, s)
        transform_shape = R_transform.dot(Mcad)
        
        verts = np.dot(transform_shape, np.hstack((verts, np.ones((verts.shape[0], 1)))).T).T[:, :3]
        obj_mesh.vertices = o3d.utility.Vector3dVector(verts)
        vis.add_geometry(obj_mesh)
    
    vis.run()
    vis.destroy_window()

def makeShapeAndLayoutMesh(shape_dir: str, scan_layout_filename: str, axis_align_matrix: np.ndarray, shape_annot: List[Dict]) -> Union[o3d.geometry.TriangleMesh, None]:
    """Create mesh combining shapes and layout."""
    scan_layout_and_object_mesh = o3d.geometry.TriangleMesh()
    
    if osp.exists(scan_layout_filename):
        scan_layout = load_utils.load_json(scan_layout_filename)
        layout_corners, layout_edges, _ = scan_layout['verts'], scan_layout['edges'], scan_layout['quads']
        for i in range(0,len(layout_corners)):
            temp = layout_corners[i][1]
            layout_corners[i][1] = - layout_corners[i][2]
            layout_corners[i][2] = temp
        
        layout_edges = np.array(layout_edges)
        
        layout_corners = point_cloud.transform(np.array(layout_corners), axis_align_matrix)
        layout_mesh = line_mesh.LineMesh(layout_corners, layout_edges, [1, 1, 1], radius=0.02)
        for cylinder_segment in layout_mesh.cylinder_segments:
            if scan_layout_and_object_mesh is None:
                scan_layout_and_object_mesh = cylinder_segment
            else:
                scan_layout_and_object_mesh += cylinder_segment
    
    if len(shape_annot) > 0:
        shape_annot = shape_annot[0]
        Mscan = se3.make_M_from_tqs(shape_annot["trs"]["translation"], shape_annot["trs"]["rotation"], shape_annot["trs"]["scale"])
        for model in shape_annot['aligned_models']:
            # read corresponding shapenet scanned points
            catid_cad = model["catid_cad"]
            id_cad = model["id_cad"]
            obj_path = osp.join(shape_dir, catid_cad, id_cad + '/models/model_normalized.obj')
            
            assert osp.exists(obj_path)
        
            obj_verts, obj_faces = point_cloud.load_obj(obj_path)
            '''transform shapenet obj to scannet'''
            
            t = model["trs"]["translation"]
            q = model["trs"]["rotation"]
            s = model["trs"]["scale"]
            
            Mcad = se3.make_M_from_tqs(t, q, s)
            transform_shape = axis_align_matrix.dot(np.linalg.inv(Mscan).dot(Mcad))
            
            obj_verts_aligned = np.hstack([obj_verts, np.ones((obj_verts.shape[0], 1))]).dot(transform_shape.T)[..., :3]
            
            obj_mesh = o3d.geometry.TriangleMesh()
            obj_mesh.vertices = o3d.utility.Vector3dVector(obj_verts_aligned)
            obj_mesh.triangles = o3d.utility.Vector3iVector(obj_faces)
            
            obj_mesh.paint_uniform_color([1., 1., 1.])
            scan_layout_and_object_mesh += obj_mesh

    if np.asarray(scan_layout_and_object_mesh.vertices).shape[0] == 0: 
        return None
    
    rotation_angle = -np.pi / 2  # 90 degrees in radians
    rotation_matrix = scan_layout_and_object_mesh.get_rotation_matrix_from_xyz((0, 0, rotation_angle))
    scan_layout_and_object_mesh.rotate(rotation_matrix, center=(0, 0, 0))
    
    return scan_layout_and_object_mesh
