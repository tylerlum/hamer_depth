import pdb
import numpy as np
from typing import Tuple, List, Optional

import open3d as o3d
import trimesh 
from pycpd import RigidRegistration
from sklearn.neighbors import NearestNeighbors

def cpd_registration(source_points: o3d.geometry.PointCloud, 
                     target_points: o3d.geometry.PointCloud) -> o3d.geometry.PointCloud:
    """
    Register two point clouds using Coherent Point Drift (CPD) algorithm.
    """
    reg = RigidRegistration(X=target_points, Y=source_points)
    transformed_points, _ = reg.register()
    return transformed_points


def preprocess_point_cloud(pcd: o3d.geometry.PointCloud, 
                           voxel_size: float) -> Tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
    """
    Downsample point cloud to desired voxel resolution and compute FPFH features.
    """
    pcd_down = pcd.voxel_down_sample(voxel_size)
    radius_normal = voxel_size * 2
    pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
    radius_feature = voxel_size * 5
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def global_registration(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud, 
                        voxel_size: float) -> o3d.pipelines.registration.RegistrationResult:
    """
    Register two point clouds using global registration with RANSAC.
    """
    source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    
    distance_threshold = voxel_size * 1.5
    result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        4,  # RANSAC iterations
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
    return result_ransac


def icp_registration(source_pcd: o3d.geometry.PointCloud, target_pcd: o3d.geometry.PointCloud, 
                     voxel_size: float=0.05, use_global_registration:bool=True, 
                     init_transform:Optional[np.ndarray]=None) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Register two point clouds using ICP algorithm. 
    """
    # Optional global registration using RANSAC
    if use_global_registration:
        if init_transform is None:
            result_ransac = global_registration(source_pcd, target_pcd, voxel_size)
            init_transform = result_ransac.transformation
    else:
        init_transform = np.eye(4)
    
    # Refine alignment using ICP
    max_correspondence_distance = voxel_size * 5
    result_icp = o3d.pipelines.registration.registration_icp(
        source=source_pcd, target=target_pcd, max_correspondence_distance=max_correspondence_distance, 
        init=init_transform,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    if np.array_equal(init_transform, result_icp.transformation):
        result_ransac = global_registration(source_pcd, target_pcd, voxel_size)
        init_transform = result_ransac.transformation
        result_icp = o3d.pipelines.registration.registration_icp(
            source=source_pcd, target=target_pcd, max_correspondence_distance=max_correspondence_distance, 
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
    aligned_source_pcd = source_pcd.transform(result_icp.transformation)

    return aligned_source_pcd, result_icp.transformation


def get_visible_points(mesh, origin: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return list of points in mesh that are visible from origin.
    """
    intersector = trimesh.ray.ray_triangle.RayMeshIntersector(mesh)
    pts = mesh.vertices
    vectors = pts - origin
    directions = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    visible_triangle_indices = intersector.intersects_first(np.tile(origin, (pts.shape[0], 1)), directions)
    visible_triangles = mesh.faces[visible_triangle_indices]
    visible_vertex_indices = np.unique(visible_triangles)
    visible_points = pts[visible_vertex_indices]
    return np.array(visible_points).astype(np.float32), np.array(visible_vertex_indices)


def get_pcd_from_points(points: np.ndarray, colors: Optional[np.ndarray]=None) -> o3d.geometry.PointCloud:
    """
    Convert a list of points to an Open3D point cloud.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(colors)
    pcd.remove_non_finite_points()
    return pcd


def visualize_pcds(list_pcds: list, visible: bool=True) -> np.ndarray:
    """
    Visualize a list of point clouds.
    """
    visualization_image = None
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=visible)
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])
    for pcd in list_pcds:
        vis.add_geometry(pcd)
    # Set camera
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()
    camera_params.extrinsic = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],  # Move camera 2 units back along z-axis
        [0, 0, 0, 1]
    ])
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)
    vis.poll_events()
    vis.update_renderer()
    if not visible:
        visualization_image = vis.capture_screen_float_buffer(do_render=True)
        visualization_image = (255.0 * np.asarray(visualization_image)).astype(np.uint8)
    if visible:
        vis.run()
    vis.destroy_window()
    if visualization_image is None:
        visualization_image = np.array([])
    return visualization_image



def radius_outlier_detection(points: np.ndarray, radius: float=5, 
                             min_neighbors: int=5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Detect outliers in a point cloud using radius-based outlier detection.
    """
    # Fit the NearestNeighbors model
    nbrs = NearestNeighbors(radius=radius).fit(points)
    
    # Get the number of neighbors for each point within the specified radius
    distances, indices = nbrs.radius_neighbors(points)
    
    # Detect points with fewer neighbors than the minimum threshold
    outliers_mask = np.array([len(neigh) < min_neighbors for neigh in indices])

    outlier_pts = points[outliers_mask]
    
    return outliers_mask, outlier_pts


def remove_outliers(pcd: o3d.geometry.PointCloud, radius: float=5, 
                    min_neighbors: int=5) -> Tuple[o3d.geometry.PointCloud, np.ndarray]:
    """
    Remove outliers from a point cloud using radius-based outlier detection.
    """
    outlier_indices, outlier_pts = radius_outlier_detection(np.asarray(pcd.points), 
                                                            radius=radius, min_neighbors=min_neighbors)
    # outlier_colors = np.ones_like(outlier_pts) * [1, 0, 0]
    # pcd_outliers = get_pcd_from_points(outlier_pts, colors=outlier_colors)
    filtered_pts = np.asarray(pcd.points)[~outlier_indices]
    filtered_colors = np.asarray(pcd.colors)[~outlier_indices]
    filtered_pcd = get_pcd_from_points(filtered_pts, colors=filtered_colors)
    return filtered_pcd, outlier_indices


def get_3D_point_from_pixel(px: int, py: int, depth: float, intrinsics: dict) -> np.ndarray:
    """
    Convert pixel coordinates and depth to 3D point.
    """
    x = (px - intrinsics["cx"]) / intrinsics["fx"]
    y = (py - intrinsics["cy"]) / intrinsics["fy"]
    
    depth = depth / 1000

    X = x * depth
    Y = y * depth
    if len(X.shape) == 0:
        p = np.array([X, Y, depth])
    else:
        p = np.stack((X, Y, depth), axis=1)

    return p


def get_3D_points_from_pixels(pixels_2d: np.ndarray, depth_map: np.ndarray, intrinsics: dict) -> np.ndarray:
    """
    Convert an array of pixel coordinates and depth map to 3D points.
    """
    px = pixels_2d[:, 0]
    py = pixels_2d[:, 1]

    x = (px - intrinsics["cx"]) / intrinsics["fx"]
    y = (py - intrinsics["cy"]) / intrinsics["fy"]

    depth = depth_map[py, px] / 1000

    X = x * depth
    Y = y * depth

    points_3d = np.stack((X, Y, depth), axis=1)
    return points_3d


def get_point_cloud_of_segmask(mask: np.ndarray, depth_img: np.ndarray, img: np.ndarray, 
                               intrinsics: dict, visualize: bool=False) -> o3d.geometry.PointCloud:
    """
    Return the point cloud that corresponds to the segmentation mask in the depth image.
    """
    idxs_y, idxs_x, _ = mask.nonzero()
    depth_masked = depth_img[idxs_y, idxs_x]
    seg_points = get_3D_point_from_pixel(idxs_x, idxs_y, depth_masked, intrinsics)
    seg_colors = img[idxs_y, idxs_x, :] / 255.0  # Normalize to [0,1] for cv2

    pcd = get_pcd_from_points(seg_points, colors=seg_colors)

    if visualize:
        visualize_pcds([pcd])

    return pcd