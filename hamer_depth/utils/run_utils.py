import copy
from pathlib import Path
from typing import List, Optional, Tuple

import networkx as nx
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

from hamer_depth.detectors.detector_hamer import (
    INDEX_FINGER_VERTEX,
    INDEX_KNUCKLE_VERTEX_BACK,
    INDEX_KNUCKLE_VERTEX_FRONT,
    MIDDLE_FINGER_VERTEX,
    MIDDLE_KNUCKLE_VERTEX_BACK,
    MIDDLE_KNUCKLE_VERTEX_FRONT,
    RING_FINGER_VERTEX,
    RING_KNUCKLE_VERTEX_BACK,
    RING_KNUCKLE_VERTEX_FRONT,
    THUMB_VERTEX,
    WRIST_VERTEX_BACK,
    WRIST_VERTEX_FRONT,
    DetectorHamer,
)
from hamer_depth.utils.pcd_utils import (
    get_3D_points_from_pixels,
    get_pcd_from_points,
    get_point_cloud_of_segmask,
    get_visible_points,
    icp_registration,
)


def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    pts = np.dot(T, pts.T).T
    return pts[:, :3]


def convert_intrinsics_matrix_to_dict(camera_matrix: np.ndarray) -> dict:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    return intrinsics


def get_camera_matrix_from_file(file_path: Path) -> np.ndarray:
    with open(file_path, "r") as f:
        camera_matrix = np.loadtxt(f)

    assert camera_matrix.shape == (3, 3), (
        f"Camera matrix shape {camera_matrix.shape} is not (3, 3)"
    )

    return camera_matrix


def find_connected_clusters(points, distance_threshold=0.05):
    """
    Find clusters of points that are connected within a distance threshold
    and return the largest cluster.

    Args:
        points: numpy array of shape (N, 3) containing 3D points
        distance_threshold: float, maximum distance for points to be considered connected

    Returns:
        largest_cluster_points: numpy array of points in the largest cluster
        largest_cluster_indices: indices of points in the largest cluster
    """
    # Build KD-tree for efficient nearest neighbor search
    tree = cKDTree(points)

    # Find all pairs of points within distance_threshold
    pairs = tree.query_pairs(distance_threshold, output_type="ndarray")

    # Create graph
    G = nx.Graph()
    G.add_nodes_from(range(len(points)))
    G.add_edges_from(pairs)

    # Find connected components (clusters)
    clusters = list(nx.connected_components(G))

    # Get sizes of all clusters
    cluster_sizes = [len(c) for c in clusters]

    if not clusters:
        return np.array([]), np.array([])

    # Find the largest cluster
    largest_cluster = list(clusters[np.argmax(cluster_sizes)])
    largest_cluster_indices = np.array(largest_cluster)
    largest_cluster_points = points[largest_cluster_indices]

    return largest_cluster_points, largest_cluster_indices


def refine_3d_pts_with_depth(
    visible_hamer_points_3d_inaccurate: np.ndarray,
    detector_hamer: DetectorHamer,
    hamer_out: dict,
    img_depth: np.ndarray,
    cam_intrinsics: dict,
) -> np.ndarray:
    """
    Identify points in the depth image that belong to hamer vertices that are visible from the camera viewpoint
    """
    visible_hamer_points_2d = detector_hamer.project_3d_kpt_to_2d(
        kpts_3d=(
            visible_hamer_points_3d_inaccurate - hamer_out["T_cam_pred"].cpu().numpy()
        ).astype(np.float32),
        img_w=hamer_out["img_w"],
        img_h=hamer_out["img_h"],
        scaled_focal_length=hamer_out["scaled_focal_length"],
        camera_center=hamer_out["camera_center"],
        T_cam=hamer_out["T_cam_pred"],
    )
    visible_hamer_points_3d = get_3D_points_from_pixels(
        pixels_2d=visible_hamer_points_2d,
        depth_map=img_depth,
        intrinsics=cam_intrinsics,
    )
    return visible_hamer_points_3d


def get_initial_transformation_estimate(
    visible_hamer_points_3d_inaccurate: np.ndarray,
    visible_hamer_points_3d_depth: np.ndarray,
) -> np.ndarray:
    """
    Estimate 4x4 transformation that will fix the hamer hand pose predictions:

    Args:
        visible_hamer_points_3d_inaccurate (np.ndarray):
            HaMeR's predicted 3d points of the hand, which may have accurate XY coordinates but inaccurate Z coordinates
        visible_hamer_points_3d_depth (np.ndarray):
            Same corresponding 3d points, but with depth (Z) derived from the depth image

    Notes:
     * In a perfect world, we could simply get the distance between the two point clouds and use that as the translation.
     * However, `visible_hamer_points_3d_depth` is imperfect. Because of depth image errors/noise, the points may not be right
     * Example 1: If hamer is slightly off in XY, then we would be getting the corresponding Z values from the wrong pixel (e.g., far away background)
     * Example 2: If the depth image has noise or issues at the boundaries between of the hand, it can get very large depth values from the background instead of the hand
     * To fix this, we do some filtering on the points before doing the translation estimate.
     * We assume orientation is the same (only translation is different)
    """
    assert (
        visible_hamer_points_3d_inaccurate.shape == visible_hamer_points_3d_depth.shape
    ), (
        f"Visible hamer points 3d inaccurate shape {visible_hamer_points_3d_inaccurate.shape} "
        f"and visible hamer points 3d depth shape {visible_hamer_points_3d_depth.shape} are not the same"
    )

    # Compute the distances from the points in the depth image to the hand center
    hand_center = np.mean(visible_hamer_points_3d_inaccurate, axis=0)
    distances = np.linalg.norm(
        visible_hamer_points_3d_depth - hand_center[None], axis=1
    )

    # Filter out 0s and nans
    valid_idxs = visible_hamer_points_3d_depth[:, 2] > 0 & ~np.isnan(
        visible_hamer_points_3d_depth[:, 2]
    )

    # Filter out far away points (this assumes that the hamer inaccuracy is smaller than this distance)
    MAX_DIST = 0.5
    close_idxs = distances < MAX_DIST

    valid_visible_points_3d = visible_hamer_points_3d_depth[valid_idxs & close_idxs]
    valid_visible_hamer_points_3d_inaccurate = visible_hamer_points_3d_inaccurate[
        valid_idxs & close_idxs
    ]

    # From the depth image, we may still have points all over the place
    # E.g., if the mask is poor, it may include points from the background (very large depth)
    # Thus, we find the largest connected cluster of points and assume that is the hand
    largest_cluster_points, largest_cluster_indices = find_connected_clusters(
        valid_visible_points_3d, distance_threshold=0.05
    )

    # Get the median distance between the hamer predicted points and the remaining depth image points
    translation = np.nanmedian(
        valid_visible_points_3d[largest_cluster_indices]
        - valid_visible_hamer_points_3d_inaccurate[largest_cluster_indices],
        axis=0,
    )

    assert not np.isnan(translation).any(), "Translation is nan"

    T_0 = np.eye(4)
    T_0[:3, 3] = translation
    return T_0


def get_transformation_estimate(
    visible_hamer_pcd_inaccurate: o3d.geometry.PointCloud,
    pcd: o3d.geometry.PointCloud,
    T_0: np.ndarray,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Align the hamer point cloud (with only visible points) with the full arm point cloud using initial translation prediction
    """
    try:
        aligned_hamer_pcd, T = icp_registration(
            copy.deepcopy(visible_hamer_pcd_inaccurate),
            pcd,
            voxel_size=0.005,
            init_transform=T_0,
        )

        # HaMeR predictions' orientation should be very accurate, so if the ICP output is flipped, we use the initial prediction
        roll_pitch_yaw = np.absolute(
            R.from_matrix(T[:3, :3]).as_euler("xyz", degrees=True)
        )
        if (roll_pitch_yaw > 45).any():
            print(
                f"ICP result has too much rotation, reverting to initial prediction: T = {T}, roll_pitch_yaw = {roll_pitch_yaw}"
            )
            T = T_0
            aligned_hamer_pcd = visible_hamer_pcd_inaccurate.transform(T)
    except:
        print("ICP failed")
        return T_0, visible_hamer_pcd_inaccurate
    return T, aligned_hamer_pcd


def get_hand_keypoints(
    mesh: trimesh.Trimesh, T: Optional[np.ndarray] = None
) -> Tuple[dict, o3d.geometry.PointCloud]:
    """
    Get the 3D locations of the thumb, index finger, and hand end effector points in the world frame.
    """
    if T is None:
        T = np.eye(4)

    thumb_pt = mesh.vertices[THUMB_VERTEX]
    index_pt = mesh.vertices[INDEX_FINGER_VERTEX]
    middle_pt = mesh.vertices[MIDDLE_FINGER_VERTEX]
    ring_pt = mesh.vertices[RING_FINGER_VERTEX]
    index_knuckle_front, index_knuckle_back = (
        mesh.vertices[INDEX_KNUCKLE_VERTEX_FRONT],
        mesh.vertices[INDEX_KNUCKLE_VERTEX_BACK],
    )
    middle_knuckle_front, middle_knuckle_back = (
        mesh.vertices[MIDDLE_KNUCKLE_VERTEX_FRONT],
        mesh.vertices[MIDDLE_KNUCKLE_VERTEX_BACK],
    )
    ring_knuckle_front, ring_knuckle_back = (
        mesh.vertices[RING_KNUCKLE_VERTEX_FRONT],
        mesh.vertices[RING_KNUCKLE_VERTEX_BACK],
    )
    wrist_front, wrist_back = (
        mesh.vertices[WRIST_VERTEX_FRONT],
        mesh.vertices[WRIST_VERTEX_BACK],
    )

    hand_keypoints = np.vstack(
        [
            wrist_back,
            wrist_front,
            index_knuckle_back,
            index_knuckle_front,
            middle_knuckle_back,
            middle_knuckle_front,
            ring_knuckle_back,
            ring_knuckle_front,
            index_pt,
            middle_pt,
            ring_pt,
            thumb_pt,
        ]
    )
    hand_keypoints = transform_pts(hand_keypoints, T)
    hand_keypoints_pcd = get_pcd_from_points(
        hand_keypoints, colors=np.ones_like(hand_keypoints) * [1, 0, 0]
    )
    hand_keypoints_dict = {
        "wrist_back": hand_keypoints[0],
        "wrist_front": hand_keypoints[1],
        "index_0_back": hand_keypoints[2],
        "index_0_front": hand_keypoints[3],
        "middle_0_back": hand_keypoints[4],
        "middle_0_front": hand_keypoints[5],
        "ring_0_back": hand_keypoints[6],
        "ring_0_front": hand_keypoints[7],
        "index_3": hand_keypoints[8],
        "middle_3": hand_keypoints[9],
        "ring_3": hand_keypoints[10],
        "thumb_3": hand_keypoints[11],
    }
    return hand_keypoints_dict, hand_keypoints_pcd

def visualize_geometries(
    width: int,
    height: int,
    cam_intrinsics: dict,
    geometries: List[o3d.geometry.Geometry],
):
    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height)

    # Add point clouds to visualizer
    for geom in geometries:
        vis.add_geometry(geom)

    # Get ViewControl and current camera parameters
    view_control = vis.get_view_control()
    camera_params = view_control.convert_to_pinhole_camera_parameters()

    # Update intrinsic matrix
    camera_params.intrinsic.set_intrinsics(
        width=width, height=height,
        fx=cam_intrinsics["fx"], fy=cam_intrinsics["fy"],
        cx=cam_intrinsics["cx"], cy=cam_intrinsics["cy"]
    )

    # Set up camera extrinsics (camera at origin with Z forward and Y down)
    extrinsics = np.eye(4)
    extrinsics[:3, 3] = np.array([0, 0, 0])  # origin
    extrinsics[:3, 0] = np.array([1, 0, 0])  # X-right
    extrinsics[:3, 1] = np.array([0, 1, 0])  # Y-down
    extrinsics[:3, 2] = np.array([0, 0, 1])  # Z-forward
    camera_params.extrinsic = extrinsics

    # Apply updated parameters
    view_control.convert_from_pinhole_camera_parameters(camera_params, allow_arbitrary=True)

    # Render and show
    vis.run()
    breakpoint()
    vis.destroy_window()



def process_image_with_hamer(
    img_rgb: np.ndarray,
    img_depth: np.ndarray,
    mask: np.ndarray,
    cam_intrinsics: dict,
    detector_hamer: DetectorHamer,
    debug: bool = False,
) -> Tuple[
    o3d.geometry.PointCloud,
    dict,
    trimesh.Trimesh,
    o3d.geometry.PointCloud,
    dict,
    o3d.geometry.PointCloud,
    trimesh.Trimesh,
    o3d.geometry.PointCloud,
]:
    # Get masked hand point cloud
    # These are accurate points in 3D space
    masked_hand_pcd = get_point_cloud_of_segmask(
        mask=mask,
        depth_img=img_depth,
        img=img_rgb,
        intrinsics=cam_intrinsics,
        visualize=False,
    )

    # Run HaMeR to get an initial estimate of the hand pose
    # It is accurate in 2D space, but not in 3D space
    hamer_out = detector_hamer.detect_hand_keypoints(
        img=img_rgb,
        img_mask=mask,
        camera_params=cam_intrinsics,
    )
    if hamer_out is None or not hamer_out.get("success", False):
        raise ValueError("No hand detected in image")
    hand_mesh_inaccurate = trimesh.Trimesh(
        hamer_out["verts"].copy(), detector_hamer.faces_right.copy()
    )

    # Figure out which hamer points are visible from the camera
    # These are inaccurate points in 3D space
    visible_hamer_points_3d_inaccurate, _ = get_visible_points(
        hand_mesh_inaccurate, origin=np.array([0, 0, 0])
    )
    visible_hamer_pcd_inaccurate = get_pcd_from_points(
        visible_hamer_points_3d_inaccurate,
        colors=np.ones_like(visible_hamer_points_3d_inaccurate) * [0, 1, 0],
    )

    # Refine the 3D points using the depth image
    visible_hamer_points_3d_depth = refine_3d_pts_with_depth(
        visible_hamer_points_3d_inaccurate=visible_hamer_points_3d_inaccurate,
        detector_hamer=detector_hamer,
        hamer_out=hamer_out,
        img_depth=img_depth,
        cam_intrinsics=cam_intrinsics,
    )

    # Make initial transformation estimate
    T_0 = get_initial_transformation_estimate(
        visible_hamer_points_3d_inaccurate=visible_hamer_points_3d_inaccurate,
        visible_hamer_points_3d_depth=visible_hamer_points_3d_depth,
    )

    # DEBUG
    full_pcd = get_point_cloud_of_segmask(
        mask=np.ones_like(mask),
        depth_img=img_depth,
        img=img_rgb,
        intrinsics=cam_intrinsics,
        visualize=False,
    )

    # Align the inaccurate hand point cloud with the masked hand point cloud
    T, aligned_hamer_pcd = get_transformation_estimate(
        visible_hamer_pcd_inaccurate=visible_hamer_pcd_inaccurate,
        pcd=masked_hand_pcd,
        T_0=T_0,
    )

    # Get the hand keypoints
    hand_mesh_accurate = hand_mesh_inaccurate.apply_transform(T)
    hand_keypoints_dict, hand_keypoints_pcd = get_hand_keypoints(
        mesh=hand_mesh_accurate,
    )

    if debug:
        width, height = img_rgb.shape[1], img_rgb.shape[0]
        RESCALE_FACTOR = 2.0
        width, height = int(width * RESCALE_FACTOR), int(height * RESCALE_FACTOR)

        # Set colors
        RED, GREEN, BLUE, YELLOW = [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 1, 0]
        masked_hand_pcd.paint_uniform_color(RED)
        hand_keypoints_pcd.paint_uniform_color(BLUE)
        visible_hamer_pcd_inaccurate.paint_uniform_color(YELLOW)
        aligned_hamer_pcd.paint_uniform_color(GREEN)

        visualize_geometries(
            width=width,
            height=height,
            cam_intrinsics=cam_intrinsics,
            geometries=[
                full_pcd,
                visible_hamer_pcd_inaccurate,
                aligned_hamer_pcd,
                hand_keypoints_pcd,
                masked_hand_pcd,
            ],
        )

    return (
        masked_hand_pcd,
        hamer_out,
        hand_mesh_inaccurate,
        aligned_hamer_pcd,
        hand_keypoints_dict,
        hand_keypoints_pcd,
        hand_mesh_accurate,
        full_pcd,
    )
