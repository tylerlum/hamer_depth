import argparse
import copy
import os
from typing import Tuple

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import json

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


def convert_intrinsics_matrix_to_dict(camera_matrix):
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


def get_intrinsics_from_json(json_path: str):
    with open(json_path, "r") as f:
        camera_matrix = np.loadtxt(f)

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


def get_visible_pts_from_hamer(
    detector_hamer: DetectorHamer,
    hamer_out: dict,
    mesh: trimesh.Trimesh,
    img_depth: np.ndarray,
    cam_intrinsics: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify points in the depth image that belong to hamer vertices that are visible from the camera viewpoint
    """
    visible_hamer_vertices, _ = get_visible_points(mesh, origin=np.array([0, 0, 0]))
    visible_points_2d = detector_hamer.project_3d_kpt_to_2d(
        (visible_hamer_vertices - hamer_out["T_cam_pred"].cpu().numpy()).astype(
            np.float32
        ),
        hamer_out["img_w"],
        hamer_out["img_h"],
        hamer_out["scaled_focal_length"],
        hamer_out["camera_center"],
        hamer_out["T_cam_pred"],
    )
    visible_points_3d = get_3D_points_from_pixels(
        visible_points_2d, img_depth, cam_intrinsics
    )
    return visible_points_3d, visible_hamer_vertices


def get_initial_transformation_estimate(
    visible_points_3d: np.ndarray, visible_hamer_vertices: np.ndarray
) -> np.ndarray:
    """
    Get estimate of transformation from HaMeR's predicted 3d point cloud of the hand (using only the visible points) to the point cloud of the arm obtained from the depth image. Assume orientation is the same (only translation is different)
    """
    hand_center = np.mean(visible_hamer_vertices, axis=0)
    distances = np.linalg.norm(visible_points_3d - hand_center[None], axis=1)
    valid_idxs = visible_points_3d[:, 2] > 0
    close_idxs = (
        distances < 0.5
    )  # Screening out points more than 0.5m away to not affect initial transform estimate
    valid_visible_points_3d = visible_points_3d[valid_idxs & close_idxs]
    valid_visible_hamer_vertices = visible_hamer_vertices[valid_idxs & close_idxs]
    largest_cluster_points, largest_cluster_indices = find_connected_clusters(
        valid_visible_points_3d, distance_threshold=0.05
    )
    translation = np.nanmedian(
        valid_visible_points_3d[largest_cluster_indices]
        - valid_visible_hamer_vertices[largest_cluster_indices],
        axis=0,
    )

    T_0 = np.eye(4)
    if not np.isnan(translation).any():
        T_0[:3, 3] = translation
    return T_0


def get_transformation_estimate(
    visible_points_3d: np.ndarray,
    visible_hamer_vertices: np.ndarray,
    pcd: o3d.geometry.PointCloud,
    full_pcd: o3d.geometry.PointCloud,
) -> Tuple[np.ndarray, o3d.geometry.PointCloud, o3d.geometry.PointCloud]:
    """
    Align the hamer point cloud (with only visible points) with the full arm point cloud using initial translation prediction
    """
    T_0 = get_initial_transformation_estimate(visible_points_3d, visible_hamer_vertices)
    visible_hamer_pcd = get_pcd_from_points(
        visible_hamer_vertices, colors=np.ones_like(visible_hamer_vertices) * [0, 1, 0]
    )
    visible_hamer_pcd_copy = copy.deepcopy(visible_hamer_pcd)
    try:
        aligned_hamer_pcd, T = icp_registration(
            visible_hamer_pcd, pcd, voxel_size=0.005, init_transform=T_0
        )

        T_copy = T.copy()
        if (
            np.absolute(R.from_matrix(T_copy[:3, :3]).as_euler("xyz", degrees=True))
            > 45
        ).any():  # Checking ICP output is not flipped
            print("Old T", T)
            print("Hand is flipped or too far away - using HaMeR prediction")
            T = T_0
            aligned_hamer_pcd = visible_hamer_pcd_copy.transform(T)
            print("New T", T)
    except:
        print("ICP failed")
        return T_0, None, visible_hamer_pcd
    return T, aligned_hamer_pcd, visible_hamer_pcd


def get_finger_pose(
    mesh: trimesh.Trimesh, T: np.ndarray
) -> Tuple[dict, o3d.geometry.PointCloud]:
    """
    Get the 3D locations of the thumb, index finger, and hand end effector points in the world frame.
    """
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

    finger_pts = np.vstack(
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
    finger_pts = transform_pts(finger_pts, T)
    finger_pcd = get_pcd_from_points(
        finger_pts, colors=np.ones_like(finger_pts) * [1, 0, 0]
    )
    result_dict = {
        "wrist_back": finger_pts[0],
        "wrist_front": finger_pts[1],
        "index_0_back": finger_pts[2],
        "index_0_front": finger_pts[3],
        "middle_0_back": finger_pts[4],
        "middle_0_front": finger_pts[5],
        "ring_0_back": finger_pts[6],
        "ring_0_front": finger_pts[7],
        "index_3": finger_pts[8],
        "middle_3": finger_pts[9],
        "ring_3": finger_pts[10],
        "thumb_3": finger_pts[11],
    }
    return result_dict, finger_pcd


def process_image_with_hamer(
    img_rgb: np.ndarray,
    img_depth: np.ndarray,
    mask: np.ndarray,
    cam_intrinsics: dict,
    detector_hamer: DetectorHamer,
    vis,
) -> Tuple[dict, np.ndarray, np.ndarray]:
    pcd = get_point_cloud_of_segmask(
        mask, img_depth, img_rgb, cam_intrinsics, visualize=False
    )
    full_pcd = get_point_cloud_of_segmask(
        np.ones_like(mask), img_depth, img_rgb, cam_intrinsics, visualize=False
    )
    hamer_out = detector_hamer.detect_hand_keypoints(
        img_rgb, mask, camera_params=cam_intrinsics
    )
    if hamer_out is None or not hamer_out.get("success", False):
        raise ValueError("No hand detected in image")
    mesh = trimesh.Trimesh(hamer_out["verts"].copy(), detector_hamer.faces_right.copy())
    visible_points_3d, visible_hamer_vertices = get_visible_pts_from_hamer(
        detector_hamer, hamer_out, mesh, img_depth, cam_intrinsics
    )
    T, aligned_hamer_pcd, visible_hamer_pcd = get_transformation_estimate(
        visible_points_3d, visible_hamer_vertices, pcd, full_pcd
    )
    finger_pts, finger_pcd = get_finger_pose(mesh, T)
    transformed_mesh = mesh.apply_transform(T)

    return (
        pcd,
        hamer_out,
        mesh,
        aligned_hamer_pcd,
        finger_pts,
        finger_pcd,
        transformed_mesh,
    )


def main(demo_path):
    detector_hamer = DetectorHamer()

    rgb_paths = sorted(
        [
            os.path.join(demo_path, "rgb", file)
            for file in os.listdir(os.path.join(demo_path, "rgb"))
            if file.lower().endswith(".jpg")
        ]
    )
    depth_paths = sorted(
        [
            os.path.join(demo_path, "depth", file)
            for file in os.listdir(os.path.join(demo_path, "depth"))
        ]
    )
    mask_paths = sorted(
        [
            os.path.join(demo_path, "masks_hand2", file)
            for file in os.listdir(os.path.join(demo_path, "masks_hand2"))
        ]
    )
    assert len(rgb_paths) == len(depth_paths)
    assert len(rgb_paths) == len(mask_paths)
    cam_intrinsics_path = (
        "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/cam_K.txt"
    )

    # Get intrinsics
    camera_matrix = get_intrinsics_from_json(cam_intrinsics_path)
    cam_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)

    for i in tqdm(range(len(rgb_paths))):
        idx = rgb_paths[i][-9:-4]
        print(idx)

        # Get data
        img_rgb = np.array(Image.open(rgb_paths[i]))
        img_depth = np.array(Image.open(depth_paths[i]))
        mask = np.array(Image.open(mask_paths[i]))

        # Adjust for size of images
        img_w, img_h = img_rgb.shape[:2]

        try:
            (
                pcd,
                hamer_out,
                mesh,
                aligned_hamer_pcd,
                finger_pts,
                finger_pcd,
                transformed_mesh,
            ) = process_image_with_hamer(
                img_rgb, img_depth, mask, cam_intrinsics, detector_hamer, vis=None
            )
        except Exception as e:
            print(e)
            continue

        # Save data
        out_path = demo_path.replace(
            "/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/final_scene/",
            "/juno/u/oliviayl/repos/cross_embodiment/human_shadow/outputs_depth_final_CLUSTER/",
        )
        print(out_path)
        if not os.path.exists(os.path.join(out_path)):
            os.makedirs(os.path.join(out_path))
        transformed_mesh.export(os.path.join(out_path, f"{idx}.obj"))
        cv2.imwrite(os.path.join(out_path, f"{idx}.png"), hamer_out["annotated_img"])
        joint_poses = hamer_out["kpts_3d"]
        assert joint_poses.shape == (21, 3)

        joint_names = [
            "wrist_back",
            "wrist_front",
            "index_0_back",
            "index_0_front",
            "middle_0_back",
            "middle_0_front",
            "ring_0_back",
            "ring_0_front",
            "index_3",
            "middle_3",
            "ring_3",
            "thumb_3",
        ]
        frame_data = {}

        for j in joint_names:
            frame_data[j] = list(finger_pts[j])
        frame_data["global_orient"] = hamer_out["global_orient"].tolist()
        with open(os.path.join(out_path, f"{idx}.json"), "w") as json_file:
            json.dump(frame_data, json_file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--demo_path",
        type=str,
        default="/juno/u/oliviayl/repos/cross_embodiment/FoundationPose/demo_data/final_scene/plate_pivotrack",
    )
    args = parser.parse_args()

    main(args.demo_path)
