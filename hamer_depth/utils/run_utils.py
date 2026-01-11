import copy
from copy import deepcopy
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial.transform import Rotation as R
from termcolor import colored

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
from hamer_depth.utils.hand_type import HandType
from hamer_depth.utils.pcd_utils import (
    get_pcd_from_points,
    get_point_cloud_of_segmask,
    get_visible_points,
    icp_registration,
    visualize_geometries,
)


def transform_pts(pts: np.ndarray, T: np.ndarray) -> np.ndarray:
    pts = np.hstack([pts, np.ones((len(pts), 1))])
    pts = np.dot(T, pts.T).T
    return pts[:, :3]


def get_transformation_estimate(
    source_pcd: o3d.geometry.PointCloud,
    target_pcd: o3d.geometry.PointCloud,
) -> Tuple[o3d.geometry.PointCloud, np.ndarray, np.ndarray]:
    """
    Align the source pcd to the target pcd
    Returns the aligned source pcd and the transformation matrix T that aligns the source pcd to the target pcd
    """
    # Make initial transformation estimate
    # By getting the rough translation between the source pcd and the target pcd
    T_0 = np.eye(4)
    T_0[:3, 3] = np.nanmedian(np.asarray(target_pcd.points), axis=0) - np.nanmedian(
        source_pcd.points, axis=0
    )

    aligned_source_pcd, T = icp_registration(
        source_pcd=copy.deepcopy(source_pcd),
        target_pcd=copy.deepcopy(target_pcd),
        init_transform=T_0,
    )

    # HaMeR predictions' orientation should be very accurate, so if the ICP output is flipped, we use the initial prediction
    angle_deg = np.absolute(np.rad2deg(R.from_matrix(T[:3, :3]).magnitude()))
    MAX_ANGLE_DEG = 35
    if (angle_deg > MAX_ANGLE_DEG).any():
        print(
            f"ICP result has too much rotation, reverting to initial prediction: T = {T}, angle_deg = {angle_deg}"
        )
        T = T_0
        aligned_source_pcd = copy.deepcopy(source_pcd).transform(T)

    return aligned_source_pcd, T, T_0


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
    hand_keypoints_pcd = get_pcd_from_points(hand_keypoints)
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


def create_annotated_img_with_keypoints(
    hamer_out: dict,
    T: np.ndarray,
    cam_intrinsics: dict,
    img_rgb: np.ndarray,
) -> np.ndarray:
    # --- NEW CODE START ---
    # 1. Get the original 3D keypoints from HaMeR (Shape: 21x3)
    # These are in the original "inaccurate" frame
    original_kpts_3d = hamer_out["kpts_3d"]

    # 2. Apply the transformation T to these points
    # We need to use homogeneous coordinates for the matrix multiplication
    # (x, y, z) -> (x, y, z, 1)
    ones = np.ones((original_kpts_3d.shape[0], 1))
    kpts_hom = np.hstack([original_kpts_3d, ones])  # Shape: 21x4

    # Apply T: (21x4) @ (4x4).T -> (21x4)
    # We transpose T because we are multiplying a row vector
    transformed_kpts_hom = kpts_hom @ T.T

    # Drop the 4th column to get back to (x, y, z)
    refined_kpts_3d = transformed_kpts_hom[:, :3]

    # 3. Project the new 3D points back to 2D
    # Handle both Dict and Matrix intrinsics
    if isinstance(cam_intrinsics, dict):
        fx, fy = cam_intrinsics["fx"], cam_intrinsics["fy"]
        cx, cy = cam_intrinsics["cx"], cam_intrinsics["cy"]
    else:  # Assumes 3x3 matrix
        fx, fy = cam_intrinsics[0, 0], cam_intrinsics[1, 1]
        cx, cy = cam_intrinsics[0, 2], cam_intrinsics[1, 2]

    refined_kpts_2d = np.zeros((refined_kpts_3d.shape[0], 2))
    z_coords = refined_kpts_3d[:, 2]

    # Simple pinhole projection
    refined_kpts_2d[:, 0] = (refined_kpts_3d[:, 0] * fx / z_coords) + cx
    refined_kpts_2d[:, 1] = (refined_kpts_3d[:, 1] * fy / z_coords) + cy

    # 4. Create the new annotated image
    refined_annotated_img_bgr = DetectorHamer.visualize_2d_kpt_on_img(
        kpts_2d=refined_kpts_2d, img=img_rgb
    )
    refined_annotated_img = cv2.cvtColor(refined_annotated_img_bgr, cv2.COLOR_BGR2RGB)
    # --- NEW CODE END ---
    return refined_annotated_img


def process_image_with_hamer(
    img_rgb: np.ndarray,
    img_depth: np.ndarray,
    mask: np.ndarray,
    cam_intrinsics: dict,
    detector_hamer: DetectorHamer,
    hand_type: HandType = HandType.RIGHT,
    debug: bool = False,
) -> Tuple[dict, dict, trimesh.Trimesh, np.ndarray, np.ndarray]:
    full_pcd = get_point_cloud_of_segmask(
        mask=np.ones_like(mask),
        depth_img=img_depth,
        img=img_rgb,
        intrinsics=cam_intrinsics,
        visualize=False,
    )

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
        mask=mask,
        camera_params=cam_intrinsics,
        hand_type=hand_type,
    )
    if hamer_out is None or not hamer_out.get("success", False):
        raise ValueError("No hand detected in image")

    if hand_type == HandType.RIGHT:
        faces = detector_hamer.faces_right.copy()
    elif hand_type == HandType.LEFT:
        faces = detector_hamer.faces_left.copy()
    else:
        raise ValueError(f"Invalid hand type: {hand_type}")

    hand_mesh_inaccurate = trimesh.Trimesh(hamer_out["verts"].copy(), faces)

    # Figure out which hamer points are visible from the camera
    # These are inaccurate points in 3D space
    visible_hamer_points_3d, _ = get_visible_points(
        hand_mesh_inaccurate, origin=np.array([0, 0, 0])
    )
    visible_hamer_pcd = get_pcd_from_points(visible_hamer_points_3d)

    # Create annotated image with inaccurate hand keypoints
    annotated_rgb_img_inaccurate = create_annotated_img_with_keypoints(
        hamer_out=hamer_out,
        T=np.eye(4),
        cam_intrinsics=cam_intrinsics,
        img_rgb=img_rgb,
    )

    if debug:
        plt.imshow(annotated_rgb_img_inaccurate.astype(np.uint8))
        plt.title("Inaccurate Hand Keypoints")
        plt.show()

        RED, GREEN = [1, 0, 0], [0, 1, 0]

        # Inputs
        visible_hamer_pcd.paint_uniform_color(RED)  # Initial hamer points to refine
        masked_hand_pcd.paint_uniform_color(GREEN)  # Real hand points to align to

        print("Showing debug information for inputs")
        print("INPUTS:")
        print(colored("RED: Initial hamer points to refine", "red"))
        print(colored("GREEN: Masked hand points to align to", "green"))

        visualize_geometries(
            width=img_rgb.shape[1],
            height=img_rgb.shape[0],
            cam_intrinsics=cam_intrinsics,
            geometries={
                "full_pcd": full_pcd,  # Full point cloud of the scene
                "masked_hand_pcd": masked_hand_pcd,  # 3D points of the masked hand
                "visible_hamer_pcd": visible_hamer_pcd,  # Raw hamer prediction of points visible from the camera
            },
            meshes={
                "hand_mesh_inaccurate": hand_mesh_inaccurate,
            },
        )

    # Align the inaccurate hand point cloud with the masked hand point cloud
    # Using ICP registration
    visible_hamer_pcd_aligned, T, T_0 = get_transformation_estimate(
        source_pcd=visible_hamer_pcd,
        target_pcd=masked_hand_pcd,
    )

    # Get the hand keypoints
    hand_mesh = deepcopy(hand_mesh_inaccurate).apply_transform(T)
    hand_keypoints_dict_inaccurate, hand_keypoints_pcd_inaccurate = get_hand_keypoints(
        mesh=hand_mesh_inaccurate,
    )
    hand_keypoints_dict, hand_keypoints_pcd = get_hand_keypoints(
        mesh=hand_mesh,
    )

    annotated_rgb_img = create_annotated_img_with_keypoints(
        hamer_out=hamer_out,
        T=T,
        cam_intrinsics=cam_intrinsics,
        img_rgb=img_rgb,
    )

    if debug:
        # Visualize the inaccurate and refined hand keypoints
        fig, axes = plt.subplots(1, 2)
        axes = axes.flatten()
        axes[0].imshow(annotated_rgb_img_inaccurate.astype(np.uint8))
        axes[0].set_title("Inaccurate Hand Keypoints")
        axes[1].imshow(annotated_rgb_img.astype(np.uint8))
        axes[1].set_title("Refined Hand Keypoints")
        plt.show()

        RED, GREEN = [1, 0, 0], [0, 1, 0]

        # Inputs
        visible_hamer_pcd.paint_uniform_color(RED)  # Initial hamer points to refine
        masked_hand_pcd.paint_uniform_color(GREEN)  # Real hand points to align to

        BLUE, YELLOW = [0, 0, 1], [1, 1, 0]

        # Intermediate output
        visible_hamer_pcd_initial_estimate = deepcopy(visible_hamer_pcd).transform(T_0)
        visible_hamer_pcd_initial_estimate.paint_uniform_color(YELLOW)

        # Final output
        visible_hamer_pcd_aligned.paint_uniform_color(BLUE)

        print(
            "Showing debug information for inputs and outputs of hamer depth refinement"
        )
        print("INPUTS:")
        print(colored("RED: Initial hamer points to refine", "red"))
        print(colored("GREEN: Masked hand points to align to", "green"))
        print("INTERMEDIATE OUTPUT:")
        print(colored("YELLOW: Initial estimate of aligned hamer points", "yellow"))
        print("FINAL OUTPUT:")
        print(colored("BLUE: Final aligned hamer points", "blue"))

        hand_mesh_initial_estimate = deepcopy(hand_mesh_inaccurate).apply_transform(T_0)
        visualize_geometries(
            width=img_rgb.shape[1],
            height=img_rgb.shape[0],
            cam_intrinsics=cam_intrinsics,
            geometries={
                "full_pcd": full_pcd,  # Full point cloud of the scene
                "masked_hand_pcd": masked_hand_pcd,  # 3D points of the masked hand
                "visible_hamer_pcd": visible_hamer_pcd,  # Raw hamer prediction of points visible from the camera
                "visible_hamer_pcd_initial_estimate": visible_hamer_pcd_initial_estimate,  # Initial estimate of aligned hamer points
                "visible_hamer_pcd_aligned": visible_hamer_pcd_aligned,  # Final aligned hamer points
            },
            meshes={
                "hand_mesh_inaccurate": hand_mesh_inaccurate,
                "hand_mesh_initial_estimate": hand_mesh_initial_estimate,
                "hand_mesh": hand_mesh,
            },
        )

    return (
        hamer_out,
        hand_keypoints_dict,
        hand_mesh,
        annotated_rgb_img_inaccurate,
        annotated_rgb_img,
    )
