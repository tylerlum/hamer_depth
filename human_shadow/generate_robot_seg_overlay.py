
import pdb
import os
import json
from tqdm import tqdm
import numpy as np 
import logging
import argparse 
from typing import Tuple

import pandas as pd
import logging
import mediapy as media
from scipy.spatial.transform import Rotation

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.camera.zed_utils import ZED_RESOLUTIONS
from human_shadow.utils.transform_utils import transform_pt
from human_shadow.virtual_twin.twin_robot import *


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_human_orientation_to_robot_orientation(thumb_pos: np.ndarray, index_pos: np.ndarray) -> np.ndarray:
    """Convert orientation of human hand to orientation of robot gripper."""
    finger_vector = index_pos - thumb_pos
    base_vector = np.array([0, 1, 0])
    angle = np.arccos(np.dot(finger_vector, base_vector) / (np.linalg.norm(finger_vector) * np.linalg.norm(base_vector)))

    base_ori_xyzw = np.array([1, 0, 0, 0])
    r_base = Rotation.from_quat(base_ori_xyzw, scalar_first=False)
    r_180 = Rotation.from_euler("z", np.pi, degrees=False)
    r_human = Rotation.from_euler("z", -angle, degrees=False)
    robot_rot =  r_human * r_180 * r_base
    robot_ori_xyzw = robot_rot.as_quat(scalar_first=False)
    return robot_ori_xyzw


def convert_human_gripper_to_robot_gripper(thumb_pos: np.ndarray, index_pos: np.ndarray) -> float:
    """Convert human gripper opening to robot gripper opening."""
    finger_vector = index_pos - thumb_pos
    gripper_pos = np.linalg.norm(finger_vector)
    return float(gripper_pos)


def convert_human_to_robot_action(thumb_pos: np.ndarray, index_pos: np.ndarray, hand_ee_pos: np.ndarray, 
                                  T_cam2robot: np.ndarray) -> dict:
    """Convert human hand action to robot gripper action."""
    thumb_pos = transform_pt(thumb_pos, T_cam2robot)
    hand_ee_pos = transform_pt(hand_ee_pos, T_cam2robot)
    index_pos = transform_pt(index_pos, T_cam2robot)

    robot_ori_xyzw = convert_human_orientation_to_robot_orientation(thumb_pos, index_pos)
    gripper_action = convert_human_gripper_to_robot_gripper(thumb_pos, index_pos)

    return {
        "pos": hand_ee_pos,
        "quat_xyzw": robot_ori_xyzw,
        "qpos": None,
        "gripper_pos": gripper_action,
    }


def get_finger_poses(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get human finger poses from csv file."""
    finger_poses = pd.read_csv(path)
    finger_poses["thumb"] = finger_poses["thumb"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["index"] = finger_poses["index"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["hand_ee"] = finger_poses["hand_ee"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    thumb_poses = finger_poses["thumb"].values
    index_poses = finger_poses["index"].values
    hand_ee_poses = finger_poses["hand_ee"].values
    return np.vstack(thumb_poses), np.vstack(index_poses), np.vstack(hand_ee_poses)


def get_first_nonzero_index(arr: np.ndarray) -> int:
    """Get first row index where all values in row of array are nonzero."""
    nonzero_index = np.where(arr.sum(axis=1) != 0)[0]
    if len(nonzero_index) > 0:
        return nonzero_index[0]
    return 0

def get_start_idx(thumb_poses: np.ndarray, index_poses: np.ndarray, hand_ee_poses: np.ndarray) -> int:
    """Get index of first frame where hand is visible."""
    thumb_zero_index = get_first_nonzero_index(thumb_poses)
    index_zero_index = get_first_nonzero_index(index_poses)
    hand_ee_zero_index = get_first_nonzero_index(hand_ee_poses)
    return max(thumb_zero_index, index_zero_index, hand_ee_zero_index)


def main(args):
    resolution = "HD1080"
    project_folder = get_parent_folder_of_package("human_shadow")

    # Extrinsics
    camera_extrinsics_path = os.path.join(project_folder, f"human_shadow/camera/camera_calibration_data/hand_calib_{resolution}/cam_cal.json")
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)

    # Intrinsics
    camera_intrinsics_path = os.path.join(project_folder, f"human_shadow/camera/intrinsics/camera_intrinsics_{resolution}.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)

    # Initialize mujoco camera
    camera_params = get_mujoco_camera_params(resolution, camera_extrinsics, camera_intrinsics)

    # Get data paths 
    if args.use_shared:
        videos_folder = os.path.join("/juno/group/human_shadow/processed_data/", args.demo_name)
    else:
        videos_folder = os.path.join(project_folder, f"human_shadow/data/videos/processed/{args.demo_name}/")
    all_video_folders = [f for f in os.listdir(videos_folder) if os.path.isdir(os.path.join(videos_folder, f))]
    all_video_folders = sorted(all_video_folders, key=lambda x: int(x))

    twin_robot = TwinRobot("PandaReal", "Robotiq85", camera_params, camera_res=1080, render=args.render, n_steps_short=3)

    for human_data_subfolder in all_video_folders:
        human_data_folder = os.path.join(videos_folder, human_data_subfolder)

        # Load human finger poses
        finger_poses_path = os.path.join(human_data_folder, "finger_poses.csv")
        thumb_poses, index_poses, hand_ee_poses = get_finger_poses(finger_poses_path)

        # Load human masks
        human_masks_path = os.path.join(human_data_folder, "imgs_hand_masks.mkv")
        human_masks = np.array(media.read_video(human_masks_path, output_format="gray"))
        human_masks[human_masks > 0] = 1


        # Initialize virtual twin robot in mujoco
        start_idx = get_start_idx(thumb_poses, index_poses, hand_ee_poses)
        twin_robot.reset()
        
        # Generate masked images
        list_masked_imgs = []
        list_robot_masks = []
        for idx in tqdm(range(len(thumb_poses))):
            if idx < start_idx:
                list_masked_imgs.append(human_masks[idx])
                continue

            human_mask = human_masks[idx]
            if human_mask.shape != (1080, 1080):
                human_mask = human_mask[:,420:-420]

            target_robot_state = convert_human_to_robot_action(thumb_poses[idx], index_poses[idx], 
                                                            hand_ee_poses[idx], T_cam2robot)

            robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(target_robot_state, init=(idx == start_idx))
            masked_img = np.ones_like(robot_mask) * 255
            masked_img[(robot_mask == 1) | (gripper_mask == 1) | (human_mask == 1)] = 0

            masked_img_robot = np.ones_like(robot_mask) * 255
            masked_img_robot[(robot_mask == 1) | (gripper_mask == 1)] = 0

            list_masked_imgs.append(np.squeeze(masked_img))
            list_robot_masks.append(np.squeeze(masked_img_robot))


        # Save masked images to video
        masks_path = os.path.join(human_data_folder, f"imgs_masks_overlay.mkv")
        media.write_video(masks_path, list_masked_imgs, fps=30, codec="ffv1", input_format="gray")

        robot_masks_path = os.path.join(human_data_folder, f"imgs_robot_masks.mkv")
        media.write_video(robot_masks_path, list_robot_masks, fps=30, codec="ffv1", input_format="gray")
        print(f"Saved masked images to {masks_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_shared", action="store_true")
    parser.add_argument("--resolution", type=str, default="HD1080")
    parser.add_argument("--demo_name", type=str)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()
    main(args)