
import pdb
import pickle
import os
from collections import deque
import copy
import time
import json
from tqdm import tqdm
import pandas as pd
import cv2
import numpy as np 
import logging
import mediapy as media
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple
from scipy.spatial.transform import Rotation

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.camera.zed_utils import ZED_RESOLUTIONS, ZEDResolution
from human_shadow.utils.transform_utils import transform_pt
from robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat
from robosuite.controllers import load_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils
from human_shadow.virtual_twin.twin_robot import TwinRobot, CameraParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def convert_real_camera_ori_to_mujoco(camera_ori_matrix):
    r = Rotation.from_matrix(camera_ori_matrix)
    camera_ori_wxyz = r.as_quat(scalar_first=True)
    r1 = Rotation.from_quat(camera_ori_wxyz)
    r2 = Rotation.from_euler("z", 180, degrees=True)
    new_rot =  r2 * r1
    camera_ori_wxyz = new_rot.as_quat()
    return camera_ori_wxyz


def get_transformation_matrix_from_extrinsics(camera_extrinsics):
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)
    return T_cam2robot


def get_mujoco_camera_params(camera_extrinsics, camera_intrinsics):
    camera_pos = camera_extrinsics[0]["camera_base_pos"]
    camera_ori_matrix = np.array(camera_extrinsics[0]["camera_base_ori"])
    camera_ori_wxyz = convert_real_camera_ori_to_mujoco(camera_ori_matrix)

    fx = camera_intrinsics["left"]["fx"]
    fy = camera_intrinsics["left"]["fy"]
    cx = camera_intrinsics["left"]["cx"]
    cy = camera_intrinsics["left"]["cy"]
    v_fov = camera_intrinsics["left"]["v_fov"]
    camera_resolution = ZED_RESOLUTIONS[resolution]
    img_w = camera_resolution.value[1]
    img_h = camera_resolution.value[0]
    sensor_width = img_w / fy / 1000
    sensor_height = img_h / fx / 1000

    camera_params = CameraParams(
        name="frontview",
        pos=camera_pos,
        ori_wxyz=np.array(camera_ori_wxyz),
        fov=v_fov,
        resolution=ZED_RESOLUTIONS[resolution],
        sensorsize=np.array([sensor_width, sensor_height]),
        principalpixel=np.array([img_w/2-cx, cy-img_h/2]),
        focalpixel=np.array([fx, fy])
    )
    return camera_params


def convert_human_orientation_to_robot_orientation(thumb_pos, index_pos, hand_ee_pos):
    finger_vector = index_pos - thumb_pos
    base_vector = np.array([0, 1, 0])
    angle = np.arccos(np.dot(finger_vector, base_vector) / (np.linalg.norm(finger_vector) * np.linalg.norm(base_vector)))

    base_ori_xyzw = np.array([1, 0, 0, 0])
    r_base = Rotation.from_quat(base_ori_xyzw, scalar_first=False)
    r_human = Rotation.from_euler("z", -angle, degrees=False)
    robot_rot =  r_human * r_base
    robot_ori_xyzw = robot_rot.as_quat(scalar_first=False)
    return robot_ori_xyzw


def convert_human_gripper_to_robot_gripper(thumb_pos, index_pos):
    finger_vector = index_pos - thumb_pos
    gripper_pos = np.linalg.norm(finger_vector)
    return gripper_pos


def convert_human_to_robot_action(thumb_pos, index_pos, hand_ee_pos, T_cam2robot):
    thumb_pos = transform_pt(thumb_pos, T_cam2robot)
    hand_ee_pos = transform_pt(hand_ee_pos, T_cam2robot)
    index_pos = transform_pt(index_pos, T_cam2robot)

    robot_ori_xyzw = convert_human_orientation_to_robot_orientation(thumb_pos, index_pos, hand_ee_pos)
    gripper_action = convert_human_gripper_to_robot_gripper(thumb_pos, index_pos)

    return {
        "pos": hand_ee_pos,
        "quat_xyzw": robot_ori_xyzw,
        "qpos": None,
        "gripper_pos": gripper_action,
    }


def get_finger_poses(path):
    finger_poses = pd.read_csv(finger_poses_path)
    finger_poses["thumb"] = finger_poses["thumb"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["index"] = finger_poses["index"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["hand_ee"] = finger_poses["hand_ee"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    thumb_poses = finger_poses["thumb"].values
    index_poses = finger_poses["index"].values
    hand_ee_poses = finger_poses["hand_ee"].values
    return np.vstack(thumb_poses), np.vstack(index_poses), np.vstack(hand_ee_poses)


def get_first_zero_index(arr):
    zero_index = np.where(arr.sum(axis=1) == 0)[0]
    if len(zero_index) > 0:
        return zero_index[0]
    return 0

def get_start_idx(thumb_poses, index_poses, hand_ee_poses):
    thumb_zero_index = get_first_zero_index(thumb_poses)
    index_zero_index = get_first_zero_index(index_poses)
    hand_ee_zero_index = get_first_zero_index(hand_ee_poses)
    return max(thumb_zero_index, index_zero_index, hand_ee_zero_index)


if __name__ == "__main__":
    resolution = "HD1080"
    project_folder = get_parent_folder_of_package("human_shadow")

    # Extrinsics
    camera_extrinsics_path = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/cam_cal.json")
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    T_cam2robot = get_transformation_matrix_from_extrinsics(camera_extrinsics)

    # Intrinsics
    camera_intrinsics_path = os.path.join(project_folder, f"human_shadow/camera/intrinsics/camera_intrinsics_HD1080.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)


    camera_params = get_mujoco_camera_params(camera_extrinsics, camera_intrinsics)

    # Load human finger poses
    human_data_folder = os.path.join(project_folder, "human_shadow/data/videos/demo_marion_calib_2/0")
    finger_poses_path = os.path.join(human_data_folder, "finger_poses.csv")
    thumb_poses, index_poses, hand_ee_poses = get_finger_poses(finger_poses_path)

    # Load human masks
    human_masks_path = "sam_masks2.mkv"
    human_masks = np.array(media.read_video(human_masks_path, output_format="gray"))
    human_masks[human_masks > 0] = 1


    start_idx = get_start_idx(thumb_poses, index_poses, hand_ee_poses)
    print(f"Start idx: {start_idx}")
    robot_state = convert_human_to_robot_action(thumb_poses[start_idx], index_poses[start_idx], 
                                                hand_ee_poses[start_idx], T_cam2robot)

 
    twin_robot = TwinRobot("PandaReal", "Robotiq85", camera_params, render=False, mode="ee",
                           real_initial_state=robot_state)
    
    list_masked_imgs = []
    for idx in tqdm(range(len(thumb_poses))):
        if idx < start_idx:
            list_masked_imgs.append(human_masks[idx])
            continue

        human_mask = human_masks[idx][:,420:-420]

        robot_state = convert_human_to_robot_action(thumb_poses[idx], index_poses[idx], hand_ee_poses[idx], T_cam2robot)

        # # robot_base_pos = np.array([-0.56, 0, 0.912])
        # # twin_robot.env.set_indicator_pos("thumb", thumb_pos + robot_base_pos)
        # # twin_robot.env.set_indicator_pos("hand_ee", hand_ee_pos + robot_base_pos)
        # # twin_robot.env.set_indicator_pos("index", index_pos + robot_base_pos)

        robot_mask, gripper_mask, rgb_img = twin_robot.get_obs(robot_state)
        masked_img = np.ones_like(robot_mask) * 255
        masked_img[(robot_mask == 1) | (gripper_mask == 1) | (human_mask == 1)] = 0
        list_masked_imgs.append(np.squeeze(masked_img))


    masks_path = os.path.join(project_folder, f"imgs_masks_overlay.mkv")
    media.write_video(masks_path, list_masked_imgs, fps=30, codec="ffv1", input_format="gray")
