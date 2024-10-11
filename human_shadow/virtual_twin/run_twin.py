import pdb
import os
from collections import deque
import copy
import time
import json
import pickle
from tqdm import tqdm
import cv2
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from dataclasses import dataclass
from typing import Tuple
from scipy.spatial.transform import Rotation

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.camera.zed_utils import ZED_RESOLUTIONS
from human_shadow.virtual_twin.twin_robot import TwinRobot, CameraParams
from robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat
from robosuite.controllers import load_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils


def plot(obs):
    img = obs["frontview_image"]
    # pdb.set_trace()
    img = img.transpose(1, 2, 0)
    seg_img = obs["frontview_segmentation_instance"]
    mask = np.zeros_like(seg_img)
    mask[seg_img == 1] = 1
    mask[seg_img == 3] = 1

    mask_rgb_img = copy.deepcopy(real_img)
    mask_rgb_img[mask.squeeze() == 1] = 0

    fig = plt.figure()
    plt.imshow(mask_rgb_img, aspect="equal")
    # ax = fig.add_subplot(231)
    # ax.imshow(real_img, aspect="equal")
    # ax.axis("off")
    # ax = fig.add_subplot(232)
    # ax.imshow(img, aspect="equal")
    # ax.axis("off")
    # ax = fig.add_subplot(233)
    # ax.imshow(seg_img, aspect="equal")
    # ax.axis("off")
    # ax = fig.add_subplot(234)
    # ax.imshow(mask, aspect="equal")
    # ax.axis("off")
    # ax = fig.add_subplot(235)
    # ax.imshow(mask_rgb_img, aspect="equal")
    # ax.axis("off")
    plt.show()
    plt.savefig("mask_rgb_img.png")
    pdb.set_trace()


if __name__ == "__main__":
    resolution = "HD1080"
    project_folder = get_parent_folder_of_package("human_shadow")
    camera_extrinsics_path = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/cam_cal.json")
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)

    camera_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    r = Rotation.from_matrix(camera_ori)
    camera_ori_wxyz = r.as_quat(scalar_first=True)
    camera_ori_wxyz = np.array([-0.82049655, -0.56472998, 0.0544513, 0.07000374])

    camera_intrinsics_path = os.path.join(project_folder, f"human_shadow/camera/intrinsics/camera_intrinsics_HD1080.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)

    camera_params = CameraParams(
        name="frontview",
        pos=np.array(camera_extrinsics[0]["camera_base_pos"]),
        ori_wxyz=np.array(camera_ori_wxyz),
        fov=camera_intrinsics["left"]["v_fov"],
        resolution=ZED_RESOLUTIONS[resolution],
    )


    # Load calibration pickle
    project_folder = get_parent_folder_of_package("human_shadow")
    cal_pkl = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/calibration_data.pkl")
    with open(cal_pkl, "rb") as f:
        data_list = pickle.load(f)
    img_num = 14
    robot_qpos = data_list[img_num]["qpos"]
    robot_pos = data_list[img_num]["pos"]
    robot_ori_xyzw = data_list[img_num]["ori"]
    real_img = data_list[img_num]["imgs"][0]


    real_initial_state = {
        "pos": robot_pos,
        "ori": robot_ori_xyzw,
        "qpos": robot_qpos,
    }


    twin_robot = TwinRobot("Panda", "Robotiq85", camera_params, render=True, real_initial_state=real_initial_state)
    robot_mask, gripper_mask, rgb_img = twin_robot.get_obs(real_initial_state, init=True)


    fig, axs = plt.subplots(1, 3)
    axs[0].imshow(rgb_img)
    axs[1].imshow(robot_mask)
    axs[2].imshow(gripper_mask)
    plt.show()
