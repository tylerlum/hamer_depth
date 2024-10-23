
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



if __name__ == "__main__":
    resolution = "HD1080"
    project_folder = get_parent_folder_of_package("human_shadow")

    # Extrinsics
    camera_extrinsics_path = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/cam_cal.json")
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    camera_pos = camera_extrinsics[0]["camera_base_pos"]
    camera_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    r = Rotation.from_matrix(camera_ori)
    camera_ori_wxyz = r.as_quat(scalar_first=True)
    r1 = Rotation.from_quat(camera_ori_wxyz)
    r2 = Rotation.from_euler("z", 180, degrees=True)
    new_rot =  r2 * r1
    camera_ori_wxyz = new_rot.as_quat()

    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)


    # Intrinsics
    camera_intrinsics_path = os.path.join(project_folder, f"human_shadow/camera/intrinsics/camera_intrinsics_HD1080.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)
    fx = camera_intrinsics["left"]["fx"]
    fy = camera_intrinsics["left"]["fy"]
    cx = camera_intrinsics["left"]["cx"]
    cy = camera_intrinsics["left"]["cy"]
    v_fov = camera_intrinsics["left"]["v_fov"]
    h_fov = camera_intrinsics["left"]["h_fov"]
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


    human_data_folder = os.path.join(project_folder, "human_shadow/data/videos/demo_marion_calib_2/0")

    # Load finger poses
    finger_poses_path = os.path.join(human_data_folder, "finger_poses.csv")
    finger_poses = pd.read_csv(finger_poses_path)
    n_steps = len(finger_poses)
    finger_poses["thumb"] = finger_poses["thumb"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["index"] = finger_poses["index"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    finger_poses["hand_ee"] = finger_poses["hand_ee"].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
    idx = 0 
    thumb_pos = finger_poses["thumb"].iloc[idx]
    hand_ee_pos = finger_poses["hand_ee"].iloc[idx]
    index_pos = finger_poses["index"].iloc[idx]

    thumb_pos = transform_pt(thumb_pos, T_cam2robot)
    hand_ee_pos = transform_pt(hand_ee_pos, T_cam2robot)
    index_pos = transform_pt(index_pos, T_cam2robot)

    # Load calibration pickle
    project_folder = get_parent_folder_of_package("human_shadow")
    cal_pkl = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/calibration_data.pkl")
    with open(cal_pkl, "rb") as f:
        data_list = pickle.load(f)
    img_num = 0
    robot_qpos = data_list[img_num]["qpos"]
    robot_pos = data_list[img_num]["pos"]
    robot_ori_xyzw = data_list[img_num]["ori"]
    real_img = data_list[img_num]["imgs"][0]
    real_img = real_img[:,420:-420]

    real_initial_state = {
        "pos": robot_pos,
        "quat_xyzw": robot_ori_xyzw,
        "qpos": robot_qpos,
        "gripper_pos": 0.0
    }

    twin_robot = TwinRobot("PandaReal", "Robotiq85", camera_params, render=False, mode="ee",
                           real_initial_state=real_initial_state)
    

    list_masked_imgs = []
    for idx in tqdm(range(n_steps)):
        thumb_pos = finger_poses["thumb"].iloc[idx]
        hand_ee_pos = finger_poses["hand_ee"].iloc[idx]
        index_pos = finger_poses["index"].iloc[idx]

        thumb_pos = transform_pt(thumb_pos, T_cam2robot)
        hand_ee_pos = transform_pt(hand_ee_pos, T_cam2robot)
        index_pos = transform_pt(index_pos, T_cam2robot)

        robot_base_pos = np.array([-0.56, 0, 0.912])
        # twin_robot.env.set_indicator_pos("thumb", thumb_pos + robot_base_pos)
        # twin_robot.env.set_indicator_pos("hand_ee", hand_ee_pos + robot_base_pos)
        # twin_robot.env.set_indicator_pos("index", index_pos + robot_base_pos)


        finger_vector = index_pos - thumb_pos
        base_vector = np.array([0, 1, 0])
        angle = np.arccos(np.dot(finger_vector, base_vector) / (np.linalg.norm(finger_vector) * np.linalg.norm(base_vector)))
        print("Angle", angle)
        print("Dist: ", np.linalg.norm(finger_vector))

        robot_ori_xyzw = np.array([1, 0, 0, 0])
        r_robot = Rotation.from_quat(robot_ori_xyzw, scalar_first=False)
        r_new = Rotation.from_euler("z", -angle, degrees=False)
        new_rot =  r_new * r_robot
        new_ori_xyzw = new_rot.as_quat(scalar_first=False)

        if idx == 0:
            n_rollout_steps = 100
        else:
            n_rollout_steps = 3

        # if np.linalg.norm(finger_vector) > 0.03:
        #     gripper_action = 0.0
        # else:
        #     gripper_action = 255
        # gripper_action = twin_robot._convert_gripper_pos_to_action(np.linalg.norm(finger_vector))
        # obs = twin_robot.move_to_pose(hand_ee_pos, new_ori_xyzw, gripper_action, n_rollout_steps)

        state = {
            "pos": hand_ee_pos,
            "quat_xyzw": new_ori_xyzw,
            "gripper_pos" : np.linalg.norm(finger_vector),
        }
        robot_mask, gripper_mask, rgb_img = twin_robot.get_obs(state)
        masked_img = np.zeros_like(robot_mask)
        masked_img[(robot_mask == 1) | (gripper_mask == 1)] = 1
        masked_img = np.tile(masked_img, (1,1,3))
        masked_img = masked_img * 255
        list_masked_imgs.append(masked_img)

        # pdb.set_trace()
        # plt.imshow(masked_img)
        # plt.show()
        # img = obs["frontview_image"]
        # img = np.transpose(img, (1, 2, 0))
        # plt.imshow(img)
        # plt.show()
        # pdb.set_trace()
        # twin_robot.env.env.viewer.set_camera(camera_id=2)
        # frontview', 'birdview', 'agentview', 'sideview', 'robot0_robotview', 'robot0_eye_in_hand'
        # for _ in range(1000):
        #     twin_robot.env.render(camera_name="sideview")
            
        # for img_idx in tqdm(range(len(data_list))):
        #     robot_qpos = data_list[img_idx]["qpos"]
        #     robot_pos = data_list[img_idx]["pos"]
        #     robot_ori_xyzw = data_list[img_idx]["ori"]
        #     real_img = data_list[img_idx]["imgs"][0]
        #     real_img = real_img[:,420:-420]

        #     state = {
        #         "pos": robot_pos,
        #         "quat_xyzw": robot_ori_xyzw,
        #         "qpos": robot_qpos,
        #         "gripper_pos": 0.0
        #     }

        #     robot_mask, gripper_mask, rgb_img = twin_robot.get_obs(state)

        #     plt.imshow(rgb_img)
        #     plt.show()

            # masked_img = np.copy(real_img)

            # try:
            #     robot_mask = np.squeeze(robot_mask)
            #     gripper_mask = np.squeeze(gripper_mask)
            #     masked_img[(robot_mask == 1) | (gripper_mask == 1)] = 0
            # except:
            #     pdb.set_trace()


            # robot_mask_img = np.repeat(robot_mask[:, :, np.newaxis], 3, axis=2)*255
            # rgb_img = rgb_img * 255
            # rgb_img = rgb_img.astype(np.uint8)
            # debug_image_1 = np.hstack([robot_mask_img, rgb_img])
            # debug_image_2 = np.hstack([real_img, masked_img])
            # debug_image = np.vstack([debug_image_1, debug_image_2])
            # # cv2.imwrite(f"debug_images4/img_{img_idx}.png", debug_image)
            # plt.imshow(debug_image)
            # plt.show()

    print("Writing video")
    media.write_video("masked_imgs.avi", list_masked_imgs, fps=30, codec="ffv1")
    print("Done")