import pdb
import os
from collections import deque
import copy
import time
import json
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
from robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat
from robosuite.controllers import load_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils


def get_action_from_ee_pose_panda(ee_pos, ee_quat_xyzw, gripper_action, gripper, mode="ee", use_base_offset=False):
    if len(ee_pos.shape) > 1:
        ee_pos = ee_pos[-1]
        ee_quat_xyzw = ee_quat_xyzw[-1]
    if use_base_offset:
        robot_base_pos = np.array([-0.56, 0, 0.912])
        ee_pos = ee_pos + robot_base_pos
    mat = quat2mat(ee_quat_xyzw)
    if mode == "ee" and gripper == "Robotiq85":
        quat_90deg = np.array([0, 0, -0.7071068, 0.7071068])
        quat_45deg = np.array([0, 0, -0.3826834, 0.9238795])
        mat_90deg = quat2mat(quat_90deg)
        mat_45deg = quat2mat(quat_45deg)
        mat_rotated = np.dot(mat, mat_90deg)
        mat_rotated = np.dot(mat_rotated, mat_45deg)
    else:
        quat_90deg = np.array([0, 0, -0.7071068, 0.7071068])
        mat_90deg = quat2mat(quat_90deg)
        mat_rotated = np.dot(mat, mat_90deg)

    quat_rotated = mat2quat(mat_rotated)
    axis_angle = quat2axisangle(quat_rotated)
    action = np.concatenate([ee_pos, axis_angle, np.array([gripper_action])])
    return action

def convert_action(action_pos, action_quat, T_conversion):
    # print("Action before: ", action_pos)
    pos = np.array([action_pos[0], action_pos[1], action_pos[2], 1])
    new_pos = T_conversion @ pos

    rot = Rotation.from_quat(action_quat)
    rot_matrix = rot.as_matrix()
    new_rot_matrix = T_conversion[:3, :3] @ rot_matrix
    rot2 = Rotation.from_matrix(new_rot_matrix)
    new_ori = rot2.as_quat()
    # print("Action after: ", new_pos[:3])
    return new_pos[:3], new_ori

@dataclass
class CameraParams:
    name: str
    pos: np.ndarray
    ori_wxyz: np.ndarray
    fov: float
    resolution: int


class TwinRobot:
    def __init__(self, robot_name, gripper_name, camera_params, render, real_initial_state): 
        self.robot_name = robot_name
        self.gripper_name = gripper_name
        self.camera_params = camera_params
        self.render = render


        # Create environment
        obs_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[f"{self.camera_params.name}_image"],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(
            obs_modality_specs=obs_spec)
        
        pdb.set_trace()
        
        options = {}
        options["env_name"] = "Twin"
        options["robots"] = [self.robot_name]
        options["gripper_types"] = [f"{self.gripper_name}GripperRealPanda"]

        options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
        options["controller_configs"]["control_delta"] = False
        options["controller_configs"]["uncouple_pos_ori"] = False
        options["camera_heights"] =  self.camera_params.resolution[0]
        options["camera_widths"] = self.camera_params.resolution[1]
        options["camera_segmentations"] = "instance"
        options["direct_gripper_control"] = True
        options["camera_pos"] = self.camera_params.pos
        options["camera_quat_wxyz"] = self.camera_params.ori_wxyz
        options["camera_fov"] = self.camera_params.fov

        control_freq = 20

        self.env = EnvRobosuite(
            **options,
            has_renderer=True,
            has_offscreen_renderer=True,
            ignore_done=True,
            use_camera_obs=True,
            use_image_obs=True,
            camera_names=self.camera_params.name,
            control_freq=control_freq,
        )

        self.env.reset()
        self.initial_state = self.env.get_state()["states"]

        # Initialize the twin sim to the pose of the current pose of the real robot
        real_ee_pos = real_initial_state["pos"]
        real_ee_quat_xyzw = real_initial_state["quat_xyzw"]
        real_qpos = real_initial_state["qpos"]
        real_gripper_pos = real_initial_state["gripper_pos"]

        if self.mode == "qpos":
            self.reset_to_qpos(real_qpos, real_gripper_pos)
        else:
            self.move_to_pose(real_ee_pos, real_ee_quat_xyzw, real_gripper_pos, self.n_steps_long)

        self.initial_state = self.env.get_state()["states"]
        self.obs_history = self._get_initial_obs_history(real_initial_state)


    def _get_initial_obs_history(self, state):
        obs_history = deque(
                [self.get_obs(state, init=True) for _ in range(self.num_frames)], 
                maxlen=self.num_frames,
            )
        return obs_history
    
    def get_obs_history(self, state):
        self.obs_history.append(self.get_obs(state))
        return list(self.obs_history)
    
    def get_obs(self, state, init=False):
        gripper_action = self._convert_gripper_pos_to_action(state["gripper_pos"])
        if self.mode == "qpos":
            if init:
                obs = self.reset_to_qpos(state["qpos"], gripper_action)
            else:
                if self.T_conversion is not None:
                    new_pos, new_ori = convert_action(state["pos"], state["quat_xyzw"], self.T_conversion)
                    obs = self.move_to_pose(new_pos, new_ori, gripper_action, self.n_steps_short)
                else:
                    obs = self.move_to_pose(state["pos"], state["quat_xyzw"], gripper_action, self.n_steps_short)
        else:
            if self.T_conversion is not None:
                pos_command, ori_command = convert_action(state["pos"], state["quat_xyzw"], self.T_conversion)
            else:
                pos_command = state["pos"]
                ori_command = state["quat_xyzw"]
            if init:
                obs = self.move_to_pose(pos_command, ori_command, gripper_action, self.n_steps_long)
            else:
                obs = self.move_to_pose(pos_command, ori_command, gripper_action, self.n_steps_short)

        robot_mask = self.get_robot_mask(obs)
        gripper_mask = self.get_gripper_mask(obs)
        rgb_img = self.get_image(obs, self.camera_res)

        return (robot_mask, gripper_mask, rgb_img)


    def move_to_pose(self, ee_pos, ee_ori, gripper_action, n_steps):
        action = get_action_from_ee_pose_panda(ee_pos, ee_ori, gripper_action, self.gripper, use_base_offset=True)
        for _ in range(n_steps):
            obs, _, _, _ = self.env.step(action)
        return obs


    def get_image(self, obs, camera_res):
        img = obs[f"{self.camera_name}_image"]
        img = img.transpose(1, 2, 0)
        height = img.shape[0]
        width = img.shape[1]
        n_remove = int((width - height)/2)
        img = img[:,n_remove:-n_remove,:]   
        img = cv2.resize(img, (camera_res,camera_res))
        return img
    
    def get_seg_image(self, obs):
        img = obs["frontview_segmentation_instance"]
        height = img.shape[0]
        width = img.shape[1]
        n_remove = int((width - height)/2)
        img = img[:,n_remove:-n_remove,:]  
        img = img.astype(np.uint8)
        return img
    
    def get_robot_mask(self, obs):
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 1] = 1
        return mask
    
    def get_gripper_mask(self, obs):
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 3] = 1
        return mask




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

    twin_robot = TwinRobot("Panda", "Robotiq85", camera_params, render=True)
    pdb.set_trace()
