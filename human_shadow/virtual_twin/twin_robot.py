import pdb
import pickle
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
from human_shadow.camera.zed_utils import ZED_RESOLUTIONS, ZEDResolution
from robosuite.utils.transform_utils import quat2axisangle, quat2mat, mat2quat
from robosuite.controllers import load_controller_config
from robomimic.envs.env_robosuite import EnvRobosuite
import robomimic.utils.obs_utils as ObsUtils

print("done with imports")


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
    resolution: ZEDResolution
    sensorsize: np.ndarray
    principalpixel: np.ndarray
    focalpixel: np.ndarray


class TwinRobot:
    def __init__(self, robot_name, gripper_name, camera_params, render, mode,
                 real_initial_state): 
        self.robot_name = robot_name
        self.gripper_name = gripper_name
        self.camera_params = camera_params
        self.mode = mode
        self.render = render
        self.n_steps_long = 100
        self.n_steps_short=20
        self.num_frames = 2
        self.T_conversion = None
        self.camera_res = 1080
        self.camera_name = "frontview"

        # Create environment
        obs_spec = dict(
            obs=dict(
                low_dim=["robot0_eef_pos"],
                rgb=[f"{self.camera_params.name}_image"],
            ),
        )
        ObsUtils.initialize_obs_utils_with_obs_specs(
            obs_modality_specs=obs_spec)
                        
        options = {}
        options["env_name"] = "Twin"
        options["robots"] = [self.robot_name]
        options["gripper_types"] = [f"{self.gripper_name}GripperRealPanda"]

        options["controller_configs"] = load_controller_config(default_controller="OSC_POSE")
        options["controller_configs"]["control_delta"] = False
        options["controller_configs"]["uncouple_pos_ori"] = False
        options["camera_heights"] =  self.camera_params.resolution.value[0]
        options["camera_widths"] = self.camera_params.resolution.value[1]
        options["camera_segmentations"] = "instance"
        options["direct_gripper_control"] = True
        options["camera_pos"] = self.camera_params.pos
        options["camera_quat_wxyz"] = self.camera_params.ori_wxyz
        # options["camera_fov"] = self.camera_params.fov
        options["camera_sensorsize"] = self.camera_params.sensorsize
        options["camera_principalpixel"] = self.camera_params.principalpixel
        options["camera_focalpixel"] = self.camera_params.focalpixel


        control_freq = 20

        print("Before env")

        self.env = EnvRobosuite(
            **options,
            has_renderer=render,
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

        print("Initializing twin robot")

        # if self.mode == "qpos":
        #     self.reset_to_qpos(real_qpos, real_gripper_pos)
        # else:
        #     self.move_to_pose(real_ee_pos, real_ee_quat_xyzw, real_gripper_pos, self.n_steps_long)

        print("Done moving")

        self.initial_state = self.env.get_state()["states"]
        self.obs_history = self._get_initial_obs_history(real_initial_state)




    def _get_initial_obs_history(self, state):
        # obs_history = deque(
        #         [self.get_obs(state, init=True) for _ in range(self.num_frames)], 
        #         maxlen=self.num_frames,
        #     )
        obs_history = deque(
                [self.get_obs(state, init=True)], 
                maxlen=self.num_frames,
        )
        for _ in range(self.num_frames-1):
            obs_history.append(self.get_obs(state))
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

    def _convert_gripper_pos_to_action(self, gripper_pos): 
        min_gripper_pos = 0.0
        max_gripper_pos = 0.085
        gripper_pos = np.clip(gripper_pos, min_gripper_pos, max_gripper_pos)
        closed_gripper_action = 255
        open_gripper_action = 0

        # gripper pos is 0.085, gripper action is 0
        # gripper pos is 0, gripper action is 255
        return np.interp(gripper_pos, [min_gripper_pos, max_gripper_pos], [closed_gripper_action, open_gripper_action])


    def move_to_pose(self, ee_pos, ee_ori, gripper_action, n_steps):
        print("EE pos: ", ee_pos, "EE ori: ", ee_ori)
        action = get_action_from_ee_pose_panda(ee_pos, ee_ori, gripper_action, self.gripper_name, use_base_offset=True)
        print("Action: ", action)
        for _ in tqdm(range(n_steps)):
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

    twin_robot = TwinRobot("PandaReal", "Robotiq85", camera_params, render=True, mode="ee",
                           real_initial_state=real_initial_state)
    
    for img_idx in tqdm(range(len(data_list))):
        robot_qpos = data_list[img_idx]["qpos"]
        robot_pos = data_list[img_idx]["pos"]
        robot_ori_xyzw = data_list[img_idx]["ori"]
        real_img = data_list[img_idx]["imgs"][0]
        real_img = real_img[:,420:-420]

        state = {
            "pos": robot_pos,
            "quat_xyzw": robot_ori_xyzw,
            "qpos": robot_qpos,
            "gripper_pos": 0.0
        }

        robot_mask, gripper_mask, rgb_img = twin_robot.get_obs(state)

        masked_img = np.copy(real_img)

        try:
            robot_mask = np.squeeze(robot_mask)
            gripper_mask = np.squeeze(gripper_mask)
            masked_img[(robot_mask == 1) | (gripper_mask == 1)] = 0
        except:
            pdb.set_trace()


        robot_mask_img = np.repeat(robot_mask[:, :, np.newaxis], 3, axis=2)*255
        rgb_img = rgb_img * 255
        rgb_img = rgb_img.astype(np.uint8)
        debug_image_1 = np.hstack([robot_mask_img, rgb_img])
        debug_image_2 = np.hstack([real_img, masked_img])
        debug_image = np.vstack([debug_image_1, debug_image_2])
        cv2.imwrite(f"debug_images4/img_{img_idx}.png", debug_image)



    pdb.set_trace()

    print("plotting")
    
    fig = plt.figure()
    ax = fig.add_subplot(151)
    ax.imshow(robot_mask, aspect="equal")
    ax.axis("off")
    ax = fig.add_subplot(152)
    ax.imshow(gripper_mask, aspect="equal")
    ax.axis("off")
    ax = fig.add_subplot(153)
    ax.imshow(rgb_img, aspect="equal")
    ax.axis("off")
    ax = fig.add_subplot(154)
    ax.imshow(real_img, aspect="equal")
    ax.axis("off")
    ax = fig.add_subplot(155)
    ax.imshow(masked_img, aspect="equal")
    ax.axis("off")
    print("Showing images")
    plt.show()
    
    twin_robot.move_to_pose
    pdb.set_trace()
