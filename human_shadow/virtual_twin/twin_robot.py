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


@dataclass
class MujocoCameraParams:
    name: str
    pos: np.ndarray
    ori_wxyz: np.ndarray
    fov: float
    resolution: ZEDResolution
    sensorsize: np.ndarray
    principalpixel: np.ndarray
    focalpixel: np.ndarray


THUMB_COLOR = [0, 1, 0, 1]
INDEX_COLOR = [1, 0, 0, 1]
HAND_EE_COLOR = [0, 0, 1, 1]

def convert_real_camera_ori_to_mujoco(camera_ori_matrix: np.ndarray) -> np.ndarray:
    """Convert camera orientation from real world to mujoco xml format."""
    r = Rotation.from_matrix(camera_ori_matrix)
    camera_ori_wxyz = r.as_quat(scalar_first=True)
    r1 = Rotation.from_quat(camera_ori_wxyz)
    r2 = Rotation.from_euler("z", 180, degrees=True)
    new_rot =  r2 * r1
    camera_ori_wxyz = new_rot.as_quat()
    return camera_ori_wxyz

def get_mujoco_camera_params(resolution: str, camera_extrinsics: list[dict], 
                             camera_intrinsics: dict) -> MujocoCameraParams:
    """Get mujoco camera parameters from real world camera parameters."""
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

    camera_params = MujocoCameraParams(
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

def get_transformation_matrix_from_extrinsics(camera_extrinsics: list[dict]) -> np.ndarray:
    """Get homogeneous transformation matrix from camera extrinsics."""
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)
    return T_cam2robot

class TwinRobot:
    """Virtual twin robot of Franka panda in mujoco"""
    def __init__(self, robot_name: str, gripper_name: str, camera_params: MujocoCameraParams, 
                 camera_res: int, render: bool, n_steps_short: int, visualize_sites=False): 
        self.robot_name = robot_name
        self.gripper_name = gripper_name
        self.camera_params = camera_params
        self.render = render
        self.n_steps_long = 100
        self.n_steps_short= n_steps_short
        self.num_frames = 2 # Observation history length
        self.camera_res = camera_res
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
        if not visualize_sites:
            options["camera_segmentations"] = "instance"
        options["direct_gripper_control"] = True
        options["camera_pos"] = self.camera_params.pos
        options["camera_quat_wxyz"] = self.camera_params.ori_wxyz
        options["camera_sensorsize"] = self.camera_params.sensorsize
        options["camera_principalpixel"] = self.camera_params.principalpixel
        options["camera_focalpixel"] = self.camera_params.focalpixel

        self.env = EnvRobosuite(
            **options,
            render=render,
            render_offscreen=True,
            use_image_obs=True,
            camera_names=self.camera_params.name,
            control_freq=20,
            visualize_sites=False,
            indicator_configs=TwinRobot.generate_hand_sites(0.01),
        )

        self.reset()
        self.robot_base_pos = np.array([-0.56, 0, 0.912])

    def reset(self):
        """Reset environment."""
        self.env.reset()
        self.obs_history = deque()


    def get_action_from_ee_pose_panda(self, ee_pos: np.ndarray, ee_quat_xyzw: np.ndarray, gripper_action: float,
                                       use_base_offset: bool=False) -> np.ndarray:
        """Convert panda's end effector pose to action."""
        if len(ee_pos.shape) > 1:
            ee_pos = ee_pos[-1]
            ee_quat_xyzw = ee_quat_xyzw[-1]
        if use_base_offset:
            ee_pos = ee_pos + self.robot_base_pos
        mat = quat2mat(ee_quat_xyzw)

        # TODO: Fix this!!!
        quat_90deg = np.array([0, 0, -0.7071068, 0.7071068])
        quat_45deg = np.array([0, 0, -0.3826834, 0.9238795])
        mat_90deg = quat2mat(quat_90deg)
        mat_45deg = quat2mat(quat_45deg)
        mat_rotated = np.dot(mat, mat_90deg)
        mat_rotated = np.dot(mat_rotated, mat_45deg)

        quat_rotated = mat2quat(mat_rotated)
        axis_angle = quat2axisangle(quat_rotated)
        action = np.concatenate([ee_pos, axis_angle, np.array([gripper_action])])
        return action

    def _get_initial_obs_history(self, state: dict) -> deque:
        """Get initial observation history by repeating the first observation."""
        obs_history = deque(
                [self.move_to_target_state(state, init=True)], 
                maxlen=self.num_frames,
        )
        for _ in range(self.num_frames-1):
            obs_history.append(self.move_to_target_state(state))
        return obs_history
    
    def get_obs_history(self, state: dict) -> list:
        """Get observation history. History length is self.num_frames."""
        if len(self.obs_history) == 0:
            self.obs_history = self._get_initial_obs_history(state)
        else:
            self.obs_history.append(self.move_to_target_state(state))
        return list(self.obs_history)
    
    def move_to_target_state(self, state: dict, init=False) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Move to target state and return observation."""
        gripper_action = self._convert_handgripper_pos_to_action(state["gripper_pos"])
        n_steps = self.n_steps_long if init else self.n_steps_short
        obs = self.move_to_pose(state["pos"], state["quat_xyzw"], gripper_action, n_steps)

        robot_mask = np.squeeze(self.get_robot_mask(obs))
        gripper_mask = np.squeeze(self.get_gripper_mask(obs))
        rgb_img = self.get_image(obs, self.camera_res)

        return (robot_mask, gripper_mask, rgb_img)

    def _convert_handgripper_pos_to_action(self, gripper_pos: float) -> float:
        """Convert hand gripper position to robot gripper action."""
        min_gripper_pos, max_gripper_pos = 0.0, 0.085
        gripper_pos = np.clip(gripper_pos, min_gripper_pos, max_gripper_pos)
        open_gripper_action, closed_gripper_action = 0, 255
        return np.interp(gripper_pos, [min_gripper_pos, max_gripper_pos], [closed_gripper_action, open_gripper_action])

    def move_to_pose(self, ee_pos: np.ndarray, ee_ori: np.ndarray, gripper_action: float, n_steps: int) -> dict:
        """Move to target end effector pose and return observation."""
        action = self.get_action_from_ee_pose_panda(ee_pos, ee_ori, gripper_action, use_base_offset=True)
        for _ in tqdm(range(n_steps), leave=False):
            obs, _, _, _ = self.env.step(action)
            if self.render:
                self.env.render(camera_name="sideview")
        return obs

    def get_image(self, obs: dict, camera_res: int) -> np.ndarray:
        """Get rgb image from observation."""
        img = obs[f"{self.camera_name}_image"]
        img = img.transpose(1, 2, 0)
        height = img.shape[0]
        width = img.shape[1]
        n_remove = int((width - height)/2)
        img = img[:,n_remove:-n_remove,:]   
        img = cv2.resize(img, (camera_res,camera_res))
        return img
    
    def get_seg_image(self, obs: dict) -> np.ndarray:
        """Get segmentation image from observation."""
        img = obs["frontview_segmentation_instance"]
        height = img.shape[0]
        width = img.shape[1]
        n_remove = int((width - height)/2)
        img = img[:,n_remove:-n_remove,:]  
        img = img.astype(np.uint8)
        return img
    
    def get_robot_mask(self, obs: dict) -> np.ndarray:
        """Get robot mask from observation."""
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 1] = 1
        return mask
    
    def get_gripper_mask(self, obs: dict) -> np.ndarray:
        """Get gripper mask from observation."""
        seg_img = self.get_seg_image(obs)
        mask = np.zeros_like(seg_img)
        mask[seg_img == 3] = 1
        return mask

    @staticmethod
    def generate_hand_sites(size: float):
        """Generate hand keypoint sites for visualization."""
        thumb_site_config={"type": "sphere", "size": [size], "rgba": THUMB_COLOR, "name": "thumb",}
        index_site_config={"type": "sphere", "size": [size], "rgba": INDEX_COLOR, "name": "index",}
        hand_ee_site_config={"type": "sphere", "size": [size],"rgba": HAND_EE_COLOR, "name": "hand_ee",}
        return [thumb_site_config, index_site_config, hand_ee_site_config]
        

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

    # Initialize mujoco camera
    camera_params = get_mujoco_camera_params(resolution, camera_extrinsics, camera_intrinsics)

    # Load calibration pickle
    project_folder = get_parent_folder_of_package("human_shadow")
    cal_pkl = os.path.join(project_folder, f"human_shadow/camera/camera_calibration_data/hand_calib_{resolution}/calibration_data.pkl")
    with open(cal_pkl, "rb") as f:
        data_list = pickle.load(f)

    twin_robot = TwinRobot("PandaReal", "Robotiq85", camera_params, camera_res=1080, render=False, 
                           n_steps_short=50)
    
    for img_idx in tqdm(range(len(data_list))):
        robot_qpos = data_list[img_idx]["qpos"]
        robot_pos = data_list[img_idx]["pos"]
        robot_ori_xyzw = data_list[img_idx]["ori"]
        real_img = data_list[img_idx]["imgs"][0]
        real_img = real_img[:,420:-420]

        target_robot_state = {
            "pos": robot_pos,
            "quat_xyzw": robot_ori_xyzw,
            "qpos": robot_qpos,
            "gripper_pos": 0.0
        }

        robot_mask, gripper_mask, rgb_img = twin_robot.move_to_target_state(target_robot_state)

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
        cv2.imwrite(f"debug_images/img_{img_idx}.png", debug_image)


