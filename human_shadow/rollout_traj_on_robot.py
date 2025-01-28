import pdb 
import matplotlib.pyplot as plt
import numpy as np
import json
import argparse
import time
from tqdm import tqdm
from scipy.spatial.transform import Rotation

from franka_utils.opspace_client import FrankaPanda, decode_matlab
from human_shadow.robots.robotiq85_gripper import Robotiq85Gripper
from human_shadow.config.redis_keys import *

def transform_pt(pt, T):
    pt = np.array(pt)
    pt = np.append(pt, 1)
    pt = np.dot(T, pt)
    return pt[:3]


def main(args):
    # Camera extrinsics
    camera_extrinsics_path = "camera/camera_calibration_data/hand_calib_HD1080/cam_cal.json"
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)

    tip_points_path = "/juno/u/lepertm/shadow/human_shadow/human_shadow/data/videos/demo_marion_calib_2/0/tip_points.npy"
    tip_points = np.load(tip_points_path)
    tip_points = np.squeeze(tip_points)

    thumb_points_path = "/juno/u/lepertm/shadow/human_shadow/human_shadow/data/videos/demo_marion_calib_2/0/thumb_tip_points.npy"
    thumb_points = np.load(thumb_points_path)
    thumb_points = np.squeeze(thumb_points)

    index_points_path = "/juno/u/lepertm/shadow/human_shadow/human_shadow/data/videos/demo_marion_calib_2/0/index_tip_points.npy"
    index_points = np.load(index_points_path)
    index_points = np.squeeze(index_points)

    gripper_thresh = 0.03
    hand_ee_pos = []
    hand_thumb_pos = []
    hand_index_pos = []
    gripper_distance = []
    robotiq_command = []
    for tip, thumb, index in zip(tip_points, thumb_points, index_points):
        hand_ee_pt = transform_pt(tip, T_cam2robot)
        hand_index_pt = transform_pt(index, T_cam2robot)
        hand_thumb_pt = transform_pt(thumb, T_cam2robot)

        hand_ee_pos.append(hand_ee_pt)
        hand_index_pos.append(hand_index_pt)
        hand_thumb_pos.append(hand_thumb_pt)
        
        dist = np.linalg.norm(hand_thumb_pt - hand_index_pt)
        if dist > gripper_thresh:
            robotiq_command.append(0)
        else:
            robotiq_command.append(1)
        gripper_distance.append(dist)


# 0.01277384 -0.02124954  0.65372316
    hand_ee_pos = np.array(hand_ee_pos)
    robotiq_command = np.array(robotiq_command)

    n_pts = len(hand_ee_pos)

    # Initialize connection to robot driver redis server
    if args.sim:
        robot = FrankaPanda()  # Local host, for sim
        gripper = Robotiq85Gripper(host=NUC_HOST, password=NUC_PWD)
    else:
        # Franka NUC IP address - to communicate w/ Redis
        robot = FrankaPanda(host="172.24.68.230", password="iprl")
        gripper = Robotiq85Gripper(host=NUC_HOST, password=NUC_PWD)

    robot.go_home()

    eef_pos_cur, eef_quatxyzw_cur = robot.get_pose()
    rot = Rotation.from_quat(eef_quatxyzw_cur, scalar_first=False)
    # Rotate by 90 deg about z
    rot = rot * Rotation.from_euler("z", -np.pi/2)
    eef_quatxyzw_cur = rot.as_quat()
    robot.goto_pose_with_traj(hand_ee_pos[0], eef_quatxyzw_cur)



    for idx in tqdm(range(n_pts)):
        # Get pose of eef in world frame
        # eef_pos_cur, eef_quatxyzw_cur = robot.get_pose()

        robot.goto_pose(hand_ee_pos[idx], eef_quatxyzw_cur)
        gripper.goto_pose(pos=robotiq_command[idx])

        time.sleep(0.2)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim", action="store_true", help="Run in simulation mode")
    args = parser.parse_args()
    main(args)



# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.plot(gripper_distance)
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# hand_ee_pos = np.array(hand_ee_pos)
# hand_thumb_pos = np.array(hand_thumb_pos)
# hand_index_pos = np.array(hand_index_pos)
# ax.scatter(hand_ee_pos[:,0], hand_ee_pos[:,1], hand_ee_pos[:,2], c='r', marker='o')
# ax.scatter(hand_thumb_pos[:,0], hand_thumb_pos[:,1], hand_thumb_pos[:,2], c='g', marker='o')
# ax.scatter(hand_index_pos[:,0], hand_index_pos[:,1], hand_index_pos[:,2], c='b', marker='o')
# ax.set_xlabel('X Label')
# ax.set_ylabel('Y Label')
# ax.set_zlabel('Z Label')
# plt.show()


pdb.set_trace()