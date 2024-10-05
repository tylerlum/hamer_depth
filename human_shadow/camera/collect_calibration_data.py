import numpy as np
import pdb
import msgpack_numpy as m
m.patch()
import ast
import pandas as pd
import time
import redis
import cv2
import os
import argparse
import pickle
from scipy.spatial.transform import Rotation as R
import json

from human_shadow.config.redis_keys import *

from franka_utils.opspace_client import FrankaPanda, decode_matlab
import franka_utils.traj_utils as t_utils
import franka_utils.rotation_utils as r_utils

"""
Script to collect camera calibration data with robot
Data collection setup:
1. Prepare robot. Close robotiq gripper and attach AR tag to the magnetic pins on the fingers
2. Start camera by running zed_redis.py. No visualizations will be displayed. 
3. Run script (test in sim first)

Test calibration:
"""

def main(args):
    save_dir = os.path.join("camera_calibration_data/", args.name)

    # Create save_dir for data
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Create directory for saving images
    img_dir = os.path.join(save_dir, "imgs")
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    # Initialize connection to robot driver redis server
    if args.sim:
        robot = FrankaPanda()  # Local host, for sim
    else:
        # Franka NUC IP address - to communicate w/ Redis
        robot = FrankaPanda(host="172.24.68.230", password="iprl")

    # Connect to local redis server for Zed
    _redis = redis.Redis(
        host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
    )
    redis_pipe = _redis.pipeline()

    def str_to_list(s):
        return list(map(float, s.strip('[]').split()))

    # Load target calibration poses
    csv_path = "calibration_poses.csv"
    df = pd.read_csv(csv_path, converters={"pos": str_to_list, "quat_xyzw": str_to_list})
    pos_waypt_list = df["pos"]
    quat_waypt_list = df["quat_xyzw"]

    cal_data = []  # To log calibration data at each waypoint

    # Move eef to home pose
    robot.go_home()

    for i in range(len(pos_waypt_list)):
    # for i in range(4):
    # for i in range(15):
        if i == 30 or i ==34 or i == 35 or i==36 or i == 37 or i == 38:
            continue
        # # if i >= 35:
        # #     break
        print(f"Waypoint {i+1}/{len(pos_waypt_list)}")

        pos_waypt = pos_waypt_list[i]
        pos_waypt[2] += 0.01
        quat_waypt = quat_waypt_list[i]

        # Get pose of eef in world frame
        eef_pos_cur, eef_quat_cur = robot.get_pose()

        if args.debug:
            max_vel = 2
        else:
            max_vel = 0.05

        linear_traj = t_utils.get_traj(eef_pos_cur, pos_waypt, max_vel=max_vel)
        if args.debug:
            angular_traj = t_utils.get_angular_traj(eef_quat_cur, quat_waypt, max_vel=2)
        else:
            angular_traj = t_utils.get_angular_traj(eef_quat_cur, quat_waypt)

        # if i >= 36:
        #     pdb.set_trace()

        # angular_traj = get_angular_traj(curr_ori, target_ori)
        duration = max(linear_traj.duration, angular_traj.duration)
        print("Duration of current trajectory (s)", duration)
        start_time = time.time()

        while (time.time() - start_time) < duration:
            curr_time = time.time() - start_time
            franka_pos_command = np.array(linear_traj.at_time(curr_time)[0])
            franka_ori_command = np.array(angular_traj.at_time(curr_time))
            robot.goto_pose(pos=franka_pos_command, ori=franka_ori_command)

            # # Display images
            # redis_pipe.get(KEY_CAMERA_COLOR_LEFT_BIN)
            # redis_pipe.get(KEY_CAMERA_INTRINSIC)
            # b_img, b_K = redis_pipe.execute()
            # img = m.unpackb(b_img)            
            # K = decode_matlab(b_K)

        if not args.debug:
            time.sleep(4.0)

 
        # Capture image
        start_time = time.time()
        saved = False
        pause_duration = 0.5  # seconds to pause
        while (time.time() - start_time) < pause_duration:
            if not saved:
                print(f"Logging data at waypoint {i}")

                # Get current image and save
                fname = f"{img_dir}/img_{i}.png"
                redis_pipe.get(KEY_LEFT_CAMERA_IMAGE_BIN)
                redis_pipe.get(KEY_LEFT_CAMERA_INTRINSIC)
                b_img, b_K = redis_pipe.execute()
                img = m.unpackb(b_img)
                K = decode_matlab(b_K)
                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # [720, 1280, 3]
                cv2.imwrite(fname, rgb_img)

                # Get eef pose and save
                eef_pos_cur, eef_quat_cur = robot.get_pose()  # [3,], [4,]
                qpos = robot.get_q()

                data_dict = {
                    "pos": eef_pos_cur,
                    "ori": eef_quat_cur,
                    "qpos": qpos,
                    "imgs": [rgb_img],
                    "K": K,
                }
                cal_data.append(data_dict)
                saved = True

    # Save policy data dict
    data_path = os.path.join(save_dir, "calibration_data.pkl")
    with open(data_path, "wb") as f:
        pickle.dump(cal_data, f)
    print("Saved policy data to: ", data_path)
    print(f"Done. Data has {len(cal_data)} samples.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", type=str, required=True
    )
    parser.add_argument(
        "--sim", action="store_true", help="If using flag, will run robot in sim"
    )
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        assert args.sim

    main(args)