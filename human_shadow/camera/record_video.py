"""
Record video from rgb values streamed on redis
"""
import pdb 
import os
import re
import argparse
import curses
import ctrlutils
import time
import mediapy as media
import matplotlib.pyplot as plt
import msgpack_numpy as m
m.patch()
import numpy as np

from human_shadow.config.redis_keys import *
from human_shadow.utils.file_utils import get_parent_folder_of_package


def main(stdscr, args): 
    # Create output folder 
    project_folder = get_parent_folder_of_package("human_shadow")
    save_folder = os.path.join(project_folder, "human_shadow/data", "videos", args.folder)
    print("Folder path: ", save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    left_pattern = re.compile(r"video_L(\d+).mp4")
    left_video_files = [f for f in os.listdir(save_folder) if left_pattern.match(f)]
    # right_pattern = re.compile(r"video_R(\d+).mp4")
    # right_video_files = [f for f in os.listdir(save_folder) if right_pattern.match(f)]
    folder_count = len(left_video_files)
    left_video_path = os.path.join(save_folder, f"video_{folder_count}_L.mp4")
    right_video_path = os.path.join(save_folder, f"video_{folder_count}_R.mp4")

    # Initialize redis client
    _redis = ctrlutils.RedisClient(
            host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
        )
    redis_pipe = _redis.pipeline()

    # Wait until start signal is received
    stdscr.nodelay(True)
    stdscr.clear()

    while True: 
        key = stdscr.getch()
        if key == ord(" "):
            break

    while True:
        key = stdscr.getch()
        if key != ord(" "):
            break
    print("START RECORDING \n")

    dt = 1 / args.hz

    left_imgs = []
    right_imgs = []
    depth_imgs = []
    point_clouds = []
    while True:
        start_time = time.time()

        # Get image from redis 
        redis_pipe.get(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN)
        redis_pipe.get(KEY_CAMERA_DEPTH_BIN)
        redis_pipe.get(KEY_CAMERA_POINT_CLOUD_BIN)
        img_left_right_b, depth_img_b, point_cloud_b = redis_pipe.execute()
        img_left_right = m.unpackb(img_left_right_b)
        img_left = img_left_right[0]
        img_right = img_left_right[1]
        left_imgs.append(img_left)
        right_imgs.append(img_right)

        # Get point cloud from redis
        point_cloud = m.unpackb(point_cloud_b)
        point_clouds.append(point_cloud)

        # Get depth image from point cloud
        depth_img = m.unpackb(depth_img_b)
        depth_imgs.append(depth_img)

        key = stdscr.getch()
        if key == ord(" "):
            print("STOP RECORDING \n")
            break

        while (time.time() - start_time) < dt:
            pass

    left_imgs = np.array(left_imgs)
    right_imgs = np.array(right_imgs)

    media.write_video(left_video_path, left_imgs, fps=args.hz)
    media.write_video(right_video_path, right_imgs, fps=args.hz)
    np.save(os.path.join(save_folder, f"point_clouds_{folder_count}.npy"), np.array(point_clouds))
    np.save(os.path.join(save_folder, f"depth_imgs_{folder_count}.npy"), np.array(depth_imgs))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="demo2")
    parser.add_argument("--hz", type=int, default=10)
    args = parser.parse_args()
    curses.wrapper(lambda stdscr: main(stdscr, args))           
            
            
            
            
