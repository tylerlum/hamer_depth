import pdb 
import numpy as np
import json
from tqdm import tqdm
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import mediapy as media

from human_shadow.utils.aruco_utils import *
from human_shadow.camera.zed_utils import *


def main(args):

    charuco_detector, charuco_board = init_charuco_detector(checker_size=0.020, marker_size=0.016, ncols=14, nrows=9, dictionary=cv2.aruco.DICT_5X5_50) 
    aruco_detector = init_aruco_detector(dictionary=cv2.aruco.DICT_4X4_50)


    run_folders = os.listdir(args.video_folder)
    run_folders = sorted(run_folders, key=int)
    for folder in tqdm(run_folders):
        camera_intrinsics_path = os.path.join(args.video_folder, folder, f"cam_intrinsics_{folder}.json")
        camera_matrix = get_intrinsics_from_json(camera_intrinsics_path)

        video_left_path = os.path.join(args.video_folder, folder, f"video_{folder}_L.mp4")
        # Check if video file exists
        if not os.path.exists(video_left_path):
            continue
        imgs = media.read_video(video_left_path)
        list_marker_poses = []
        marker_img = None
        for img in tqdm(imgs):
            if args.mode == "marker":
                annotated_img, rvec, tvec = get_aruco_marker_pose(img, aruco_detector, camera_matrix, marker_id=0, marker_size_mm=45, aruco_dict_type=cv2.aruco.DICT_4X4_50)
            else:
                annotated_img, rvec, tvec = get_charuco_board_pose(img, camera_matrix, charuco_detector, charuco_board)

            if tvec is None or rvec is None:
                pose = np.ones(6) * (-1)
            else:
                pose = np.concatenate([tvec[0], rvec[0]], axis=0)
                if marker_img is None:
                    marker_img = annotated_img

            list_marker_poses.append(pose)

            if args.render:
                plt.imshow(annotated_img)
                plt.show()
        
        # Save poses
        poses_path = os.path.join(args.video_folder, folder, f"marker_poses_{folder}.npy")
        np.save(poses_path, np.array(list_marker_poses))

        # Save annotated marker image
        marker_img_path = os.path.join(args.video_folder, folder, f"marker_img_{folder}.png")
        if marker_img is not None:
            marker_img_bgr = marker_img[..., ::-1]
            cv2.imwrite(marker_img_path, marker_img_bgr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--mode", type=str, default="marker", choices=["marker", "board"])
    args = parser.parse_args()

    main(args)
