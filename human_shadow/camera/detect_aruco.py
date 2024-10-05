
import pdb 
import json
import argparse
import os
import time
import cv2
import re
import curses
import numpy as np

import mediapy as media
import pyzed.sl as sl

from human_shadow.camera.zed_utils import *
from human_shadow.utils.aruco_utils import *


def main(args):
    zed = init_zed(args.resolution, args.depth_mode)
    camera_params, K_left, K_right = get_camera_params(zed)
    res = camera_params.left_cam.image_size
    img_left = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    img_right = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_img = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4)

    # Get camera matrix 
    camera_matrix = get_intrinsics_matrix(camera_params, cam_side="left")

    charuco_detector, charuco_board = init_charuco_detector(checker_size=0.020, marker_size=0.016, ncols=14, nrows=9, dictionary=cv2.aruco.DICT_5X5_50) 
    aruco_detector = init_aruco_detector(dictionary=cv2.aruco.DICT_4X4_50)

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img_left_rgb, img_right_rgb, depth_img_arr = capture_camera_data(zed, args.depth_mode, img_left, 
            img_right, depth_img)
            img_left_rgb = resize_img_to_square(img_left_rgb)
            img_right_rgb = resize_img_rgb = resize_img_to_square(img_right_rgb)
            depth_img_arr = resize_img_to_square(depth_img_arr)

            # Detect single marker
            if args.mode == "marker":
                annotated_img, rvec, tvec = get_aruco_marker_pose(img_left_rgb, aruco_detector, camera_matrix, marker_id=0, marker_size_mm=45, aruco_dict_type=cv2.aruco.DICT_4X4_50)
            elif args.mode == "board":
                annotated_img, rvec, tvec = get_charuco_board_pose(img_left_rgb, camera_matrix, charuco_detector, charuco_board)


            annotated_img_bgr = annotated_img[..., ::-1]
            annotated_img_bgr = cv2.resize(annotated_img_bgr, (720, 720))
            cv2.imshow("Left", annotated_img_bgr)
            cv2.waitKey(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="marker", choices=["marker", "board"])
    parser.add_argument("--hz", type=int, default=10, required=True)
    parser.add_argument("--depth_mode", required=True, choices=["PERFORMANCE", "ULTRA", "QUALITY", "NEURAL", "TRI", "NONE"])
    parser.add_argument("--resolution", required=True, choices=["VGA", "HD720", "HD1080", "HD2K"])
    args = parser.parse_args()
    main(args)