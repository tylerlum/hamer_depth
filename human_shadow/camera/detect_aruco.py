
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

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.camera.zed_utils import *
from human_shadow.utils.button_utils import *

def resize_image_to_rectangle(img, target_height, target_width):
    assert(img.shape[0] == target_height)
    if img.shape[0] == img.shape[1]:
        new_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        dp = (target_width - target_height) // 2
        new_img[:, dp:dp+target_height] = img
        img = new_img
    return img, dp


def detect_single_aruco_marker(img, detector, marker_id=0, marker_size_mm=45, aruco_dict_type=cv2.aruco.DICT_4X4_50):
    # Get the ArUco dictionary
    
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect markers in the image
    corners, ids, _ = detector.detectMarkers(gray)

    img = np.ascontiguousarray(img)

    # If markers are detected, check if the desired marker (ID = 0) is present
    if ids is not None:
        for i, detected_id in enumerate(ids):
            if detected_id == marker_id:
                # Draw the bounding box around the detected marker
                corners_int = np.int0(corners[i])  # Convert corners to integer values for drawing
                cv2.polylines(img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2)

                # Annotate the image with marker ID
                top_left_corner = tuple(corners_int[0][0])  # Get the top-left corner for annotation
                cv2.putText(img, f'ID: {marker_id}', top_left_corner, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Save the output image or return it
    return img


def get_charuco_board_pose(img, camera_matrix, detector, charuco_board):
    original_img = img.copy()
    original_img_height = original_img.shape[0]

    if img.shape[0] == img.shape[1]:
        if img.shape[0] == 720:
            img, dp = resize_image_to_rectangle(img, 720, 1280)
        elif img.shape[0] == 1080:
            img, dp = resize_image_to_rectangle(img, 1080, 1920)
        elif img.shape[0] == 1242:
            img, dp = resize_image_to_rectangle(img, 1242, 2208)
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers and Charuco corners in the image
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    dist_coeffs = np.zeros(12)
    # If there are enough Charuco corners detected, estimate the board pose
    if charuco_ids is not None and len(charuco_ids) > 4:
        # Initialize rvec and tvec to store the pose
        rvec = np.zeros((3, 1))  # Rotation vector
        tvec = np.zeros((3, 1))  # Translation vector

        # Estimate the pose of the Charuco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, rvec, tvec
        )

        img = np.ascontiguousarray(img)

        # Draw axes with a length of 0.1 meters
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)


        # Draw the outline of each detected marker
        if marker_corners is not None:
            for corners in marker_corners:
                corners = corners.reshape((4, 2)).astype(int)  # Ensure corners are in integer format
                cv2.polylines(img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

        # Draw the identified Charuco corners
        if charuco_corners is not None:
            for corner in charuco_corners:
                corner = tuple(corner.ravel().astype(int))  # Convert to an (x, y) tuple
                cv2.circle(img, corner, radius=5, color=(255, 0, 0), thickness=-1)  # Draw each corner

        img = img[:, dp:dp+original_img_height]

        return img, rvec, tvec

    return original_img, None, None

def get_camera_params(zed):
    camera_model = zed.get_camera_information().camera_model
    camera_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    K_left = get_intrinsics_matrix(camera_params, cam_side="left")
    K_right = get_intrinsics_matrix(camera_params, cam_side="right")
    return camera_params, K_left, K_right


def capture_camera_data(zed, depth_mode, img_left, img_right, depth_img): 
    zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU)
    zed.retrieve_image(img_right, sl.VIEW.RIGHT, sl.MEM.CPU)
    if not depth_mode == "NONE":
        if not depth_mode == "TRI": 
            zed.retrieve_measure(depth_img, sl.MEASURE.DEPTH, sl.MEM.CPU)

    # RGB image
    img_left_bgr = img_left.get_data()[:,:,:3]
    img_left_rgb = img_left_bgr[...,::-1] # bgr to rgb
    img_right_bgr = img_right.get_data()[:,:,:3]
    img_right_rgb = img_right_bgr[...,::-1] # bgr to rgb
    depth_img_arr = depth_img.get_data()
    
    return img_left_rgb, img_right_rgb, depth_img_arr


def resize_img_to_square(img):
    img_w = img.shape[1]
    img_h = img.shape[0]
    min_dim = min(img_w, img_h)
    if img_w > min_dim:
        diff = img_w - min_dim
        img = img[:, diff//2:diff//2+min_dim]
    elif img_h > min_dim:
        diff = img_h - min_dim
        img = img[diff//2:diff//2+min_dim, :]
    return img


def main(args):
    zed = init_zed(args.resolution, args.depth_mode)
    camera_params, K_left, K_right = get_camera_params(zed)
    res = camera_params.left_cam.image_size
    img_left = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    img_right = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_img = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4)

    # Get camera matrix 
    camera_matrix = get_intrinsics_matrix(camera_params, cam_side="left")

    # Define the Charuco board
    checker_size = 0.020  # Checker size in meters (20mm)
    marker_size = 0.016  # Marker size in meters (16mm)
    charuco_board = cv2.aruco.CharucoBoard((14, 9), checker_size, marker_size, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

    # Set up the detector parameters and dictionary
    # detector_parameters = cv2.aruco.DetectorParameters()
    charuco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)

    aruco_dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)

    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img_left_rgb, img_right_rgb, depth_img_arr = capture_camera_data(zed, args.depth_mode, img_left, 
            img_right, depth_img)
            img_left_rgb = resize_img_to_square(img_left_rgb)
            img_right_rgb = resize_img_rgb = resize_img_to_square(img_right_rgb)
            depth_img_arr = resize_img_to_square(depth_img_arr)

            # Detect single marker
            if args.mode == "marker":
                annotated_img = detect_single_aruco_marker(img_left_rgb, aruco_detector, marker_id=0, marker_size_mm=45, aruco_dict_type=cv2.aruco.DICT_4X4_50)
            elif args.mode == "board":
                annotated_img, rvec, tvec = get_charuco_board_pose(img_left_rgb, camera_matrix, charuco_detector, charuco_board)


            annotated_img_bgr = annotated_img[..., ::-1]

            annotated_img_bgr = cv2.resize(annotated_img_bgr, (720, 720))

            cv2.imshow("Left", annotated_img_bgr)
            cv2.waitKey(1)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test")
    parser.add_argument("--mode", type=str, default="marker", choices=["marker", "board"])
    parser.add_argument("--hz", type=int, default=10, required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--render_pcd", action="store_true")
    parser.add_argument("--intrinsics", action="store_true")
    parser.add_argument("--use_nuc_ip", action="store_true")
    parser.add_argument("--camera_calib", action="store_true")
    parser.add_argument("--depth_mode", required=True, choices=["PERFORMANCE", "ULTRA", "QUALITY", "NEURAL", "TRI", "NONE"])
    parser.add_argument("--resolution", required=True, choices=["VGA", "HD720", "HD1080", "HD2K"])
    args = parser.parse_args()
    main(args)