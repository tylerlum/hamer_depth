import pdb 
import numpy as np
import json
from tqdm import tqdm
import argparse

import cv2
import matplotlib.pyplot as plt
import mediapy as media


def get_charuco_board_pose(img, camera_matrix):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers in the image
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    dist_coeffs = np.zeros(12)
    # If there are enough Charuco corners detected, estimate the board pose
    if charuco_ids is not None and len(charuco_ids) > 6:
        # Initialize rvec and tvec to store the pose
        rvec = np.zeros((3, 1))  # Rotation vector
        tvec = np.zeros((3, 1))  # Translation vector

        # Estimate the pose of the Charuco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, rvec, tvec)

        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw axes with a length of 0.1 meters

        return img, rvec, tvec

    return img, None, None


def get_camera_intrinsics_matrix(camera_intrinsics_path):
    with open(camera_intrinsics_path, "r") as f:
        camera_intrinsics = json.load(f)

    # Get camera matrix 
    fx = camera_intrinsics["left"]["fx"]
    fy = camera_intrinsics["left"]["fy"]
    cx = camera_intrinsics["left"]["cx"]
    cy = camera_intrinsics["left"]["cy"]
    camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    return camera_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    # video_path = "data/videos/test/video_9_L.mp4"
    imgs = media.read_video(args.video_path)

    camera_matrix = get_camera_intrinsics_matrix("camera/camera_intrinsics.json")

    # Define the Charuco board
    checker_size = 0.020  # Checker size in meters (20mm)
    marker_size = 0.016  # Marker size in meters (16mm)
    charuco_board = cv2.aruco.CharucoBoard((14, 9), checker_size, marker_size, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

    # Set up the detector parameters and dictionary
    parameters = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    charuco_parameters = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)

    list_tvecs = []
    for img in tqdm(imgs):

        annotated_img, rvec, tvec = get_charuco_board_pose(img, camera_matrix)

        if tvec is None:
            tvec = np.ones((3, 1)) * (-1)

        list_tvecs.append(tvec)
        if args.render:
            plt.imshow(annotated_img)
            plt.show()

    # Save translation vectors
    tvecs = np.array(list_tvecs)
    # tvecs_path = args.video_path.split(".")[0] + "_tvecs.npy"
    # np.save(tvecs_path, tvecs)
