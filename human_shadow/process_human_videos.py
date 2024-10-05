import pdb 
import numpy as np
import json
from tqdm import tqdm
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import mediapy as media

def get_charuco_board_pose(img, camera_matrix, detector, charuco_board):
    # Load the image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect ArUco markers and Charuco corners in the image
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

    print("Len charuco ids: ", len(charuco_ids))

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

        return img, rvec, tvec

    return img, None, None

# def get_charuco_board_pose(img, camera_matrix, detector, charuco_board):
#     # Load the image
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     # Detect ArUco markers in the image
#     charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)

#     print("Len charuco ids: ", len(charuco_ids))

#     dist_coeffs = np.zeros(12)
#     # If there are enough Charuco corners detected, estimate the board pose
#     if charuco_ids is not None and len(charuco_ids) > 4:
#         # Initialize rvec and tvec to store the pose
#         rvec = np.zeros((3, 1))  # Rotation vector
#         tvec = np.zeros((3, 1))  # Translation vector

#         # Estimate the pose of the Charuco board
#         retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, rvec, tvec)

#         cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw axes with a length of 0.1 meters

#         return img, rvec, tvec

#     return img, None, None


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

def main(args):

    # Define the Charuco board
    checker_size = 0.020  # Checker size in meters (20mm)
    marker_size = 0.016  # Marker size in meters (16mm)
    charuco_board = cv2.aruco.CharucoBoard((14, 9), checker_size, marker_size, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

    # Set up the detector parameters and dictionary
    parameters = cv2.aruco.DetectorParameters()
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
    charuco_parameters = cv2.aruco.CharucoParameters()
    detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)

    run_folders = os.listdir(args.video_folder)
    run_folders = sorted(run_folders, key=int)
    for folder in run_folders:
        camera_intrinsics_path = os.path.join(args.video_folder, folder, f"cam_intrinsics_{folder}.json")
        camera_matrix = get_camera_intrinsics_matrix(camera_intrinsics_path)

        video_left_path = os.path.join(args.video_folder, folder, f"video_{folder}_L.mp4")
        imgs = media.read_video(video_left_path)
        for img in imgs:
            annotated_img, rvec, tvec = get_charuco_board_pose(img, camera_matrix, detector, charuco_board)

            plt.imshow(annotated_img)
            plt.show()

        pdb.set_trace()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_folder", type=str, required=True)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    main(args)
