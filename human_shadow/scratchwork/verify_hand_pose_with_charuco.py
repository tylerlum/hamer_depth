import pdb 
import numpy as np
import json

import cv2
import matplotlib.pyplot as plt
import mediapy as media

# pos_finger = np.array([0.20353733, 0.22991824, 0.58280803])
# np.save("pos_finger.npy", pos_finger)
# pdb.set_trace()

video_path = "data/videos/demo3_hand_correct/video_0_L.mp4"
imgs = media.read_video(video_path)
print("n imgs: ", len(imgs))
img = imgs[0]
# plt.imshow(img)
# plt.show()

# Get image
video_path = "data/videos/demo3/video_0_L.mp4"
imgs = media.read_video(video_path)
img = imgs[0]

# Get camera intrinsics 
camera_intrinsics_path = "camera/camera_intrinsics.json"
with open(camera_intrinsics_path, "r") as f:
    camera_intrinsics = json.load(f)

# Get camera matrix 
fx = camera_intrinsics["left"]["fx"]
fy = camera_intrinsics["left"]["fy"]
cx = camera_intrinsics["left"]["cx"]
cy = camera_intrinsics["left"]["cy"]
camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])


# Define the Charuco board
checker_size = 0.020  # Checker size in meters (20mm)
marker_size = 0.016  # Marker size in meters (16mm)
charuco_board = cv2.aruco.CharucoBoard((14, 9), checker_size, marker_size, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))

# Set up the detector parameters and dictionary
parameters = cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50)
# detector = cv2.aruco.ArucoDetector(dictionary, parameters)

charuco_parameters = cv2.aruco.CharucoParameters()
detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)

# Load the image
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect ArUco markers in the image
charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(gray)


dist_coeffs = np.zeros(12)
# If there are enough Charuco corners detected, estimate the board pose
if charuco_ids is not None and len(charuco_ids) > 0:
    # Initialize rvec and tvec to store the pose
    rvec = np.zeros((3, 1))  # Rotation vector
    tvec = np.zeros((3, 1))  # Translation vector

    # Estimate the pose of the Charuco board
    retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charuco_corners, charuco_ids, charuco_board, camera_matrix, dist_coeffs, rvec, tvec)

    if retval:
        # Project the 3D coordinates of each Charuco corner to the camera's coordinate system
        board_corners_3d = charuco_board.getChessboardCorners()  # 3D coordinates in the board's reference frame

        # Get the 3D positions of the detected Charuco corners
        corners_3d_positions = []
        for i in range(len(charuco_corners)):
            corner_3d_board = board_corners_3d[charuco_ids[i][0]]  # Get the 3D coordinate in the board's reference frame
            corner_3d_board = np.array([corner_3d_board], dtype=np.float32).reshape(-1, 3)

            # Transform the corner coordinates from the board's reference frame to the camera's reference frame
            corners_3d_camera, _ = cv2.projectPoints(corner_3d_board, rvec, tvec, camera_matrix, dist_coeffs)
            corners_3d_positions.append(corners_3d_camera.flatten())


        # Print the 3D positions of each detected Charuco corner
        for i, pos in enumerate(corners_3d_positions):
            print(f"Charuco corner {charuco_ids[i][0]} position (camera frame): {pos}")

        # Optionally, draw the board pose
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)  # Draw axes with a length of 0.1 meters

        print("tvec: ", tvec)

new_width = 800
aspect_ratio = new_width / img.shape[1]
new_height = int(img.shape[0] * aspect_ratio)
resized_img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)


# Display the image
resized_img_bgr = cv2.cvtColor(resized_img, cv2.COLOR_RGB2BGR)
cv2.imshow('Charuco Corners 3D Positions', resized_img_bgr)
# Save the image
cv2.imwrite("charuco_corners.jpg", resized_img_bgr)
# cv2.waitKey(0)
# cv2.destroyAllWindows() 