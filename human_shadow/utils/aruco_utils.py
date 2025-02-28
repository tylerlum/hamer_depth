import cv2
import numpy as np

from human_shadow.utils.image_utils import resize_image_to_rectangle


def init_charuco_detector(checker_size, marker_size, ncols, nrows, dictionary):
    charuco_board = cv2.aruco.CharucoBoard(
        (ncols, nrows),
        checker_size,
        marker_size,
        cv2.aruco.getPredefinedDictionary(dictionary),
    )
    charuco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    charuco_parameters = cv2.aruco.CharucoParameters()
    charuco_detector = cv2.aruco.CharucoDetector(charuco_board, charuco_parameters)
    return charuco_detector, charuco_board


def init_aruco_detector(dictionary):
    aruco_dictionary = cv2.aruco.getPredefinedDictionary(dictionary)
    aruco_parameters = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dictionary, aruco_parameters)
    return aruco_detector


def get_aruco_marker_pose(
    img,
    detector,
    camera_matrix,
    marker_id=0,
    marker_size_mm=45,
    aruco_dict_type=cv2.aruco.DICT_4X4_50,
):
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
                corners_int = np.int0(
                    corners[i]
                )  # Convert corners to integer values for drawing
                cv2.polylines(
                    img, [corners_int], isClosed=True, color=(0, 255, 0), thickness=2
                )

                # Annotate the image with marker ID
                top_left_corner = tuple(
                    corners_int[0][0]
                )  # Get the top-left corner for annotation
                cv2.putText(
                    img,
                    f"ID: {marker_id}",
                    top_left_corner,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Estimate the pose of the marker
                marker_size_m = (
                    marker_size_mm / 1000.0
                )  # Convert size from mm to meters
                dist_coeffs = np.zeros(12)
                rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(
                    corners, marker_size_m, camera_matrix, dist_coeffs
                )

                # Draw the axes on the image
                cv2.drawFrameAxes(
                    img, camera_matrix, dist_coeffs, rvec[i], tvec[i], 0.05
                )  # Axes length is set to 5 cm

                # Return the image and the pose
                return img, rvec[i], tvec[i]

    # Save the output image or return it
    return img, None, None


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
    charuco_corners, charuco_ids, marker_corners, marker_ids = detector.detectBoard(
        gray
    )

    if charuco_ids is not None:
        print("Len charuco ids: ", len(charuco_ids))

    # Draw the outline of each detected marker
    if marker_corners is not None:
        for corners in marker_corners:
            corners = corners.reshape((4, 2)).astype(
                int
            )  # Ensure corners are in integer format
            cv2.polylines(img, [corners], isClosed=True, color=(0, 255, 0), thickness=2)

    # Draw the identified Charuco corners
    if charuco_corners is not None:
        for corner in charuco_corners:
            corner = tuple(corner.ravel().astype(int))  # Convert to an (x, y) tuple
            cv2.circle(
                img, corner, radius=5, color=(255, 0, 0), thickness=-1
            )  # Draw each corner

    dist_coeffs = np.zeros(12)
    # If there are enough Charuco corners detected, estimate the board pose
    if charuco_ids is not None and len(charuco_ids) > 4:
        # Initialize rvec and tvec to store the pose
        rvec = np.zeros((3, 1))  # Rotation vector
        tvec = np.zeros((3, 1))  # Translation vector

        # Estimate the pose of the Charuco board
        retval, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
            charuco_corners,
            charuco_ids,
            charuco_board,
            camera_matrix,
            dist_coeffs,
            rvec,
            tvec,
        )

        img = np.ascontiguousarray(img)

        # Draw axes with a length of 0.1 meters
        cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)

        img = img[:, dp : dp + original_img_height]

        return img, rvec, tvec

    img = img[:, dp : dp + original_img_height]

    return img, None, None
