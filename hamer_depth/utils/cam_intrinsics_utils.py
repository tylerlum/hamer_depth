from pathlib import Path

import numpy as np


def convert_intrinsics_matrix_to_dict(camera_matrix: np.ndarray) -> dict:
    fx = camera_matrix[0, 0]
    fy = camera_matrix[1, 1]
    cx = camera_matrix[0, 2]
    cy = camera_matrix[1, 2]
    intrinsics = {
        "fx": fx,
        "fy": fy,
        "cx": cx,
        "cy": cy,
    }
    return intrinsics


def get_camera_matrix_from_file(file_path: Path) -> np.ndarray:
    with open(file_path, "r") as f:
        camera_matrix = np.loadtxt(f)

    assert camera_matrix.shape == (3, 3), (
        f"Camera matrix shape {camera_matrix.shape} is not (3, 3)"
    )

    return camera_matrix
