import pyzed.sl as sl
import numpy as np
import json

def save_intrinsics(camera_params, save_path: str=None):
    """Save intrinsics of left and right camera in json"""
    both_params = {
        "left": camera_params.left_cam,
        "right": camera_params.right_cam,
    }

    intrinsics = {}
    for cam_name, one_cam_params in both_params.items():
        intrinsics[cam_name] = {
            "fx": one_cam_params.fx,
            "fy": one_cam_params.fy,
            "cx": one_cam_params.cx,
            "cy": one_cam_params.cy,
            "disto": one_cam_params.disto.tolist(),
            "v_fov": one_cam_params.v_fov,
            "h_fov": one_cam_params.h_fov,
            "d_fov": one_cam_params.d_fov,
        }

    if save_path is not None:
        with open(save_path, "w") as f:
            json.dump(intrinsics, f, indent=4)

    return intrinsics

def get_intrinsics_matrix(camera_params, cam_side: str="left"):
    if cam_side == "left":
        one_cam_params = camera_params.left_cam
    elif cam_side == "right":
        one_cam_params = camera_params.right_cam
    else:
        raise ValueError("Invalid cam_side")

    K = np.array(
        [
            [one_cam_params.fx, 0, one_cam_params.cx],
            [0, one_cam_params.fy, one_cam_params.cy],
            [0, 0, 1],
        ]
    )

    return K

def init_zed(resolution: str, depth_mode: str):
    """Initialize Zed Camera"""
    zed_resolutions_dict = {
        "VGA": sl.RESOLUTION.VGA, "HD720": sl.RESOLUTION.HD720, 
        "HD1080": sl.RESOLUTION.HD1080, "HD2K": sl.RESOLUTION.HD2K}
    zed_depth_modes_dict = {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE, "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY, "NEURAL": sl.DEPTH_MODE.NEURAL,
        "TRI": sl.DEPTH_MODE.NEURAL, "NONE": sl.DEPTH_MODE.PERFORMANCE}

    init = sl.InitParameters(
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        camera_resolution=zed_resolutions_dict[resolution],
        camera_fps=60, 
        depth_mode=zed_depth_modes_dict[depth_mode],
    )
    # init.depth_maximum_distance = 2  # Max distance in meters

    zed = sl.Camera()
    status = zed.open(init)

    # zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    return zed