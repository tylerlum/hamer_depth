# import pyzed.sl as sl
import numpy as np
import json
import enum

class ZEDResolution(enum.Enum):
  SD = (360, 640)
  HD720 = (720, 1280)
  HD1080 = (1080, 1920)
  HD2K = (1242, 2208)

ZED_RESOLUTIONS = {
    "SD": ZEDResolution.SD,
    "HD720": ZEDResolution.HD720,
    "HD1080": ZEDResolution.HD1080,
    "HD2K": ZEDResolution.HD2K,
    }

ZED_RESOLUTIONS_SQUARE_SIZE = {
    360: ZEDResolution.SD,
    720: ZEDResolution.HD720,
    1080: ZEDResolution.HD1080,
    1242: ZEDResolution.HD2K,
    }

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

def convert_intrinsics_matrix_to_dict(camera_matrix):
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

def get_intrinsics_from_json(json_path: str):
    # with open(json_path, "r") as f:
    #     camera_intrinsics = json.load(f)

    # # Get camera matrix 
    # fx = camera_intrinsics["left"]["fx"]
    # fy = camera_intrinsics["left"]["fy"]
    # cx = camera_intrinsics["left"]["cx"]
    # cy = camera_intrinsics["left"]["cy"]
    # camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])

    with open(json_path, "r") as f:
        camera_matrix = np.loadtxt(f)

    return camera_matrix


def get_camera_params(zed):
    camera_model = zed.get_camera_information().camera_model
    camera_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    K_left = get_intrinsics_matrix(camera_params, cam_side="left")
    K_right = get_intrinsics_matrix(camera_params, cam_side="right")
    return camera_params, K_left, K_right

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