"""
Stream rgb images from Zed camera to redis   
"""
import msgpack
import msgpack_numpy as m
m.patch()
import time
from tqdm import tqdm
import pdb
import sys
import pyzed.sl as sl
import os
import cv2
import numpy as np
import json
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import ctrlutils
import human_shadow.camera.ogl_viewer.viewer as gl

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

from human_shadow.config.redis_keys import *
from franka_utils.redis_utils import encode_matlab


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

def init_zed(resolution: str):
    """Initialize Zed Camera"""
    zed_resolutions_dict = {
        "VGA": sl.RESOLUTION.VGA, "HD720": sl.RESOLUTION.HD720, 
        "HD1080": sl.RESOLUTION.HD1080, "HD2K": sl.RESOLUTION.HD2K}
    zed_depth_modes_dict = {
        "PERFORMANCE": sl.DEPTH_MODE.PERFORMANCE, "ULTRA": sl.DEPTH_MODE.ULTRA,
        "QUALITY": sl.DEPTH_MODE.QUALITY, "NEURAL": sl.DEPTH_MODE.NEURAL,
        "TRI": sl.DEPTH_MODE.NEURAL}

    init = sl.InitParameters(
        coordinate_units=sl.UNIT.METER,
        coordinate_system=sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP,
        camera_resolution=zed_resolutions_dict[resolution],
        camera_fps=60, 
        depth_mode=zed_depth_modes_dict[args.depth_mode],
    )
    # init.depth_maximum_distance = 2  # Max distance in meters

    zed = sl.Camera()
    status = zed.open(init)

    # zed.set_camera_settings(sl.VIDEO_SETTINGS.WHITEBALANCE_AUTO, 1)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    return zed


def capture_camera_data(args, zed, img_left, img_right, depth_img, point_cloud, 
                        redis_pipe, K_left, K_right, tri_stereo_to_depth=None, viewer=None):
    # Get zed data
    zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU)
    zed.retrieve_image(img_right, sl.VIEW.RIGHT, sl.MEM.CPU)
    zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)

    # RGB image
    img_left_bgr = img_left.get_data()[:,:,:3]
    img_left_rgb = img_left_bgr[...,::-1] # bgr to rgb
    img_right_bgr = img_right.get_data()[:,:,:3]
    img_right_rgb = img_right_bgr[...,::-1] # bgr to rgb
    img_left_right_rgb = np.stack([img_left_rgb, img_right_rgb])

    # Depth image 
    if args.depth_mode == "TRI":
        # Get depth image from TRI stereo to depth
        tri_depth, tri_left, tri_right = tri_stereo_to_depth.get_depth_and_bgr(
            img_left_bgr[:, :, :3], img_right_bgr[:, :, :3]
        )
        img_left_bgr = tri_left
        img_right_bgr = tri_right
        depth_img_arr = tri_depth.astype(np.float32)
    else: 
        zed.retrieve_measure(depth_img, sl.MEASURE.DEPTH, sl.MEM.CPU)
        depth_img_arr = depth_img.get_data()

    # Point cloud 
    point_cloud_arr = np.array(point_cloud.get_data())

    # Convert data to binary
    b_img_left_right = m.packb(img_left_right_rgb)
    b_depth_img = m.packb(depth_img_arr)
    b_point_cloud = m.packb(point_cloud_arr)

    if args.render:
        cv2.imshow("img_left", img_left_bgr)
        cv2.waitKey(1)

    if args.render_depth:
        depth_img_arr = depth_img_arr 
        # depth_img_arr = np.clip(depth_img_arr, 0, 255).astype(np.uint8)
        cv2.imshow("depth_img", depth_img_arr)
        cv2.waitKey(1)

    if args.render_pcd: 
        viewer.updateData(point_cloud)

    # Send to redis
    redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, b_img_left_right)
    redis_pipe.set(KEY_CAMERA_DEPTH_BIN, b_depth_img)
    redis_pipe.set(KEY_CAMERA_POINT_CLOUD_BIN, b_point_cloud)
    redis_pipe.set(KEY_LEFT_CAMERA_INTRINSIC, encode_matlab(K_left))
    redis_pipe.set(KEY_RIGHT_CAMERA_INTRINSIC, encode_matlab(K_right))
    redis_pipe.execute()


def main(args):
    print("Starting Zed driver...")
    zed = init_zed(args.resolution)

    # Get camera params
    camera_model = zed.get_camera_information().camera_model
    camera_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    K_left = get_intrinsics_matrix(camera_params, cam_side="left")
    K_right = get_intrinsics_matrix(camera_params, cam_side="right")
    save_intrinsics(camera_params, save_path="camera_intrinsics.json")

    res = camera_params.left_cam.image_size

    img_left = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    img_right = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_img = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4)
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Set up redis server
    if args.use_nuc_ip: 
        _redis = ctrlutils.RedisClient(
            host=NUC_HOST, port=NUC_PORT, password=NUC_PWD
        )
    else:
        _redis = ctrlutils.RedisClient(
            host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
        )
    redis_pipe = _redis.pipeline()

    if args.render_pcd: 
        # Create OpenGL viewer
        viewer = gl.GLViewer()
        viewer.init(1, sys.argv, camera_model, res)
    else:
        viewer = None

    if args.depth_mode == "TRI": 
        from mmt.stereo_inference.python.stereo_to_depth import StereoToDepth
        print("Initializing TRI stereo to depth model...")
        tri_stereo_to_depth = StereoToDepth(fx=K_left[0, 0], cam_baseline=0.12)
        print("TRI stereo to depth initialized.")
    else:
        tri_stereo_to_depth = None


    print("Streaming images to redis...")
    start_time = time.time()
    count = 0
    if args.render_pcd:
        while viewer.is_available():
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                capture_camera_data(args, zed, img_left, img_right, depth_img, point_cloud, 
                            redis_pipe, K_left, K_right, viewer=viewer)
                count += 1
                loop_dt = (time.time() - start_time) / count
                if count % 50 == 0:
                    print(f"Loop freq: {1/loop_dt:.3f} hz")
        viewer.exit()
    else:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                capture_camera_data(args, zed, img_left, img_right, depth_img, point_cloud, 
                            redis_pipe, K_left, K_right, tri_stereo_to_depth=tri_stereo_to_depth)
                count += 1
                loop_dt = (time.time() - start_time) / count
                if count % 50 == 0:
                    print(f"Loop freq: {1/loop_dt:.3f} hz")


    zed.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--render_depth", action="store_true")
    parser.add_argument("--render_pcd", action="store_true")
    parser.add_argument("--intrinsics", action="store_true")
    parser.add_argument("--use_nuc_ip", action="store_true")
    parser.add_argument("--depth_mode", required=True, choices=["PERFORMANCE", "ULTRA", "QUALITY", "NEURAL", "TRI"])
    parser.add_argument("--resolution", required=True, choices=["VGA", "HD720", "HD1080", "HD2K"])
    args = parser.parse_args()
    main(args)

