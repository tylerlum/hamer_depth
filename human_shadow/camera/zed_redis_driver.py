"""
Stream rgb images from Zed camera to redis   
"""
import msgpack
import msgpack_numpy as m
m.patch()
import pickle
import pyarrow as pa
import time
from tqdm import tqdm
import jax.numpy as jnp
import pdb
import sys
import pyzed.sl as sl
import os
import cv2
import numpy as np
import json
import zlib
import argparse
import matplotlib.pyplot as plt
import redis
from PIL import Image
# import ctrlutils
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
        "TRI": sl.DEPTH_MODE.NEURAL, "NONE": sl.DEPTH_MODE.PERFORMANCE}

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
                        redis_client, K_left, K_right, tri_stereo_to_depth=None, viewer=None):
    # Get zed data
    zed.retrieve_image(img_left, sl.VIEW.LEFT, sl.MEM.CPU)
    zed.retrieve_image(img_right, sl.VIEW.RIGHT, sl.MEM.CPU)
    if not args.depth_mode == "NONE":
        # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU)
        if not args.depth_mode == "TRI": 
            zed.retrieve_measure(depth_img, sl.MEASURE.DEPTH, sl.MEM.CPU)

    # RGB image
    img_left_bgr = img_left.get_data()[:,:,:3]
    img_left_rgb = img_left_bgr[...,::-1] # bgr to rgb
    img_right_bgr = img_right.get_data()[:,:,:3]
    img_right_rgb = img_right_bgr[...,::-1] # bgr to rgb
    img_left_right_rgb = np.stack([img_left_rgb, img_right_rgb])
    depth_img_arr = depth_img.get_data()

    print("RGB shape:", img_left_rgb.shape)
    pdb.set_trace()

    # start_time = time.time()
    # img_left_rgb_b = img_left_rgb.tobytes()
    # img_right_rgb_b = img_right_rgb.tobytes()
    # depth_img_b = depth_img_arr.tobytes()

    # img_left_rgb = np.ascontiguousarray(img_left_rgb.astype(np.uint8))
    # img_right_rgb = np.ascontiguousarray(img_right_rgb.astype(np.uint8))
    # depth_img_arr = np.ascontiguousarray(depth_img_arr.astype(np.float32))  

    # img_left_rgb_b = m.packb(img_left_rgb)
    # img_right_rgb_b = m.packb(img_right_rgb)
    # depth_img_b = m.packb(depth_img_arr)

    # img_left_rgb = np.ascontiguousarray(img_left_rgb.astype(np.uint8))
    # img_left_rgb = m.packb(img_left_rgb)
    # # img_left_rgb = memoryview(img_left_rgb)
    # # img_left_rgb_b = img_left_rgb.tobytes()
    # redis_client.set(KEY_LEFT_CAMERA_IMAGE_BIN, img_left_rgb)
    # print("Conversion time (s):", time.time() - start_time)

    # img_right_rgb = np.ascontiguousarray(img_right_rgb.astype(np.uint8))
    # # img_right_rgb = memoryview(img_right_rgb)
    # img_right_rgb_b = img_right_rgb.tobytes()

    # depth_img_arr = np.ascontiguousarray(depth_img_arr.astype(np.float32))
    # # depth_img_arr = memoryview(depth_img_arr)
    # depth_img_b = depth_img_arr.tobytes()

    # data = {"rgb_left": img_left_rgb_b, "rgb_right": img_right_rgb_b, "depth": depth_img_b}
    # redis_client.hset('zed_data', mapping=data)

    # img_left_rgb_b = img_left_rgb.tobytes()
    # img_left_rgb_b = pa.array(img_left_rgb.flatten())
    # img_right_rgb_b = pa.array(img_right_rgb.flatten())
    # depth_img_b = pa.array(depth_img_arr.flatten())
    # img_left_rgb_b = m.packb(img_left_rgb)
    # img_right_rgb_b = m.packb(img_right_rgb)
    # depth_img_b = m.packb(depth_img_arr)
    # img_rgba = {"rgb_left": img_left_rgb_b, "rgb_right": img_right_rgb_b, "depth": depth_img_b}
    # data = {"rgb_left": img_left_rgb_b, "rgb_right": img_right_rgb_b, "depth": depth_img_b}

    # img_data = pickle.dumps(img_rgba)
    # img_data = msgpack.packb(img_rgba)
    # img_data = img_rgba
    # redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, img_data)
    # redis_client.hset('zed_data', mapping=img_rgba)
    # marion = img_left_rgb.tobytes(order='C')

    # img_left_rgb_b = bytes(memoryview(img_left_rgb))
    # redis_client.set(KEY_LEFT_CAMERA_IMAGE_BIN, img_left_rgb_b)
    # print("Pickle time (s):", time.time() - start_time)
    

    

    # start_time = time.time()
    # # img_left_rgba = np.concatenate([img_left_rgb, depth_img_arr[..., None]], axis=-1)
    # # img_right_rgba = np.concatenate([img_right_rgb, depth_img_arr[..., None]], axis=-1)
    # # img_left_rgba = np.dstack([img_left_rgb, depth_img_arr[..., None]])
    # # img_right_rgba = np.dstack([img_right_rgb, depth_img_arr[..., None]])
    # # img_left_rgba[..., :3] = img_left_rgb
    # # img_right_rgba[..., :3] = img_right_rgb
    # # img_left_rgba[..., 3] = depth_img_arr
    # # img_right_rgba[..., 3] = depth_img_arr
    # img_left_rgb_jax = jnp.array(img_left_rgb)
    # img_right_rgb_jax = jnp.array(img_right_rgb)
    # depth_img_arr_jax = jnp.array(depth_img_arr)
    # img_left_rgba = jnp.concatenate([img_left_rgb_jax, depth_img_arr_jax[..., None]], axis=-1)
    # img_right_rgba = jnp.concatenate([img_right_rgb_jax, depth_img_arr_jax[..., None]], axis=-1)
    # print("Concatenation time (s):", time.time() - start_time)
    # pdb.set_trace()
    # img_left_right_rgba = np.stack([img_left_rgba, img_right_rgba])

    # if not args.depth_mode == "NONE":
    #     # Depth image 
    #     if args.depth_mode == "TRI":
    #         # Get depth image from TRI stereo to depth
    #         tri_depth, tri_left, tri_right = tri_stereo_to_depth.get_depth_and_bgr(
    #             img_left_bgr[:, :, :3], img_right_bgr[:, :, :3]
    #         )
    #         img_left_bgr = tri_left
    #         img_right_bgr = tri_right
    #         depth_img_arr = tri_depth.astype(np.float32)
    #     else: 
    #         depth_img_arr = depth_img.get_data()

    #     # Point cloud 
    #     # point_cloud_arr = np.array(point_cloud.get_data())

    #     # b_depth_img = m.packb(depth_img_arr)
    #     # b_point_cloud = m.packb(point_cloud_arr)

    #     img_left_rgba = np.concatenate([img_left_rgb, depth_img_arr[..., None]], axis=-1)
    #     img_right_rgba = np.concatenate([img_right_rgb, depth_img_arr[..., None]], axis=-1)
    #     img_left_right_rgba = np.stack([img_left_rgba, img_right_rgba])
    #     # b_img_left_right_rgba = m.packb(img_left_right_rgba)
    # else:
    #     if args.camera_calib:
    #         b_img_left = m.packb(img_left_rgb)
    #     else:
    #         b_img_left_right = m.packb(img_left_right_rgb)

    if args.render:
        img_w = img_left_bgr.shape[1]
        img_h = img_left_bgr.shape[0]
        min_dim = min(img_w, img_h)
        if img_w > min_dim:
            diff = img_w - min_dim
            img_left_bgr = img_left_bgr[:, diff//2:diff//2+min_dim]
        elif img_h > min_dim:
            diff = img_h - min_dim
            img_left_bgr = img_left_bgr[diff//2:diff//2+min_dim, :]
        print(img_left_bgr.shape)
        cv2.imshow("img_left", img_left_bgr)
        cv2.waitKey(1)

    # if not args.depth_mode == "NONE":
    #     if args.render_depth:
    #         depth_img_arr = depth_img_arr 
    #         # depth_img_arr = np.clip(depth_img_arr, 0, 255).astype(np.uint8)
    #         cv2.imshow("depth_img", depth_img_arr)
    #         cv2.waitKey(1)

    #     if args.render_pcd: 
    #         viewer.updateData(point_cloud)

    # # Send to redis
    # if args.camera_calib:
    #     redis_pipe.set(KEY_LEFT_CAMERA_IMAGE_BIN, b_img_left)
    # elif not args.depth_mode == "NONE":
    #     redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, b_img_left_right_rgba)
    # else:
    #     redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, b_img_left_right_rgb)

    # redis_pipe.set(KEY_LEFT_CAMERA_INTRINSIC, encode_matlab(K_left))
    # redis_pipe.set(KEY_RIGHT_CAMERA_INTRINSIC, encode_matlab(K_right))


    # redis_pipe.execute()


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

    # # Set up redis server
    # if args.use_nuc_ip: 
    #     _redis = ctrlutils.RedisClient(
    #         host=NUC_HOST, port=NUC_PORT, password=NUC_PWD
    #     )
    # else:
    #     _redis = ctrlutils.RedisClient(
    #         host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
    #     )
    # redis_pipe = _redis.pipeline()
    redis_client = redis.Redis(host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD)

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
    # img_left_rgba = np.zeros((res.height, res.width, 4), dtype=np.uint8)
    # img_right_rgba = np.zeros((res.height, res.width, 4), dtype=np.uint8)
    start_time = time.time()
    count = 0
    if args.render_pcd:
        while viewer.is_available():
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                capture_camera_data(args, zed, img_left, img_right, depth_img, point_cloud, 
                            redis_client, K_left, K_right, viewer=viewer)
                count += 1
                loop_dt = (time.time() - start_time) / count
                if count % 50 == 0:
                    print(f"Loop freq: {1/loop_dt:.3f} hz")
        viewer.exit()
    else:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                capture_camera_data(args, zed, img_left, img_right, depth_img, point_cloud, 
                            redis_client, K_left, K_right, tri_stereo_to_depth=tri_stereo_to_depth)
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
    parser.add_argument("--camera_calib", action="store_true")
    parser.add_argument("--depth_mode", required=True, choices=["PERFORMANCE", "ULTRA", "QUALITY", "NEURAL", "TRI", "NONE"])
    parser.add_argument("--resolution", required=True, choices=["VGA", "HD720", "HD1080", "HD2K"])
    args = parser.parse_args()
    main(args)

