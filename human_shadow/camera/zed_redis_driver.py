"""
Stream rgb images from Zed camera to redis
"""

import msgpack_numpy as m

m.patch()
import argparse
import os
import sys
import time

import ctrlutils
import cv2
import numpy as np
import pyzed.sl as sl

import human_shadow.camera.ogl_viewer.viewer as gl

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"


from human_shadow.camera.zed_utils import *
from human_shadow.config.redis_keys import *


def stream_camera_data(
    args,
    zed,
    img_left,
    img_right,
    depth_img,
    point_cloud,
    redis_pipe,
    K_left,
    K_right,
    tri_stereo_to_depth=None,
    viewer=None,
):
    img_left_rgb, img_right_rgb, depth_img_arr = capture_camera_data(
        zed, args.depth_mode, img_left, img_right, depth_img
    )

    img_left_bgr = img_left_rgb[..., ::-1]
    img_right_bgr = img_right_rgb[..., ::-1]

    if not args.depth_mode == "NONE":
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
            depth_img_arr = depth_img.get_data()

        # Point cloud
        # point_cloud_arr = np.array(point_cloud.get_data())

        # b_depth_img = m.packb(depth_img_arr)
        # b_point_cloud = m.packb(point_cloud_arr)

        img_left_rgba = np.concatenate(
            [img_left_rgb, depth_img_arr[..., None]], axis=-1
        )
        img_right_rgba = np.concatenate(
            [img_right_rgb, depth_img_arr[..., None]], axis=-1
        )
        img_left_right_rgba = np.stack([img_left_rgba, img_right_rgba])
        b_img_left_right_rgba = m.packb(img_left_right_rgba)
    else:
        if args.camera_calib:
            b_img_left = m.packb(img_left_rgb)
        else:
            img_left_right_rgb = np.stack([img_left_rgb, img_right_rgb])
            b_img_left_right_rgb = m.packb(img_left_right_rgb)

    if args.render:
        img_w = img_left_bgr.shape[1]
        img_h = img_left_bgr.shape[0]
        min_dim = min(img_w, img_h)
        if img_w > min_dim:
            diff = img_w - min_dim
            img_left_bgr = img_left_bgr[:, diff // 2 : diff // 2 + min_dim]
        elif img_h > min_dim:
            diff = img_h - min_dim
            img_left_bgr = img_left_bgr[diff // 2 : diff // 2 + min_dim, :]
        cv2.imshow("img_left", img_left_bgr)
        cv2.waitKey(1)

    if not args.depth_mode == "NONE":
        if args.render_depth:
            depth_img_arr = depth_img_arr
            # depth_img_arr = np.clip(depth_img_arr, 0, 255).astype(np.uint8)
            cv2.imshow("depth_img", depth_img_arr)
            cv2.waitKey(1)

        if args.render_pcd:
            viewer.updateData(point_cloud)

    # Send to redis
    if not args.depth_mode == "NONE":
        redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, b_img_left_right_rgba)
    elif args.camera_calib:
        redis_pipe.set(KEY_LEFT_CAMERA_IMAGE_BIN, b_img_left)
    else:
        redis_pipe.set(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN, b_img_left_right_rgb)

    K_left_b = m.packb(K_left)
    K_right_b = m.packb(K_right)
    redis_pipe.set(KEY_LEFT_CAMERA_INTRINSIC, K_left_b)
    redis_pipe.set(KEY_RIGHT_CAMERA_INTRINSIC, K_right_b)
    redis_pipe.execute()


def main(args):
    print("Starting Zed driver...")
    zed = init_zed(args.resolution, args.depth_mode)

    # Get camera params
    camera_model = zed.get_camera_information().camera_model
    camera_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    K_left = get_intrinsics_matrix(camera_params, cam_side="left")
    K_right = get_intrinsics_matrix(camera_params, cam_side="right")
    save_intrinsics(
        camera_params, save_path=f"intrinsics/camera_intrinsics_{args.resolution}.json"
    )

    res = camera_params.left_cam.image_size

    img_left = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    img_right = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)

    depth_img = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4)
    point_cloud = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

    # Set up redis server
    if args.use_nuc_ip:
        _redis = ctrlutils.RedisClient(host=NUC_HOST, port=NUC_PORT, password=NUC_PWD)
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
                stream_camera_data(
                    args,
                    zed,
                    img_left,
                    img_right,
                    depth_img,
                    point_cloud,
                    redis_pipe,
                    K_left,
                    K_right,
                    viewer=viewer,
                )
                count += 1
                loop_dt = (time.time() - start_time) / count
                if count % 50 == 0:
                    print(f"Loop freq: {1 / loop_dt:.3f} hz")
        viewer.exit()
    else:
        while True:
            if zed.grab() == sl.ERROR_CODE.SUCCESS:
                stream_camera_data(
                    args,
                    zed,
                    img_left,
                    img_right,
                    depth_img,
                    point_cloud,
                    redis_pipe,
                    K_left,
                    K_right,
                    tri_stereo_to_depth=tri_stereo_to_depth,
                )
                count += 1
                loop_dt = (time.time() - start_time) / count
                if count % 50 == 0:
                    print(f"Loop freq: {1 / loop_dt:.3f} hz")

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
    parser.add_argument(
        "--depth_mode",
        required=True,
        choices=["PERFORMANCE", "ULTRA", "QUALITY", "NEURAL", "TRI", "NONE"],
    )
    parser.add_argument(
        "--resolution", required=True, choices=["VGA", "HD720", "HD1080", "HD2K"]
    )
    args = parser.parse_args()
    main(args)
