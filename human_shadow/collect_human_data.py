import pdb 
import json
import argparse
import os
import time
import re
import curses
import numpy as np
from concurrent.futures import ThreadPoolExecutor

import mediapy as media
import pyzed.sl as sl

from human_shadow.utils.file_utils import get_parent_folder_of_package
from human_shadow.camera.zed_utils import *
from human_shadow.utils.button_utils import *


def create_output_folder(args):
    project_folder = get_parent_folder_of_package("human_shadow")
    save_folder = os.path.join(project_folder, "human_shadow/data", "videos", args.folder)
    os.makedirs(save_folder, exist_ok=True)
    n_folders = len([f for f in os.listdir(save_folder) if os.path.isdir(os.path.join(save_folder, f))])
    save_folder = os.path.join(save_folder, f"{n_folders}")
    os.makedirs(save_folder, exist_ok=True)

    left_video_path = os.path.join(save_folder, f"video_{n_folders}_L.mp4")
    right_video_path = os.path.join(save_folder, f"video_{n_folders}_R.mp4")
    intrinsics_path = os.path.join(save_folder, f"cam_intrinsics_{n_folders}.json")
    depth_path = os.path.join(save_folder, f"depth_{n_folders}.npy")
    return left_video_path, right_video_path, depth_path, intrinsics_path

def get_camera_params(zed):
    camera_model = zed.get_camera_information().camera_model
    camera_params = (
        zed.get_camera_information().camera_configuration.calibration_parameters
    )
    K_left = get_intrinsics_matrix(camera_params, cam_side="left")
    K_right = get_intrinsics_matrix(camera_params, cam_side="right")
    return camera_params, K_left, K_right


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

def resize_img_to_square(img):
    img_w = img.shape[1]
    img_h = img.shape[0]
    min_dim = min(img_w, img_h)
    if img_w > min_dim:
        diff = img_w - min_dim
        img = img[:, diff//2:diff//2+min_dim]
    elif img_h > min_dim:
        diff = img_h - min_dim
        img = img[diff//2:diff//2+min_dim, :]
    return img

def record_one_video(zed, camera_params, args, button):
    res = camera_params.left_cam.image_size
    img_left = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    img_right = sl.Mat(res.width, res.height, sl.MAT_TYPE.U8_C4)
    depth_img = sl.Mat(res.width, res.height, sl.MAT_TYPE.F32_C4)

    button.reset()

    print("Press the yellow button to start recording")
    button.wait_for_press()
    print("START RECORDING")

    dt = 1 / args.hz

    left_imgs = []
    right_imgs = []
    depth_imgs = []
    total_time = 0
    loop_count = 0
    button_pressed = False
    while not button_pressed:
        start_time = time.time()
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            img_left_rgb, img_right_rgb, depth_img_arr = capture_camera_data(zed, args.depth_mode, img_left, 
            img_right, depth_img)
            img_left_rgb = resize_img_to_square(img_left_rgb)
            img_right_rgb = resize_img_rgb = resize_img_to_square(img_right_rgb)
            depth_img_arr = resize_img_to_square(depth_img_arr)

            left_imgs.append(img_left_rgb.copy())
            right_imgs.append(img_right_rgb.copy())
            depth_imgs.append(depth_img_arr.copy())

            button_pressed = button.is_pressed()
    
            while (time.time() - start_time) < dt:
                pass
            loop_time = time.time() - start_time
            total_time += loop_time
            loop_count += 1

            if loop_count % 100 == 0:
                print(f"Average loop time: {total_time / loop_count}")

    fps = loop_count / total_time
    print(f"STOP RECORDING| FPS: {fps:.2f} Hz, Total time: {total_time:.2f}")

    left_imgs = np.array(left_imgs)
    right_imgs = np.array(right_imgs)

    return left_imgs, right_imgs, depth_imgs

def save_videos(left_video_path, right_video_path, left_imgs, right_imgs, args):
    executor = ThreadPoolExecutor(max_workers=2)
    future_left = executor.submit(media.write_video, left_video_path, left_imgs, fps=30)
    future_right = executor.submit(media.write_video, right_video_path, right_imgs, fps=30)
    return future_left, future_right



def init(args):
    # Initialize zed camera
    zed = init_zed(args.resolution, args.depth_mode)

    # Initialize yellow button
    button = Button()

    return zed,  button


def main(args):
    zed, button = init(args)
    future_left, future_right = None, None
    demo_idx = 0
    while True:
        print("DEMO ", demo_idx)
        left_video_path, right_video_path, depth_path, cam_intrinsics_path = create_output_folder(args)

        # Get camera parameters
        camera_params, K_left, K_right = get_camera_params(zed)
        save_intrinsics(camera_params, save_path=cam_intrinsics_path)

        # Record video
        left_imgs, right_imgs, depth_imgs = record_one_video(zed, camera_params, args, button)

        # Write videos (and wait for previous videos to finish writing)
        if future_left is not None:
            while not future_left.done():
                time.sleep(1)
        future_left, future_right = save_videos(left_video_path, right_video_path, left_imgs, right_imgs, args)

        # Save depth images
        np.save(depth_path, np.array(depth_imgs))

        print("\n")
        demo_idx += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="test")
    parser.add_argument("--hz", type=int, default=10, required=True)
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