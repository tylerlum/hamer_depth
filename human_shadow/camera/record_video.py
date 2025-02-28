import argparse
import os
import re
import time

import numpy as np

# import pygame
# from pygame.locals import *
import pyzed.sl as sl

from human_shadow.camera.zed_utils import *
from human_shadow.utils.button_utils import *
from human_shadow.utils.file_utils import get_parent_folder_of_package


def main(args):
    # Initialize zed camera
    zed = init_zed(args.resolution, args.depth_mode)

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

    # Create output folder
    project_folder = get_parent_folder_of_package("human_shadow")
    save_folder = os.path.join(
        project_folder, "human_shadow/data", "videos", args.folder
    )
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    left_pattern = re.compile(r"video_(\d+)_L.mp4")
    left_video_files = [f for f in os.listdir(save_folder) if left_pattern.match(f)]
    folder_count = len(left_video_files)
    left_video_path = os.path.join(save_folder, f"video_{folder_count}_L.mp4")
    right_video_path = os.path.join(save_folder, f"video_{folder_count}_R.mp4")

    # Initialize yellow button
    # init_button()
    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        raise Exception("No joystick found")
    joystick = pygame.joystick.Joystick(0)
    joystick.init()

    print("Press the yellow button to start recording")
    # wait_for_button_press()
    button_pressed = False
    while not button_pressed:
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                print("Pressed!")
                button_pressed = True
                break
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
            img_left_rgb, img_right_rgb, depth_img_arr = capture_camera_data(
                zed, args.depth_mode, img_left, img_right, depth_img
            )
            left_imgs.append(img_left_rgb.copy())
            right_imgs.append(img_right_rgb.copy())
            depth_imgs.append(depth_img_arr.copy())

            # Check if button is pressed
            # button_pressed = is_button_pressed()
            for event in pygame.event.get():
                if event.type == pygame.JOYBUTTONDOWN:
                    button_pressed = True
                    break

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

    # media.write_video(left_video_path, left_imgs, fps=args.hz)
    # media.write_video(right_video_path, right_imgs, fps=args.hz)
    # np.save(os.path.join(save_folder, f"depth_imgs_{folder_count}.npy"), np.array(depth_imgs))


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
