import pdb
import os
import subprocess

import cv2
import mediapy as media
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd

from human_shadow.utils.file_utils import get_parent_folder_of_package

root_folder = get_parent_folder_of_package("human_shadow")

videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_marion_calib_2/")
video_idx = 0
video_folder = os.path.join(videos_folder, str(video_idx))

full_images_folder = os.path.join(video_folder, "annotated_images")
if not os.path.exists(full_images_folder):
    os.makedirs(full_images_folder)

original_imgs = np.array(media.read_video(os.path.join(video_folder, f"video_{video_idx}_L.mp4")))
sam_imgs = np.array(media.read_video(os.path.join(video_folder, "sam_imgs.mp4")))
pcd_imgs = np.array(media.read_video(os.path.join(video_folder, "pcd_imgs.mp4")))
hamer_imgs = np.array(media.read_video(os.path.join(video_folder, "hamer_imgs.mp4")))
df = pd.read_csv(os.path.join(video_folder, "finger_poses.csv"))

success_idx = np.array(df["idx"])
n_imgs = len(original_imgs)
print("Number of images:", n_imgs)

hamer_idx = 0
for idx in tqdm(range(n_imgs)):
    original_img = original_imgs[idx]
    sam_img = sam_imgs[idx]

    if idx in success_idx:
        hamer_img = hamer_imgs[hamer_idx]
        pcd_img = pcd_imgs[hamer_idx]
        hamer_idx += 1
    else:
        pcd_img = np.zeros_like(original_img)
        hamer_img = np.zeros_like(original_img)

    fig, axs = plt.subplots(2,2, figsize=(20,20))
    axs[0,0].imshow(original_img)
    axs[0,0].set_axis_off()
    axs[0,1].imshow(sam_img)
    axs[0,1].set_axis_off()
    axs[1,0].imshow(hamer_img)
    axs[1,0].set_axis_off()
    axs[1,1].imshow(pcd_img)
    axs[1,1].set_axis_off()
    plt.tight_layout()
    plt.savefig(f"{full_images_folder}/{idx:04d}.png")
    plt.close(fig)

subprocess.run([
    'ffmpeg', '-r', '10', '-i', os.path.join(full_images_folder, 'frame_%04d.png'),
    '-vcodec', 'libx264', '-crf', '25', '-pix_fmt', 'yuv420p',
    os.path.join(videos_folder, 'annotated_imgs.mp4')
])
pdb.set_trace()


