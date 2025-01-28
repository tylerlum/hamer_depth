import pdb
import os
import numpy as np

import mediapy as media


def convert_video_to_images(video_path: str, save_folder: str):
    """Save each frame of video as an image in save_folder."""
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    imgs = np.array(media.read_video(video_path))
    n_imgs = len(imgs)
    for idx in range(n_imgs):
        img = imgs[idx]
        media.write_image(f"{save_folder}/{idx:05d}.jpg", img)