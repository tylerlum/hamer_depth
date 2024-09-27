"""
Save video as images in a folder
"""

import pdb
import mediapy as media
import numpy as np
import os
from tqdm import tqdm

video_path = "/juno/u/lepertm/human_shadow/human_shadow/data/videos/demo1/video_0_L.mp4"
save_folder = "/juno/u/lepertm/human_shadow/human_shadow/data/videos/demo1/video_0_L"

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

imgs = np.array(media.read_video(video_path))
n_imgs = len(imgs)
for idx in tqdm(range(n_imgs)):
    img = imgs[idx]
    media.write_image(f"{save_folder}/{idx:05d}.jpg", img)
