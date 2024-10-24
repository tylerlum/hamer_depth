import pdb
import os
import logging
import glob
from typing import Tuple
from tqdm import tqdm
import numpy as np 
import pandas as pd

import mediapy as media
from human_shadow.detectors.detector_dino import DetectorDino
from human_shadow.detectors.detector_detectron2 import DetectorDetectron2
from human_shadow.detectors.detector_hamer import DetectorHamer, THUMB_VERTEX, INDEX_FINGER_VERTEX
from human_shadow.detectors.detector_sam2 import DetectorSam2
from human_shadow.utils.pcd_utils import *
from human_shadow.utils.transform_utils import *
from human_shadow.utils.video_utils import *
from human_shadow.camera.zed_utils import *
from human_shadow.utils.file_utils import get_parent_folder_of_package

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def is_hand_in_image(img: np.ndarray, hamer_output: dict) -> bool:
    """ 
    Return True if a hand is detected in the image by HaMeR
    """
    return hamer_output is not None and hamer_output["success"]

def get_paths(video_folder: str, video_idx: int) -> Tuple[str, str, str]:
    video_path = os.path.join(video_folder, f"video_{video_idx}_L.mp4")
    depth_path = os.path.join(video_folder, f"depth_{video_idx}.npy")
    cam_intrinsics_path = os.path.join(video_folder, f"cam_intrinsics_{video_idx}.json")
    return video_path, depth_path, cam_intrinsics_path


def get_first_frame_with_hand(imgs_rgb: np.ndarray, detector: DetectorHamer) -> Tuple[int, dict]:
    """
    Return the index of the first frame with a hand detected by HaMeR.
    """
    for idx, img in enumerate(imgs_rgb):
        try:
            hamer_output = detector.detect_hand_keypoints(img)
        except ValueError as e:
            logger.debug(f"{e}")
            continue
        if is_hand_in_image(img, hamer_output):
            return idx, hamer_output
    return -1, {}

root_folder = get_parent_folder_of_package("human_shadow")

detector_hamer = DetectorHamer()
detector_id = "IDEA-Research/grounding-dino-base"
detector_bbox = DetectorDino(detector_id)
detector_detectron = DetectorDetectron2(root_dir=root_folder)
detector_sam = DetectorSam2()

print("Here")

# videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_jiaying_waffles_large_2/")
videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_marion_calib_2/")
# videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo1/")
video_idx = 0
video_folder = os.path.join(videos_folder, str(video_idx))
video_path, depth_path, cam_intrinsics_path = get_paths(video_folder, video_idx)

# Convert video to images if they don't exist
images_folder = os.path.join(video_folder, "images")
if not os.path.exists(images_folder):
    convert_video_to_images(video_path, images_folder)
image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))
image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split(".")[0]))


# Get intrinsics
camera_matrix = get_intrinsics_from_json(cam_intrinsics_path)
cam_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)
    
# Get data
imgs_rgb = media.read_video(video_path)
imgs_depth = np.load(depth_path)

# Adjust for size of images
img_w, img_h = imgs_rgb[0].shape[:2]
depth_w, depth_h = imgs_depth[0].shape[:2]
assert img_w == depth_w and img_h == depth_h
zed_resolution = ZED_RESOLUTIONS_SQUARE_SIZE[img_w]
if img_h < zed_resolution.value[1]:
    offset = (zed_resolution.value[1] - img_h) // 2
    cam_intrinsics["cx"] -= offset
elif img_h > zed_resolution.value[1]:
    raise ValueError("Image height is greater than ZED resolution")

# Get first frame with hand
start_idx, hamer_output = get_first_frame_with_hand(imgs_rgb, detector_hamer)

# Get bbox of arm
bbox, score = detector_detectron.get_best_bbox(imgs_rgb[start_idx], visualize=False)

# Segment out arm in video
kpts_2d = hamer_output["kpts_2d"]
masks, sam_imgs = detector_sam.segment_video(images_folder, bbox=bbox, bbox_ctr=kpts_2d.astype(np.int32), start_idx=start_idx)

# n_imgs = len(sam_imgs)
# list_masks = []
# for idx in tqdm(range(n_imgs)):
#     mask = masks[idx][0][0]
#     # mask = np.tile(mask[:, :, np.newaxis], (1, 1, 3))
#     mask = mask.astype(np.uint8) * 255
#     list_masks.append(mask)

list_masks = [(masks[idx][0][0].astype(np.uint8) * 255) for idx in tqdm(range(len(sam_imgs)))]
media.write_video("sam_masks2.mkv", list_masks, codec="ffv1", input_format="gray")