import pdb
import os
import mediapy as media
import numpy as np 
from tqdm import tqdm
import cv2
import matplotlib.pyplot as plt

from human_shadow.utils.file_utils import get_parent_folder_of_package

def convert_mask_to_color(mask, color):
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    mask_rgb[mask[:,:,0] > 0] = color
    return mask_rgb


project_folder = get_parent_folder_of_package("human_shadow")

# human_data_folder = os.path.join(project_folder, "human_shadow/data/videos/demo_marion_calib_2/0")
# human_masks_path = os.path.join(human_data_folder, "imgs_sam.mp4")
human_masks_path = "sam_masks.avi"
human_masks = media.read_video(human_masks_path)


robot_masks_path = "masked_imgs.avi"
robot_masks = media.read_video(robot_masks_path)


n_masks = len(robot_masks)

list_overlay_imgs = []
list_human_masks = []
list_robot_masks = []
for idx in tqdm(range(n_masks)):
    human_mask = human_masks[idx]
    robot_mask = robot_masks[idx]

    human_mask = human_mask[:,420:-420]
    # human_mask = np.sum(human_mask, axis=2)
    # binary_mask = human_mask > 0
    # binary_mask = binary_mask.astype(np.uint8) * 255
    # binary_mask = 255 - binary_mask
    # binary_mask = np.tile(binary_mask[:, :, np.newaxis], (1, 1, 3))
    # human_mask = binary_mask

    human_mask = convert_mask_to_color(human_mask, (0, 0, 255))
    robot_mask = convert_mask_to_color(robot_mask, (0, 255, 0))

    overlay_img = human_mask + robot_mask
    # overlay_img = np.array(overlay_img)
    # overlay_img[overlay_img > 0] = 1

    # overlay_img = overlay_img * (-1)
    # overlay_img += 1
    # overlay_img *= 255

    overlay_img = overlay_img.astype(np.uint8)
    overlay_img[(overlay_img == [0,0,0]).all(axis=-1)] = 255


    # plt.imshow(overlay_img)
    # plt.show()
    # pdb.set_trace()
    list_overlay_imgs.append(overlay_img)
    list_human_masks.append(human_mask)
    list_robot_masks.append(robot_mask)

    # if idx > 100:
    #     plt.imshow(overlay_img)
    #     plt.show()

    #     pdb.set_trace()


    # fig = plt.figure()
    # ax = fig.add_subplot(1, 2, 1)
    # ax.imshow(human_mask)
    # ax = fig.add_subplot(1, 2, 2)
    # ax.imshow(robot_mask)
    # plt.show()


    # overlay_img = cv2.add(human_mask, robot_mask)


    # overlay_img = cv2.addWeighted(human_mask, 0.5, robot_mask, 0.5, 0)

    # list_overlay_imgs.append(overlay_img)


media.write_video("overlay_imgs_multi.mp4", list_overlay_imgs, fps=30)
media.write_video("human_masks.mp4", list_human_masks, fps=30)
media.write_video("robot_masks.mp4", list_robot_masks, fps=30)




pdb.set_trace()