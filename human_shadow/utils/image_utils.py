import pdb 
import numpy as np


def resize_image_to_rectangle(img, target_height, target_width):
    assert(img.shape[0] == target_height)
    if img.shape[0] == img.shape[1]:
        new_img = np.zeros((target_height, target_width, 3), dtype=np.uint8)
        dp = (target_width - target_height) // 2
        new_img[:, dp:dp+target_height] = img
        img = new_img
    return img, dp



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