import os
import cv2

from human_shadow.detectors.detector_hamer import DetectorHamer


folder_path = "/juno/u/jyfang/human_shadow/human_shadow/data/rgb"
imgs_rgb_L_paths = sorted([file for file in os.listdir(folder_path) if file.lower().endswith('.jpg')])
imgs_rgb_L=[]
for path in imgs_rgb_L_paths:
    full_path = os.path.join(folder_path, path)
    imgs_rgb_L.append(cv2.imread(full_path)[:, :, ::-1])

# Adjust for size of images
img_w, img_h = imgs_rgb_L[0].shape[:2]
detector_hamer = DetectorHamer()

for i in range(len(imgs_rgb_L)):
    img, img_path = imgs_rgb_L[i], imgs_rgb_L_paths[i]
    assert img.shape[:2] == (img_w, img_h)
    hamer_output = detector_hamer.detect_hand_keypoints(img)
    if not os.path.exists("outputs/"):
        os.makedirs("outputs/")
    cv2.imwrite(os.path.join("outputs", img_path), hamer_output["annotated_img"])