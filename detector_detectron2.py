"""
Wrapper around detectron2 for object detection
"""

import pdb 
import os
import numpy as np
from pathlib import Path
from typing import Tuple

import cv2
import mediapy as media
import requests
import hamer
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig

from utils.file_utils import get_parent_folder_of_package


def download_detectron_ckpt(root_dir: str, ckpt_path: str) -> None:
    url = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
    save_path = Path(root_dir, ckpt_path)
    save_path.parent.mkdir(exist_ok=True, parents=True)
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        print(f"File downloaded successfully and saved to {save_path}")
    else:
        print(f"Failed to download the file. Status code: {response.status_code}")



class DetectorDetectron2:
    def __init__(self, root_dir: str):
        cfg_path = (Path(hamer.__file__).parent / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py")
        detectron2_cfg = LazyConfig.load(str(cfg_path))

        detectron2_cfg.train.init_checkpoint = os.path.join(
            root_dir, "_DATA/detectron_ckpts/model_final_f05665.pkl"
        )
        if not os.path.exists(detectron2_cfg.train.init_checkpoint):
            download_detectron_ckpt(
                root_dir, "_DATA/detectron_ckpts/model_final_f05665.pkl"
            )
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
        self.detectron2 = DefaultPredictor_Lazy(detectron2_cfg)

    def get_bboxes(self, img: np.ndarray, visualize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """ Get bounding boxes and scores for the detected hand in the image """
        det_out = self.detectron2(img)

        det_instances = det_out["instances"]
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        if visualize:
            img_rgb = img.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            for bbox, score in zip(pred_bboxes, pred_scores):
                cv2.rectangle(
                    img_bgr,
                    (int(bbox[0]), int(bbox[1])),
                    (int(bbox[2]), int(bbox[3])),
                    (0, 255, 0),
                    2,
                )
                cv2.putText(img_bgr,
                            f"{score:.4f}",
                            (int(bbox[0]), int(bbox[1])),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2,
                            cv2.LINE_AA)

            cv2.imshow(f"Detected bounding boxes", img_bgr)
            cv2.waitKey(0)

        return pred_bboxes, pred_scores
    
    def get_best_bbox(self, img: str, visualize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """ Get the best bounding box and score for the detected hand in the image """
        bboxes, scores = self.get_bboxes(img)
        if len(bboxes) == 0:
            return None
        best_idx = scores.argmax()
        best_bbox, best_score = bboxes[best_idx], scores[best_idx]

        if visualize: 
            img_rgb = img.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                img_bgr,
                (int(best_bbox[0]), int(best_bbox[1])),
                (int(best_bbox[2]), int(best_bbox[3])),
                (0, 255, 0),
                2,
            )
            cv2.putText(img_bgr,
                        f"{best_score:.4f}",
                        (int(best_bbox[0]), int(best_bbox[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2,
                        cv2.LINE_AA)

            cv2.imshow(f"Best detected bounding box", img_bgr)
            cv2.waitKey(0)
        
        return best_bbox, best_score
                

if __name__ == "__main__":
    img_path = "data/demo/00000.jpg"
    frame = media.read_image(img_path)
    root_dir = get_parent_folder_of_package("hamer") 
    detector = DetectorDetectron2(root_dir=root_dir)
    # detector.get_best_bbox(frame, visualize=True)
    detector.get_bboxes(frame, visualize=True)

