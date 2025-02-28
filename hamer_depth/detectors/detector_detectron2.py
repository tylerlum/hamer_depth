"""
Wrapper around detectron2 for object detection
"""

import glob
import os
from pathlib import Path
from typing import Tuple

import cv2
import hamer
import mediapy as media
import numpy as np
import requests
from detectron2.config import LazyConfig
from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

from human_shadow.utils.file_utils import get_parent_folder_of_package


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
        cfg_path = (
            Path(hamer.__file__).parent
            / "configs"
            / "cascade_mask_rcnn_vitdet_h_75ep.py"
        )
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

    def get_bboxes(
        self, img: np.ndarray, visualize: bool = False, visualize_wait: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Get bounding boxes and scores for the detected hand in the image"""
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
                cv2.putText(
                    img_bgr,
                    f"{score:.4f}",
                    (int(bbox[0]), int(bbox[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow("Detected bounding boxes", img_bgr)
            if visualize_wait:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)

        return pred_bboxes, pred_scores

    def get_best_bbox(
        self, img: np.ndarray, visualize: bool = False, visualize_wait: bool = True
    ) -> Tuple[np.ndarray, float]:
        """Get the best bounding box and score for the detected hand in the image"""
        bboxes, scores = self.get_bboxes(img)
        if len(bboxes) == 0:
            return np.ndarray([]), 0
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
            cv2.putText(
                img_bgr,
                f"{best_score:.4f}",
                (int(best_bbox[0]), int(best_bbox[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Best detected bounding box", img_bgr)
            if visualize_wait:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)

        return best_bbox, best_score


if __name__ == "__main__":
    root_folder = get_parent_folder_of_package("human_shadow")
    detector = DetectorDetectron2(root_dir=root_folder)

    image_paths = glob.glob(
        os.path.join(
            root_folder,
            "human_shadow/data/videos/demo_marion_calib_2/0/video_0_L/*.jpg",
        )
    )
    # image_paths = glob.glob(os.path.join(root_folder, "human_shadow/data/videos/demo1/video_0_L/*.jpg"))
    image_paths = sorted(
        image_paths, key=lambda x: int(os.path.basename(x).split(".")[0])
    )

    for img_path in image_paths:
        frame = media.read_image(img_path)

        # detector.get_best_bbox(frame, visualize=True)
        detector.get_bboxes(frame, visualize=True, visualize_wait=False)
