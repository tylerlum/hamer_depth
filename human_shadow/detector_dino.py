"""
Wrapper around DINO for object detection
"""
import os
import pdb
from typing import List, Tuple, Optional
import numpy as np

from transformers import pipeline
from PIL import Image
import mediapy as media
import cv2

from human_shadow.utils.image_utils import DetectionResult
from human_shadow.utils.file_utils import get_parent_folder_of_package

class DetectorDino:
    def __init__(self, detector_id: str):
        self.detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device="cuda",
        )


    def get_bboxes(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
                   visualize: bool = False, visualize_wait: bool = True) -> Tuple[List[np.ndarray], List[np.float32]]:
        img_pil = Image.fromarray(frame)
        labels = [f"{object_name}."]
        results = self.detector(img_pil, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
        if not results:
            return [], []
        bboxes = [np.array(result.box.xyxy) for result in results]
        scores = [result.score for result in results]

        if visualize:
            img_rgb = frame.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            for bbox, score in zip(bboxes, scores):
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
            cv2.imshow("Detection", img_bgr)
            if visualize_wait:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return bboxes, scores


    def get_best_bbox(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
               visualize: bool = False, visualize_wait: bool = True) -> Optional[np.ndarray]:
        bboxes, scores = self.get_bboxes(frame, object_name, threshold)
        if len(bboxes) == 0:
            return None
        best_idx = np.array(scores).argmax()
        best_bbox, best_score = bboxes[best_idx], scores[best_idx]

        if visualize:
            img_rgb = frame.copy()
            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.rectangle(
                img_bgr,
                (best_bbox[0], best_bbox[1]),
                (best_bbox[2], best_bbox[3]),
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
            cv2.imshow("Detection", img_bgr)
            if visualize_wait:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return best_bbox
    
    
if __name__ == "__main__":
    root_folder = get_parent_folder_of_package("human_shadow")
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector = DetectorDino(detector_id)

    indices = np.arange(13, 40)
    for idx in indices:
        img_path = os.path.join(root_folder, f"human_shadow/data/videos/demo1/video_0_L/000{idx}.jpg")

        frame = media.read_image(img_path)
        # detector.get_best_bbox(frame, "hand", visualize=True)
        detector.get_bboxes(frame, "hand", visualize=True, visualize_wait=False)