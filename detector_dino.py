"""
Wrapper around DINO for object detection
"""

import pdb
from typing import List, Tuple
import numpy as np

from transformers import pipeline
from PIL import Image
import mediapy as media
import cv2

from utils.image_utils import DetectionResult

class DetectorDino:
    def __init__(self, detector_id: str):
        self.detector = pipeline(
            model=detector_id,
            task="zero-shot-object-detection",
            device="cuda",
        )


    def get_bboxes(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
                   visualize: bool = False) -> List[np.ndarray]:
        img_pil = Image.fromarray(frame)
        labels = [f"{object_name}."]
        results = self.detector(img_pil, candidate_labels=labels, threshold=threshold)
        results = [DetectionResult.from_dict(result) for result in results]
        if not results:
            return None
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
            cv2.waitKey(0)
        return bboxes, scores


    def get_best_bbox(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
               visualize: bool = False) -> Tuple[np.ndarray, np.ndarray[np.int32]]:
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
            cv2.waitKey(0)
        return best_bbox
    
    


if __name__ == "__main__":
    img_path = "data/demo/00000.jpg"
    frame = media.read_image(img_path)
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector = DetectorDino(detector_id)
    # detector.get_best_bbox(frame, "hand", visualize=True)
    detector.get_bboxes(frame, "hand", visualize=True)