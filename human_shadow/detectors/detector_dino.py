"""
Wrapper around DINO for object detection
"""
import os
import pdb
import glob
from tqdm import tqdm
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
            batch_size=16,
        )


    def get_bboxes(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
                   visualize: bool = False, pause_visualization: bool = True) -> Tuple[List[np.ndarray], List[np.float32]]:
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
            if pause_visualization:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return bboxes, scores


    def get_best_bbox(self, frame: np.ndarray, object_name: str, threshold: float = 0.4, 
               visualize: bool = False, pause_visualization: bool = True) -> Optional[np.ndarray]:
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
            if pause_visualization:
                cv2.waitKey(0)
            else:
                cv2.waitKey(1)
        return best_bbox
    
    
if __name__ == "__main__":
    root_folder = get_parent_folder_of_package("human_shadow")
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector_dino = DetectorDino(detector_id)

    # video_folder = "/juno/u/lepertm/shadow/human_shadow/human_shadow/data/videos/demo_marion_calib/0"
    # left_video_path = os.path.join(video_folder, f"video_0_L_preprocessed.mp4")
    # left_imgs = np.array(media.read_video(left_video_path))
    # detector.get_bboxes(left_imgs[0], "hand", visualize=True, visualize_wait=True)
    # pdb.set_trace()

    image_paths = glob.glob(os.path.join(root_folder, "human_shadow/data/videos/demo1/video_0_L/*.jpg"))
    image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split(".")[0]))

    videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_marion_calib_2/")
    video_idx = 0
    video_folder = os.path.join(videos_folder, str(video_idx))
    video_path = os.path.join(video_folder, f"video_{video_idx}_L.mp4")
    imgs_rgb = media.read_video(video_path)

    for img_rgb in tqdm(imgs_rgb):
        detector_dino.get_best_bbox(img_rgb, "hand", visualize=False)
        # detector.get_bboxes(frame, "hand", threshold=0.2, visualize=True, pause_visualization=False)

    # videos_folder = os.path.join(root_folder, "human_shadow/data/videos/demo_marion_calib_2/")
    # video_idx = 0
    # video_folder = os.path.join(videos_folder, str(video_idx))
    # video_idx = 0
    # video_path = os.path.join(video_folder, f"video_{video_idx}_L.mp4")
    # imgs_rgb = media.read_video(video_path)

    # detector_id = "IDEA-Research/grounding-dino-tiny"
    # detector = pipeline(
    #             model=detector_id,
    #             task="zero-shot-object-detection",
    #             device="cuda",
    #             batch_size=16,
    #         )

    # batch_size = 16
    # object_name = "hand"
    # threshold=0.8
    # labels = [f"{object_name}." for _ in range(batch_size)]
    # images = [img for img in imgs_rgb[:batch_size]]
    # results = detector(images, candidate_labels=labels, threshold=threshold)

    # pdb.set_trace()