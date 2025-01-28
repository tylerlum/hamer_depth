"""
Wrapper around HaMeR for hand keypoint detection.

Adapted from Zi-ang-Cao's code and original HaMeR code.
"""

import os
import pdb
import numpy as np
from pathlib import Path
from typing import Optional

import cv2
import torch
import mediapy as media
from hamer.utils import recursive_to

from hamer.models import HAMER, DEFAULT_CHECKPOINT
from vitpose_model import ViTPoseModel
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
from hamer.utils.geometry import perspective_projection
from hamer.configs import get_config

from detector_detectron2 import DetectorDetectron2
from detector_dino import DetectorDino
from utils.file_utils import get_parent_folder_of_package

class DetectorHamer:
    def __init__(self):
        root_dir = get_parent_folder_of_package("hamer")
        checkpoint_path = Path(root_dir, DEFAULT_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rescale_factor = 2.0 # Factor for padding the box
        self.batch_size = 2 # Batch size for inference

        self.model, self.model_cfg = self.load_hamer_model(checkpoint_path, root_dir)
        self.model.to(self.device)
        self.model.eval()

        self.cpm = ViTPoseModel(self.device)

        # Load bounding box detectors
        self.dino_detector = DetectorDino("IDEA-Research/grounding-dino-tiny")
        self.detectron_detector = DetectorDetectron2(root_dir)


    def detect_hand_keypoints(self, img: np.ndarray, visualize: bool=False):
        # Get initial bounding boxes
        bboxes, scores = self.get_bboxes(img)

        # Detect keypoints inside bounding boxes
        vitposes_out = self.get_human_vitposes(img, bboxes, scores)

        # Refine bounding boxes based on keypoints
        bboxes, is_right, idx_to_confidence = self.get_bboxes_from_keypoints(vitposes_out)
        bboxes, is_right = DetectorHamer.sort_bboxes(bboxes, is_right, idx_to_confidence)

        dataset = ViTDetDataset(self.model_cfg, img, bboxes, is_right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
    
        list_2d_kpts = []
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self.model(batch)

            T_cam_pred_all, scaled_focal_length = self.get_all_T_cam_pred(batch, out)

            for idx in range(len(T_cam_pred_all)):
                kpts_3d = out["pred_keypoints_3d"][idx].detach().cpu().numpy()  # [21, 3]
                verts = out["pred_vertices"][idx].detach().cpu().numpy()  # [778, 3]
                is_right = batch["right"][idx].cpu().numpy()

                T_cam_pred = T_cam_pred_all[idx]

                W_H_shapes = batch["img_size"].float() 
                img_w = W_H_shapes[idx][0]
                img_h = W_H_shapes[idx][1]

                kpts_2d_hamer = DetectorHamer.project_3d_kpt_to_2d(kpts_3d, img_w, img_h, 
                                                                     T_cam_pred, scaled_focal_length)
                list_2d_kpts.append(kpts_2d_hamer)

        if visualize:
            annotated_img = DetectorHamer.visualize_2d_kpt_on_img(
                kpts_2d=list_2d_kpts[0],
                img=img,
            )
            cv2.imshow("Annotated Image", annotated_img)
            cv2.waitKey(0)


    def get_all_T_cam_pred(self, batch, out):

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        # NOTE: FOR HaMeR, they are using the img_size as (W, H)
        W_H_shapes = batch["img_size"].float()  # (2, 2): full image size (2208, 1242)

        multiplier = 2 * batch["right"] - 1
        scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE * W_H_shapes.max())

        # Get cam_t to full image (instead of bbox)
        T_cam_pred_all = cam_crop_to_full(
            pred_cam, box_center, box_size, W_H_shapes, scaled_focal_length
        )

        return T_cam_pred_all, scaled_focal_length

    
    def get_bboxes(self, img: np.ndarray, use_dino: bool=True, use_detectron: bool=True, visualize: bool=False):
        if use_dino:
            dino_bboxes, dino_scores = self.dino_detector.get_bboxes(img, "hand", visualize=visualize)

        if use_detectron:
            det_bboxes, det_scores = self.detectron_detector.get_bboxes(img, visualize=visualize)

        if use_dino and use_detectron:
            bboxes = np.vstack([dino_bboxes, det_bboxes])
            scores = np.vstack([dino_scores, det_scores])
        elif use_dino:
            bboxes, scores = dino_bboxes, dino_scores
        elif use_detectron:
            bboxes, scores = det_bboxes, det_scores

        return bboxes, scores
    
    def get_human_vitposes(self, img, bboxes, scores):
        return self.cpm.predict_pose(img, [np.concatenate([bboxes,scores], axis=1)],)
    
    
    def evaluate_hand_vitposes(self, vitposes, n_valid_pts_thresh=3):
        left_hand_keypoint = vitposes["keypoints"][-42:-21]
        right_hand_keypoint = vitposes["keypoints"][-21:]

        confidence_thresh = 0.5
        n_valid_left_pts = np.sum(left_hand_keypoint[:, 2] > confidence_thresh)
        n_valid_right_pts = np.sum(right_hand_keypoint[:, 2] > confidence_thresh)
        if (n_valid_left_pts < n_valid_pts_thresh) and (n_valid_right_pts < n_valid_pts_thresh):
            return None, None, None
        
        bboxes = []
        is_right = []
        confidences = []
        if n_valid_left_pts > n_valid_pts_thresh:
            bbox, confidence = DetectorHamer.get_bbox_from_keypoints(left_hand_keypoint, confidence_thresh)
            bboxes.append(bbox)
            is_right.append(False)
            confidences.append(confidence)

        if n_valid_right_pts > n_valid_pts_thresh:
            bbox, confidence = DetectorHamer.get_bbox_from_keypoints(right_hand_keypoint, confidence_thresh)
            bboxes.append(bbox)
            is_right.append(True)
            confidences.append(confidence)

        return bboxes, is_right, confidences
    
    def get_bboxes_from_keypoints(self, vitposes_out):
        bboxes = []
        is_right = []
        idx_to_confidence = []
        idx = 0
        for vitposes in vitposes_out:
            sub_bboxes, sub_is_right, sub_confidences = self.evaluate_hand_vitposes(vitposes)
            print("Confidences: ", sub_confidences)
            if sub_bboxes is not None:
                print("Not none")
                bboxes.extend(sub_bboxes)
                is_right.extend(sub_is_right)
                for confidence in sub_confidences:
                    idx_to_confidence.append((idx, confidence))
                    idx += 1
        if len(bboxes) == 0:
            print("No valid hand keypoints detected")

        return bboxes, is_right, idx_to_confidence
    

    @staticmethod
    def visualize_2d_kpt_on_img(kpts_2d: torch.Tensor, img: np.ndarray) -> np.array:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pts = kpts_2d.cpu().numpy().astype(np.int32)
        nfingers = len(pts) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([pts[0], pts[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors[finger_idx]
                cv2.line(
                    img_bgr,
                    tuple(finger_pts[i]),
                    tuple(finger_pts[i + 1]),
                    color,
                    thickness=5,
                )

        cv2.line(img_bgr, [1787, 1522], [1656,1400], (255,0,0), thickness=5)

        for pt in pts:
            cv2.circle(img_bgr, (pt[0], pt[1]), radius=5, color=(0,0,0), thickness=-1)

        return img_bgr
    

    @staticmethod
    def project_3d_kpt_to_2d(kpts_3d: torch.Tensor, img_w: int, img_h: int,
        T_cam: Optional[torch.Tensor] = None, scaled_focal_length: Optional[torch.Tensor] = None,
        ) -> np.array:
        batch_size = 1

        rotation = torch.eye(3).unsqueeze(0)
        assert T_cam is not None

        T_cam = T_cam.cpu()
        kpts_3d = torch.tensor(kpts_3d).cpu()

        T_cam = T_cam.clone().cuda()
        kpts_3d = kpts_3d.clone().cuda()
        rotation = rotation.cuda()

        scaled_focal_length = torch.tensor([scaled_focal_length, scaled_focal_length]).reshape(1, 2) # 43125

        camera_center = torch.tensor([img_w, img_h], dtype=torch.float).reshape(1, 2) / 2.0 # 1104, 621

        # Tcam tensor([ 0.1076, -0.0526, 15.4407], device='cuda:0')

        pdb.set_trace()
        kpts_2d = perspective_projection(
            kpts_3d.reshape(batch_size, -1, 3),
            rotation=rotation.repeat(batch_size, 1, 1),
            translation=T_cam.reshape(batch_size, -1),
            focal_length=scaled_focal_length.repeat(batch_size, 1),
            camera_center=camera_center.repeat(batch_size, 1),
            ).reshape(batch_size, -1, 2)
        kpts_2d = kpts_2d[0]

        return kpts_2d


    @staticmethod
    def sort_bboxes(bboxes, is_right, idx_to_confidence):
        idx_to_confidence = sorted(idx_to_confidence, key=lambda x: x[1], reverse=True)
        bboxes = np.array([bboxes[idx] for idx, _ in idx_to_confidence])
        is_right = np.array([is_right[idx] for idx, _ in idx_to_confidence])
        return bboxes, is_right
        
    
    @staticmethod
    def get_bbox_from_keypoints(keypoints, thresh):
        valid_idx = keypoints[:, 2] > thresh
        bbox = np.array([keypoints[valid_idx, 0].min(), keypoints[valid_idx, 1].min(), 
                         keypoints[valid_idx, 0].max(), keypoints[valid_idx, 1].max()])
        confidence = sum(keypoints[valid_idx, 2])
        return bbox, confidence


    @staticmethod
    def load_hamer_model(checkpoint_path: str, root_dir: str = None):
        model_cfg = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg, update_cachedir=True)
        # update model and params path
        if root_dir:
            model_cfg.defrost()
            model_cfg.MANO.DATA_DIR = os.path.join(root_dir, model_cfg.MANO.DATA_DIR)
            model_cfg.MANO.MODEL_PATH = os.path.join(root_dir, model_cfg.MANO.MODEL_PATH.replace("./", ""))
            model_cfg.MANO.MEAN_PARAMS = os.path.join(root_dir, model_cfg.MANO.MEAN_PARAMS.replace("./", ""))
            model_cfg.freeze()

        # Override some config values, to crop bbox correctly
        if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and ("BBOX_SHAPE" not in model_cfg.MODEL):
            model_cfg.defrost()
            assert (
                model_cfg.MODEL.IMAGE_SIZE == 256
            ), f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()

        # Update config to be compatible with demo
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()

        model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
        return model, model_cfg
    


if __name__ == "__main__":
    img_path = "data/videos/demo1/video_0_L/00013.jpg"
    img = media.read_image(img_path)
    detector = DetectorHamer()
    detector.detect_hand_keypoints(img, visualize=True)