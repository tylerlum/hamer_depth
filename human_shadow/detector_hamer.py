"""
Wrapper around HaMeR for hand keypoint detection.

Adapted from Zi-ang-Cao's code and original HaMeR code.
"""

import os
import pdb
import json
import numpy as np
from pathlib import Path
from typing import Optional, Tuple

import cv2
import torch
import mediapy as media
from hamer.utils import recursive_to
import matplotlib.pyplot as plt

from hamer.models import HAMER, DEFAULT_CHECKPOINT
from vitpose_model import ViTPoseModel
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full, Renderer
from hamer.utils.geometry import perspective_projection
from hamer.configs import get_config
from yacs.config import CfgNode as CN

from human_shadow.detector_detectron2 import DetectorDetectron2
from human_shadow.detector_dino import DetectorDino
from human_shadow.utils.file_utils import get_parent_folder_of_package

class DetectorHamer:
    def __init__(self):
        root_dir = get_parent_folder_of_package("hamer")
        checkpoint_path = Path(root_dir, DEFAULT_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rescale_factor = 2.0 # Factor for padding the box
        self.batch_size = 1 # Batch size for inference

        self.model, self.model_cfg = self.load_hamer_model(checkpoint_path, root_dir)
        self.model.to(self.device)
        self.model.eval()

        self.cpm = ViTPoseModel(self.device)

        # Load bounding box detectors
        self.dino_detector = DetectorDino("IDEA-Research/grounding-dino-base")
        self.detectron_detector = DetectorDetectron2(root_dir)
        self.faces = self.model.mano.faces
        self.renderer = Renderer(self.model_cfg, faces=self.model.mano.faces)


    def detect_hand_keypoints(self, img: np.ndarray, frame_idx, 
                              visualize: bool=False, visualize_3d: bool=False, visualize_wait: bool=True, path:str=None,
                              camera_params: Optional[dict]=None) -> Tuple[np.ndarray, bool, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Detect the hand keypoints in the image.
        """
        bboxes, is_right = self.get_bboxes_for_hamer(img)
        if bboxes is None:
            return img, False, None, None, None, None, None, None, None, None, None
        if len(bboxes) == 0:
            return img, False, None, None, None, None, None, None, None, None, None
        
        scaled_focal_length, camera_center = self.get_image_params(img, camera_params)

        dataset = ViTDetDataset(self.model_cfg, img, bboxes, is_right, rescale_factor=self.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)


        list_2d_kpts, list_3d_kpts, list_verts = [], [], []
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self.model(batch)

            T_cam_pred_all = DetectorHamer.get_all_T_cam_pred(batch, out, scaled_focal_length)

            for idx in range(len(T_cam_pred_all)):
                kpts_3d = out["pred_keypoints_3d"][idx].detach().cpu().numpy()  # [21, 3]
                verts = out["pred_vertices"][idx].detach().cpu().numpy()  # [778, 3]
                is_right = batch["right"][idx].cpu().numpy()
                global_orient = out["pred_mano_params"]["global_orient"][idx].detach().cpu().numpy()

                T_cam_pred = T_cam_pred_all[idx]

                W_H_shapes = batch["img_size"].float() 
                img_w = W_H_shapes[idx][0]
                img_h = W_H_shapes[idx][1]

                kpts_2d_hamer = DetectorHamer.project_3d_kpt_to_2d(kpts_3d, img_w, img_h, scaled_focal_length, 
                                                            camera_center, T_cam_pred)


                list_2d_kpts.append(kpts_2d_hamer)

                # Need to add the T_cam_pred to the 3D keypoints after projecting to 2D
                kpts_3d += T_cam_pred_all[idx].cpu().numpy()
                list_3d_kpts.append(kpts_3d)

                verts += T_cam_pred_all[idx].cpu().numpy()
                list_verts.append(verts)

        annotated_img = DetectorHamer.visualize_2d_kpt_on_img(
            kpts_2d=list_2d_kpts[0],
            img=img,
        )
        if visualize:
            cv2.imwrite(os.path.join(path, '%05d.png'%frame_idx), annotated_img)
            # cv2.imshow("Annotated Image", annotated_img)
            # if visualize_wait:
            #     cv2.waitKey(0)
            # else:
            #     cv2.waitKey(1)

        if visualize_3d:
            DetectorHamer.visualize_keypoints_3d(annotated_img, list_3d_kpts[0], list_verts[0])

        return annotated_img, True, list_3d_kpts[0], list_2d_kpts[0], list_verts[0], T_cam_pred_all[0], scaled_focal_length, camera_center, W_H_shapes[0][0], W_H_shapes[0][1], global_orient[0]
    

    def get_image_params(self, img: np.ndarray, camera_params: Optional[dict]) -> Tuple[float, torch.Tensor]:
        """
        Get the scaled focal length and camera center.
        """
        img_w = img.shape[1]
        img_h = img.shape[0]
        if camera_params is not None:
            scaled_focal_length = camera_params["fx"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
            camera_center = torch.tensor([img_w-cx, img_h-cy])
        else:
            scaled_focal_length = (self.model_cfg.EXTRA.FOCAL_LENGTH / self.model_cfg.MODEL.IMAGE_SIZE 
                                   * max(img_w, img_h))
            camera_center = torch.tensor([img_w, img_h], dtype=torch.float).reshape(1, 2) / 2.0
        return scaled_focal_length, camera_center

    @staticmethod
    def visualize_keypoints_3d(annotated_img: np.ndarray, kpts_3d: np.ndarray, verts: np.ndarray) -> None:
        nfingers = len(kpts_3d) - 1
        npts_per_finger = 4
        list_fingers = [np.vstack([kpts_3d[0], kpts_3d[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
        finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
        finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]
        fig, axs = plt.subplots(1,2, figsize=(20, 10))
        axs[0] = fig.add_subplot(111, projection='3d')
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors_rgb[finger_idx]
                axs[0].plot(
                    [finger_pts[i][0], finger_pts[i + 1][0]],
                    [finger_pts[i][1], finger_pts[i + 1][1]],
                    [finger_pts[i][2], finger_pts[i + 1][2]],
                    color=np.array(color)/255.0,
                )
        axs[0].scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2])
        axs[0].scatter(verts[:, 0], verts[:, 1], verts[:, 2])
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axs[1].imshow(annotated_img_rgb)
        plt.show()

    @staticmethod
    def get_all_T_cam_pred(batch: dict, out: dict, scaled_focal_length: float) -> torch.Tensor:
        """
        Get the camera transformation matrix
        """
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        # NOTE: FOR HaMeR, they are using the img_size as (W, H)
        W_H_shapes = batch["img_size"].float()  # (2, 2): full image size (2208, 1242)

        multiplier = 2 * batch["right"] - 1

        # Get cam_t to full image (instead of bbox)
        T_cam_pred_all = cam_crop_to_full(
            pred_cam, box_center, box_size, W_H_shapes, scaled_focal_length
        )

        return T_cam_pred_all
    
    def get_bboxes_for_hamer(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding boxes of the hands in the image for HaMeR.
        """
        # Get initial bounding boxes
        bboxes, scores = self.get_bboxes(img)

        if len(bboxes) == 0:
            return np.ndarray([]), np.ndarray([])
 
        # Detect keypoints inside bounding boxes
        vitposes_out = self.get_human_vitposes(img, bboxes, scores)

        # Refine bounding boxes based on keypoints
        bboxes, is_right, idx_to_confidence = self.get_bboxes_from_vitposes(vitposes_out)
        bboxes, is_right = DetectorHamer.sort_bboxes(bboxes, is_right, idx_to_confidence)

        if len(bboxes) == 0:
            return None, None
        bboxes = bboxes[is_right]
        is_right = is_right[is_right == True]
        return bboxes, is_right

    
    def get_bboxes(self, img: np.ndarray, use_dino: bool=True, 
                   use_detectron: bool=True, visualize: bool=False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get bounding boxes around the hands using the Dino or Detectron detectors
        """
        # img = img[81:1161, 144:2064, :]
        if use_dino:
            dino_bboxes, dino_scores = self.dino_detector.get_bboxes(img, "hand", threshold=0.2, visualize=visualize)

        if use_detectron:
            det_bboxes, det_scores = self.detectron_detector.get_bboxes(img, visualize=visualize)

        if (use_dino and dino_bboxes is not None) and (use_detectron and det_bboxes is not None):
            bboxes = np.vstack([dino_bboxes, det_bboxes])
            scores = np.concatenate([dino_scores, det_scores])
        elif use_dino and dino_bboxes is not None:
            bboxes, scores = np.array(dino_bboxes), np.array(dino_scores)
        elif use_detectron and det_bboxes is not None:
            bboxes, scores = det_bboxes, det_scores

        if len(scores.shape) == 1:
            scores = scores[:, None]

        return bboxes, scores
    
    def get_human_vitposes(self, img: np.ndarray, bboxes: np.ndarray, scores: np.ndarray) -> list:
        """
        Get the human keypoints using the ViTPose model.
        """
        return self.cpm.predict_pose(img, [np.concatenate([bboxes,scores], axis=1)],)
    
    
    def evaluate_hand_vitposes(self, vitposes: dict, 
                               n_valid_pts_thresh: int=3) -> Tuple[Optional[list], Optional[list], Optional[list]]:
        """
        Evaluate the hand keypoints predicted by vitposes.
        """
        left_hand_keypoint = vitposes["keypoints"][-42:-21]
        right_hand_keypoint = vitposes["keypoints"][-21:]

        confidence_thresh = 0.5
        n_valid_left_pts = np.sum(left_hand_keypoint[:, 2] > confidence_thresh)
        n_valid_right_pts = np.sum(right_hand_keypoint[:, 2] > confidence_thresh)
        if (n_valid_left_pts < n_valid_pts_thresh) and (n_valid_right_pts < n_valid_pts_thresh):
            return None, None, None
        
        bboxes = []
        is_right: list[bool] = []
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
    
    def get_bboxes_from_vitposes(self, vitposes_out) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get bounding boxes around the hand key points predicted by vitposes.
        """
        bboxes = []
        is_right: list[bool] = []
        idx_to_confidence = []
        idx = 0
        for vitposes in vitposes_out:
            sub_bboxes, sub_is_right, sub_confidences = self.evaluate_hand_vitposes(vitposes)
            if sub_bboxes is not None and sub_is_right is not None and sub_confidences is not None:
                bboxes.extend(sub_bboxes)
                is_right.extend(sub_is_right)
                for confidence in sub_confidences:
                    idx_to_confidence.append((idx, confidence))
                    idx += 1

        return np.array(bboxes), np.array(is_right), np.array(idx_to_confidence)
    

    @staticmethod
    def visualize_2d_kpt_on_img(kpts_2d: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Plot 2D keypoints on the image.
        """
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pts = kpts_2d.astype(np.int32)
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
    def project_3d_kpt_to_2d(kpts_3d: torch.Tensor, img_w: int, img_h: int, scaled_focal_length: float,
                                camera_center: torch.Tensor, T_cam: Optional[torch.Tensor] = None,) -> np.ndarray:
        """
        Project 3D keypoints to 2D using camera parameters.
        """
        batch_size = 1

        rotation = torch.eye(3).unsqueeze(0)
        assert T_cam is not None

        T_cam = T_cam.cpu()
        kpts_3d = torch.tensor(kpts_3d).cpu()

        T_cam = T_cam.clone().cuda()
        kpts_3d = kpts_3d.clone().cuda()
        rotation = rotation.cuda()

        scaled_focal_length_full = torch.tensor([scaled_focal_length, scaled_focal_length]).reshape(1, 2)

        # IMPORTANT: The perspective_projection function assumes T_cam has not been added to kpts_3d already!
        kpts_2d = perspective_projection(
            kpts_3d.reshape(batch_size, -1, 3),
            rotation=rotation.repeat(batch_size, 1, 1),
            translation=T_cam.reshape(batch_size, -1),
            focal_length=scaled_focal_length_full.repeat(batch_size, 1),
            camera_center=camera_center.repeat(batch_size, 1),
            ).reshape(batch_size, -1, 2)
        kpts_2d = kpts_2d[0].cpu().numpy()

        return kpts_2d


    @staticmethod
    def sort_bboxes(bboxes: np.ndarray, is_right: np.ndarray, 
                    idx_to_confidence: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort the bounding boxes based on confidence.
        """
        idx_to_confidence = np.array(sorted(idx_to_confidence, key=lambda x: x[1], reverse=True))
        bboxes = np.array([bboxes[int(idx)] for idx, _ in idx_to_confidence])
        is_right = np.array([is_right[int(idx)] for idx, _ in idx_to_confidence])
        return bboxes, is_right
        
    
    @staticmethod
    def get_bbox_from_keypoints(keypoints: np.ndarray, thresh: float) -> Tuple[np.ndarray, float]:
        """
        Return the bounding box and confidence of the keypoints.
        """
        valid_idx = keypoints[:, 2] > thresh
        bbox = np.array([keypoints[valid_idx, 0].min(), keypoints[valid_idx, 1].min(), 
                         keypoints[valid_idx, 0].max(), keypoints[valid_idx, 1].max()])
        confidence = sum(keypoints[valid_idx, 2])
        return bbox, confidence


    @staticmethod
    def load_hamer_model(checkpoint_path: str, root_dir: Optional[str] = None) -> Tuple[HAMER, CN]:
        """
        Load the HaMeR model from the checkpoint path.
        """
        model_cfg_path = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
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

    # Get camera intrinsics
    # root_folder = get_parent_folder_of_package("human_shadow")
    root_folder = "/juno/u/jyfang/human_shadow"
    camera_intrinsics_path = os.path.join(root_folder, "human_shadow/camera/camera_intrinsics.json")
    with open(camera_intrinsics_path, "r") as f:
        camera_params = json.load(f)

    detector = DetectorHamer()

    indices = np.arange(13, 40)
    for idx in indices:
        img_path = os.path.join(root_folder, f"data/videos/demo1/video_0_L/000{idx}.jpg")
        img = media.read_image(img_path)
        detector.detect_hand_keypoints(img, visualize=True, visualize_3d=False, visualize_wait=False)
        # detector.detect_hand_keypoints(img, camera_params=camera_params["left"], visualize=True) # TOD0: understand why real params don't work




