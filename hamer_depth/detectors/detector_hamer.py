"""
Wrapper around HaMeR for hand keypoint detection.

Adapted from Zi-ang-Cao's code and original HaMeR code.
"""

import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from hamer.configs import get_config
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.models import DEFAULT_CHECKPOINT, HAMER
from hamer.utils import recursive_to
from hamer.utils.geometry import perspective_projection
from hamer.utils.renderer import cam_crop_to_full
from yacs.config import CfgNode as CN

from hamer_depth.detectors.detector_detectron2 import DetectorDetectron2
from hamer_depth.detectors.detector_dino import DetectorDino
from hamer_depth.utils.file_utils import get_parent_folder_of_package
from hamer_depth.utils.hand_type import HandType
from hamer_depth.utils.vitpose_model import ViTPoseModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


THUMB_VERTEX = 744
INDEX_FINGER_VERTEX = 333
MIDDLE_FINGER_VERTEX = 444
RING_FINGER_VERTEX = 555
INDEX_KNUCKLE_VERTEX_BACK = 274
INDEX_KNUCKLE_VERTEX_FRONT = 62
MIDDLE_KNUCKLE_VERTEX_BACK = 220
MIDDLE_KNUCKLE_VERTEX_FRONT = 268
RING_KNUCKLE_VERTEX_BACK = 290
RING_KNUCKLE_VERTEX_FRONT = 275
WRIST_VERTEX_BACK = 279
WRIST_VERTEX_FRONT = 118


def calculate_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
    """
    Calculate the Intersection over Union (IOU) between two bounding boxes.

    Args:
        bbox1: numpy array of shape (4,) with [min_x, min_y, max_x, max_y]
        bbox2: numpy array of shape (4,) with [min_x, min_y, max_x, max_y]

    Returns:
        float: IOU score between 0 and 1
    """
    # Get intersection rectangle coordinates
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    # Check if there is no intersection
    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # Calculate intersection area
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # Calculate areas of both bounding boxes
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    # Calculate union area
    union_area = bbox1_area + bbox2_area - intersection_area

    # Calculate IOU
    iou = intersection_area / union_area

    return iou


class DetectorHamer:
    def __init__(self):
        root_dir = get_parent_folder_of_package("hamer")
        checkpoint_path = Path(root_dir, DEFAULT_CHECKPOINT)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.rescale_factor = 2.0  # Factor for padding the box
        self.batch_size = 1  # Batch size for inference

        self.model, self.model_cfg = self.load_hamer_model(checkpoint_path, root_dir)
        self.model.to(self.device)
        self.model.eval()

        self.cpm = ViTPoseModel(self.device)

        # Load bounding box detectors
        self.dino_detector = DetectorDino("IDEA-Research/grounding-dino-base")
        self.detectron_detector = DetectorDetectron2(root_dir)
        self.faces_right = self.model.mano.faces
        self.faces_left = self.faces_right[:, [0, 2, 1]]

    def detect_hand_keypoints(
        self,
        img: np.ndarray,
        img_mask: np.ndarray,
        visualize: bool = False,
        visualize_3d: bool = False,
        pause_visualization: bool = True,
        hand_type: HandType = HandType.RIGHT,
        camera_params: Optional[dict] = None,
    ) -> Optional[dict]:
        """ "
        Detect the hand keypoints in the image.
        """
        bboxes, is_right, debug_bboxes = self.get_bboxes_for_hamer(
            img, img_mask, hand_type=hand_type
        )
        scaled_focal_length, camera_center = self.get_image_params(
            img=img, camera_params=camera_params
        )

        dataset = ViTDetDataset(
            cfg=self.model_cfg,
            img_cv2=img,
            boxes=bboxes,
            right=is_right,
            rescale_factor=self.rescale_factor,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=False, num_workers=0
        )

        list_2d_kpts, list_3d_kpts, list_verts, list_global_orient, T_cam_pred_all = (
            [],
            [],
            [],
            [],
            [],
        )
        kpts_2d_hamer = None
        for batch in dataloader:
            batch = recursive_to(batch, "cuda")
            with torch.no_grad():
                out = self.model(batch)

            batch_T_cam_pred_all = DetectorHamer.get_all_T_cam_pred(
                batch=batch, out=out, scaled_focal_length=scaled_focal_length
            )
            for idx in range(len(batch_T_cam_pred_all)):
                kpts_3d = (
                    out["pred_keypoints_3d"][idx].detach().cpu().numpy()
                )  # [21, 3]
                verts = out["pred_vertices"][idx].detach().cpu().numpy()  # [778, 3]
                is_right = batch["right"][idx].cpu().numpy()
                global_orient = (
                    out["pred_mano_params"]["global_orient"][idx].detach().cpu().numpy()
                )
                hand_pose = (
                    out["pred_mano_params"]["hand_pose"][idx].detach().cpu().numpy()
                )

                if hand_type == HandType.LEFT:
                    kpts_3d, verts = (
                        DetectorHamer.convert_right_hand_keypoints_to_left_hand(
                            kpts=kpts_3d, verts=verts
                        )
                    )

                T_cam_pred = batch_T_cam_pred_all[idx]

                img_w, img_h = batch["img_size"][idx].float()

                kpts_2d_hamer = DetectorHamer.project_3d_kpt_to_2d(
                    kpts_3d=kpts_3d,
                    img_w=img_w,
                    img_h=img_h,
                    scaled_focal_length=scaled_focal_length,
                    camera_center=camera_center,
                    T_cam=T_cam_pred,
                )

                T_cam_pred = T_cam_pred.cpu().numpy()
                list_2d_kpts.append(kpts_2d_hamer)
                list_3d_kpts.append(kpts_3d + T_cam_pred)
                list_verts.append(verts + T_cam_pred)
                list_global_orient.append(global_orient)

            T_cam_pred_all += batch_T_cam_pred_all

        annotated_img = DetectorHamer.visualize_2d_kpt_on_img(
            kpts_2d=list_2d_kpts[0],
            img=img,
        )
        annotated_img = DetectorHamer.annotate_bboxes_on_img(
            annotated_img, debug_bboxes
        )
        if visualize:
            cv2.imshow("Annotated Image", annotated_img)
            cv2.waitKey(0 if pause_visualization else 1)

        if visualize_3d:
            DetectorHamer.visualize_keypoints_3d(
                annotated_img, list_3d_kpts[0], list_verts[0]
            )

        return {
            "annotated_img": annotated_img,
            "success": len(list_2d_kpts[0]) == 21,
            "kpts_3d": list_3d_kpts[0],
            "kpts_2d": np.rint(list_2d_kpts[0]).astype(np.int32),
            "verts": list_verts[0],
            "T_cam_pred": T_cam_pred_all[0],
            "scaled_focal_length": scaled_focal_length,
            "camera_center": camera_center,
            "img_w": img_w,
            "img_h": img_h,
            "global_orient": list_global_orient[0],
            "hand_pose": hand_pose,
        }

    def get_image_params(
        self, img: np.ndarray, camera_params: Optional[dict]
    ) -> Tuple[float, torch.Tensor]:
        """
        Get the scaled focal length and camera center.
        """
        img_w = img.shape[1]
        img_h = img.shape[0]
        if camera_params is not None:
            scaled_focal_length = camera_params["fx"]
            cx = camera_params["cx"]
            cy = camera_params["cy"]
            camera_center = torch.tensor([img_w - cx, img_h - cy])
        else:
            scaled_focal_length = (
                self.model_cfg.EXTRA.FOCAL_LENGTH
                / self.model_cfg.MODEL.IMAGE_SIZE
                * max(img_w, img_h)
            )
            camera_center = (
                torch.tensor([img_w, img_h], dtype=torch.float).reshape(1, 2) / 2.0
            )
        return scaled_focal_length, camera_center

    def get_bboxes_for_hamer(
        self,
        img: np.ndarray,
        img_mask: np.ndarray,
        hand_type: HandType,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Get bounding boxes of the hands in the image for HaMeR.
        """
        assert hand_type in [HandType.LEFT, HandType.RIGHT], (
            f"Invalid hand type: {hand_type}"
        )
        is_right = np.array([hand_type == HandType.RIGHT])

        # Get dino bounding boxes
        dino_bboxes, _dino_scores, debug_bboxes = self.get_bboxes(
            img, use_dino=True, use_detectron=False
        )  # Turned detectron off cuz bad

        # Get sam bounding boxes
        y_indices, x_indices = np.where(img_mask[:, :, 0])
        min_x = max(x_indices.min() - 5, 0)
        max_x = min(x_indices.max() + 5, img.shape[1] - 1)
        min_y = max(y_indices.min() - 5, 0)
        max_y = min(y_indices.max() + 5, img.shape[0] - 1)
        sam_bboxes = np.array([[min_x, min_y, max_x, max_y]])

        debug_bboxes["sam_bboxes"] = (sam_bboxes, np.array([1.0]))

        if dino_bboxes.size == 0:
            # If no DINO bounding boxes, use SAM
            print("Dino and Detectron failed - using SAM")
            return sam_bboxes, is_right, debug_bboxes

        # Get the dino bounding box that has the highest IOU with the SAM bounding box
        ious = [
            calculate_iou(np.array(bbox), np.array(sam_bboxes[0]))
            for bbox in dino_bboxes
        ]
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        if max_iou < 0.1:
            # No good IOU, use SAM
            return sam_bboxes, is_right, debug_bboxes

        # Good IOU, use DINO bounding box with max IOU with SAM
        return np.array([dino_bboxes[max_iou_idx]]), is_right, debug_bboxes

    def get_bboxes(
        self,
        img: np.ndarray,
        use_dino: bool = True,
        use_detectron: bool = False,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """
        Get bounding boxes around the hands using the Dino or Detectron detectors
        """
        debug_bboxes = {}

        if use_dino:
            dino_bboxes, dino_scores = self.dino_detector.get_bboxes(
                img, "hand", threshold=0.8, visualize=visualize
            )
            debug_bboxes["dino_bboxes"] = (np.array(dino_bboxes), dino_scores)

        if use_detectron:
            det_bboxes, det_scores = self.detectron_detector.get_bboxes(
                img, visualize=visualize
            )
            debug_bboxes["det_bboxes"] = (np.array(det_bboxes), det_scores)

        if (use_dino and len(dino_bboxes) > 0) and (
            use_detectron and len(det_bboxes) > 0
        ):
            bboxes = np.vstack([dino_bboxes, det_bboxes])
            scores = np.concatenate([dino_scores, det_scores])
        elif use_dino and dino_bboxes is not None:
            bboxes, scores = np.array(dino_bboxes), np.array(dino_scores)
        elif use_detectron and det_bboxes is not None:
            bboxes, scores = det_bboxes, det_scores

        if len(scores.shape) == 1:
            scores = scores[:, None]

        return bboxes, scores, debug_bboxes

    def get_human_vitposes(
        self, img: np.ndarray, bboxes: np.ndarray, scores: np.ndarray
    ) -> list:
        """
        Get the human keypoints using the ViTPose model.
        """
        return self.cpm.predict_pose(
            img,
            [np.concatenate([bboxes, scores], axis=1)],
        )

    @staticmethod
    def _filter_bboxes_by_hand(
        bboxes: np.ndarray, is_right: np.ndarray, hand_type: HandType
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Filter the bounding boxes by hand type.
        """
        if hand_type == HandType.LEFT:
            filtered_bboxes = bboxes[~is_right]
            is_right = is_right[is_right == False]
        else:
            filtered_bboxes = bboxes[is_right]
            is_right = is_right[is_right == True]
        return filtered_bboxes, is_right

    @staticmethod
    def _assign_hand_type(bboxes: np.ndarray, hand_type: HandType) -> np.ndarray:
        """
        Assign the hand type to the bounding boxes.
        """
        if hand_type == HandType.LEFT:
            is_right = np.array([False] * len(bboxes))
        else:
            is_right = np.array([True] * len(bboxes))
        return is_right

    @staticmethod
    def evaluate_hand_vitposes(
        vitposes: dict, n_valid_pts_thresh: int = 3
    ) -> Tuple[Optional[list], Optional[list], Optional[list]]:
        """
        Evaluate the hand keypoints predicted by vitposes.
        """
        left_hand_keypoint = vitposes["keypoints"][-42:-21]
        right_hand_keypoint = vitposes["keypoints"][-21:]

        confidence_thresh = 0.5
        n_valid_left_pts = np.sum(left_hand_keypoint[:, 2] > confidence_thresh)
        n_valid_right_pts = np.sum(right_hand_keypoint[:, 2] > confidence_thresh)
        if (n_valid_left_pts < n_valid_pts_thresh) and (
            n_valid_right_pts < n_valid_pts_thresh
        ):
            return None, None, None

        bboxes = []
        is_right: list[bool] = []
        confidences = []
        if n_valid_left_pts > n_valid_pts_thresh:
            bbox, confidence = DetectorHamer.get_bbox_from_keypoints(
                left_hand_keypoint, confidence_thresh
            )
            bboxes.append(bbox)
            is_right.append(False)
            confidences.append(confidence)

        if n_valid_right_pts > n_valid_pts_thresh:
            bbox, confidence = DetectorHamer.get_bbox_from_keypoints(
                right_hand_keypoint, confidence_thresh
            )
            bboxes.append(bbox)
            is_right.append(True)
            confidences.append(confidence)

        return bboxes, is_right, confidences

    @staticmethod
    def get_bboxes_from_vitposes(
        vitposes_out,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get bounding boxes around the hand key points predicted by vitposes.
        """
        bboxes = []
        is_right: list[bool] = []
        idx_to_confidence = []
        idx = 0
        for vitposes in vitposes_out:
            sub_bboxes, sub_is_right, sub_confidences = (
                DetectorHamer.evaluate_hand_vitposes(vitposes)
            )
            if (
                sub_bboxes is not None
                and sub_is_right is not None
                and sub_confidences is not None
            ):
                bboxes.extend(sub_bboxes)
                is_right.extend(sub_is_right)
                for confidence in sub_confidences:
                    idx_to_confidence.append((idx, confidence))
                    idx += 1

        return np.array(bboxes), np.array(is_right), np.array(idx_to_confidence)

    @staticmethod
    def convert_right_hand_keypoints_to_left_hand(kpts, verts):
        kpts[:, 0] = -kpts[:, 0]
        verts[:, 0] = -verts[:, 0]
        return kpts, verts

    @staticmethod
    def visualize_keypoints_3d(
        annotated_img: np.ndarray, kpts_3d: np.ndarray, verts: np.ndarray
    ) -> None:
        nfingers = len(kpts_3d) - 1
        npts_per_finger = 4
        list_fingers = [
            np.vstack([kpts_3d[0], kpts_3d[i : i + npts_per_finger]])
            for i in range(1, nfingers, npts_per_finger)
        ]
        finger_colors_bgr = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
        finger_colors_rgb = [
            (color[2], color[1], color[0]) for color in finger_colors_bgr
        ]
        fig, axs = plt.subplots(1, 2, figsize=(20, 10))
        axs[0] = fig.add_subplot(111, projection="3d")
        for finger_idx, finger_pts in enumerate(list_fingers):
            for i in range(len(finger_pts) - 1):
                color = finger_colors_rgb[finger_idx]
                axs[0].plot(
                    [finger_pts[i][0], finger_pts[i + 1][0]],
                    [finger_pts[i][1], finger_pts[i + 1][1]],
                    [finger_pts[i][2], finger_pts[i + 1][2]],
                    color=np.array(color) / 255.0,
                )
        axs[0].scatter(kpts_3d[:, 0], kpts_3d[:, 1], kpts_3d[:, 2])
        axs[0].scatter(verts[:, 0], verts[:, 1], verts[:, 2])
        annotated_img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        axs[1].imshow(annotated_img_rgb)
        plt.show()

    @staticmethod
    def get_all_T_cam_pred(
        batch: dict, out: dict, scaled_focal_length: float
    ) -> torch.Tensor:
        """
        Get the camera transformation matrix
        """
        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        # NOTE: FOR HaMeR, they are using the img_size as (W, H)
        W_H_shapes = batch["img_size"].float()

        # Get cam_t to full image (instead of bbox)
        T_cam_pred_all = cam_crop_to_full(
            cam_bbox=pred_cam,
            box_center=box_center,
            box_size=box_size,
            img_size=W_H_shapes,
            focal_length=scaled_focal_length,
        )

        return T_cam_pred_all

    @staticmethod
    def visualize_2d_kpt_on_img(kpts_2d: np.ndarray, img: np.ndarray) -> np.ndarray:
        """
        Plot 2D keypoints on the image.
        """
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        pts = kpts_2d.astype(np.int32)
        nfingers = len(pts) - 1
        npts_per_finger = 4
        list_fingers = [
            np.vstack([pts[0], pts[i : i + npts_per_finger]])
            for i in range(1, nfingers, npts_per_finger)
        ]
        finger_colors = [
            (0, 255, 0),
            (0, 0, 255),
            (255, 0, 0),
            (255, 0, 255),
            (0, 255, 255),
        ]
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

        cv2.line(img_bgr, [1787, 1522], [1656, 1400], (255, 0, 0), thickness=5)

        for pt in pts:
            cv2.circle(img_bgr, (pt[0], pt[1]), radius=5, color=(0, 0, 0), thickness=-1)

        return img_bgr

    @staticmethod
    def project_3d_kpt_to_2d(
        kpts_3d: torch.Tensor,
        img_w: int,
        img_h: int,
        scaled_focal_length: float,
        camera_center: torch.Tensor,
        T_cam: Optional[torch.Tensor] = None,
    ) -> np.ndarray:
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

        scaled_focal_length_full = torch.tensor(
            [scaled_focal_length, scaled_focal_length]
        ).reshape(1, 2)

        # IMPORTANT: The perspective_projection function assumes T_cam has not been added to kpts_3d already!
        kpts_2d = perspective_projection(
            kpts_3d.reshape(batch_size, -1, 3),
            rotation=rotation.repeat(batch_size, 1, 1),
            translation=T_cam.reshape(batch_size, -1),
            focal_length=scaled_focal_length_full.repeat(batch_size, 1),
            camera_center=camera_center.repeat(batch_size, 1),
        ).reshape(batch_size, -1, 2)
        kpts_2d = kpts_2d[0].cpu().numpy()

        return np.rint(kpts_2d).astype(np.int32)

    @staticmethod
    def refine_bboxes(vitposes_out) -> Tuple[np.ndarray, np.ndarray]:
        refined_bboxes, is_right, idx_to_confidence = (
            DetectorHamer.get_bboxes_from_vitposes(vitposes_out)
        )
        refined_bboxes, is_right = DetectorHamer.sort_bboxes(
            refined_bboxes, is_right, idx_to_confidence
        )
        return refined_bboxes, is_right

    @staticmethod
    def sort_bboxes(
        bboxes: np.ndarray, is_right: np.ndarray, idx_to_confidence: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Sort the bounding boxes based on confidence.
        """
        idx_to_confidence = np.array(
            sorted(idx_to_confidence, key=lambda x: x[1], reverse=True)
        )
        bboxes = np.array([bboxes[int(idx)] for idx, _ in idx_to_confidence])
        is_right = np.array([is_right[int(idx)] for idx, _ in idx_to_confidence])
        return bboxes, is_right

    @staticmethod
    def get_bbox_from_keypoints(
        keypoints: np.ndarray, thresh: float
    ) -> Tuple[np.ndarray, float]:
        """
        Return the bounding box and confidence of the keypoints.
        """
        valid_idx = keypoints[:, 2] > thresh
        bbox = np.array(
            [
                keypoints[valid_idx, 0].min(),
                keypoints[valid_idx, 1].min(),
                keypoints[valid_idx, 0].max(),
                keypoints[valid_idx, 1].max(),
            ]
        )
        confidence = sum(keypoints[valid_idx, 2])
        return bbox, confidence

    @staticmethod
    def annotate_bboxes_on_img(img: np.ndarray, debug_bboxes: dict) -> np.ndarray:
        """
        Annotate bounding boxes on the image.

        :param img: Input image (numpy array)
        :param debug_bboxes: Dictionary containing different sets of bounding boxes and optional scores
        :return: Annotated image
        """
        color_dict = {
            "dino_bboxes": (0, 255, 0),
            "det_bboxes": (0, 0, 255),
            "sam_bboxes": (255, 0, 0),
            "refined_bboxes": (255, 0, 0),
            "filtered_bboxes": (255, 255, 0),
        }
        corner_dict = {
            "dino_bboxes": "top_left",
            "det_bboxes": "top_right",
            "sam_bboxes": "bottom_left",
            "refined_bboxes": "bottom_left",
            "filtered_bboxes": "bottom_right",
        }

        def draw_bbox_and_label(bbox, label, color, label_pos, include_label=True):
            """Helper function to draw the bounding box and add label"""
            cv2.rectangle(
                img,
                (int(bbox[0]), int(bbox[1])),
                (int(bbox[2]), int(bbox[3])),
                color,
                2,
            )
            if include_label:
                cv2.putText(
                    img,
                    label,
                    label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2,
                    cv2.LINE_AA,
                )

        label_pos_dict = {
            "top_left": lambda bbox: (int(bbox[0]), int(bbox[1]) - 10),
            "bottom_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[3]) - 10),
            "top_right": lambda bbox: (int(bbox[2]) - 150, int(bbox[1]) - 10),
            "bottom_left": lambda bbox: (int(bbox[0]), int(bbox[3]) - 10),
        }

        for key, value in debug_bboxes.items():
            # Unpack bboxes and scores
            if key in ["dino_bboxes", "det_bboxes", "sam_bboxes"]:
                bboxes, scores = value
            else:
                bboxes = value
                scores = [None] * len(bboxes)

            color = color_dict.get(key, (0, 0, 0))
            label_pos_fn = label_pos_dict[corner_dict.get(key, "top_left")]

            # Draw each bounding box and its label
            for idx, bbox in enumerate(bboxes):
                score_text = f" {scores[idx]:.3f}" if scores[idx] is not None else ""
                label = key.split("_")[0] + score_text

                # Draw bounding box and label on the image
                label_pos = label_pos_fn(bbox)
                if key in ["dino_bboxes", "det_bboxes", "sam_bboxes"] or idx == 0:
                    draw_bbox_and_label(bbox, label, color, label_pos)
        return img

    @staticmethod
    def load_hamer_model(
        checkpoint_path: str, root_dir: Optional[str] = None
    ) -> Tuple[HAMER, CN]:
        """
        Load the HaMeR model from the checkpoint path.
        """
        model_cfg_path = str(Path(checkpoint_path).parent.parent / "model_config.yaml")
        model_cfg = get_config(model_cfg_path, update_cachedir=True)
        # update model and params path
        if root_dir:
            model_cfg.defrost()
            model_cfg.MANO.DATA_DIR = os.path.join(root_dir, model_cfg.MANO.DATA_DIR)
            model_cfg.MANO.MODEL_PATH = os.path.join(
                root_dir, model_cfg.MANO.MODEL_PATH.replace("./", "")
            )
            model_cfg.MANO.MEAN_PARAMS = os.path.join(
                root_dir, model_cfg.MANO.MEAN_PARAMS.replace("./", "")
            )
            model_cfg.freeze()

        # Override some config values, to crop bbox correctly
        if (model_cfg.MODEL.BACKBONE.TYPE == "vit") and (
            "BBOX_SHAPE" not in model_cfg.MODEL
        ):
            model_cfg.defrost()
            assert model_cfg.MODEL.IMAGE_SIZE == 256, (
                f"MODEL.IMAGE_SIZE ({model_cfg.MODEL.IMAGE_SIZE}) should be 256 for ViT backbone"
            )
            model_cfg.MODEL.BBOX_SHAPE = [192, 256]
            model_cfg.freeze()

        # Update config to be compatible with demo
        if "PRETRAINED_WEIGHTS" in model_cfg.MODEL.BACKBONE:
            model_cfg.defrost()
            model_cfg.MODEL.BACKBONE.pop("PRETRAINED_WEIGHTS")
            model_cfg.freeze()

        model = HAMER.load_from_checkpoint(checkpoint_path, strict=False, cfg=model_cfg)
        return model, model_cfg
