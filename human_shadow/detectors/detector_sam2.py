"""
Wrapper around SAM2 for object segmentation
"""

import os
from typing import Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.axes import Axes
from PIL import Image
from sam2.build_sam import build_sam2, build_sam2_video_predictor
from sam2.sam2_image_predictor import SAM2ImagePredictor

from human_shadow.detectors.detector_dino import DetectorDino
from human_shadow.detectors.detector_hamer import DetectorHamer
from human_shadow.utils.file_utils import get_parent_folder_of_package


class DetectorSam2:
    def __init__(self):
        checkpoint = "/juno/u/jyfang/human_shadow/segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        self.device = "cuda"
        self.image_predictor = SAM2ImagePredictor(
            build_sam2(model_cfg, checkpoint, device=self.device)
        )
        self.video_predictor = build_sam2_video_predictor(
            model_cfg, checkpoint, device=self.device
        )

    def segment_frame(
        self,
        frame: np.ndarray,
        positive_pts: Optional[np.ndarray] = None,
        negative_pts: Optional[np.ndarray] = None,
        bbox_pts: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        visualize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        img = frame.copy()
        if positive_pts is not None and negative_pts is not None:
            point_coords = np.concatenate([positive_pts, negative_pts], axis=0)
            point_labels = np.concatenate(
                [np.ones(len(positive_pts)), np.zeros(len(negative_pts))], axis=0
            )
        elif positive_pts is not None:
            point_coords = positive_pts
            point_labels = np.ones(len(positive_pts))
        elif negative_pts is not None:
            point_coords = negative_pts
            point_labels = np.zeros(len(negative_pts))
        else:
            point_coords = None
            point_labels = None
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            self.image_predictor.set_image(img)
            masks, scores, _ = self.image_predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                box=bbox_pts,
                multimask_output=multimask_output,
            )
        img_arr = np.array(img)
        mask = masks[2]
        img_arr[mask.astype(bool)] = (0, 0, 0)

        if visualize:
            DetectorSam2.show_masks(
                img, masks, scores, point_coords=point_coords, box_coords=bbox_pts
            )

        return masks, scores, img_arr

    def segment_video(
        self, video_dir: str, bbox, bbox_ctr: np.ndarray, start_idx: int = 0
    ):
        frame_names = os.listdir(video_dir)
        frame_names = sorted(frame_names)
        with torch.inference_mode(), torch.autocast(self.device, dtype=torch.bfloat16):
            state = self.video_predictor.init_state(video_path=video_dir)
            self.video_predictor.reset_state(state)

            for obj_id, point in enumerate([bbox_ctr]):
                self.video_predictor.add_new_points_or_box(
                    state,
                    frame_idx=start_idx,
                    obj_id=obj_id,
                    box=np.array(bbox),
                    # points=np.array([point]),
                    # labels=np.array([1], np.int32),
                    points=np.array(point),
                    labels=np.ones(len(point)),
                )

            video_segments = {}
            mask_prob = {}
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(state):
                mask_prob[out_frame_idx] = torch.mean(torch.sigmoid(out_mask_logits))
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        frame_indices = list(video_segments.keys())
        frame_indices.sort()
        list_annotated_imgs = []
        for out_frame_idx in frame_indices:
            img = Image.open(os.path.join(video_dir, frame_names[out_frame_idx]))
            img_arr = np.array(img)
            mask = video_segments[out_frame_idx][0]
            img_arr[mask[0]] = (0, 0, 0)
            list_annotated_imgs.append(img_arr)

        return video_segments, np.array(list_annotated_imgs)

    @staticmethod
    def show_video_mask(
        mask: np.ndarray,
        ax: Axes,
        obj_id: Optional[int] = None,
        random_color: bool = False,
    ) -> None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            cmap = plt.get_cmap("tab10")
            cmap_idx = 0 if obj_id is None else obj_id
            color = np.array([*cmap(cmap_idx)[:3], 0.6])
        h, w = mask.shape[-2:]
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        ax.imshow(mask_image)
        scatter = None
        if point_coords is not None:
            scatter = ax.scatter(
                point_coords[:, 0],
                point_coords[:, 1],
                color="green",
                marker="o",
                s=5,
                edgecolor="white",
                linewidth=1,
            )
        plt.axis("off")
        # plt.show()
        ax.figure.savefig("outputs/sam_out/" + str(idx) + "_" + str(obj_idx) + ".jpg")
        if scatter is not None:
            scatter.remove()

    @staticmethod
    def show_mask(
        mask: np.ndarray, ax: Axes, random_color: bool = False, borders: bool = True
    ) -> None:
        if random_color:
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        else:
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        h, w = mask.shape[-2:]
        mask = mask.astype(np.uint8)
        mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
        if borders:
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Try to smooth contours
            contours = [
                cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                for contour in contours
            ]
            mask_image = cv2.drawContours(
                mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
            )
        ax.imshow(mask_image)

    @staticmethod
    def show_masks(
        image: np.ndarray,
        masks: np.ndarray,
        scores: np.ndarray,
        point_coords: Optional[np.ndarray] = None,
        box_coords: Optional[np.ndarray] = None,
        input_labels: Optional[np.ndarray] = None,
        borders: bool = True,
    ) -> None:
        n_masks = len(masks)
        fig, axs = plt.subplots(1, n_masks, figsize=(10 * n_masks, 10))
        for i, (mask, score) in enumerate(zip(masks, scores)):
            axs[i].imshow(image)
            DetectorSam2.show_mask(mask, axs[i], borders=borders)
            if point_coords is not None:
                assert input_labels is not None
                DetectorSam2.show_points(point_coords, input_labels, axs[i])
            if box_coords is not None:
                # boxes
                DetectorSam2.show_box(box_coords, axs[i])
            if len(scores) > 1:
                axs[i].set_title(f"Mask {i + 1}, Score: {score:.3f}", fontsize=18)
            axs[i].axis("off")
        plt.show()

    @staticmethod
    def show_box(box: np.ndarray, ax: Axes) -> None:
        x0, y0 = box[0], box[1]
        w, h = box[2] - box[0], box[3] - box[1]
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2
            )
        )

    @staticmethod
    def show_points(
        coords: np.ndarray, labels: np.ndarray, ax: Axes, marker_size: int = 375
    ) -> None:
        pos_points = coords[labels == 1]
        neg_points = coords[labels == 0]
        ax.scatter(
            pos_points[:, 0],
            pos_points[:, 1],
            color="green",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )
        ax.scatter(
            neg_points[:, 0],
            neg_points[:, 1],
            color="red",
            marker="*",
            s=marker_size,
            edgecolor="white",
            linewidth=1.25,
        )


if __name__ == "__main__":
    root_folder = get_parent_folder_of_package("human_shadow")
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector = DetectorDino(detector_id)
    segmentor = DetectorSam2()

    # # Frame by frame
    # indices = np.arange(13, 40)
    # for idx in indices:
    #     video_folder = os.path.join(root_folder, f"human_shadow/data/videos/demo1/video_0_L")
    #     img_path = os.path.join(root_folder, video_folder, f"000{idx}.jpg")
    #     frame_bgr = cv2.imread(img_path)
    #     frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    #     bbox = detector.get_best_bbox(frame, "hand")
    #     bbox_center = np.mean(np.reshape(bbox, (2, 2)), axis=0)
    #     segmentor.segment_frame(frame, bbox_pts=bbox, visualize=True)

    # Entire video at once
    # TODO: why does this fail when start_idx=13??
    idx = 71
    video_folder = os.path.join(root_folder, "outputs/demos/video_4/image")
    img_path = os.path.join(root_folder, video_folder, f"000{idx}.jpg")
    frame_bgr = cv2.imread(img_path)
    frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    bbox = detector.get_best_bbox(frame, "arm")
    if bbox is None:
        bbox = detector.get_best_bbox(frame, "hand")
    if bbox is not None:
        detector_hamer = DetectorHamer()
        (
            annotated_img,
            success,
            kpts_3d,
            kpts_2d,
            verts,
            T_cam_pred,
            scaled_focal_length,
            camera_center,
            img_w,
            img_h,
        ) = detector_hamer.detect_hand_keypoints(frame, visualize=False)

        segmentor.segment_video(
            video_folder, bbox=bbox, bbox_ctr=kpts_2d.astype(np.int32), start_idx=71
        )
