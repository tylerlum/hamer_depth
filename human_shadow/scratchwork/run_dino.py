import argparse
import numpy as np
import torch
import os
import pdb
import cv2
import mediapy as media
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import pipeline
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from sam2.build_sam import build_sam2_video_predictor

from dataclasses import dataclass
from typing import Dict, List, Optional

points = []
current_image = None
original_image = None


@dataclass
class BoundingBox:
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def xyxy(self) -> List[float]:
        return [self.xmin, self.ymin, self.xmax, self.ymax]


@dataclass
class DetectionResult:
    score: float
    label: str
    box: BoundingBox
    mask: Optional[np.array] = None

    @classmethod
    def from_dict(cls, detection_dict: Dict) -> "DetectionResult":
        return cls(
            score=detection_dict["score"],
            label=detection_dict["label"],
            box=BoundingBox(
                xmin=detection_dict["box"]["xmin"],
                ymin=detection_dict["box"]["ymin"],
                xmax=detection_dict["box"]["xmax"],
                ymax=detection_dict["box"]["ymax"],
            ),
        )
    
def get_highest_box(results):
    return max(results, key=lambda r: r.score)


def select_points_dino(frame, object_name, detector):
    img_pil = Image.fromarray(frame)
    labels = [f"{object_name}."]
    results = detector(img_pil, candidate_labels=labels, threshold=0.4)
    results = [DetectionResult.from_dict(result) for result in results]
    if not results:
        return None
    highest_box = get_highest_box(results).box.xyxy
    x_min, y_min, x_max, y_max = highest_box
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    return [(int(center_x), int(center_y))], highest_box


def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    


def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def show_mask2(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    plt.axis('off')
    plt.show()

def run_dino(input_path, output_path, object_name):
    frames = media.read_video(input_path)

    detector_id = "IDEA-Research/grounding-dino-tiny"
    object_detector = pipeline(
        model=detector_id,
        task="zero-shot-object-detection",
        device="cuda",
    )

    all_points = []


    # Run SAM2 in video prediction mode
    original_image = frames[0].copy()
    current_image = original_image.copy()

    # Run dino
    bbox_ctr, bbox_pts = select_points_dino(frames[0], object_name, object_detector)
    if bbox_ctr is None:
        print(
            f"No object detected for viewpoint 0. Skipping..."
        )
    
    checkpoint = "/juno/u/lepertm/segment-anything-2/checkpoints/sam2_hiera_large.pt"
    model_cfg = "sam2_hiera_l.yaml"
    predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda")

    # Process with SAM
    video_dir = "demo/"
    frame_names = os.listdir(video_dir)
    frame_names = sorted(frame_names)
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        state = predictor.init_state(video_path=video_dir)
        predictor.reset_state(state)

        for obj_id, point in enumerate([bbox_ctr]):
            predictor.add_new_points_or_box(
                state,
                frame_idx=0,
                obj_id=obj_id,
                points=np.array([point]),
                labels=np.array([1], np.int32),
            )

        video_segments = {}
        for (
            out_frame_idx,
            out_obj_ids,
            out_mask_logits,
        ) in predictor.propagate_in_video(state):
            video_segments[out_frame_idx] = {
                out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                for i, out_obj_id in enumerate(out_obj_ids)
            }

    vis_frame_stride = 30
    plt.close("all")
    for out_frame_idx in range(0, len(frames), vis_frame_stride):
        plt.figure(figsize=(6, 4))
        plt.title(f"frame {out_frame_idx}")
        plt.imshow(Image.open(os.path.join(video_dir, frame_names[out_frame_idx])))
        for out_obj_id, out_mask in video_segments[out_frame_idx].items():
            show_mask2(out_mask, plt.gca(), obj_id=out_obj_id)


        
    for frame_idx, frame in tqdm(enumerate(frames)):
        original_image = frame.copy()
        current_image = original_image.copy()

        # Run dino
        bbox_ctr, bbox_pts = select_points_dino(frame, object_name, object_detector)
        if bbox_ctr is None:
            print(
                f"No object detected for viewpoint {frame_idx}. Skipping..."
            )
            all_points.append(np.array([]))
            continue

        # Run SAM2
        checkpoint = "/juno/u/lepertm/segment-anything-2/checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda"))

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            predictor.set_image(current_image)
            masks, scores, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox_pts,
                multimask_output=True,
            )

        show_masks(current_image, masks, scores, point_coords=None, box_coords=bbox_pts, input_labels=None)


        # Visualize the selected points
        bgr_image = cv2.cvtColor(current_image, cv2.COLOR_RGB2BGR)
        cv2.circle(bgr_image, bbox_ctr[0], 3, (0, 255, 0), -1)
        cv2.rectangle(
            bgr_image,
            (bbox_pts[0], bbox_pts[1]),
            (bbox_pts[2], bbox_pts[3]),
            (0, 255, 0),
            2,
        )
        cv2.imshow("Frame", bgr_image)
        cv2.waitKey(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive point selection")
    parser.add_argument(
        "-d",
        "--video_path",
        default="demo.MOV",
        help="Path to the input video file",
    )
    parser.add_argument(
        "-o",
        "--object_name",
        type=str,
        default="object",
        help="Name of the object to detect when using DINO-v2",
    )
    args = parser.parse_args()

    # npz_path = args.npz_path
    # if npz_path.endswith("_tracks"):
    #     npz_path = f"{npz_path}/train/first_frames.npz"
    # if "data/" not in npz_path:
    #     npz_path = f"data/{npz_path}"

    # print(f"Parsing {npz_path}")

    # output_path = npz_path.replace(".npz", "_points.npz")
    output_path = None
    run_dino(args.video_path, output_path, args.object_name)