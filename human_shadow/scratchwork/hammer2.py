import os
import pdb
import cv2
import requests
from copy import deepcopy as dcp

import torch
import mediapy as media
import numpy as np
from hamer.utils import recursive_to

from pathlib import Path

from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
from detectron2.config import LazyConfig
import hamer
from hamer.models import HAMER, DEFAULT_CHECKPOINT, load_hamer
from vitpose_model import ViTPoseModel
from hamer.datasets.vitdet_dataset import ViTDetDataset
from hamer.utils.renderer import cam_crop_to_full
import hamer_utils as hamer_vis
from pathlib import Path
from hamer.configs import get_config

from detector_detectron2 import Detectron2Detector
from dino import DinoDetector

def get_parent_folder_of_package(package_name):
    # Import the package
    package = __import__(package_name)

    # Get the absolute path of the imported package
    package_path = os.path.abspath(package.__file__)

    # Get the parent directory of the package directory
    return os.path.dirname(os.path.dirname(package_path))


def filter_boxes(boxes, right):
    # Initialize variables to store the index of the smallest box for right and not-right
    right_index = None
    not_right_index = None
    right_min_area = float("inf")
    not_right_min_area = float("inf")

    for i in range(len(boxes)):
        # Calculate the area of the current box
        box = boxes[i]
        area = (box[2] - box[0]) * (box[3] - box[1])

        # Check if the current box is for right or not-right and update the indices and min areas accordingly
        if right[i]:  # Right hand
            if area < right_min_area:
                right_min_area = area
                right_index = i
        else:  # Not-right hand
            if area < not_right_min_area:
                not_right_min_area = area
                not_right_index = i

    # Filter the boxes and right arrays to include only the boxes with the smallest areas for right and not-right
    filtered_boxes = []
    filtered_right = []
    if right_index is not None:
        filtered_boxes.append(boxes[right_index])
        filtered_right.append(right[right_index])
    if not_right_index is not None:
        filtered_boxes.append(boxes[not_right_index])
        filtered_right.append(right[not_right_index])

    if len(filtered_boxes) < 1:
        import pdb

        pdb.set_trace()
    return np.array(filtered_boxes), np.array(filtered_right)

if __name__ == "__main__":
    # from pathlib import Path
    # from hamer.configs import get_config

    ROOT_DIR = get_parent_folder_of_package(
                "hamer"
            ) 
    checkpoint_path = Path(ROOT_DIR, DEFAULT_CHECKPOINT)

    root_dir = ROOT_DIR

    # model, model_cfg = load_hamer(checkpoint_path)

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

    model = model.to("cuda")
    model.eval()



    # Keypoint detector
    # TODO: need to update the ROOT_DIR
    cpm = ViTPoseModel("cuda")

    rgb_img = media.read_image("demo/00000.jpg")


    # Detect bbox using dino 
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector = DinoDetector(detector_id)
    dino_bboxes, dino_scores = detector.get_bboxes(rgb_img, "hand", visualize=False)


    # Detect bbox using detectron
    detector = Detectron2Detector(root_dir=ROOT_DIR)
    det_bboxes, det_scores = detector.get_bboxes(rgb_img, visualize=False)

    bboxes = np.vstack([dino_bboxes, det_bboxes])
    scores = np.vstack([dino_scores, det_scores])



    # Detect human keypoints for each person
    vitposes_out = cpm.predict_pose(
        rgb_img,
        [np.concatenate([bboxes,scores], axis=1)],
    )


    ### ============= [2] Hand area box ================== ###
    bboxes = []
    is_right = []
    idx_to_confidence = []

    # Use hands based on hand keypoint detections
    idx = 0
    for vitposes in vitposes_out:
        left_hand_keyp = vitposes["keypoints"][-42:-21]
        right_hand_keyp = vitposes["keypoints"][-21:]

        # Rejecting not confident detections
        num_left_valid_pts = sum(left_hand_keyp[:, 2] > 0.5)
        num_right_valid_pts = sum(right_hand_keyp[:, 2] > 0.5)
        num_valid_pts_threshold = 3
        if (
            num_left_valid_pts < num_valid_pts_threshold
            and num_right_valid_pts < num_valid_pts_threshold
        ):
            break
        if num_left_valid_pts > num_valid_pts_threshold:
            keyp = left_hand_keyp
            valid = keyp[:, 2] > 0.5
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(0)
            idx_to_confidence.append((idx, sum(keyp[valid, 2])))
            idx += 1
        if num_right_valid_pts > num_valid_pts_threshold:
            keyp = right_hand_keyp  # [21,3], pixel coords
            valid = keyp[:, 2] > 0.5
            bbox = [
                keyp[valid, 0].min(),
                keyp[valid, 1].min(),
                keyp[valid, 0].max(),
                keyp[valid, 1].max(),
            ]
            bboxes.append(bbox)
            is_right.append(1)
            idx_to_confidence.append((idx, sum(keyp[valid, 2])))
            idx += 1

    if len(bboxes) == 0:
        print("No hands detected")



    # Sort the boxes by confidence
    idx_to_confidence = sorted(idx_to_confidence, key=lambda x: x[1], reverse=True)

    top_idx, top_confidence = idx_to_confidence[0]
    bboxes_top = np.array([bboxes[top_idx]])
    right_top = np.array([is_right[top_idx]])

    boxes = np.stack(bboxes_top)
    right = np.stack(right_top)

    # If there are multiple left or right boxes, filter larger boxes.
    filtered_boxes, filtered_right = filter_boxes(boxes, right)

    pdb.set_trace()





    hamer_cfg = model_cfg
    rescale_factor = 2.0
    batch_size = 2
    # Run reconstruction on all detected hands
    dataset = ViTDetDataset(
        hamer_cfg,
        rgb_img,
        filtered_boxes,
        filtered_right,
        rescale_factor=rescale_factor,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    for batch in dataloader:
        batch = recursive_to(batch, "cuda")


    

    kpts_HaMeR_cam_Dict = {
            "is_right": [],
            "2d": [],
            "3d": [],
            "camera_translation": [],
            "verts": [],
            "confidence": top_confidence,
        }

    ### ============= [3] HaMeR ================== ###
    # TODO: Check if dataloader is None ---  check if there is a way to bypass the issue.
    # Say, change dataloader to a not iterable object!
    for batch in dataloader:
        batch = recursive_to(batch, "cuda")
        with torch.no_grad():
            out = model(batch)

        multiplier = 2 * batch["right"] - 1
        pred_cam = out["pred_cam"]
        pred_cam[:, 1] = multiplier * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        # NOTE: FOR HaMeR, they are using the img_size as (W, H)
        W_H_shapes = batch[
            "img_size"
        ].float()  # (2, 2): full image size (2208, 1242)

        multiplier = 2 * batch["right"] - 1
        scaled_focal_length = (
            model_cfg.EXTRA.FOCAL_LENGTH
            / model_cfg.MODEL.IMAGE_SIZE
            * W_H_shapes.max()
        )

        # Get cam_t to full image (instead of bbox)
        pred_cam_t_full = cam_crop_to_full(
            pred_cam, box_center, box_size, W_H_shapes, scaled_focal_length
        )

        put_result_on_cuda_tensor = False
        verbose = True
        device = "cuda"

        batch_size = pred_cam_t_full.shape[0]
        for n in range(batch_size):
            kpts_3d = out["pred_keypoints_3d"][n].detach().cpu().numpy()  # [21, 3]
            verts = out["pred_vertices"][n].detach().cpu().numpy()  # [778, 3]

            is_right = batch["right"][n].cpu().numpy()
            # NOTE: For the left hand, must flip the x-axis of the kpts_3d_H_world!!!
            kpts_3d[:, 0] = (2 * is_right - 1) * kpts_3d[:, 0]

            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]

            if put_result_on_cuda_tensor:
                kpts_3d = torch.tensor(kpts_3d).to(device)
                verts = torch.tensor(verts).to(device)
                pred_cam_t_full = pred_cam_t_full.to(device)
            else:
                kpts_3d = torch.tensor(kpts_3d).cpu()
                verts = torch.tensor(verts).cpu()
                pred_cam_t_full = pred_cam_t_full.cpu()

            camera_translation = pred_cam_t_full[n].unsqueeze(0)

            img_W = W_H_shapes[n][0]
            img_H = W_H_shapes[n][1]

            # we are hoping to get H, W here!!
            kpts_2d_HaMeR_cam = hamer_vis.project_hamer_3D_to_2D(
                kpts_3d, img_W, img_H, camera_translation, scaled_focal_length
            )
            if verbose:
                print(
                    "finish vision_module.utils.hamer_utils.project_hamer_3D_to_2D"
                )

            kpts_3d_HaMeR_cam = kpts_3d + camera_translation
            verts_HaMeR_cam = verts + camera_translation

            kpts_HaMeR_cam_Dict["is_right"].append(is_right)

            kpts_HaMeR_cam_Dict["2d"].append(kpts_2d_HaMeR_cam)
            kpts_HaMeR_cam_Dict["3d"].append(kpts_3d_HaMeR_cam)
            kpts_HaMeR_cam_Dict["camera_translation"].append(camera_translation)

            kpts_HaMeR_cam_Dict["verts"].append(verts_HaMeR_cam)

        kpts_HaMeR_cam_Dict["pred_mano_params"] = out["pred_mano_params"]



        for kpts_ix, kpts in enumerate(kpts_HaMeR_cam_Dict["2d"]):
            color = tuple([255 if i == kpts_ix else 0 for i in range(3)])
            color = [(255, 0, 0) for _ in range(21)]
            print("Color: ", color, kpts_ix)
            annotated_img = hamer_vis.label_2D_kpt_on_img(
                kpts_2d=kpts_HaMeR_cam_Dict["2d"][kpts_ix],
                img_cv2=rgb_img,
                color=color,
            )
        print("annotated_img.shape: ", annotated_img.shape)
        # save the annotated_img to a file
        cv2.imshow("annotated_img", annotated_img[..., ::-1])
        cv2.waitKey(0)
        # parent_dir = os.path.join(os.getcwd(), "hamer_vis")
        # os.makedirs(parent_dir, exist_ok=True)
        # cv2.imwrite(
        #     f"{parent_dir}/labeled_img_test.jpg", annotated_img[..., ::-1].copy()
        # )


    pdb.set_trace()