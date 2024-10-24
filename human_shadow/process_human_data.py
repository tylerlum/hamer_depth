import pdb
import os
import logging
import glob
from typing import Tuple
from tqdm import tqdm
import numpy as np 
import pandas as pd
import argparse

import mediapy as media

from human_shadow.detectors.detector_dino import DetectorDino
from human_shadow.detectors.detector_detectron2 import DetectorDetectron2
from human_shadow.detectors.detector_hamer import DetectorHamer, THUMB_VERTEX, INDEX_FINGER_VERTEX
from human_shadow.detectors.detector_sam2 import DetectorSam2
from human_shadow.utils.pcd_utils import *
from human_shadow.utils.transform_utils import *
from human_shadow.utils.video_utils import *
from human_shadow.camera.zed_utils import *
from human_shadow.utils.file_utils import get_parent_folder_of_package

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

def is_hand_in_image(img: np.ndarray, hamer_output: dict) -> bool:
    """ 
    Return True if a hand is detected in the image by HaMeR
    """
    return hamer_output is not None and hamer_output["success"]


def get_first_frame_with_hand(imgs_rgb: np.ndarray, detector: DetectorHamer) -> Tuple[int, dict]:
    """
    Return the index of the first frame with a hand detected by HaMeR.
    """
    for idx, img in enumerate(imgs_rgb):
        try:
            hamer_output = detector.detect_hand_keypoints(img)
        except ValueError as e:
            logger.debug(f"{e}")
            continue
        if is_hand_in_image(img, hamer_output):
            return idx, hamer_output
    return -1, {}

def get_visible_pts_from_hamer(detector_hamer: DetectorHamer, hamer_out: dict, mesh: trimesh.Trimesh,
                               img_depth: np.ndarray, cam_intrinsics: dict) -> Tuple[np.ndarray, np.ndarray]:
    """
    Identify points in the depth image that belong to hamer vertices that are visible from the camera viewpoint
    """
    visible_hamer_vertices, _ = get_visible_points(mesh, origin=np.array([0,0,0]))
    visible_points_2d = detector_hamer.project_3d_kpt_to_2d(
        (visible_hamer_vertices-hamer_out["T_cam_pred"].cpu().numpy()).astype(np.float32), 
        hamer_out["img_w"], hamer_out["img_h"], hamer_out["scaled_focal_length"], 
        hamer_out["camera_center"], hamer_out["T_cam_pred"])
    visible_points_3d = get_3D_points_from_pixels(visible_points_2d, img_depth, cam_intrinsics)
    return visible_points_3d, visible_hamer_vertices


def get_initial_transformation_estimate(visible_points_3d: np.ndarray, 
                                        visible_hamer_vertices: np.ndarray) -> np.ndarray:
    """
    Get estimate of transformation from HaMeR's predicted 3d point cloud of the hand (using only the visible points) to the point cloud of the arm obtained from the depth image. Assume orientation is the same (only translation is different)
    """
    translation = np.nanmedian(visible_points_3d - visible_hamer_vertices, axis=0)
    T_0 = np.eye(4)
    if not np.isnan(translation).any():
        T_0[:3, 3] = translation
    return T_0

def get_transformation_estimate(visible_points_3d: np.ndarray, 
                                visible_hamer_vertices: np.ndarray, 
                                pcd: o3d.geometry.PointCloud) -> Tuple[np.ndarray, o3d.geometry.PointCloud]:
    """
    Align the hamer point cloud (with only visible points) with the full arm point cloud using initial translation prediction
    """
    T_0 = get_initial_transformation_estimate(visible_points_3d, visible_hamer_vertices)
    visible_hamer_pcd = get_pcd_from_points(visible_hamer_vertices, colors=np.ones_like(visible_hamer_vertices) * [0, 1, 0])
    try: 
        aligned_hamer_pcd, T = icp_registration(visible_hamer_pcd, pcd, voxel_size=0.005, init_transform=T_0)
    except:
        return T_0, visible_hamer_pcd
    return T, aligned_hamer_pcd


def get_finger_pose(mesh: trimesh.Trimesh, T: np.ndarray) -> Tuple[dict, o3d.geometry.PointCloud]:
    """
    Get the 3D locations of the thumb, index finger, and hand end effector points in the world frame.
    """
    thumb_pt = mesh.vertices[THUMB_VERTEX]
    index_pt = mesh.vertices[INDEX_FINGER_VERTEX]
    hand_ee_pt = np.mean([thumb_pt, index_pt], axis=0)
    finger_pts = np.vstack([thumb_pt, index_pt, hand_ee_pt])
    finger_pts = transform_pts(finger_pts, T)
    finger_pcd = get_pcd_from_points(finger_pts, colors=np.ones_like(finger_pts) * [1, 0, 0])
    return {"thumb": finger_pts[0], "index": finger_pts[1], "hand_ee": finger_pts[2]}, finger_pcd


def process_image_with_hamer(img_rgb: np.ndarray, img_depth:np.ndarray, mask: np.ndarray, cam_intrinsics: dict,
                              detector_hamer: DetectorHamer, vis) -> Tuple[dict, np.ndarray, np.ndarray]:
    pcd = get_point_cloud_of_segmask(mask, img_depth, img_rgb, cam_intrinsics, visualize=False) 
    
    # Detect hand
    hamer_out = detector_hamer.detect_hand_keypoints(img_rgb)

    if hamer_out is None or not hamer_out.get("success", False):  
        raise ValueError("No hand detected in image")

    mesh = trimesh.Trimesh(hamer_out["verts"].copy(), detector_hamer.faces_left.copy())
    visible_points_3d, visible_hamer_vertices = get_visible_pts_from_hamer(detector_hamer, hamer_out, mesh, img_depth, cam_intrinsics)
    T, aligned_hamer_pcd = get_transformation_estimate(visible_points_3d, visible_hamer_vertices, pcd)
    finger_pts, finger_pcd = get_finger_pose(mesh, T)

    # Display with open3d
    pcd_img = None
    if vis is not None:
        vis.add_geometry(pcd)
        vis.add_geometry(aligned_hamer_pcd)
        vis.add_geometry(finger_pcd)
        vis.poll_events()
        vis.update_renderer()
        pcd_img = vis.capture_screen_float_buffer(do_render=True)
        pcd_img = (255.0 * np.asarray(pcd_img)).astype(np.uint8)

        vis.remove_geometry(pcd)
        vis.remove_geometry(aligned_hamer_pcd)
        vis.remove_geometry(finger_pcd)

    return finger_pts, pcd_img, hamer_out["annotated_img"][:,:,::-1]

def get_paths(video_folder: str, video_idx: int) -> Tuple[str, str, str]:
    video_path = os.path.join(video_folder, f"video_{video_idx}_L.mp4")
    depth_path = os.path.join(video_folder, f"depth_{video_idx}.npy")
    cam_intrinsics_path = os.path.join(video_folder, f"cam_intrinsics_{video_idx}.json")
    return video_path, depth_path, cam_intrinsics_path


def main(args):
    root_folder = get_parent_folder_of_package("human_shadow")

    detector_hamer = DetectorHamer()
    detector_id = "IDEA-Research/grounding-dino-base"
    detector_bbox = DetectorDino(detector_id)
    detector_detectron = DetectorDetectron2(root_dir=root_folder)
    detector_sam = DetectorSam2()

    if args.use_shared:
        videos_folder = os.path.join("/juno/group/human_shadow/raw_data/", args.demo_name)
        save_videos_folder = os.path.join("/juno/group/human_shadow/processed_data/", args.demo_name)
        if not os.path.exists(save_videos_folder):
            os.makedirs(save_videos_folder)
    else:
        videos_folder = os.path.join(root_folder, f"human_shadow/data/videos/{args.demo_name}/")
        save_videos_folder = os.path.join(root_folder, f"human_shadow/data/videos/processed/{args.demo_name}/")
        if not os.path.exists(save_videos_folder):
            os.makedirs(save_videos_folder)

    # Get all folders in videos_folder
    all_video_folders = [f for f in os.listdir(videos_folder) if os.path.isdir(os.path.join(videos_folder, f))]
    all_video_folders = sorted(all_video_folders, key=lambda x: int(x))
    # all_video_folders = all_video_folders[27:]

    for video_sub_folder in tqdm(all_video_folders):
        video_folder = os.path.join(videos_folder, video_sub_folder)
        video_idx = int(video_sub_folder)
        video_path, depth_path, cam_intrinsics_path = get_paths(video_folder, video_idx)

        # Save processed data in video folder
        save_folder = os.path.join(save_videos_folder, video_sub_folder)
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        os.system(f"cp -r {video_folder}/* {save_folder}/")

        # Convert video to images if they don't exist
        images_folder = os.path.join(video_folder, "images")
        if not os.path.exists(images_folder):
            convert_video_to_images(video_path, images_folder)
        image_paths = glob.glob(os.path.join(images_folder, "*.jpg"))
        image_paths = sorted(image_paths, key=lambda x: int(os.path.basename(x).split(".")[0]))

        # Get intrinsics
        camera_matrix = get_intrinsics_from_json(cam_intrinsics_path)
        cam_intrinsics = convert_intrinsics_matrix_to_dict(camera_matrix)
            
        # Get data
        imgs_rgb = media.read_video(video_path)
        imgs_depth = np.load(depth_path)

        # Adjust for size of images
        img_w, img_h = imgs_rgb[0].shape[:2]
        depth_w, depth_h = imgs_depth[0].shape[:2]
        assert img_w == depth_w and img_h == depth_h
        zed_resolution = ZED_RESOLUTIONS_SQUARE_SIZE[img_w]
        if img_h < zed_resolution.value[1]:
            offset = (zed_resolution.value[1] - img_h) // 2
            cam_intrinsics["cx"] -= offset
        elif img_h > zed_resolution.value[1]:
            raise ValueError("Image height is greater than ZED resolution")

        # Get first frame with hand
        start_idx, hamer_output = get_first_frame_with_hand(imgs_rgb, detector_hamer)

        if start_idx == -1:
            logger.warning("No hand detected in video")
            return

        # Get bbox of arm
        bbox, score = detector_detectron.get_best_bbox(imgs_rgb[start_idx], visualize=False)

        # Segment out arm in video
        kpts_2d = hamer_output["kpts_2d"]
        masks, sam_imgs = detector_sam.segment_video(images_folder, bbox=bbox, bbox_ctr=kpts_2d.astype(np.int32), start_idx=start_idx)
        list_human_masks = [(masks[idx][0][0].astype(np.uint8) * 255) 
                            for idx in tqdm(range(len(sam_imgs)), leave=False)]

        # Initialize open3d renderer
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False)
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0.2, 0.2, 0.2])
        
        # Process images
        df = pd.DataFrame(columns=["idx", "thumb", "index", "hand_ee"])
        list_annot_imgs, list_pcd_imgs, list_hamer_imgs = [], [], []
        for img_idx in tqdm(range(len(image_paths)), leave=False):
            img_rgb = imgs_rgb[img_idx]
            img_depth = imgs_depth[img_idx]
            img_sam = sam_imgs[img_idx]

            if img_idx < start_idx:
                df.loc[img_idx] = [img_idx, np.zeros(3), np.zeros(3), np.zeros(3)]
                img_pcd = np.zeros_like(img_rgb)
                img_hamer = np.zeros_like(img_rgb)
                img_annot = np.vstack((np.hstack((img_rgb, img_sam)), np.hstack((img_hamer, img_pcd))))
                list_annot_imgs.append(img_annot)
                list_pcd_imgs.append(img_pcd)
                list_hamer_imgs.append(img_hamer)
                continue

            h, w = masks[img_idx][0].shape[-2:]
            mask = masks[img_idx][0].reshape(h,w)

            try:
                finger_pts, img_pcd, img_hamer = process_image_with_hamer(img_rgb, img_depth, mask, cam_intrinsics, detector_hamer, vis)
                df.loc[img_idx] = [img_idx, finger_pts["thumb"], finger_pts["index"], finger_pts["hand_ee"]]
                if img_pcd.shape[1] > img_rgb.shape[1]:
                    offset = (img_pcd.shape[1] - img_rgb.shape[1]) // 2
                    img_pcd = img_pcd[:, offset:-offset]
            except ValueError as e:
                logger.error(f"Error: {e}")
                img_pcd = np.zeros_like(img_rgb)
                img_hamer = np.zeros_like(img_rgb)
                df.loc[img_idx] = [img_idx, np.zeros(3), np.zeros(3), np.zeros(3)]

            try:
                img_annot = np.vstack((np.hstack((img_rgb, img_sam)), np.hstack((img_hamer, img_pcd))))
            except:
                pdb.set_trace()
            list_pcd_imgs.append(img_pcd)
            list_hamer_imgs.append(img_hamer)
            list_annot_imgs.append(img_annot)

        vis.destroy_window()

        # Save data
        df.to_csv(os.path.join(save_folder, "finger_poses.csv"))
        media.write_video(os.path.join(save_folder, "imgs_pcd.mkv"), np.array(list_pcd_imgs), fps=10, codec="ffv1")
        media.write_video(os.path.join(save_folder, "imgs_hamer.mkv"), np.array(list_hamer_imgs), fps=10, codec="ffv1")
        media.write_video(os.path.join(save_folder, "imgs_sam.mkv"), np.array(sam_imgs), fps=10, codec="ffv1")
        media.write_video(os.path.join(save_folder, "imgs_hand_masks.mkv"), list_human_masks, 
                        codec="ffv1", input_format="gray")
        media.write_video(os.path.join(save_folder, f"imgs_annot.mkv"), np.array(list_annot_imgs), fps=10, codec="ffv1")
            
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo_name", type=str)
    parser.add_argument("--use_shared", action="store_true")
    args = parser.parse_args()
    main(args)

