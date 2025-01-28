import pdb
import json
import os
import numpy as np 

import matplotlib.pyplot as plt
import mediapy as media
import mmt.stereo_inference.python.simple_inference as stereo_to_depth

from human_shadow.detector_dino import DetectorDino
from human_shadow.detector_hamer import DetectorHamer
from human_shadow.detector_sam2 import DetectorSam2
from human_shadow.utils.pcd_utils import *

if __name__ == "__main__":
    depth_mode = "ZED" # "TRI"

    video_folder = "/juno/u/lepertm/human_shadow/human_shadow/data/videos/demo1/"
    video_num = 0

    # Camera intrinsics
    intrinsics_path = "/juno/u/lepertm/human_shadow/human_shadow/camera/camera_intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    fx = intrinsics["left"]["fx"]
    CAM_BASELINE = 0.12  # Meters, TODO: what is this?

    # Load data
    left_video_path = os.path.join(video_folder, f"video_{video_num}_L.mp4")
    left_imgs = np.array(media.read_video(left_video_path))
    right_video_path = os.path.join(video_folder, f"video_{video_num}_R.mp4")
    right_imgs = np.array(media.read_video(right_video_path))
    n_imgs = len(left_imgs)
    zed_depth_imgs = np.load(os.path.join(video_folder, f"depth_imgs_{video_num}.npy"))

    idx = 30
    img_left_rgb = left_imgs[idx]
    img_right_rgb = right_imgs[idx]

    # Get depth
    if depth_mode == "TRI":
        img_left_bgr = img_left_rgb[..., ::-1]
        img_right_bgr = img_right_rgb[..., ::-1]
        depth, _ = stereo_to_depth.get_depth_and_bgr(
                    img_left_bgr.copy(), img_right_bgr.copy(), fx, CAM_BASELINE
                )
    else:
        depth = zed_depth_imgs[idx]

    img_rgb = img_left_rgb.copy()
    img_bgr = img_rgb[..., ::-1]

    # Bounding box
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector_bbox = DetectorDino(detector_id)
    bbox = detector_bbox.get_best_bbox(img_left_rgb, "hand", visualize=False)

    # Segment hand: TODO frame by frame segmentation not reliable as implemented
    segmentor = DetectorSam2()
    masks, scores = segmentor.segment_frame(img_rgb, bbox_pts=bbox, visualize=False)

    # Visualize pcd
    pcd_noisy = get_point_cloud_of_segmask(masks[2], depth, img_rgb, intrinsics["left"], visualize=False) # TODO: which of the three masks do we use?
    pcd = remove_outliers(pcd_noisy, radius=0.01, min_neighbors=200)
    # visualize_pcds([pcd])

    # Detect hand keypoints
    detector_hamer = DetectorHamer()
    annotated_img, success, kpts_3d, kpts_2d, verts = detector_hamer.detect_hand_keypoints(img_rgb, 
                                                                                           visualize=False)
    if success:
        kpts_2d = np.rint(kpts_2d).astype(np.int32)

        # METHOD 1: Lift 2D keypoints to 3D using depth
        kpts_3d_from_2d = []
        for pt_2d in kpts_2d:
            pt_3d = get_3D_point_from_pixel(pt_2d[0], pt_2d[1], depth[pt_2d[1], pt_2d[0]], intrinsics["left"])
            kpts_3d_from_2d.append(pt_3d)

        pcd_kpts_3d_from_2d = get_pcd_from_points(kpts_3d_from_2d, colors=np.ones_like(kpts_3d_from_2d) * [0,1,0])
        # visualize_pcds([pcd, pcd_kpts_3d_from_2d])

        # METHOD 2: Use 3D keypoints from hand detector
        hamer_pcd = get_pcd_from_points(kpts_3d, colors=np.ones_like(kpts_3d) * [1,0,0])
        # visualize_pcds([pcd, hamer_pcd])

        faces = detector_hamer.faces
        faces_new = np.array([[92, 38, 234],
                              [234, 38, 239],
                              [38, 122, 239],
                              [239, 122, 279],
                              [122, 118, 279],
                              [279, 118, 215],
                              [118, 117, 215],
                              [215, 117, 214],
                              [117, 119, 214],
                              [214, 119, 121],
                              [119, 120, 121],
                              [121, 120, 78],
                              [120, 108, 78],
                              [78, 108, 79]])
        faces = np.concatenate([faces, faces_new], axis=0)

        # METHOD 3: Use 3D keypoints from hand detector and align with CPD (Coherent Point Drift) registration
        # TODO: this was working ok on a different frame, but for some reason cpd_registration is squashing the points here
        mesh = trimesh.Trimesh(verts.copy(), faces.copy())
        mesh.unmerge_vertices()
        # Remove the point cloud points that are not visible from the camera to avoid confusing ICP
        camera_position = np.array([0,0,0])
        visible_points = get_visible_points(mesh, camera_position)
        visible_pcd = get_pcd_from_points(visible_points, colors=np.ones_like(visible_points) * [0, 1, 0])
        aligned_pts = cpd_registration(np.array(visible_pcd.points), np.array(pcd.points))
        aligned_hamer_pcd = get_pcd_from_points(aligned_pts, colors=np.ones_like(aligned_pts) * [0, 1, 0])
        visualize_pcds([pcd, aligned_hamer_pcd])
        

        # METHOD 4: Use 3D keypoints from hand detector and align with Open3D ICP
        # faces from HaMeR (utils.renderer.py) that make the hand mesh watertight
        mesh = trimesh.Trimesh(verts.copy(), faces.copy())
        mesh.unmerge_vertices()
        # Remove the point cloud points that are not visible from the camera to avoid confusing ICP
        camera_position = np.array([0,0,0])
        visible_points = get_visible_points(mesh, camera_position)
        visible_pcd = get_pcd_from_points(visible_points, colors=np.ones_like(visible_points) * [0, 1, 0])
        aligned_hamer_pcd, transformation = icp_registration(visible_pcd, pcd, voxel_size=0.005)
        visualize_pcds([pcd, aligned_hamer_pcd])
        

    
