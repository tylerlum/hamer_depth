import pdb
import json
import os
import numpy as np 
import argparse
import msgpack_numpy as m
m.patch()

import time
import cv2
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm
import mediapy as media
import ctrlutils
import mmt.stereo_inference.python.simple_inference as stereo_to_depth

from human_shadow.detector_dino import DetectorDino
from human_shadow.detector_detectron2 import DetectorDetectron2
from human_shadow.detector_hamer import DetectorHamer
from human_shadow.detector_sam2 import DetectorSam2
from human_shadow.utils.pcd_utils import *
from human_shadow.config.redis_keys import *
from human_shadow.camera.zed_utils import *

def get_transition(vertex_idx_list, mesh, visible_points_3d):
    '''
    Get the initial global transition between hamer prediction and point cloud
    '''
    index = np.arange(0, len(vertex_idx_list))
    verts_indx = vertex_idx_list[index]
    trans = []
    trans_pcd = []
    pred_verts = (mesh.vertices).copy()
    for i in range(len(index)):
        if not np.isnan(np.sum(visible_points_3d[index[i]])):
            trans.append(pred_verts[verts_indx[i]])
            trans_pcd.append(visible_points_3d[index[i]])
    trans_pcd = np.array(trans_pcd)
    trans = np.array(trans)
    transition = trans_pcd - trans
    transition = np.median(transition, axis=0)
    return transition

def get_hand_pose(detector_bbox, detector_hamer, segmentor, image, depth, intrinsics):
    '''
    Get hand pose
    '''
    fx = intrinsics["fx"]
    img_rgb = image.copy()
    img_bgr = img_rgb[..., ::-1]

    # Bounding box
    bbox = detector_bbox.get_best_bbox(img_rgb, "arm", threshold=0.2)

    # Detect hand keypoints
    annotated_img, success, kpts_3d, kpts_2d, verts, T_cam_pred, scaled_focal_length, camera_center, img_w, img_h, global_orient = detector_hamer.detect_hand_keypoints(img_rgb, frame_idx=0,
                                                                                           visualize=False)

    if not success:
        print('Failed!')
        return None, None, None, None, None


    # Segment hand
    masks, scores, img_arr = segmentor.segment_frame(img_rgb, positive_pts=kpts_2d.astype(np.int32), bbox_pts=bbox, visualize=False)

    # Initial transform for pcd
    init_transform = np.eye(4)

    # Get segmented out pcd
    pcd = get_point_cloud_of_segmask(masks[2], depth, img_rgb, intrinsics, visualize=False)
    
    # mesh faces
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

    # Initialize mesh and camera position
    mesh = trimesh.Trimesh(verts.copy(), faces.copy())
    camera_position = np.array([0,0,0])

    # Get visible vertices
    visible_points, visible_vertex_indices = get_visible_points(mesh, camera_position)
    visible_points = visible_points.astype(np.float32)

    # Get pcd of visible vertices of original hamer hand
    visible_pcd = get_pcd_from_points(visible_points, colors=np.ones_like(visible_points) * [0, 1, 0])
    if len(visible_pcd.points) == 0:
       print('Failed!')
       return None, None, None, None, None

    # Project visible vertices to 2d
    visible_points_2d = detector_hamer.project_3d_kpt_to_2d((visible_points-T_cam_pred.cpu().numpy()).astype(np.float32), img_w, img_h, scaled_focal_length, 
                                                        camera_center, T_cam_pred)
    visible_points_2d = np.rint(visible_points_2d).astype(np.int32)
    
    # Get the new visible vertices 3d by lifting 2d visible vertices to 3d using depth
    visible_points_3d = []
    vertex_idx_list = []
    for vertex_idx, pt_2d in enumerate(visible_points_2d):
        try:
            pt_3d = get_3D_point_from_pixel(pt_2d[0], pt_2d[1], depth[pt_2d[1], pt_2d[0]], intrinsics)
        except:
            print("Failed to get 3d point")
            return None, None, None, None, None
        visible_points_3d.append(pt_3d)
        vertex_idx_list.append(visible_vertex_indices[vertex_idx])

    vertex_idx_list = np.array(vertex_idx_list)
    if len(vertex_idx_list) == 0:
        print('Failed!')
        return None, None, None, None, None
    
    # Get the pcd of new visible vertices in 3d
    pcd_visible_points_3d = get_pcd_from_points(visible_points_3d, colors=np.ones_like(visible_points_3d) * [0,1,0])
    # Remove outliers
    pcd_visible_points_3d, outlier_indices = remove_outliers(pcd_visible_points_3d, radius=0.01, min_neighbors=5)

    # Obtain the global transition of the predicted hand using the lifted visible vertices
    transition = get_transition(vertex_idx_list, mesh, visible_points_3d)
    
    # Use the global transition obtaned above as the initial transform for ICP
    if not np.isnan(np.sum(transition)):
        init_transform[:3, 3] = transition.reshape(3,)

    # Do ICP between the pcd of the original hamer hand visible vertices and pcd of the segmented out hand.
    aligned_hamer_pcd, transformation = icp_registration(visible_pcd, pcd, voxel_size=0.005,init_transform=init_transform)
    if transformation is None:
        print('Failed!')
        return None, None, None, None, None

    # Get the pcd of whole hamer predicted vertices
    all_pcd = get_pcd_from_points(mesh.vertices, colors=np.ones_like(mesh.vertices) * [0, 1, 0])
    # Apply the transformation obtained from ICP
    all_pcd = all_pcd.transform(transformation)

    # Get the thumb tip, middle finger tip and control point
    thumb_tip_points = [mesh.vertices[756]]
    middle_tip_points = [mesh.vertices[455]]
    tip_points_control = [(mesh.vertices[756] + mesh.vertices[455])/2]

    # Apply the transformation obtained from ICP
    thumb_tip_points = get_pcd_from_points(thumb_tip_points, colors=np.ones_like(thumb_tip_points) * [1, 0, 0])
    thumb_tip_points = thumb_tip_points.transform(transformation)
    middle_tip_points = get_pcd_from_points(middle_tip_points, colors=np.ones_like(middle_tip_points) * [1, 0, 0])
    middle_tip_points = middle_tip_points.transform(transformation)
    tip_points_control = get_pcd_from_points(tip_points_control, colors=np.ones_like(tip_points_control) * [1, 0, 0])
    tip_points_control = tip_points_control.transform(transformation)

    return np.asarray(tip_points_control.points), np.asarray(thumb_tip_points.points), np.asarray(middle_tip_points.points), annotated_img, img_arr

def main(args):

    # Set up redis server
    _redis = ctrlutils.RedisClient(
        host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
    )
    redis_pipe = _redis.pipeline()

    # Initialize detectors
    detector_id = "IDEA-Research/grounding-dino-base"
    detector_bbox = DetectorDino(detector_id)
    detector_hamer = DetectorHamer()
    segmentor = DetectorSam2()

    print("Starting hand detection")
    while True: 
        redis_pipe.get(KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN)
        redis_pipe.get(KEY_LEFT_CAMERA_INTRINSIC)
        img_left_right_rgba_b, K_left_b = redis_pipe.execute()
        img_left_right_rgba = m.unpackb(img_left_right_rgba_b)
        img_left_rgba = img_left_right_rgba[0]
        img_right_rgba = img_left_right_rgba[1]
        img_left_rgb = img_left_rgba[..., :3].astype(np.uint8)
        img_right_rgb = img_right_rgba[..., :3].astype(np.uint8)
        depth_img_arr = img_left_rgba[..., 3]
        K_left = m.unpackb(K_left_b)
        
        # Detect hand pose 
        tip_points_control, thumb_tip_points, middle_tip_points, hamer_image, sam2_image = get_hand_pose(detector_bbox, detector_hamer, segmentor, img_left_rgb, depth_img_arr, convert_intrinsics_matrix_to_dict(K_left))

        if tip_points_control is not None:
            print("HAND DETECTED")

            hand_points = np.vstack([tip_points_control, thumb_tip_points, middle_tip_points])
            hand_points_b = m.packb(hand_points)

            hand_imgs = np.vstack([hamer_image[None,...], sam2_image[None,...]])
            hand_imgs_b = m.packb(hand_imgs)

            redis_pipe.set(KEY_HAND_EE_POS, hand_points_b)
            redis_pipe.set(KEY_HAMER_IMAGE, hand_imgs_b)
            redis_pipe.execute()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    main(args)

