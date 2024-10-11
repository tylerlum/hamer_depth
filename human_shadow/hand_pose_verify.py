import pdb
import json
import os
import numpy as np 

# import matplotlib.pyplot as plt
import mediapy as media
import mmt.stereo_inference.python.simple_inference as stereo_to_depth

from human_shadow.detector_dino import DetectorDino
from human_shadow.detector_detectron2 import DetectorDetectron2
from human_shadow.detector_hamer import DetectorHamer
from human_shadow.detector_sam2 import DetectorSam2
from human_shadow.utils.pcd_utils import *
import time
import cv2
import matplotlib
matplotlib.use('agg')
from tqdm import tqdm

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

def get_hand_pose(detector_bbox, detector_hamer, segmentor, video_folder, video_num):
    '''
    Get hand pose
    The outputs are:
    1. tip_points.npy: The control point
    2. thumb_tip_points.npy: The thumb point
    3. index_tip_points.npy: The index finger point
    4. transformations.npy: The transformation recorded from ICP
    5. hamer_transformations.npy: The global_orient predicted from hamer
    6. start_idx.npy: The start frame
    '''

    # Camera intrinsics
    intrinsics_path = os.path.join(video_folder, f"cam_intrinsics_{video_num}.json") # "/juno/u/jyfang/human_shadow/human_shadow/camera/camera_intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    fx = intrinsics["left"]["fx"]
    intrinsics["left"]["cx"] = intrinsics["left"]["cx"] - 80
    CAM_BASELINE = 0.12  # Meters, TODO: what is this?

    # Load data
    left_video_path = os.path.join(video_folder, f"video_{video_num}_L_preprocessed.mp4")
    left_imgs = np.array(media.read_video(left_video_path))
    n_imgs = len(left_imgs)
    zed_depth_imgs = np.load(os.path.join(video_folder, f"depth_{video_num}.npy"))

    idx = 0
    img_left_rgb = left_imgs[idx]
    depth = zed_depth_imgs[idx][:1000, 80:]

    img_rgb = img_left_rgb.copy()
    img_bgr = img_rgb[..., ::-1]

    bbox = detector_bbox.get_best_bbox(img_rgb, "arm", threshold=0.2)

    # Detect hand keypoints
    annotated_img, success, kpts_3d, kpts_2d, verts, T_cam_pred, scaled_focal_length, camera_center, img_w, img_h, global_orient = detector_hamer.detect_hand_keypoints(img_rgb, frame_idx=idx,
                                                                                           visualize=True, path=os.path.join(video_folder, 'hamer_image'))
    
    # try next frame if not success
    while not success:
        idx += 1
        img_left_rgb = left_imgs[idx]
        depth = zed_depth_imgs[idx][:1000, 80:]

        img_rgb = img_left_rgb.copy()
        img_bgr = img_rgb[..., ::-1]
        bbox = detector_bbox.get_best_bbox(img_rgb, "arm", threshold=0.2)
        annotated_img, success, kpts_3d, kpts_2d, verts, T_cam_pred, scaled_focal_length, camera_center, img_w, img_h, global_orient = detector_hamer.detect_hand_keypoints(img_rgb, frame_idx=idx,
                                                                                           visualize=True, path=os.path.join(video_folder, 'hamer_image'))


    # Segment hand
    masks = segmentor.segment_video(os.path.join(video_folder,f"video_{video_num}_L"), bbox=bbox, bbox_ctr=kpts_2d.astype(np.int32), visualize=True, start_idx=idx, path=os.path.join(video_folder, 'sam2_image'))

    # getting open3d to display the video
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    ctr  = vis.get_view_control()
    view_param = None
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])

    # Initialization
    init_transform = np.eye(4)
    tip_points_list = []
    hamer_transformation_list = []
    transformation_list = []
    thumb_tip_points_list = []
    index_finger_tip_points_list = []

    # Save the start frame
    np.save(os.path.join(video_folder, 'start_idx.npy'), np.array([idx]))

    for n_idx in tqdm(range(idx, n_imgs)):
        img_left_rgb = left_imgs[n_idx]

        depth = zed_depth_imgs[n_idx][:1000, 80:]
       

        img_rgb = img_left_rgb.copy()
        img_bgr = img_rgb[..., ::-1]

        h, w = masks[n_idx][0].shape[-2:]
        mask = masks[n_idx][0].reshape(h,w)
     
        pcd = get_point_cloud_of_segmask(mask, depth, img_rgb, intrinsics["left"], visualize=False) # TODO: which of the three masks do we use?

        if success:
            annotated_img, success, kpts_3d, kpts_2d, verts, T_cam_pred, scaled_focal_length, camera_center, img_w, img_h, global_orient = detector_hamer.detect_hand_keypoints(img_rgb, frame_idx=n_idx,
                                                                                           visualize=True, path=os.path.join(video_folder, 'hamer_image'))
            
            # Return 0 if failed
            if not success:
                tip_points_list.append(np.zeros((1,3)))
                hamer_transformation_list.append(np.zeros((3,3)))
                transformation_list.append(np.zeros((4,4)))
                thumb_tip_points_list.append(np.zeros((1,3)))
                index_finger_tip_points_list.append(np.zeros((1,3)))
                continue
            
            kpts_2d = np.rint(kpts_2d).astype(np.int32)

            # faces = detector_hamer.faces
            # faces_new = np.array([[92, 38, 234],
            #                     [234, 38, 239],
            #                     [38, 122, 239],
            #                     [239, 122, 279],
            #                     [122, 118, 279],
            #                     [279, 118, 215],
            #                     [118, 117, 215],
            #                     [215, 117, 214],
            #                     [117, 119, 214],
            #                     [214, 119, 121],
            #                     [119, 120, 121],
            #                     [121, 120, 78],
            #                     [120, 108, 78],
            #                     [78, 108, 79]])
            # faces = np.concatenate([faces, faces_new], axis=0)

            # Use left face
            faces = detector_hamer.faces_left

            mesh = trimesh.Trimesh(verts.copy(), faces.copy())
            camera_position = np.array([0,0,0])
            
            visible_points, visible_vertex_indices = get_visible_points(mesh, camera_position)

            visible_points = visible_points.astype(np.float32)
            visible_points_2d = detector_hamer.project_3d_kpt_to_2d((visible_points-T_cam_pred.cpu().numpy()).astype(np.float32), img_w, img_h, scaled_focal_length, 
                                                                camera_center, T_cam_pred)
            visible_points_3d = []
            visible_points_2d = np.rint(visible_points_2d).astype(np.int32)
            vertex_idx_list = []
            for vertex_idx, pt_2d in enumerate(visible_points_2d):
                pt_3d = get_3D_point_from_pixel(pt_2d[0], pt_2d[1], depth[pt_2d[1], pt_2d[0]], intrinsics["left"])
                visible_points_3d.append(pt_3d)
                vertex_idx_list.append(visible_vertex_indices[vertex_idx])
            vertex_idx_list = np.array(vertex_idx_list)

            # Return 0 if failed
            if len(vertex_idx_list) == 0:
                tip_points_list.append(np.zeros((1,3)))
                hamer_transformation_list.append(np.zeros((3,3)))
                transformation_list.append(np.zeros((4,4)))
                thumb_tip_points_list.append(np.zeros((1,3)))
                index_finger_tip_points_list.append(np.zeros((1,3)))
                continue

            pcd_visible_points_3d = get_pcd_from_points(visible_points_3d, colors=np.ones_like(visible_points_3d) * [0,1,0])
            pcd_visible_points_3d, outlier_indices = remove_outliers(pcd_visible_points_3d, radius=0.01, min_neighbors=5)

            transition = get_transition(vertex_idx_list, mesh, visible_points_3d)


            visible_pcd = get_pcd_from_points(visible_points, colors=np.ones_like(visible_points) * [0, 1, 0])
            
            # Return 0 if failed
            if len(visible_pcd.points) == 0:
                tip_points_list.append(np.zeros((1,3)))
                hamer_transformation_list.append(np.zeros((3,3)))
                transformation_list.append(np.zeros((4,4)))
                thumb_tip_points_list.append(np.zeros((1,3)))
                index_finger_tip_points_list.append(np.zeros((1,3)))
                continue
            
            if not np.isnan(np.sum(transition)):
                init_transform[:3, 3] = transition.reshape(3,)
            
            # ICP alignment
            aligned_hamer_pcd, transformation = icp_registration(visible_pcd, pcd, voxel_size=0.005,init_transform=init_transform)
            
            # Return 0 if failed
            if transformation is None:
                tip_points_list.append(np.zeros((1,3)))
                hamer_transformation_list.append(np.zeros((3,3)))
                transformation_list.append(np.zeros((4,4)))
                thumb_tip_points_list.append(np.zeros((1,3)))
                index_finger_tip_points_list.append(np.zeros((1,3)))
                continue
            
            all_pcd = get_pcd_from_points(mesh.vertices, colors=np.ones_like(mesh.vertices) * [0, 1, 0])
            all_pcd = all_pcd.transform(transformation)

            thumb_tip_points = []
            index_tip_points = []
            tip_points_middle = []
            # tip_points.append(mesh.vertices[333])
            # 756, 455
            # 745, 444
            thumb_tip_points.append(mesh.vertices[756])
            index_tip_points.append(mesh.vertices[350])
            tip_points_middle.append((mesh.vertices[756] + mesh.vertices[350])/2)

            # Apply ICP transformation to fingertip points
            thumb_tip_points = get_pcd_from_points(thumb_tip_points, colors=np.ones_like(thumb_tip_points) * [1, 0, 0])
            thumb_tip_points = thumb_tip_points.transform(transformation)
            index_tip_points = get_pcd_from_points(index_tip_points, colors=np.ones_like(index_tip_points) * [1, 0, 0])
            index_tip_points = index_tip_points.transform(transformation)
            tip_points_middle = get_pcd_from_points(tip_points_middle, colors=np.ones_like(tip_points_middle) * [1, 0, 0])
            tip_points_middle = tip_points_middle.transform(transformation)

            # print('tip_point', np.asarray(tip_points_middle.points))
            # print('thumb_tip_point', np.asarray(thumb_tip_points.points))
            # print('index_tip_point', np.asarray(index_tip_points.points))
            # print('orient', global_orient)
            # print('transform', np.array(transformation))

            tip_points_list.append(np.asarray(tip_points_middle.points))
            hamer_transformation_list.append(global_orient)
            transformation_list.append(np.array(transformation))
            thumb_tip_points_list.append(np.asarray(thumb_tip_points.points))
            index_finger_tip_points_list.append(np.asarray(index_tip_points.points))
            # visualize_pcds([pcd, all_pcd, index_tip_points, thumb_tip_points, tip_points_middle])

            # add pcd to visualizer
            vis.add_geometry(pcd)
            vis.add_geometry(all_pcd)
            vis.add_geometry(thumb_tip_points)
            vis.add_geometry(index_tip_points)
            vis.add_geometry(tip_points_middle)
           
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.1)
            visualization_image = vis.capture_screen_float_buffer(do_render=True)
            visualization_image = (255.0 * np.asarray(visualization_image)).astype(np.uint8)
            visualization_image = visualization_image[..., ::-1]
            # vis.capture_screen_image(os.path.join(video_folder, 'point_cloud_image', '%05d.png'%n_idx), True)
            cv2.imwrite(os.path.join(video_folder, 'point_cloud_image', '%05d.png'%n_idx), visualization_image)
            vis.remove_geometry(pcd, False)
            vis.remove_geometry(all_pcd, False)
            vis.remove_geometry(thumb_tip_points,False)
            vis.remove_geometry(index_tip_points,False)
            vis.remove_geometry(tip_points_middle, False)

    vis.destroy_window()
    tip_points_list = np.array(tip_points_list)
    hamer_transformation_list = np.array(hamer_transformation_list)
    transformation_list = np.array(transformation_list)
    thumb_tip_points_list = np.array(thumb_tip_points_list)
    index_finger_tip_points_list = np.array(index_finger_tip_points_list)

    # Save everything
    np.save(os.path.join(video_folder, 'tip_points.npy'), tip_points_list)
    np.save(os.path.join(video_folder, 'thumb_tip_points.npy'), thumb_tip_points_list)
    np.save(os.path.join(video_folder, 'index_tip_points.npy'), index_finger_tip_points_list)
    np.save(os.path.join(video_folder, 'transformations.npy'), transformation_list)
    np.save(os.path.join(video_folder, 'hamer_transformations.npy'), hamer_transformation_list)

if __name__ == '__main__':
    # Bounding box
    detector_id = "IDEA-Research/grounding-dino-base"
    detector_bbox = DetectorDino(detector_id)
    detector_hamer = DetectorHamer()
    segmentor = DetectorSam2()

    # Change the dir_path
    dir_path = "/juno/u/jyfang/human_shadow/data/demo_jiaying_waffles_large_2"
    for video_dir in tqdm(sorted(os.listdir(dir_path))):
        video_folder = os.path.join(dir_path, video_dir)
        video_num = int(video_dir)
        get_hand_pose(detector_bbox, detector_hamer, segmentor, video_folder, video_num)
