import pdb 
import os
import cv2
from tqdm import tqdm
import json
import time
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import mediapy as media
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import mmt.stereo_inference.python.simple_inference as stereo_to_depth

from detector_dino import DetectorDino
from detector_sam2 import DetectorSam2
from detector_hamer import DetectorHamer
from human_shadow.utils.pcd_utils import *

# from pycpd import RigidRegistration

# def cpd_registration(source_points, target_points):
#     # Initialize CPD registration
#     reg = RigidRegistration(X=target_points, Y=source_points)
    
#     # Perform registration
#     transformed_points, _ = reg.register()
    
#     return transformed_points


# def preprocess_point_cloud(pcd, voxel_size):
#     pcd_down = pcd.voxel_down_sample(voxel_size)
#     radius_normal = voxel_size * 2
#     pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
#     radius_feature = voxel_size * 5
#     pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
#         pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
#     return pcd_down, pcd_fpfh

# def global_registration(source_pcd, target_pcd, voxel_size):
#     source_down, source_fpfh = preprocess_point_cloud(source_pcd, voxel_size)
#     target_down, target_fpfh = preprocess_point_cloud(target_pcd, voxel_size)
    
#     distance_threshold = voxel_size * 1.5
#     result_ransac = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
#         source_down, target_down, source_fpfh, target_fpfh, True,
#         distance_threshold,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
#         4,  # RANSAC iterations
#         [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
#          o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)],
#         o3d.pipelines.registration.RANSACConvergenceCriteria(4000000, 500))
    
#     return result_ransac

# def align_point_clouds_with_icp(source_points, target_points, voxel_size=0.05):
#     # Convert numpy arrays to Open3D point clouds
#     source_pcd = o3d.geometry.PointCloud()
#     target_pcd = o3d.geometry.PointCloud()
#     source_pcd.points = o3d.utility.Vector3dVector(source_points)
#     target_pcd.points = o3d.utility.Vector3dVector(target_points)
    
#     # Global registration using RANSAC
#     result_ransac = global_registration(source_pcd, target_pcd, voxel_size)
    
#     # Refine alignment using ICP
#     max_correspondence_distance = voxel_size * 5
#     result_icp = o3d.pipelines.registration.registration_icp(
#         source_pcd, target_pcd, max_correspondence_distance, result_ransac.transformation,
#         o3d.pipelines.registration.TransformationEstimationPointToPoint())
    
#     # Get the aligned source point cloud
#     aligned_source_pcd = source_pcd.transform(result_icp.transformation)
    
#     return np.asarray(aligned_source_pcd.points), result_icp.transformation




# # Function to align two point clouds using ICP
# def align_point_clouds(source_points, target_points, max_correspondence_distance=0.02, max_iterations=50):
#     # Convert numpy arrays to Open3D point clouds
#     source_pcd = o3d.geometry.PointCloud()
#     target_pcd = o3d.geometry.PointCloud()
#     source_pcd.points = o3d.utility.Vector3dVector(source_points)
#     target_pcd.points = o3d.utility.Vector3dVector(target_points)

#     # Perform ICP alignment
#     icp_result = o3d.pipelines.registration.registration_icp(
#         source_pcd, target_pcd, max_correspondence_distance,
#         np.eye(4),  # Initial transformation
#         o3d.pipelines.registration.TransformationEstimationPointToPoint(),
#         o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
#     )

#     # Get the aligned source point cloud
#     aligned_source_pcd = source_pcd.transform(icp_result.transformation)

#     return np.asarray(aligned_source_pcd.points), icp_result.transformation

# def radius_outlier_detection(points, radius=5, min_neighbors=5):
#     # Fit the NearestNeighbors model
#     nbrs = NearestNeighbors(radius=radius).fit(points)
    
#     # Get the number of neighbors for each point within the specified radius
#     distances, indices = nbrs.radius_neighbors(points)
    
#     # Detect points with fewer neighbors than the minimum threshold
#     outliers_mask = np.array([len(neigh) < min_neighbors for neigh in indices])

#     outlier_pts = points[outliers_mask]
    
#     return outliers_mask, outlier_pts

# def get_point_from_pixel(px, py, depth, intrinsics):
#     x = (px - intrinsics["cx"]) / intrinsics["fx"]
#     y = (py - intrinsics["cy"]) / intrinsics["fy"]

#     X = x * depth
#     Y = y * depth
#     if len(X.shape) == 0:
#         p = np.array([X, Y, depth])
#     else:
#         p = np.stack((X, Y, depth), axis=1)

#     return p


def plot_kpts(ax, kpts, color):
    nfingers = len(kpts) - 1
    npts_per_finger = 4
    list_fingers = [np.vstack([kpts[0], kpts[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
    # finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    # finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]

    for finger_idx, finger_pts in enumerate(list_fingers):
        for i in range(len(finger_pts) - 1):
            # color = finger_colors_rgb[finger_idx]
            ax.plot(
                [finger_pts[i][0], finger_pts[i + 1][0]],
                [finger_pts[i][1], finger_pts[i + 1][1]],
                [finger_pts[i][2], finger_pts[i + 1][2]],
                color=np.array(color)/255.0,
            )
    ax.scatter(kpts[:, 0], kpts[:, 1], kpts[:, 2])

# def segment_pcd(mask, depth_img, img, intrinsics, visualize=False):
#     idxs_y, idxs_x = mask.nonzero()
#     depth_masked = depth_img[idxs_y, idxs_x]
#     seg_points = get_point_from_pixel(idxs_x, idxs_y, depth_masked, intrinsics)
#     seg_colors = img_left_rgb[idxs_y, idxs_x, :] / 255.0  # Normalize to [0,1] for cv2

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(seg_points)
#     pcd.colors = o3d.utility.Vector3dVector(seg_colors)
#     pcd.remove_non_finite_points()

#     if visualize:
#         visualize_pcd(pcd)

#     return pcd

if __name__ == "__main__":

    video_folder = "/juno/u/lepertm/human_shadow/human_shadow/data/videos/demo2/"
    video_num = 0

    # Camera intrinsics
    intrinsics_path = "/juno/u/lepertm/human_shadow/human_shadow/camera/camera_intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    fx = intrinsics["left"]["fx"]
    CAM_BASELINE = 0.12  # Meters

    left_video_path = os.path.join(video_folder, f"video_{video_num}_L.mp4")
    left_imgs = np.array(media.read_video(left_video_path))
    right_video_path = os.path.join(video_folder, f"video_{video_num}_R.mp4")
    right_imgs = np.array(media.read_video(right_video_path))
    n_imgs = len(left_imgs)

    idx = 20

    img_left_rgb = left_imgs[idx]
    img_right_rgb = right_imgs[idx]

    # fig, axs = plt.subplots(1, 2)
    # axs[0].imshow(img_left_rgb)
    # axs[1].imshow(img_right_rgb)
    # plt.show()

    # Depth 
    depth_imgs = np.load(os.path.join(video_folder, f"depth_imgs_{video_num}.npy"))
    img_left_bgr = img_left_rgb[..., ::-1]
    img_right_bgr = img_right_rgb[..., ::-1]
    depth_tri, _ = stereo_to_depth.get_depth_and_bgr(
                img_left_bgr.copy(), img_right_bgr.copy(), fx, CAM_BASELINE
            )
    depth_zed = depth_imgs[idx]


    # Bounding box
    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector_bbox = DetectorDino(detector_id)
    bbox = detector_bbox.get_best_bbox(img_left_rgb, "hand")


    detector_hamer = DetectorHamer()
    annotated_img, _, kpts_3d, kpts_2d, verts = detector_hamer.detect_hand_keypoints(img_left_rgb, visualize=False)
    kpts_2d = kpts_2d.cpu().numpy()
    kpts_2d = np.rint(kpts_2d).astype(np.int32)

    pcd_verts = get_pcd_from_points(verts, colors=np.ones_like(verts) * [0, 0, 1])

    # pcd_verts = o3d.geometry.PointCloud()
    # pcd_verts.points = o3d.utility.Vector3dVector(np.array(verts))
    # pcd_verts.colors = o3d.utility.Vector3dVector(np.ones_like(verts) * [0, 0, 1])
    # pcd_verts.remove_non_finite_points()

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(annotated_img)
    # plt.show()

    # Detect pcd
    segmentor = DetectorSam2()
    masks, scores = segmentor.segment_frame(img_left_rgb, bbox_pts=bbox)



    # seg_points, seg_colors = segment_pcd(masks[0], depth_imgs[idx], img_left_rgb, intrinsics["left"], visualize=True)

    img_left_bgr = img_left_rgb[..., ::-1]
    img_right_bgr = img_right_rgb[..., ::-1]
    depth_tri, img_tri_bgr = stereo_to_depth.get_depth_and_bgr(
        img_left_bgr.copy(), img_right_bgr.copy(), fx, CAM_BASELINE
    )
    pcd = get_point_cloud_of_segmask(masks[0], depth_tri, img_tri_bgr, intrinsics["left"], visualize=False)
    # pcd = segment_pcd(masks[0], depth_zed, img_left_rgb, intrinsics["left"], visualize=False)

    outlier_indices, outlier_pts = radius_outlier_detection(np.asarray(pcd.points), radius=0.01, min_neighbors=200)
    outlier_colors = np.ones_like(outlier_pts) * [1, 0, 0]

    pcd_outliers = get_pcd_from_points(outlier_pts, colors=outlier_colors)

    # pcd_outliers = o3d.geometry.PointCloud()
    # pcd_outliers.points = o3d.utility.Vector3dVector(outlier_pts)
    # pcd_outliers.colors = o3d.utility.Vector3dVector(outlier_colors)

    filtered_pts = np.asarray(pcd.points)[~outlier_indices]
    filtered_colors = np.asarray(pcd.colors)[~outlier_indices]
    pcd.points = o3d.utility.Vector3dVector(filtered_pts)
    pcd.colors = o3d.utility.Vector3dVector(filtered_colors)

    # voxel_size = 0.005
    # verts_down, source_fpfh = preprocess_point_cloud(pcd_verts, voxel_size)
    # pcd_down, target_fpfh = preprocess_point_cloud(pcd, voxel_size)


    # vis = o3d.visualization.Visualizer()
    # vis.create_window()
    # opt = vis.get_render_option()
    # opt.background_color = np.asarray([0.2, 0.2, 0.2])
    # vis.add_geometry(pcd_down)
    # vis.add_geometry(verts_down)
    # vis.update_renderer()
    # vis.run()
    # vis.destroy_window()

    verts = np.array(verts)
    pcd_pts = np.array(pcd.points)

    np.save("verts.npy", verts)
    np.save("pcd.npy", pcd_pts)

    # pdb.set_trace()


    # aligned_pts, transformation = align_point_clouds(np.array(verts), np.array(pcd.points), max_correspondence_distance=0.01, max_iterations=1000)

    aligned_pts, transformation = align_point_clouds_with_icp(np.array(verts), np.array(pcd.points), voxel_size=0.005)

    # aligned_pts, transformation = align_point_clouds(np.array(pcd.points), np.array(verts), max_correspondence_distance=0.01, max_iterations=1000)
    aligned_pcd = o3d.geometry.PointCloud()
    aligned_pcd.points = o3d.utility.Vector3dVector(aligned_pts)
    aligned_pcd.colors = o3d.utility.Vector3dVector(np.ones_like(aligned_pts) * [0, 1, 0])
    aligned_pcd.remove_non_finite_points()

    # Get keypoints from depth
    kpts_3d_d = []
    for pt_2d in kpts_2d:
        pt_3d = get_point_from_pixel(pt_2d[0], pt_2d[1], depth_tri[pt_2d[1], pt_2d[0]], intrinsics["left"])
        # pt_3d = get_point_from_pixel(pt_2d[0], pt_2d[1], depth_zed[pt_2d[1], pt_2d[0]], intrinsics["left"])
        kpts_3d_d.append(pt_3d)

    kpts_3d_d = np.array(kpts_3d_d)
    kpts_3d_d_colors = np.ones_like(kpts_3d_d) * [0, 0, 1]

    pcd_3d_d = o3d.geometry.PointCloud()
    pcd_3d_d.points = o3d.utility.Vector3dVector(kpts_3d_d)
    pcd_3d_d.colors = o3d.utility.Vector3dVector(kpts_3d_d_colors)
    pcd_3d_d.remove_non_finite_points()


    kpts_3d_colors = np.ones_like(kpts_3d) * [0, 1, 0]
    pcd_3d = o3d.geometry.PointCloud()
    pcd_3d.points = o3d.utility.Vector3dVector(kpts_3d)
    pcd_3d.colors = o3d.utility.Vector3dVector(kpts_3d_colors)
    pcd_3d.remove_non_finite_points()


    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])
    vis.add_geometry(pcd)
    vis.add_geometry(pcd_verts)
    # vis.add_geometry(aligned_pcd)
    # vis.add_geometry(pcd_outliers)
    # vis.add_geometry(pcd_3d_d)
    # vis.add_geometry(pcd_3d)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()


    fig = plt.figure()
    ax  = fig.add_subplot(111, projection='3d')
    plot_kpts(ax, kpts_3d, color=(0, 255, 0))
    plot_kpts(ax, kpts_3d_d, color=(255, 0, 0))

    # nfingers = len(kpts_3d_d) - 1
    # npts_per_finger = 4
    # list_fingers = [np.vstack([kpts_3d_d[0], kpts_3d_d[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
    # finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    # finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]

    # for finger_idx, finger_pts in enumerate(list_fingers):
    #     for i in range(len(finger_pts) - 1):
    #         color = finger_colors_rgb[finger_idx]
    #         ax.plot(
    #             [finger_pts[i][0], finger_pts[i + 1][0]],
    #             [finger_pts[i][1], finger_pts[i + 1][1]],
    #             [finger_pts[i][2], finger_pts[i + 1][2]],
    #             color=np.array(color)/255.0,
    #         )
    # ax.scatter(kpts_3d_d[:, 0], kpts_3d_d[:, 1], kpts_3d_d[:, 2])

    ax.set_xlim([0,1])
    ax.set_ylim([0,1])
    ax.set_zlim([0,1])
        
    plt.show()

