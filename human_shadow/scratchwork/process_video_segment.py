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
import mmt.stereo_inference.python.simple_inference as stereo_to_depth

from detector_dino import DetectorDino
from detector_sam2 import DetectorSam2

def get_point_from_pixel(px, py, depth, intrinsics):
    x = (px - intrinsics["cx"]) / intrinsics["fx"]
    y = (py - intrinsics["cy"]) / intrinsics["fy"]

    # For inverse brown conrady distortion?
    # From rs2_deproject_pixel_to_point()
    # Coeffs of camera I'm using are all 0 anyway.
    # r2 = x*x + y*y
    # f = 1 + intrinsics.coeffs[0]*r2 + intrinsics.coeffs[1]*r2*r2 + intrinsics.coeffs[4]*r2*r2*r2
    # ux = x*f + 2*intrinsics.coeffs[2]*x*y + intrinsics.coeffs[3]*(r2 + 2*x*x)
    # uy = y*f + 2*intrinsics.coeffs[3]*x*y + intrinsics.coeffs[2]*(r2 + 2*y*y)
    # x = ux
    # y = uy

    X = x * depth
    Y = y * depth
    p = np.stack((X, Y, depth), axis=1)

    return p

def visualize_pcd(pcd):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])
    vis.add_geometry(pcd)
    vis.update_renderer()
    vis.run()
    vis.destroy_window()

def segment_pcd(mask, depth_img, img, intrinsics, visualize=False):
    idxs_y, idxs_x = mask.nonzero()
    depth_masked = depth_img[idxs_y, idxs_x]
    seg_points = get_point_from_pixel(idxs_x, idxs_y, depth_masked, intrinsics)
    seg_colors = img_left_rgb[idxs_y, idxs_x, :] / 255.0  # Normalize to [0,1] for cv2

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(seg_points)
    pcd.colors = o3d.utility.Vector3dVector(seg_colors)
    pcd.remove_non_finite_points()

    if visualize:
        visualize_pcd(pcd)

    return pcd



if __name__ == "__main__":

    depth_mode = "TRI"
    
    # Get camera intrinsics
    intrinsics_path = "/juno/u/lepertm/human_shadow/human_shadow/camera/camera_intrinsics.json"
    with open(intrinsics_path, "r") as f:
        intrinsics = json.load(f)
    fx = intrinsics["left"]["fx"]
    CAM_BASELINE = 0.12  # Meters

    video_folder = "/juno/u/lepertm/human_shadow/human_shadow/data/videos/demo2/"
    video_num = 0

    left_video_path = os.path.join(video_folder, f"video_{video_num}_L.mp4")
    left_imgs = np.array(media.read_video(left_video_path))
    right_video_path = os.path.join(video_folder, f"video_{video_num}_R.mp4")
    right_imgs = np.array(media.read_video(right_video_path))
    n_imgs = len(left_imgs)

    depth_imgs = np.load(os.path.join(video_folder, f"depth_imgs_{video_num}.npy"))
    point_clouds = np.load(os.path.join(video_folder, f"point_clouds_{video_num}.npy"))

    detector_id = "IDEA-Research/grounding-dino-tiny"
    detector = DetectorDino(detector_id)

    # list_annotated_imgs_0 = []
    # list_annotated_imgs_1 = []
    # list_annotated_imgs_2 = []
    # for idx in tqdm(range(n_imgs)):
    #     img_left_rgb = left_imgs[idx]
    #     img_right_rgb = right_imgs[idx]

    #     bbox = detector.get_best_bbox(img_left_rgb, "hand")

    #     img_left_rgb_0 = img_left_rgb.copy()
    #     img_left_rgb_1 = img_left_rgb.copy()
    #     img_left_rgb_2 = img_left_rgb.copy()

    #     if bbox is not None:
    #         x = bbox[0]
    #         y = bbox[1]
    #         w = bbox[2] - bbox[0]
    #         h = bbox[3] - bbox[1]
    #         cv2.rectangle(img_left_rgb_0, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         cv2.rectangle(img_left_rgb_1, (x, y), (x+w, y+h), (0, 255, 0), 2)
    #         cv2.rectangle(img_left_rgb_2, (x, y), (x+w, y+h), (0, 255, 0), 2)

    #         segmentor = DetectorSam2()
    #         masks, scores = segmentor.segment_frame(img_left_rgb, bbox_pts=bbox)

    #         img_left_rgb_0[masks[0] == 1] = [0, 0, 0]
    #         img_left_rgb_1[masks[0] == 1] = [0, 0, 0]
    #         img_left_rgb_2[masks[0] == 1] = [0, 0, 0]
    #     list_annotated_imgs_0.append(img_left_rgb_0)
    #     list_annotated_imgs_1.append(img_left_rgb_1)
    #     list_annotated_imgs_2.append(img_left_rgb_2)

    # media.write_video("annotated_0_2.mp4", list_annotated_imgs_0, fps=10)
    # media.write_video("annotated_1_2.mp4", list_annotated_imgs_1, fps=10)
    # media.write_video("annotated_2_2.mp4", list_annotated_imgs_2, fps=10)

    # pdb.set_trace()



    list_pcds = []
    for idx in tqdm(range(0, 190, 40)):
        img_left_rgb = left_imgs[idx]
        img_right_rgb = right_imgs[idx]


        bbox = detector.get_best_bbox(img_left_rgb, "hand")

        if bbox is not None:

            segmentor = DetectorSam2()
            masks, scores = segmentor.segment_frame(img_left_rgb, bbox_pts=bbox)

            # seg_points, seg_colors = segment_pcd(masks[0], depth_imgs[idx], img_left_rgb, intrinsics["left"], visualize=True)

            img_left_bgr = img_left_rgb[..., ::-1]
            img_right_bgr = img_right_rgb[..., ::-1]
            depth_tri, img_tri_bgr = stereo_to_depth.get_depth_and_bgr(
                img_left_bgr.copy(), img_right_bgr.copy(), fx, CAM_BASELINE
            )
            pcd = segment_pcd(masks[0], depth_tri, img_left_rgb, intrinsics["left"], visualize=False)
            list_pcds.append(pcd)
    

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.background_color = np.asarray([0.2, 0.2, 0.2])
    for _ in range(100):
        for pcd in list_pcds:
            vis.add_geometry(pcd)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(1.0)
            vis.clear_geometries()

    list_seg_points.append(seg_points)

    all_seg_points = np.concatenate(list_seg_points, axis=0)
    visualize_pcd(all_seg_points, np.ones_like(all_seg_points))

    pdb.set_trace()


    fig, axs = plt.subplots(2, 2)
    img_left_rgb[masks[0] == 1] = [0, 0, 0]
    axs[0,0].imshow(img_left_rgb)
    # plot bbox 
    bbox = np.array(bbox).astype(int)
    bbox = patches.Rectangle((bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1], linewidth=1, edgecolor='r', facecolor='none')
    axs[0,0].add_patch(bbox)

    color = (0,0,0)
    
    
    axs[0,1].imshow(depth_imgs[idx], vmin=0, vmax=2)

    img_tri_rgb = img_tri_bgr[..., ::-1]
    axs[1,0].imshow(img_tri_rgb)
    axs[1,1].imshow(depth_tri, vmin=0, vmax=2)
    plt.show()

    pdb.set_trace()
