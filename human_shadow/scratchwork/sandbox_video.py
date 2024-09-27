import pdb 
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import mediapy as media
from tqdm import tqdm
from human_shadow.detector_hamer import DetectorHamer



if __name__ == "__main__":

    video_folder = "/juno/u/lepertm/human_shadow/data/videos/demo1/"
    video_num = 8

    video_path = os.path.join(video_folder, f"video_{video_num}.mp4")
    imgs = media.read_video(video_path)
    imgs = np.array(imgs)
    n_imgs = len(imgs)

    detector_hamer = DetectorHamer()
    annotated_imgs = []
    failed_imgs = []
    index_traj = []
    list_kpts_3d = []
    list_kpts_2d = []

    for img_idx, img in enumerate(tqdm(imgs)):
        annotated_img, success, kpts_3d, kpts_2d = detector_hamer.detect_hand_keypoints(img, visualize=False)
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
        annotated_imgs.append(annotated_img)

        if kpts_3d is not None:
            index_traj.append(kpts_3d[8,:])
            list_kpts_3d.append(kpts_3d)
            list_kpts_2d.append(kpts_2d.cpu().numpy())
        else:
            list_kpts_3d.append(np.zeros((21, 3)))
            list_kpts_2d.append(np.zeros((21, 2)))

        if not success:
            failed_imgs.append(annotated_img)

    index_traj = np.array(index_traj)
    list_kpts_3d = np.array(list_kpts_3d)
    list_kpts_2d = np.array(list_kpts_2d)
    list_kpts_2d = np.rint(list_kpts_2d).astype(int)
    np.save(os.path.join(video_folder, f"video_{video_num}_kpts_3d.npy"), list_kpts_3d)
    np.save(os.path.join(video_folder, f"video_{video_num}_kpts_2d.npy"), list_kpts_2d)

    point_cloud = np.load(os.path.join(video_folder, f"point_clouds_8.npy"))
    


    # fig = plt.figure()
    # n_kpts = len(list_kpts)
    # vis_indices = np.linspace(0, n_kpts-1, 3).astype(int)

    # ax = fig.add_subplot(111, projection='3d')
    # for i in vis_indices:
    #     kpts_3d = list_kpts[i]
    #     ax.scatter(kpts_3d[:,0], kpts_3d[:,1], kpts_3d[:,2])

    #     nfingers = len(kpts_3d) - 1
    #     npts_per_finger = 4
    #     list_fingers = [np.vstack([kpts_3d[0], kpts_3d[i:i + npts_per_finger]]) for i in range(1, nfingers, npts_per_finger)]
    #     finger_colors_bgr = [(0, 255, 0), (0, 0, 255), (255, 0, 0), (255, 0, 255), (0, 255, 255)]
    #     finger_colors_rgb = [(color[2], color[1], color[0]) for color in finger_colors_bgr]
    #     for finger_idx, finger_pts in enumerate(list_fingers):
    #         for i in range(len(finger_pts) - 1):
    #             color = finger_colors_rgb[finger_idx]
    #             ax.plot(
    #                 [finger_pts[i][0], finger_pts[i + 1][0]],
    #                 [finger_pts[i][1], finger_pts[i + 1][1]],
    #                 [finger_pts[i][2], finger_pts[i + 1][2]],
    #                 color=np.array(color)/255.0,
    #             )


    # ax.plot(index_traj[:,0], index_traj[:,1], index_traj[:,2])
    # plt.show()
    
    annotated_video_path = os.path.join(video_folder, f"video_{video_num}_annotated.mp4")

    media.write_video(annotated_video_path, annotated_imgs)
    media.write_video(os.path.join(video_folder, f"video_{video_num}_failed.mp4"), failed_imgs)

    pdb.set_trace()

