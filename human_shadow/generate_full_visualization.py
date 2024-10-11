import cv2
import os
import mediapy as media
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

def generate_visualization(dir_path):
    '''
    Generate visualization for each video, also generate a full visualization of all sequences.
    '''
    images_all_full = []
    for video_dir in tqdm(sorted(os.listdir(dir_path), key=lambda x: int(os.path.splitext(x)[0]))):
        video_folder = os.path.join(dir_path, video_dir)
        video_num = int(video_dir)
        root_rgb_path = os.path.join(video_folder, "video_"+video_dir+"_L")
        root_sam_path = os.path.join(video_folder, "sam2_image")
        root_hamer_path = os.path.join(video_folder, "hamer_image")
        root_pcd_path = os.path.join(video_folder, "point_cloud_image")
        y_predicted = np.load(os.path.join(video_folder, "tip_points.npy"))
        y_predicted = y_predicted.squeeze()

        y_predicted_index = np.load(os.path.join(video_folder, "index_tip_points.npy"))
        y_predicted_index = y_predicted_index.squeeze()
        y_predicted_thumb = np.load(os.path.join(video_folder, "thumb_tip_points.npy"))
        y_predicted_thumb = y_predicted_thumb.squeeze()

        images_all = []

        fig, ax = plt.subplots(1,1,subplot_kw=dict(projection='3d'),figsize=(10.0, 10.0))
        ax.view_init(elev=-10., azim=100)
        plt.gca().set_aspect('equal', adjustable='box')
        ax.set_xlim(-0.4, 0)
        ax.set_zlim(-0.1, 0.3)
        ax.set_ylim(0.3, 0.7)
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.gca().invert_zaxis()


        lines = []
        ax.set_xlabel('x')
        ax.set_ylabel('z')
        ax.set_zlabel('y')
        for index in range(1):
            lobj = ax.plot3D([],[],[], marker='o', color='r',label='control_point_line')[0]
            lpoint1 = ax.plot3D([],[],[], marker='o', linestyle='', color='cyan',label='thumb_point')[0]
            lpoint2 = ax.plot3D([],[],[], marker='o', linestyle='', color='blue',label='index_point')[0]
            lpoint3 = ax.plot3D([],[],[], marker='o', linestyle='', color='g',label='control_point')[0]
            lines.append(lobj)
            lines.append(lpoint1)
            lines.append(lpoint2)
            lines.append(lpoint3)

        ax.legend()
        ax.grid()
        # Read images from files or generate them
        for i, image_path in tqdm(enumerate(sorted(os.listdir(root_hamer_path)))):
            img1 = cv2.imread(os.path.join(root_sam_path, image_path))
            img1 = cv2.resize(img1, (1000, 1000))
            img2 = cv2.imread(os.path.join(root_hamer_path, image_path))
            img2 = cv2.resize(img2, (1000, 1000))
            img3 = cv2.imread(os.path.join(root_pcd_path, image_path))
            img3 = cv2.resize(img3, (1000, 1000))
        
            lines[0].set_data(y_predicted[:i+1,0], y_predicted[:i+1,2])
            lines[0].set_3d_properties(y_predicted[:i+1,1])
            lines[1].set_data(y_predicted_thumb[i:i+1,0], y_predicted_thumb[i:i+1,2])
            lines[1].set_3d_properties(y_predicted_thumb[i:i+1,1])
            lines[2].set_data(y_predicted_index[i:i+1,0], y_predicted_index[i:i+1,2])
            lines[2].set_3d_properties(y_predicted_index[i:i+1,1])
            lines[3].set_data(y_predicted[i:i+1,0], y_predicted[i:i+1,2])
            lines[3].set_3d_properties(y_predicted[i:i+1,1])
            fig.canvas.draw()
            img4 = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            img4 = img4.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            img4 = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)
            images1 = cv2.hconcat([img1, img2])
            images2 = cv2.hconcat([img3, img4])
            images = cv2.vconcat([images1, images2])
            images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
            images_all.append(images)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(images, video_dir, (20, 60), font, 3, (0, 255, 0), 2, cv2.LINE_AA)
            images_all_full.append(images)

        media.write_video(os.path.join(video_folder, "visualization.mp4"), images_all, fps=30)
    media.write_video(os.path.join(dir_path, "full_visualization.mp4"), images_all_full, fps=30)

if __name__ == '__main__':
    # Change path
    dir_path = "/juno/u/jyfang/human_shadow/data/demo_jiaying_waffles_large"
    generate_visualization(dir_path)