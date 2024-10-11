import os
import cv2
from tqdm import tqdm
import mediapy as media

def video_preprocessing(dir_path):
    for video_dir in tqdm(os.listdir(dir_path)):
        video_folder = os.path.join(dir_path, video_dir)
        print(video_folder)
        video_path = os.path.join(video_folder, "video_"+video_dir+"_L.mp4")
        output_dir = os.path.join(video_folder, "video_"+video_dir+"_L")
        hamer_dir = os.path.join(video_folder, "hamer_image")
        sam2_dir = os.path.join(video_folder, "sam2_image")
        pcd_dir = os.path.join(video_folder, "point_cloud_image")
        # Create the output directory if it doesn't exist
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.path.exists(hamer_dir):
            os.makedirs(hamer_dir)
        if not os.path.exists(sam2_dir):
            os.makedirs(sam2_dir)
        if not os.path.exists(pcd_dir):
            os.makedirs(pcd_dir)

        # Open the video file
        cap = cv2.VideoCapture(video_path)

        # Frame counter
        frame_num = 0

        images = []
        # Loop through the video frames
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            filename = f"{frame_num:05}.jpg"
            
            # Crop frame
            frame = frame[:1000, 80:, :]
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            images.append(frame[..., ::-1].copy())
            
            frame_num += 1

        cap.release()
        media.write_video(os.path.join(video_folder, "video_"+video_dir+"_L_preprocessed.mp4"), images, fps=30)

if __name__ == '__main__':
    # Change the dir_path
    dir_path = "/juno/u/jyfang/human_shadow/data/demo_jiaying_waffles_large_2"
    video_preprocessing(dir_path)
