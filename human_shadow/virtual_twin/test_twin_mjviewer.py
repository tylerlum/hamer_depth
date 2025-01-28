import pdb
import numpy as np 
import mujoco 
import os
import cv2
import pickle
from tqdm import tqdm

from mujoco import viewer
import mediapy as media
import matplotlib.pyplot as plt
import enum
from scipy.spatial.transform import Rotation


from human_shadow.utils.file_utils import get_parent_folder_of_package


class Resolution(enum.Enum):
  SD = (480, 640)
  HD = (720, 1280)
  HD2 = (1080, 1920)
  UHD = (2160, 3840)


def main():
    # original_quat_xyzw = np.array([0.8268784921283836, -0.07776511222665376, 0.058362171949838414, -0.553911909477639])
    # original_quat_wxyz = np.array([-0.553911909477639, 0.8268784921283836, -0.07776511222665376, 0.058362171949838414])

    # original_quat_xyzw = np.array([0.764365774715041, 0.012246957244567166, -0.0010757563131612176, -0.6446656631393358])
    # original_quat_wxyz = np.array([original_quat_xyzw[1], original_quat_xyzw[2], original_quat_xyzw[3], original_quat_xyzw[0]])

    original_quat_xyzw = np.array([0.8204965462375373, -0.07000374049084156, 0.054451304871138306, -0.564729979129313])
    original_quat_wxyz = np.array([original_quat_xyzw[3], original_quat_xyzw[0], original_quat_xyzw[1], original_quat_xyzw[2]])

    r1 = Rotation.from_quat(original_quat_wxyz)
    r2 = Rotation.from_euler("z", 180, degrees=True)
    new_rot =  r2 * r1
    new_quat = new_rot.as_quat()
    print(new_quat) # copy this value to scene.xml

    # model_path = "assets/franka_emika_panda/scene.xml"
    # model_path = "assets/universal_robots_ur5e/scene.xml"
    model_path = "assets/panda_robotiq_2f85/scene.xml"

    model = mujoco.MjModel.from_xml_path(model_path)

    data = mujoco.MjData(model)

    res = Resolution.HD2
    h, w = res.value

    model.vis.global_.offheight = h
    model.vis.global_.offwidth = w

    rgb_renderer = mujoco.Renderer(model, height=h, width=w)
    seg_renderer = mujoco.Renderer(model, height=h, width=w)
    seg_renderer.enable_segmentation_rendering() # output (object ID, object type)

    n_links = 8
    body_ids = []
    geom_ids = []
    for link_idx in range(n_links):
        id = model.body(f'link{link_idx}').id
        ngeoms = model.body(f'link{link_idx}').geomnum
        geom_adr = model.body(f'link{link_idx}').geomadr
        body_geom_ids = np.arange(ngeoms) + geom_adr
        geom_ids.append(body_geom_ids)
        body_ids.append(id)
    geom_ids = np.concatenate(geom_ids)
    body_ids = np.array(body_ids)
    

    # for key in range(model.nkey):
    #     mujoco.mj_resetDataKeyframe(model, data, key)
    #     mujoco.mj_forward(model, data)
    #     renderer.update_scene(data, camera="zed2")
    #     img = renderer.render()
    # #     plt.imshow(img)
    # #     plt.show()


    # Load calibration pickle
    project_folder = get_parent_folder_of_package("human_shadow")
    cal_pkl = os.path.join(project_folder, "human_shadow/camera/camera_calibration_data/hand_calib_HD1080/calibration_data.pkl")
    with open(cal_pkl, "rb") as f:
        data_list = pickle.load(f)
    img_num = 0
    robot_qpos = data_list[img_num]["qpos"]
    robot_pos = data_list[img_num]["pos"]
    robot_ori_xyzw = data_list[img_num]["ori"]
    real_img = data_list[img_num]["imgs"][0]
    real_img = real_img[:,420:-420]

    real_initial_state = {
        "pos": robot_pos,
        "quat_xyzw": robot_ori_xyzw,
        "qpos": robot_qpos,
        "gripper_pos": 0.0
    }



    mujoco.mj_step(model, data)



    n_steps = 1000



    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as vwr:
        while vwr.is_running():
            vwr.sync()
            mujoco.mj_step(model, data)

            for img_idx in tqdm(range(len(data_list))):
                robot_qpos = data_list[img_idx]["qpos"]
                real_img = data_list[img_idx]["imgs"][0]
                # real_img = real_img[:,420:-420]

                for i in range(10000):
                    data.qpos[:7] = robot_qpos
                    if img_idx % 3 == 0:
                        data.ctrl[-1] = 255
                        print(255)
                    elif img_idx % 3 == 1: 
                        data.ctrl[-1] = 128
                        print(128)
                    else: 
                        data.ctrl[-1] = 0
                        print(0)

    
                    vwr.sync()
                    data.qpos[:7] = robot_qpos
                    if img_idx % 3 == 0:
                        data.ctrl[-1] = 255
                    elif img_idx % 3 == 1: 
                        data.ctrl[-1] = 128
                    else: 
                        data.ctrl[-1] = 0
                    mujoco.mj_step(model, data)

                rgb_renderer.update_scene(data, camera="zed2")
                img = rgb_renderer.render()
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                seg_renderer.update_scene(data, camera="zed2")
                seg_img = seg_renderer.render()[:,:,0]
                seg_mask = np.isin(seg_img, geom_ids).astype(np.uint8)*255

                masked_img = real_img.copy()
                masked_img[seg_mask == 255] = 0
                masked_img_bgr = masked_img[:,:,::-1]

                cv2.imwrite(f"debug_images3/img_{img_idx}.png",  masked_img_bgr)


                # if img_idx == 36: 
                #     plt.imshow(seg_img)
                #     plt.show()

                #     plt.imshow(seg_mask)
                #     plt.show()


                # cv2.imwrite(f"debug_images2/img_{img_idx}.png", img_bgr)

                # cv2.imwrite(f"debug_images2/seg_img_{img_idx}.png", seg_img)
                # cv2.imwrite(f"debug_images2/seg_mask_{img_idx}.png", seg_mask)






if __name__ == "__main__":
    main()