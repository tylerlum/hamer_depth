import pdb
import numpy as np 
import mujoco 
from mujoco import viewer
import mediapy as media
import matplotlib.pyplot as plt
import enum
from scipy.spatial.transform import Rotation

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

    model_path = "assets/franka_emika_panda/scene.xml"
    # model_path = "assets/universal_robots_ur5e/scene.xml"

    model = mujoco.MjModel.from_xml_path(model_path)

    data = mujoco.MjData(model)

    res = Resolution.HD2
    h, w = res.value

    model.vis.global_.offheight = h
    model.vis.global_.offwidth = w

    renderer = mujoco.Renderer(model, height=h, width=w)

    for key in range(model.nkey):
        mujoco.mj_resetDataKeyframe(model, data, key)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera="zed2")
        img = renderer.render()
    #     plt.imshow(img)
    #     plt.show()


    mujoco.mj_step(model, data)

    with viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as vwr:
        while vwr.is_running():
            vwr.sync()
            mujoco.mj_step(model, data)



if __name__ == "__main__":
    main()