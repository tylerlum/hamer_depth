import pdb
import numpy as np 
import time
import redisgl
import argparse
import ctrlutils
import json
import cv2
import msgpack_numpy as m
m.patch()

from human_shadow.config.redis_keys import *
from human_shadow.camera.zed_utils import *


def update_marker_pos(redis_pipe, key_obj, pos=None, ori=None):
    redis_pipe.get(key_obj)
    obj_string = redis_pipe.execute()[0]
    new_obj_string = edit_graphics_params(obj_string, pos=pos, ori=ori)
    redis_pipe.set(key_obj, new_obj_string)
    redis_pipe.execute()

def edit_graphics_params(json_string=None, pos=None, ori=None, radius=None, rgba=None):
    if json_string is None:
        json_string = "{\"graphics\": [{\"name\": \"sphere\", \"T_to_parent\": {\"pos\": [0.0, 0.0, 0.0], \"ori\": {\"x\": 0.0, \"y\": 0.0, \"z\": 0.0, \"w\": 1.0}}, \"geometry\": {\"type\": \"sphere\", \"radius\": 0.1}, \"material\": {\"name\": \"\", \"rgba\": [1, 0, 0, 1], \"texture\": \"\"}}], \"key_pos\": \"redisgl::human_shadow::markers::pos\", \"key_ori\": \"redisgl::human_shadow::markers::ori\", \"key_scale\": \"redisgl::human_shadow::markers::scale\", \"key_matrix\": \"redisgl::human_shadow::markers::matrix\", \"axis_size\": 0.1}"

    data = json.loads(json_string)
    
    if pos is not None:
        data["graphics"][0]["T_to_parent"]["pos"] = pos
    if ori is not None:
        data["graphics"][0]["T_to_parent"]["ori"] = ori
    if radius is not None:
        data["graphics"][0]["geometry"]["radius"] = radius
    if rgba is not None:
        data["graphics"][0]["material"]["rgba"] = rgba
    
    return json.dumps(data)


def init_marker(redis_pipe, name, color):
    APP_NAMESPACE = "redisgl::human_shadow"
    KEY_OBJECTS_PREFIX = f"{APP_NAMESPACE}::markers_{name}"
    model_keys = redisgl.ModelKeys(KEY_OBJECTS_PREFIX)
    redisgl.register_model_keys(redis_pipe, model_keys)

    # Create sphere 
    mat = redisgl.Material(rgba=color)
    KEY_POS = f"{KEY_OBJECTS_PREFIX}::pos"
    KEY_ORI = f"{KEY_OBJECTS_PREFIX}::ori"
    KEY_SCALE = f"{KEY_OBJECTS_PREFIX}::scale"
    KEY_MATRIX = f"{KEY_OBJECTS_PREFIX}::matrix"
    graph = redisgl.Graphics(
        name=name,
        geometry=redisgl.Sphere(radius=0.01), 
        material=mat,
    )
    object_model = redisgl.ObjectModel(
        name=name,
        graphics=graph,
        key_pos=KEY_POS, 
        key_ori=KEY_ORI,
        key_scale=KEY_SCALE,
        key_matrix=KEY_MATRIX,
    )
    redisgl.register_object(redis_pipe, model_keys, object_model)
    redis_pipe.execute()

def transform_pt(pt, T):
    pt = np.array(pt)
    pt = np.append(pt, 1)
    pt = np.dot(T, pt)
    return pt[:3]


def main(args):
    # Initialize redis
    _redis = ctrlutils.RedisClient(
        host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
    )
    redis_pipe = _redis.pipeline()

    # Initialize redisgl
    init_marker(redis_pipe, "hand_ee", color=(1, 0, 0, 1))
    init_marker(redis_pipe, "thumb", color=(0, 1, 0, 1))
    init_marker(redis_pipe, "middle", color=(0, 0, 1, 1))

    # Camera extrinsics
    camera_extrinsics_path = "camera/camera_calibration_data/hand_calib_HD1080/cam_cal.json"
    with open(camera_extrinsics_path, "r") as f:
        camera_extrinsics = json.load(f)
    cam_base_pos = np.array(camera_extrinsics[0]["camera_base_pos"])
    cam_base_ori = np.array(camera_extrinsics[0]["camera_base_ori"])
    T_cam2robot = np.eye(4)
    T_cam2robot[:3, 3] = cam_base_pos
    T_cam2robot[:3, :3] = np.array(cam_base_ori).reshape(3, 3)

    print("Running visualization")
    while True: 
        redis_pipe.get(KEY_HAND_EE_POS)
        redis_pipe.get(KEY_HAMER_IMAGE)
        hand_points_b, hand_imgs_b = redis_pipe.execute()
        hand_points = m.unpackb(hand_points_b)
        hand_ee_pt = hand_points[0]
        thumb_pt = hand_points[1]
        middle_pt = hand_points[2]
        hand_imgs = m.unpackb(hand_imgs_b)
        hamer_img_bgr = hand_imgs[0]
        sam2_img_rgb = hand_imgs[1]

        hand_ee_pt = transform_pt(hand_ee_pt, T_cam2robot)
        thumb_pt = transform_pt(thumb_pt, T_cam2robot)
        middle_pt = transform_pt(middle_pt, T_cam2robot)

        update_marker_pos(redis_pipe, f"redisgl::human_shadow::markers_hand_ee::model::object::hand_ee", pos=hand_ee_pt.tolist())
        update_marker_pos(redis_pipe, f"redisgl::human_shadow::markers_thumb::model::object::thumb", pos=thumb_pt.tolist())
        update_marker_pos(redis_pipe, f"redisgl::human_shadow::markers_middle::model::object::middle", pos=middle_pt.tolist())

        if args.render:
            sam2_img_bgr = sam2_img_rgb[..., ::-1]

            img_h = int(hamer_img_bgr.shape[0]/2)
            img_w = int(hamer_img_bgr.shape[1]/2)
            hamer_img_bgr = cv2.resize(hamer_img_bgr, (img_w, img_h))
            sam2_img_bgr = cv2.resize(sam2_img_bgr, (img_w, img_h))

            image_vis = cv2.hconcat([hamer_img_bgr, sam2_img_bgr])
            cv2.imshow("hand", image_vis)
            cv2.waitKey(1)

        




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--render", action="store_true", help="render")
    args = parser.parse_args()
    main(args)

