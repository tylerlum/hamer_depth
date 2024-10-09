import pdb
import numpy as np 
import time
import redisgl
import redis
import ctrlutils
import json

from human_shadow.config.redis_keys import *

def update_marker_pos(redis_pipe, key_obj, pos, ori):
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


def init_marker(redis_pipe):
    APP_NAMESPACE = "redisgl::human_shadow"
    KEY_OBJECTS_PREFIX = f"{APP_NAMESPACE}::markers"
    model_keys = redisgl.ModelKeys(KEY_OBJECTS_PREFIX)
    redisgl.register_model_keys(redis_pipe, model_keys)

    # Create sphere 
    color = (1, 0, 0, 1)
    mat = redisgl.Material(rgba=color)
    KEY_POS = f"{KEY_OBJECTS_PREFIX}::pos"
    KEY_ORI = f"{KEY_OBJECTS_PREFIX}::ori"
    KEY_SCALE = f"{KEY_OBJECTS_PREFIX}::scale"
    KEY_MATRIX = f"{KEY_OBJECTS_PREFIX}::matrix"
    object_name = "hand_pos"
    graph = redisgl.Graphics(
        name="sphere",
        geometry=redisgl.Sphere(radius=0.01), 
        material=mat,
    )
    object_model = redisgl.ObjectModel(
        name=object_name,
        graphics=graph,
        key_pos=KEY_POS, 
        key_ori=KEY_ORI,
        key_scale=KEY_SCALE,
        key_matrix=KEY_MATRIX,
    )
    redisgl.register_object(redis_pipe, model_keys, object_model)
    redis_pipe.execute()


def main(): 
    # Initialize redis
    _redis = ctrlutils.RedisClient(
        host=BOHG_FRANKA_HOST, port=BOHG_FRANKA_PORT, password=BOHG_FRANKA_PWD
    )
    redis_pipe = _redis.pipeline()

    init_marker(redis_pipe)

    start_time = time.time()
    while True: 
        marker_x = np.sin(time.time() - start_time)
        marker_y = np.cos(time.time() - start_time)
        marker_z = 0.5
        pos = [marker_x, marker_y, marker_z]
        ori = {"x": 0, "y": 0, "z": 0, "w": 1}
        update_marker_pos(redis_pipe, f"redisgl::human_shadow::markers::model::object::hand_pos", pos, ori)





if __name__ == "__main__":
    main()

