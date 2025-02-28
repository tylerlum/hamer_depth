# Local bohg-franka redis server HOST and PORT
# BOHG_FRANKA_HOST = "127.0.0.1"
BOHG_FRANKA_HOST = "172.24.69.91"
BOHG_FRANKA_PORT = 6379
BOHG_FRANKA_PWD = ""

NUC_HOST = "172.24.68.230"
NUC_PORT = 6379
NUC_PWD = "iprl"

# Robotiq Gripper keys for redis
KEY_ROBOTIQ_SENSOR_POS = "robotiq_gripper::sensor::pos"
KEY_ROBOTIQ_SENSOR_Q = "robotiq_gripper::sensor::q"
KEY_ROBOTIQ_CONTROL_POS_DES = "robotiq_gripper:control::pos_des"
KEY_ROBOTIQ_CONTROL_COMMAND = "robotiq_gripper::control::command"

# Franka keys for redis
KEY_SENSOR_Q = "franka_panda::sensor::q"
KEY_CONTROL_POS = "franka_panda::control::pos"
KEY_CONTROL_ORI = "franka_panda::control::ori"

# Camera keys for redis
APP_NAMESPACE = "rgbd"
LEFT_CAMERA_NAME = "left_camera"
KEY_LEFT_CAMERA_POS = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::pos"
KEY_LEFT_CAMERA_ORI = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::ori"
KEY_LEFT_CAMERA_INTRINSIC = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::intrinsic"
KEY_LEFT_CAMERA_DEPTH_MM = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::depth_mm"
KEY_LEFT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::{LEFT_CAMERA_NAME}::image_bin"

RIGHT_CAMERA_NAME = "right_camera"
KEY_RIGHT_CAMERA_POS = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::pos"
KEY_RIGHT_CAMERA_ORI = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::ori"
KEY_RIGHT_CAMERA_INTRINSIC = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::intrinsic"
KEY_RIGHT_CAMERA_DEPTH_MM = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::depth_mm"
KEY_RIGHT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::{RIGHT_CAMERA_NAME}::image_bin"

KEY_LEFT_RIGHT_CAMERA_IMAGE_BIN = f"{APP_NAMESPACE}::left_right_camera::image_bin"


KEY_CAMERA_DEPTH_BIN = f"{APP_NAMESPACE}::camera::depth_bin"
KEY_CAMERA_POINT_CLOUD_BIN = f"{APP_NAMESPACE}::camera::point_cloud_bin"


# Hand detection
KEY_HAND_EE_POS = "hand::ee_pos"
KEY_HAMER_IMAGE = "hand::hamer_image"
