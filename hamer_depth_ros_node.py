#!/usr/bin/env python

import argparse
import time
from typing import Dict

import numpy as np
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROSImage
from std_msgs.msg import Float64MultiArray, MultiArrayDimension, MultiArrayLayout
from termcolor import colored

from hamer_depth.utils.hand_type import HandType


class HamerDepthROS:
    def __init__(self, hand_type: HandType):
        self.hand_type = hand_type

        # Put these imports here to avoid heavy import at the top
        # As this makes --help slow and ugly
        from hamer_depth.detectors.detector_hamer import (
            DetectorHamer,
        )

        self.detector_hamer = DetectorHamer()

        # Variables for storing the latest images
        self.latest_rgb = None
        self.latest_depth = None
        self.latest_cam_K = None
        self.latest_mask = None

        rospy.init_node("hamer_depth_node")
        self.bridge = CvBridge()

        # Check camera parameter
        camera = rospy.get_param("/camera", None)
        if camera is None:
            DEFAULT_CAMERA = "zed"
            print(
                colored(
                    f"No /camera parameter found, using default camera {DEFAULT_CAMERA}",
                    "yellow",
                )
            )
            camera = DEFAULT_CAMERA
        print(colored(f"Using camera: {camera}", "green"))
        if camera == "zed":
            self.rgb_sub_topic = "/zed/zed_node/rgb/image_rect_color"
            self.depth_sub_topic = "/zed/zed_node/depth/depth_registered"
            self.camera_info_sub_topic = "/zed/zed_node/rgb/camera_info"
        elif camera == "realsense":
            self.rgb_sub_topic = "/camera/color/image_raw"
            self.depth_sub_topic = "/camera/aligned_depth_to_color/image_raw"
            self.camera_info_sub_topic = "/camera/color/camera_info"
        else:
            raise ValueError(f"Unknown camera: {camera}")

        # Subscribers for RGB, depth, and mask images
        self.rgb_sub = rospy.Subscriber(
            self.rgb_sub_topic,
            ROSImage,
            self.rgb_callback,
            queue_size=1,
        )
        self.depth_sub = rospy.Subscriber(
            self.depth_sub_topic,
            # "/depth_anything_v2/depth",
            ROSImage,
            self.depth_callback,
            queue_size=1,
        )
        self.mask_sub = rospy.Subscriber(
            "/sam2_mask", ROSImage, self.mask_callback, queue_size=1
        )
        self.cam_K_sub = rospy.Subscriber(
            self.camera_info_sub_topic,
            CameraInfo,
            self.cam_K_callback,
            queue_size=1,
        )

        # Publisher for the object pose
        self.right_hand_keypoints_pub = rospy.Publisher(
            "/right_hand_keypoints", Float64MultiArray, queue_size=1
        )
        self.left_hand_keypoints_pub = rospy.Publisher(
            "/left_hand_keypoints", Float64MultiArray, queue_size=1
        )

    def rgb_callback(self, data):
        try:
            self.latest_rgb = self.bridge.imgmsg_to_cv2(data, "rgb8")
        except CvBridgeError as e:
            print(colored(f"Could not convert RGB image: {e}", "red"))

    def depth_callback(self, data):
        try:
            self.latest_depth = self.bridge.imgmsg_to_cv2(data, "64FC1")
        except CvBridgeError as e:
            print(colored(f"Could not convert depth image: {e}", "red"))

    def mask_callback(self, data):
        try:
            self.latest_mask = self.bridge.imgmsg_to_cv2(data, "mono8")
        except CvBridgeError as e:
            print(colored(f"Could not convert mask image: {e}", "red"))

    def cam_K_callback(self, data: CameraInfo):
        self.latest_cam_K = np.array(data.K).reshape(3, 3)

    def run(self):
        # Put these imports here to avoid heavy import at the top
        # As this makes --help slow and ugly
        from hamer_depth.utils.run_utils import (
            convert_intrinsics_matrix_to_dict,
            process_image_with_hamer,
        )

        ##############################
        # Wait for the first images
        ##############################
        while not rospy.is_shutdown() and (
            self.latest_rgb is None
            or self.latest_depth is None
            or self.latest_mask is None
            or self.latest_cam_K is None
        ):
            print(
                colored(
                    "Missing one of the required images (RGB, depth, mask, cam_K). Waiting...",
                    "yellow",
                )
            )
            rospy.sleep(0.1)

        assert self.latest_rgb is not None
        assert self.latest_depth is not None
        assert self.latest_mask is not None
        assert self.latest_cam_K is not None

        while not rospy.is_shutdown():
            ##############################
            # Register
            ##############################
            print(colored("Running registration", "green"))

            rgb = self.process_rgb(self.latest_rgb)
            depth = self.process_depth(self.latest_depth)
            mask = self.process_mask(self.latest_mask)
            cam_K = self.latest_cam_K.copy()

            camera_intrinsics = convert_intrinsics_matrix_to_dict(cam_K)

            # Hand detection
            t0 = time.time()
            (
                _,
                hamer_out,
                _,
                _,
                hand_keypoints_dict,
                _,
                _,
                _,
            ) = process_image_with_hamer(
                img_rgb=rgb,
                img_depth=depth,
                mask=mask,
                cam_intrinsics=camera_intrinsics,
                detector_hamer=self.detector_hamer,
                hand_type=self.hand_type,
                debug=False,
            )

            print(
                colored(
                    f"time for hamer depth = {(time.time() - t0) * 1000} ms",
                    "green",
                )
            )
            print(colored(f"hand_keypoints_dict = {hand_keypoints_dict}", "green"))
            if self.hand_type == HandType.RIGHT:
                self.publish_right_hand_keypoints(
                    keypoints_dict=hand_keypoints_dict,
                    orientation=hamer_out["global_orient"],
                )
            elif self.hand_type == HandType.LEFT:
                self.publish_left_hand_keypoints(
                    keypoints_dict=hand_keypoints_dict,
                    orientation=hamer_out["global_orient"].tolist(),
                )
            else:
                raise ValueError(f"Unknown hand type: {self.hand_type}")

    def process_rgb(self, rgb):
        return rgb

    def process_depth(self, depth):
        # Turn nan values into 0
        depth[np.isnan(depth)] = 0
        depth[np.isinf(depth)] = 0

        # depth is either in meters or millimeters
        # Need to convert to meters
        # If the max value is greater than 100, then it's likely in mm
        in_mm = depth.max() > 100
        if in_mm:
            # print(colored(f"Converting depth from mm to m since max = {depth.max()}", "green"))
            depth = depth / 1000
        else:
            pass
            # print(colored(f"Depth is in meters since max = {depth.max()}", "green"))

        # Clamp
        depth[depth < 0.1] = 0
        depth[depth > 4] = 0

        return depth

    def process_mask(self, mask):
        mask = mask.astype(bool)
        return mask

    def publish_right_hand_keypoints(
        self, keypoints_dict: Dict[str, np.ndarray], orientation: np.ndarray
    ):
        keypoints = np.stack([v for v in keypoints_dict.values()], axis=0)
        N_KEYPOINTS = 12
        assert keypoints.shape == (N_KEYPOINTS, 3), (
            f"keypoints.shape = {keypoints.shape}"
        )

        # Publish the keypoints
        msg = Float64MultiArray()
        msg.layout = MultiArrayLayout(
            dim=[
                MultiArrayDimension(
                    label="keypoints", size=N_KEYPOINTS, stride=N_KEYPOINTS
                )
            ],
            data_offset=0,
        )
        msg.data = keypoints.flatten().tolist()
        self.right_hand_keypoints_pub.publish(msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hand_type", type=HandType, default=HandType.RIGHT)
    args = parser.parse_args()

    node = HamerDepthROS(hand_type=args.hand_type)
    node.run()
