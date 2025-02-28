"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
Modified from: https://github.com/facebookresearch/fairo/blob/main/perception/sandbox/eyehandcal/src/eyehandcal

Top-level script for computing camera calibration given .pkl data file
"""

import argparse
import json
import os
import pickle
from collections import namedtuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

import human_shadow.camera.calibration_utils as c_utils

PROJ_FUNC = "hand_marker_proj_world_camera"
PROJ_FUNC_DICT = {
    "hand_marker_proj_world_camera": c_utils.hand_marker_proj_world_camera,
}

os.environ["DISPLAY"] = ":1"


def extract_obs_data_std(data, camera_index):
    obs_data_std = []
    for d in data:
        if d["corners"][camera_index] is not None:
            # Convert ftrot (xyzw quat) to rotvec
            ftrot_vec = R.from_quat(d["ori"]).as_rotvec()
            obs_data_std.append(
                (
                    torch.tensor(d["corners"][camera_index], dtype=torch.float64),
                    torch.tensor(d["pos"]).double(),
                    torch.tensor(ftrot_vec).double(),
                )
            )

    ic = list(data[0]["intrinsics"])[camera_index]
    K = c_utils.build_proj_matrix(
        fx=ic["fx"], fy=ic["fy"], ppx=ic["ppx"], ppy=ic["ppy"]
    )
    return obs_data_std, K


def get_data_with_corners(data_list, target_idx=0, save_dir=None):
    data_with_corners = c_utils.detect_corners(data_list, target_idx=0)
    num_imgs_detected = 0
    corner_data = []
    for i, data in enumerate(data_with_corners):
        # Check if sample has any corners detected
        any_corners = True in (x is not None for x in data["corners"])

        if any_corners:
            num_imgs_detected += 1
            intrinsics = data["K"]
            if isinstance(intrinsics, np.ndarray):
                intrinsics_dict = {
                    "fx": intrinsics[0, 0],
                    "fy": intrinsics[1, 1],
                    "ppx": intrinsics[0, 2],
                    "ppy": intrinsics[1, 2],
                }
                data["intrinsics"] = [intrinsics_dict]  # Put intrinsics in list
            else:
                data["intrinsics"] = [intrinsics]  # Put intrinsics in list

            corner_data.append(data)

            # Save image with corners visualized
            if save_dir is not None:
                assert len(data["corners"]) == 1
                x, y = data["corners"][0]
                plt.figure()
                fig_save_path = os.path.join(save_dir, f"corner_img_{i}.png")
                plt.imshow(data["imgs"][0])
                plt.plot(x, y, "r+")
                plt.savefig(fig_save_path)
                plt.close()
        else:
            print(f"No corners detected in image: {i}")

    print(f"Detected marker in {num_imgs_detected} images")
    return corner_data


def compute_cal(corner_data, pixel_tolerance, save_dir):
    intrinsics = corner_data[0]["intrinsics"]
    num_of_camera = len(intrinsics)
    CalibrationResult = namedtuple(
        "CalibrationResult",
        field_names=[
            "num_marker_seen",
            "stage2_retry",
            "pixel_error",
            "param",
            "proj_func",
        ],
        defaults=[None] * 5,
    )
    cal_results = []
    for i in range(num_of_camera):
        print(f"Solve camera {i}/{num_of_camera} pose")
        obs_data_std, K = extract_obs_data_std(corner_data, i)
        print("number of images with keypoint", len(obs_data_std))
        if len(obs_data_std) < 3:
            print("too few keypoint found for this camera, skip this camera")
            cal_results.append(CalibrationResult(num_marker_seen=len(obs_data_std)))
            continue

        # stage 1 - assuming marker is attached to EE origin, solve camera pose first
        if PROJ_FUNC == "hand_marker_proj_world_camera":
            p3d = torch.stack([p[1] for p in obs_data_std]).detach().numpy()
        elif PROJ_FUNC == "world_marker_proj_hand_camera":
            p3d = (
                torch.stack([c_utils.rotmat(-p[2]).matmul(-p[1]) for p in obs_data_std])
                .detach()
                .numpy()
            )

        p2d = torch.stack([p[0] for p in obs_data_std]).detach().numpy()
        retval, rvec, tvec = cv2.solvePnP(
            p3d, p2d, K.numpy(), distCoeffs=None, flags=cv2.SOLVEPNP_SQPNP
        )
        rvec_cam = torch.tensor(-rvec.reshape(-1))
        tvec_cam = -c_utils.rotmat(rvec_cam).matmul(torch.tensor(tvec.reshape(-1)))
        pixel_error = c_utils.mean_loss(
            obs_data_std,
            torch.cat([rvec_cam, tvec_cam, torch.zeros(3)]),
            K,
            PROJ_FUNC_DICT[PROJ_FUNC],
        ).item()
        print("stage 1 mean pixel error", pixel_error)

        # stage 2 - allow marker to move, joint optimize camera pose and marker
        max_stage2_retry = 10
        stage2_retry_count = 0

        while True:
            stage2_retry_count += 1
            if stage2_retry_count > max_stage2_retry:
                cal_results.append(
                    CalibrationResult(
                        num_marker_seen=len(obs_data_std),
                        stage2_retry=stage2_retry_count,
                        param=param_star,
                        pixel_error=pixel_error,
                        proj_func=PROJ_FUNC,
                    )
                )
                print("Maximum stage2 retry execeeded, bailing out")
                break

            marker_max_displacement = 0.1  # meter
            param = (
                torch.cat(
                    [rvec_cam, tvec_cam, torch.randn(3) * marker_max_displacement]
                )
                .clone()
                .detach()
            )
            param.requires_grad = True
            L = lambda param: c_utils.mean_loss(
                obs_data_std, param, K, PROJ_FUNC_DICT[PROJ_FUNC]
            )
            try:
                param_star = c_utils.find_parameter(param, L)
            except Exception as e:
                print(e)
                continue

            pixel_error = L(param_star).item()
            print("stage 2 mean pixel error", pixel_error)
            if pixel_error > pixel_tolerance:
                print(
                    f"Try again {stage2_retry_count}/{max_stage2_retry} because of poor solution {pixel_error} > {pixel_tolerance}"
                )
            else:
                print(f"Good solution {pixel_error} <= {pixel_tolerance}")
                cal_results.append(
                    CalibrationResult(
                        num_marker_seen=len(obs_data_std),
                        stage2_retry=stage2_retry_count,
                        param=param_star,
                        pixel_error=pixel_error,
                        proj_func=PROJ_FUNC,
                    )
                )
                break

    with torch.no_grad():
        param_list = []
        for i, cal in enumerate(cal_results):
            result = cal._asdict().copy()
            result.update({"intrinsics": intrinsics[i]})
            del result["param"]  # pytorch vector
            if cal.param is not None:
                if cal.proj_func == "world_marker_proj_hand_camera":
                    camera_ee_ori_rotvec = cal.param[:3]
                    camera_ee_ori = c_utils.rotmat(camera_ee_ori_rotvec)
                    result.update(
                        {
                            "camera_ee_ori": camera_ee_ori.numpy().tolist(),
                            "camera_ee_ori_rotvec": camera_ee_ori_rotvec.numpy().tolist(),
                            "camera_ee_pos": cal.param[3:6].numpy().tolist(),
                            "marker_base_pos": cal.param[6:9].numpy().tolist(),
                        }
                    )
                elif cal.proj_func == "hand_marker_proj_world_camera":
                    camera_base_ori_rotvec = cal.param[:3]
                    camera_base_ori = c_utils.rotmat(camera_base_ori_rotvec)
                    camera_base_quat = R.from_matrix(
                        camera_base_ori.cpu().numpy()
                    ).as_quat()
                    result.update(
                        {
                            "camera_base_ori": camera_base_ori.cpu().numpy().tolist(),
                            "camera_base_ori_rotvec": camera_base_ori_rotvec.cpu()
                            .numpy()
                            .tolist(),
                            "camera_base_pos": cal.param[3:6].cpu().numpy().tolist(),
                            "p_marker_ee": cal.param[6:9].cpu().numpy().tolist(),
                            "camera_base_quat": camera_base_quat.tolist(),
                        }
                    )
                else:
                    raise ArgumentError("shouldn't reach here")

            param_list.append(result)
            print(f"Camera {i} calibration: {result}")

        calibration_file = os.path.join(save_dir, "cam_cal.json")
        with open(calibration_file, "w") as f:
            print(f"Saving calibrated parameters to {calibration_file}")
            json.dump(param_list, f, indent=4)

        # Re-projection error
        plt.figure()
        marker_proj = PROJ_FUNC_DICT[PROJ_FUNC]
        for camera_index in range(num_of_camera):
            ax = plt.subplot(1, 1, camera_index + 1)
            obs_data_std, K = extract_obs_data_std(corner_data, camera_index)
            err_list = []
            for obs_marker, pos_ee_base, ori_ee_base in obs_data_std:
                with torch.no_grad():
                    proj_marker = marker_proj(
                        cal_results[camera_index].param, pos_ee_base, ori_ee_base, K
                    )

                err = (proj_marker - obs_marker).norm()
                err_list.append(err)
                plt.plot(
                    (obs_marker[0], proj_marker[0]),
                    (obs_marker[1], proj_marker[1]),
                    "-",
                )
                plt.plot((obs_marker[0]), (obs_marker[1]), ".")
            ax.set(xlim=(0, 1280), ylim=(720, 0))
            ax.set_aspect("equal", "box")
            errs = torch.stack(err_list)
            ax.set_title(
                f"cam{camera_index} reproj_err\nmean:{errs.mean():.2f} max:{errs.max():.2} px"
            )
            print(
                f"cam{camera_index} reproj_err\nmean:{errs.mean():.2f} max:{errs.max():.2} px"
            )
        fig_path = os.path.join(save_dir, "cam_cal_error.png")
        plt.savefig(fig_path)


def main(args):
    with open(args.cal_pkl, "rb") as f:
        data_list = pickle.load(f)

    if args.save_imgs:
        save_dir = os.path.join(os.path.dirname(args.cal_pkl), "calibration_imgs")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = None

    # fig, axs = plt.subplots(1, 4)

    # for idx in range(4):
    #     axs[idx].imshow(data_list[idx]["imgs"][0])
    # plt.show()
    # import pdb; pdb.set_trace()

    # Get list of data with corner data
    corner_data = get_data_with_corners(data_list, target_idx=0, save_dir=save_dir)

    # Compute calibration from data with corners
    cal_save_dir = os.path.dirname(args.cal_pkl)
    compute_cal(corner_data, args.pixel_tolerance, cal_save_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_imgs", "-s", action="store_true", help="Use flag to save imgs"
    )
    parser.add_argument("cal_pkl", help="Path of calibration pickle file to load")

    parser.add_argument(
        "--seed", default=0, type=int, help="random seed for initializing solution"
    )
    parser.add_argument(
        "--pixel-tolerance",
        default=2.5,
        type=float,
        help="mean pixel error tolerance (stage 2)",
    )
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    main(args)
