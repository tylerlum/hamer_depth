# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# From: https://github.com/facebookresearch/fairo/blob/main/perception/sandbox/eyehandcal/src/eyehandcal/utils.py

import torch
import cv2
import math
import numpy as np
import pdb


def detect_corners(data, target_idx=0):
    """
    data: [{'img': [np.ndarray]}]
    return: [{'corners', [(x,y)]}]
    """

    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    aruco_param = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_param)

    aruco_param.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    for i, d in enumerate(data):
        d["corners"] = []
        for j, img in enumerate(d["imgs"]):
            result = detector.detectMarkers(img.astype(np.uint8))
            corners, idx, rej = result
            if idx is not None and target_idx in idx:
                corner_i = idx.squeeze(axis=1).tolist().index(target_idx)
                target_corner = corners[corner_i][
                    0, 0, :
                ].tolist()  # Get top right corner
                d["corners"].append(target_corner)
            else:
                d["corners"].append(None)
    return data


def rotmat(v):
    assert len(v) == 3
    v_ss = skewsym(v)
    return torch.matrix_exp(v_ss)


def skewsym(v):
    """
    pytorch backwark() compatible
    """
    zero = torch.tensor([0.0])
    return torch.stack(
        [zero, -v[2:3], v[1:2], v[2:3], zero, -v[0:1], -v[1:2], v[0:1], zero]
    ).reshape(3, 3)


def build_proj_matrix(fx, fy, ppx, ppy, coeff=None):
    # consider handle distortion here
    return torch.DoubleTensor([[fx, 0.0, ppx], [0.0, fy, ppy], [0.0, 0.0, 1.0]])


def hand_marker_proj_world_camera(param, pos_ee_base, ori_ee_base, K):
    camera_base_ori = param[:3]
    camera_base_pos = param[3:6]
    p_marker_ee = param[6:9]
    p_marker_camera = rotmat(-camera_base_ori).matmul(
        (rotmat(ori_ee_base).matmul(p_marker_ee) + pos_ee_base) - camera_base_pos
    )
    p_marker_image = K.matmul(p_marker_camera)
    return p_marker_image[:2] / p_marker_image[2]


def world_marker_proj_hand_camera(param, pos_ee_base, ori_ee_base, K):
    ori_camera_ee = param[:3]
    pos_camera_ee = param[3:6]
    pos_marker_base = param[6:9]
    pos_marker_camera = rotmat(-ori_camera_ee).matmul(
        (rotmat(-ori_ee_base).matmul(pos_marker_base - pos_ee_base) - pos_camera_ee)
    )
    pos_marker_image = K.matmul(pos_marker_camera)
    return pos_marker_image[:2] / pos_marker_image[2]


def pointloss(param, obs_marker_2d, pos_ee_base, ori_ee_base, K, proj_func):
    proj_marker_2d = proj_func(param, pos_ee_base, ori_ee_base, K)
    return (obs_marker_2d - proj_marker_2d).norm()


def mean_loss(data, param, K, proj_func=hand_marker_proj_world_camera):
    losses = []
    for d in data:
        corner = d[0]
        ee_base_pos = d[1]
        ee_base_ori = d[2]
        ploss = pointloss(param, corner, ee_base_pos, ee_base_ori, K, proj_func)
        losses.append(ploss)
    return torch.stack(losses).mean()


def find_parameter(param, L):
    optimizer = torch.optim.LBFGS(
        [param], max_iter=1000, lr=1, line_search_fn="strong_wolfe"
    )

    def closure():
        optimizer.zero_grad()
        loss = L(param)
        loss.backward()
        return loss

    optimizer.step(closure)
    return param.detach()