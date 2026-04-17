# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: CC-BY-NC-4.0

"""Wrist-camera depth + intrinsics + extrinsics observation terms for DROID.

Consumed by planners like TipTop that need point clouds / camera poses in
addition to RGB. Safe to include on envs whose policies only read RGB — the
extra terms are ignored.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass


def wrist_cam_depth(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Wrist camera depth image (distance to image plane)."""
    sensor = env.scene[sensor_cfg.name]
    try:
        return sensor.data.output["distance_to_image_plane"]
    except (RuntimeError, KeyError):
        h, w = sensor.cfg.height, sensor.cfg.width
        return torch.zeros((env.num_envs, h, w, 1), device=env.device, dtype=torch.float32)


def wrist_cam_intrinsics(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.intrinsic_matrices


def wrist_cam_pos_w(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.pos_w


def wrist_cam_quat_w(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_cam")
):
    """Wrist camera quaternion in world frame (ROS convention: w, x, y, z)."""
    sensor = env.scene[sensor_cfg.name]
    return sensor.data.quat_w_ros


@configclass
class CameraParamsObservationCfg(ObsGroup):
    wrist_depth = ObsTerm(func=wrist_cam_depth)
    wrist_intrinsics = ObsTerm(func=wrist_cam_intrinsics)
    wrist_cam_pos_w = ObsTerm(func=wrist_cam_pos_w)
    wrist_cam_quat_w = ObsTerm(func=wrist_cam_quat_w)

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False
