from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """計算目標物相對於機器人基座的位置向量"""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]

    # 取得機器人基座位置
    robot_root_pos = robot.data.root_pos_w
    # 取得目標物位置
    target_pos = target.data.root_pos_w

    # 回傳相對向量 (Target - Robot)
    return target_pos - robot_root_pos