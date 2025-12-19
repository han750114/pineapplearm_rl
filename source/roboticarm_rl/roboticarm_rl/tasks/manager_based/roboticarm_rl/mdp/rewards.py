from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply  

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def distance_to_target_tcp_v2(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
    std: float = 1.0,
    tcp_offset: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> torch.Tensor:

    robot = env.scene[robot_cfg.name]
    target = env.scene[target_cfg.name]
    
    body_idx = robot.find_bodies(robot_cfg.body_names)[0]
    wrist_pos = robot.data.body_pos_w[:, body_idx[0]]
    wrist_quat = robot.data.body_quat_w[:, body_idx[0]]
    
    offset_vec = torch.tensor(tcp_offset, device=env.device).repeat(env.num_envs, 1)
    tcp_pos = wrist_pos + quat_apply(wrist_quat, offset_vec)

    target_pos = target.data.root_pos_w
    distance_sq = torch.sum(torch.square(target_pos - tcp_pos), dim=-1)
    
    raw_score = torch.exp(-distance_sq / (std**2))

    if env.common_step_counter % 100 == 0:
        mean_score = torch.mean(raw_score).item()
        has_nan = torch.isnan(raw_score).any().item()
        
        print(f"\n------")
        print(f"Env[0] 分數: {raw_score[0].item():.4f} (距離: {torch.sqrt(distance_sq[0]).item():.3f}m)")
        print(f"平均: {mean_score:.4f} ")
        print(f"是否有 NaN: {has_nan}")
        print(f"----------------------\n")
    return raw_score
