from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg


if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

def reset_target_position(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
):
    asset: RigidObject = env.scene[asset_cfg.name]

    range_x_min, range_x_max = x_range
    rand_x = torch.rand(len(env_ids), device=env.device) * (range_x_max - range_x_min) + range_x_min
    
    range_y_min, range_y_max = y_range
    rand_y = torch.rand(len(env_ids), device=env.device) * (range_y_max - range_y_min) + range_y_min

    range_z_min, range_z_max = z_range
    rand_z = torch.rand(len(env_ids), device=env.device) * (range_z_max - range_z_min) + range_z_min

    pos = torch.stack([rand_x, rand_y, rand_z], dim=-1)

    pos += env.scene.env_origins[env_ids]

    default_root_state = asset.data.default_root_state[env_ids].clone()
    default_root_state[:, :3] = pos
    
    default_root_state[:, 7:] = 0.0 

    asset.write_root_state_to_sim(default_root_state, env_ids)
    
def reset_target_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
):
    asset: RigidObject = env.scene[asset_cfg.name]

    range_min, range_max = velocity_range
    
    rand_vel_x = (torch.rand(len(env_ids), device=env.device) * 2 - 1)
    rand_vel_y = (torch.rand(len(env_ids), device=env.device) * 2 - 1)
    rand_vel_z = (torch.rand(len(env_ids), device=env.device) * 2 - 1)
    
    velocities = torch.stack([rand_vel_x, rand_vel_y, rand_vel_z], dim=-1)
    
    speed = torch.rand(len(env_ids), device=env.device) * (range_max - range_min) + range_min
    velocities = torch.nn.functional.normalize(velocities, dim=-1) * speed.unsqueeze(-1)

    root_vel = asset.data.root_vel_w[env_ids].clone()
    root_vel[:, :3] = velocities
    
    asset.write_root_velocity_to_sim(root_vel, env_ids)
    

def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    
    root_pos = asset.data.root_pos_w
    if env.scene.env_origins is not None:
        env_origins = env.scene.env_origins
    else:
        env_origins = torch.zeros_like(root_pos)

    # 使用二範數 (L2 norm) 計算 3D 距離
    distance = torch.norm(root_pos - env_origins, dim=-1)

    return distance > threshold

def apply_boundary_constraint(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
):
    asset: RigidObject = env.scene[asset_cfg.name]

    root_pos_w = asset.data.root_pos_w[env_ids]
    root_vel_w = asset.data.root_vel_w[env_ids]

    if env.scene.env_origins is not None:
        env_origins = env.scene.env_origins[env_ids]
        local_pos = root_pos_w - env_origins
    else:
        env_origins = torch.zeros_like(root_pos_w)
        local_pos = root_pos_w

    out_x_high = local_pos[:, 0] > x_range[1]
    local_pos[out_x_high, 0] = x_range[1]        # 拉回邊界
    root_vel_w[out_x_high, 0] *= -1.0            # 速度反轉 (反彈)
    
    out_x_low = local_pos[:, 0] < x_range[0]
    local_pos[out_x_low, 0] = x_range[0]
    root_vel_w[out_x_low, 0] *= -1.0

    out_y_high = local_pos[:, 1] > y_range[1]
    local_pos[out_y_high, 1] = y_range[1]
    root_vel_w[out_y_high, 1] *= -1.0
    
    out_y_low = local_pos[:, 1] < y_range[0]
    local_pos[out_y_low, 1] = y_range[0]
    root_vel_w[out_y_low, 1] *= -1.0

    out_z_high = local_pos[:, 2] > z_range[1]
    local_pos[out_z_high, 2] = z_range[1]
    root_vel_w[out_z_high, 2] *= -1.0

    out_z_low = local_pos[:, 2] < z_range[0]
    local_pos[out_z_low, 2] = z_range[0]
    root_vel_w[out_z_low, 2] *= -1.0 

    asset.write_root_pose_to_sim(torch.cat([local_pos + env_origins, asset.data.root_quat_w[env_ids]], dim=-1), env_ids)
    asset.write_root_velocity_to_sim(root_vel_w, env_ids)