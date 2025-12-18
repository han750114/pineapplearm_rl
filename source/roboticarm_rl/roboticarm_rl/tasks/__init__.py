# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for the extension."""

import gymnasium as gym

from isaaclab.envs import ManagerBasedRLEnv

# 1. 匯入環境設定 (Env Config)
# 注意這個路徑是根據你的檔案結構來的
from .manager_based.roboticarm_rl.roboticarm_rl_env_cfg import RoboticarmRlEnvCfg

# 2. 匯入訓練參數 (Agent Config)
# 我們預設模板裡應該有這個檔案，我們嘗試匯入它
from .manager_based.roboticarm_rl.agents.rsl_rl_ppo_cfg import RoboticarmRlPPORunnerCfg

##
# Register Gym environments.
##

# 這裡我們明確定義任務名稱叫做 "roboticarm_rl-v0"
gym.register(
    id="roboticarm_rl-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": RoboticarmRlEnvCfg,
        "rsl_rl_cfg_entry_point": RoboticarmRlPPORunnerCfg,
    },
)