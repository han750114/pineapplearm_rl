# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# source/roboticarm_rl/roboticarm_rl/tasks/manager_based/roboticarm_rl/agents/rsl_rl_ppo_cfg.py

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

@configclass
class RoboticarmRlPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Configuration for the RSL-RL PPO agent."""
    
    # 1. 基礎設定
    num_steps_per_env = 24  # 每個環境採樣步數 (24 * 4096 envs = 每次更新的資料量)
    max_iterations = 1500   # 總訓練迭代次數
    save_interval = 50      # 每 50 次存檔一次
    experiment_name = "roboticarm_reach" # 實驗名稱 (會顯示在 Log 資料夾)
    run_name = ""           # 留空，系統會自動加時間戳記

    # 2. 策略網路 (Policy Network) 設定
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,  # 初始探索雜訊 (太小會動不了，太大會亂動)
        actor_obs_normalization=True, # 【重要】正規化觀測值，對手臂訓練很有幫助
        critic_obs_normalization=True,
        # 網路架構：加深加寬，因為手臂控制比倒單擺複雜得多
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[256, 128, 64],
        activation="elu",    # ELU 激活函數通常比 ReLU 表現更好
    )

    # 3. PPO 演算法參數
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,   # 熵係數：控制探索程度 (0.01 是標準值)
        num_learning_epochs=5,
        num_mini_batches=4,  # 將資料切成 4 份來更新
        learning_rate=1.0e-3, # 學習率
        schedule="adaptive", # 自適應學習率調整
        gamma=0.99,          # 折扣因子 (看重未來獎勵的程度)
        lam=0.95,
        desired_kl=0.01,     # 目標 KL 散度 (用於自適應調整)
        max_grad_norm=1.0,
    )