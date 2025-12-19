import math
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.envs import mdp
import roboticarm_rl.tasks.manager_based.roboticarm_rl.mdp as self_mdp

from roboticarm_rl.assets.config import MY_ROBOT_CFG


@configclass
class RoboticarmRlSceneCfg(InteractiveSceneCfg):
    """Configuration for the robotic arm reach scene."""

    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            size=(100.0, 100.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.1,  # 靜摩擦 (越低越容易被推動)
                dynamic_friction=0.1, # 動摩擦 (越低滑越遠)
                restitution=0.5,      # 彈性 (撞到牆會彈回來)
            ),
        ),
    )
    # # 地板
    # ground = AssetBaseCfg(
    #     prim_path="/World/ground",
    #     spawn=sim_utils.GroundPlaneCfg(size=(100.0, 100.0)),
    # )

    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(color=(0.9, 0.9, 0.9), intensity=500.0),
    )

    robot: ArticulationCfg = MY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    # target = RigidObjectCfg(
    #     prim_path="{ENV_REGEX_NS}/Target",
    #     spawn=sim_utils.SphereCfg(
    #         radius=0.03,
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(
    #             disable_gravity=True, 
    #             kinematic_enabled=True,
    #         ),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #     ),
    #     # 初始化設定
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos=(0.3, 0.0, 0.3), # 預設放在手臂前面
    #     ),
    # )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.SphereCfg(
            radius=0.03,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, 
                kinematic_enabled=False,
                
                # (Air Resistance)
                linear_damping=1.0, 
                angular_damping=0.5,
            ),
            

            mass_props=sim_utils.MassPropertiesCfg(mass=1.0), 

            # collision_props=sim_utils.CollisionPropertiesCfg(),
            
            # physics_material=sim_utils.RigidBodyMaterialCfg(
            #     static_friction=0.0,  
            #     dynamic_friction=0.0, 
            #     restitution=0.0,      
            # ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.1, 0.0, 0.1),
        ),
    )

@configclass
class ActionsCfg:
    """Action specifications for the MDP."""
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot", 
        joint_names=["joint.*"], 
        scale=2.0, 
        use_default_offset=True
    )

@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""
        
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)

        target_pos = ObsTerm(func=self_mdp.object_position_in_robot_root_frame)

        def __post_init__(self) -> None:
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""
    # reset_target = EventTerm(
    #     func=self_mdp.reset_target_position,
    #     mode="reset",
    #     params={
    #         "x_range": (0.3, 0.6),
            
    #         "y_range": (-0.4, 0.4),
            
    #         "z_range": (0.1, 0.45),

    #         "asset_cfg": SceneEntityCfg("target"),
    #     },
    # )
    
    reset_target = EventTerm(
        func=self_mdp.reset_target_position,
        mode="reset",
        params={
            "x_range": (0.45, 0.55),
            "y_range": (-0.05, 0.05),
            "z_range": (0.35, 0.45),
            "asset_cfg": SceneEntityCfg("target"),
        },
    )
    # reset_ball_velocity_start = EventTerm(
    #     func=self_mdp.reset_target_velocity,
    #     mode="reset",
    #     params={
    #         "velocity_range": (0.3, 0.8), 
    #         "asset_cfg": SceneEntityCfg("target"),
    #     },
    # )

    # change_ball_direction = EventTerm(
    #     func=self_mdp.reset_target_velocity, 
        
    #     mode="interval", 

    #     interval_range_s=(0.3, 0.8), 
        
    #     params={
    #         "velocity_range": (0.3, 0.8), 
    #         "asset_cfg": SceneEntityCfg("target"),
    #     },
    # )
    # enforce_ball_bounds = EventTerm(
    #     func=self_mdp.apply_boundary_constraint,
    #     mode="interval",
    #     interval_range_s=(0.05, 0.05),
    #     params={
    #         "x_range": (0.3, 0.65), 
    #         "y_range": (-0.4, 0.4),
    #         "z_range": (0.05, 0.5),
            
    #         "asset_cfg": SceneEntityCfg("target"),
    #     },
    # )
    
    move_target_circle = EventTerm(
        func=self_mdp.update_target_circular_motion,
        mode="interval", 
        interval_range_s=(0.02, 0.02),
        params={
            "radius": 0.16,          # 半徑
            "speed": 2.0,            # 角速度 ( rad/s)
            "center_pos": (0.5, 0.0, 0.4), # 圓心位置
            "asset_cfg": SceneEntityCfg("target"),
        },
    )

@configclass
class RewardsCfg:
    """Reward terms for the MDP."""
    
    track_target = RewTerm(
        func=self_mdp.distance_to_target_tcp_v2,
        weight=5.0, 
        params={
            "robot_cfg": SceneEntityCfg("robot", body_names=["link7"]), 
            "target_cfg": SceneEntityCfg("target"),
            "std": 0.5, 
            "tcp_offset": (0.084, 0.0, 0.0),
        },
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    
    # joint_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0)

    # object_is_reached = RewTerm(
    #    func=mdp.object_reached, 
    #    weight=10.0,
    #    params={"threshold": 0.05}
    # )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    # joint_limits = DoneTerm(func=mdp.joint_limits_rel, params={"threshold": 0.95})
    object_out_of_bounds = DoneTerm(
        func=self_mdp.object_out_of_bounds,
        params={
            "asset_cfg": SceneEntityCfg("target"),
            "threshold": 2.0, 
        },
        time_out=True, 
    )

@configclass
class RoboticarmRlEnvCfg(ManagerBasedRLEnvCfg):
    # Scene settings
    scene: RoboticarmRlSceneCfg = RoboticarmRlSceneCfg(num_envs=4096, env_spacing=3.0)
    
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()
    
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    # Post initialization
    def __post_init__(self) -> None:
        """Post initialization."""
        self.decimation = 2  #(Control Frequency)
        self.episode_length_s = 5.0 # 每個回合 5 秒
        
        self.viewer.eye = (2.0, 2.0, 2.0)
        self.viewer.lookat = (0.0, 0.0, 0.0)
        
        self.sim.dt = 0.01 # 100Hz