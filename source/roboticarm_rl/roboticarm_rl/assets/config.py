import os
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg

CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
USD_PATH = os.path.join(CURRENT_DIR, "usd", "robot.usd")

MY_ROBOT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        # pos=(-94.7, 94.5, 0.0), 
        joint_pos={".*": 0.0},
    ),
    actuators={
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["joint.*"],
            
            stiffness=80.0, 
            
            damping=10.0,
            
            effort_limit=100.0,  
            
            velocity_limit=5.0, 
        ),
    },
)