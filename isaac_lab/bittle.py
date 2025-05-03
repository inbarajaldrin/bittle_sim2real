# /home/aaugus11/Projects/bittle_bot/src/Bittle_assets/bittle.usd
        # usd_path="omniverse://localhost/Library/ur5e_rg2_articulation.usd",

# import isaaclab.sim as sim_utils
# from isaaclab.actuators import DCMotorCfg
# from isaaclab.assets.articulation import ArticulationCfg
# from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

# BITTLE_CFG = ArticulationCfg(
#     spawn=sim_utils.UsdFileCfg(
#         usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/A1/a1.usd",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=1000.0,
#             max_angular_velocity=1000.0,
#             max_depenetration_velocity=1.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False,
#             solver_position_iteration_count=4,
#             solver_velocity_iteration_count=0
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.42),
#         joint_pos={
#             ".*L_hip_joint": 0.1,
#             ".*R_hip_joint": -0.1,
#             "F[L,R]_thigh_joint": 0.8,
#             "R[L,R]_thigh_joint": 1.0,
#             ".*_calf_joint": -1.5,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "base_legs": DCMotorCfg(
#             joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
#             effort_limit=33.5,
#             saturation_effort=33.5,
#             velocity_limit=21.0,
#             stiffness=25.0,
#             damping=0.5,
#             friction=0.0,
#         ),
#     },
# )
# """Configuration of Unitree A1 using DC motor.

# Note: Specifications taken from: https://www.trossenrobotics.com/a1-quadruped#specifications
# """

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

BITTLE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="omniverse://localhost/Library/bittle_edit.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.7),
            joint_pos={
                "right_front_shoulder_joint": -0.5236,
                "right_front_knee_joint": -0.5236,
                "right_back_shoulder_joint": -0.5236,
                "right_back_knee_joint": -0.5236,
                "left_front_shoulder_joint": 0.5236,
                "left_front_knee_joint": 0.5236,
                "left_back_shoulder_joint": 0.5236,
                "left_back_knee_joint": 0.5236,
            },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "base_legs": DCMotorCfg(
            joint_names_expr=[
                "right_front_shoulder_joint",
                "right_front_knee_joint",
                "right_back_shoulder_joint",
                "right_back_knee_joint",
                "left_front_shoulder_joint",
                "left_front_knee_joint",
                "left_back_shoulder_joint",
                "left_back_knee_joint",
            ],
            effort_limit=33.5,
            saturation_effort=33.5,
            velocity_limit=21.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Bittle using DC motor.

"""
