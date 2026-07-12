import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from RobotGPT.utils.asset_root_path import ROBOTGPT_ASSETS_PATH


##
# Taken from pal_mjlab - START
##

##
# Actuator Parameters (BeyondMimic methodology)
##

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0
FACTOR = 0.05


def _calc_actuator_params(
  gear_ratio: float, motor_inertia: float, effort: float
) -> dict:
  """Calculate armature, stiffness, and damping for an actuator."""
  armature = FACTOR * motor_inertia * gear_ratio**2
  stiffness = round(armature * NATURAL_FREQ**2, 3)
  damping = round(2.0 * DAMPING_RATIO * armature * NATURAL_FREQ, 3)
  return {
    "armature": armature,
    "stiffness": stiffness,
    "damping": damping,
    "effort_limit": effort,
  }


# Motor parameters: (gear_ratio, motor_inertia, effort_limit)
S_PLUS = _calc_actuator_params(121, 1.728e-5, 50)
S_MINUS = _calc_actuator_params(101, 1.3e-5, 25)
XS = _calc_actuator_params(101, 1.3e-5, 25)
TORSO = {"armature": 0.1, "stiffness": 1500.0, "damping": 300.0, "effort_limit": 2200.0}

##
# Taken from pal_mjlab - END
##


#usd_path=f"{ROBOTGPT_ASSETS_PATH}/pal_tiago_pro/usd_conversion/tiago_pro/tiago_pro.usda",


TIAGO_PRO_CFG = ArticulationCfg(
    spawn=sim_utils.MjcfFileCfg(
        asset_path=f"{ROBOTGPT_ASSETS_PATH}/pal_tiago_pro/pal_mjlab/pal_tiago_pro/xmls/tiago_pro.xml",
        usd_dir=f"{ROBOTGPT_ASSETS_PATH}/pal_tiago_pro/usd_conversion",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        fix_base=True,
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # "torso_lift_joint": 0.1,
            "arm_left_1_joint": 0.3578,
            "arm_right_1_joint": -0.3578,
            "arm_left_2_joint": -1.8266,
            "arm_right_2_joint": -1.8266,
            "arm_left_3_joint": 0.4698,
            "arm_right_3_joint": -0.4698,
            "arm_left_4_joint": -2.3409,
            "arm_right_4_joint": -2.3409,
            "arm_left_6_joint": -1.2006,
            "arm_right_6_joint": -1.2006,
        },
    ),
    actuators={
        "joints_12": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_[1-2]_joint"],
            effort_limit_sim=S_PLUS["effort_limit"],
            stiffness=S_PLUS["stiffness"],
            damping=S_PLUS["damping"],
            armature=S_PLUS["armature"],
        ),
        "joints_345": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_[3-5]_joint"],
            effort_limit_sim=S_MINUS["effort_limit"],
            stiffness=S_MINUS["stiffness"],
            damping=S_MINUS["damping"],
            armature=S_MINUS["armature"],
        ),
        "joints_67": ImplicitActuatorCfg(
            joint_names_expr=["arm_.*_[6-7]_joint"],
            effort_limit_sim=XS["effort_limit"],
            stiffness=XS["stiffness"],
            damping=XS["damping"],
            armature=XS["armature"],
        ),
        "gripper_joints": ImplicitActuatorCfg( # XS actuator is an assumption
            joint_names_expr=["gripper_.*_outer_finger_.*_joint"],
            effort_limit_sim=XS["effort_limit"],
            stiffness=XS["stiffness"],
            damping=XS["damping"],
            armature=XS["armature"],
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Franka Emika Panda robot."""


# FRANKA_PANDA_HIGH_PD_CFG = FRANKA_PANDA_CFG.copy()
# FRANKA_PANDA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_shoulder"].damping = 80.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].stiffness = 400.0
# FRANKA_PANDA_HIGH_PD_CFG.actuators["panda_forearm"].damping = 80.0
# """Configuration of Franka Emika Panda robot with stiffer PD control.

# This configuration is useful for task-space control using differential IK.
# """