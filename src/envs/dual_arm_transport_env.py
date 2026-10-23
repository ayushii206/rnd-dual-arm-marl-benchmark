from pathlib import Path

import mujoco
import numpy as np

from src.controllers.classical_controller import ClassicalController
from src.controllers.ik_controller import solve_pose_ik
from src.controllers.jacobian_action import calculate_cartesian_action

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "assets"
    / "models"
    / "dual_panda"
    / "transport_scene.xml"
)


# ============================================================
# Validated cooperative task configuration
# ============================================================

INITIAL_ARM_Q = np.array([
    0.0,
    -0.4,
    0.0,
    -1.8,
    0.0,
    1.4,
    0.8,
])


LEFT_PREGRASP_POSITION = np.array([
    0.0,
    0.185,
    0.560,
])

RIGHT_PREGRASP_POSITION = np.array([
    0.0,
    -0.185,
    0.560,
])


LEFT_GRASP_POSITION = np.array([
    0.0,
    0.145,
    0.560,
])

RIGHT_GRASP_POSITION = np.array([
    0.0,
    -0.145,
    0.560,
])


LEFT_LIFT_POSITION = np.array([
    0.0,
    0.145,
    0.580,
])

RIGHT_LIFT_POSITION = np.array([
    0.0,
    -0.145,
    0.580,
])


# ============================================================
# MARL suspended-transport target
# ============================================================

TRANSPORT_TARGET_POSITION = np.array([
    0.380,
    0.0,
    0.580,
])

# ============================================================
# MARL action configuration
# ============================================================

ACTION_DIM_PER_AGENT = 7
CARTESIAN_ACTION_DIM = 3
CARTESIAN_ACTION_SCALE_M = 0.0005

ACTION_SCALE_RAD = 0.005

PHYSICS_STEPS_PER_ACTION = 10

# Episode configuration
MAX_EPISODE_STEPS = 500
SUCCESS_DISTANCE_M = 0.03
SUCCESS_LINEAR_SPEED_M_S = 0.05
SUCCESS_ANGULAR_SPEED_RAD_S = 0.20
SUCCESS_ORIENTATION_RAD = 0.20
DROP_HEIGHT_M = 0.48

# Reward weights
PROGRESS_WEIGHT = 50.0
DISTANCE_WEIGHT = 0.1
ORIENTATION_WEIGHT = 0.05
ANGULAR_VELOCITY_WEIGHT = 0.02
ACTION_WEIGHT = 0.001
GRASP_LOSS_PENALTY = 0.02
SUCCESS_BONUS = 20.0
DROP_PENALTY = 20.0

LEFT_EE_SITE_NAME = "left_ee_site"
RIGHT_EE_SITE_NAME = "right_ee_site"

OPEN_GRIPPER_CTRL = 255.0
CLOSE_GRIPPER_CTRL = 0.0


# ============================================================
# Validated initialization durations
# ============================================================

INITIAL_OPEN_HOLD_STEPS = 500

APPROACH_PREGRASP_STEPS = 2000
PREGRASP_HOLD_STEPS = 500

APPROACH_GRASP_STEPS = 1500
PRE_CLOSE_HOLD_STEPS = 1000

GRIPPER_CLOSE_STEPS = 2000
POST_GRASP_HOLD_STEPS = 1000

LIFT_STEPS = 2500
PRE_TRANSPORT_HOLD_STEPS = 1000


class DualArmTransportEnv:
    """
    MuJoCo environment for cooperative dual-arm object transport.

    The RL task will begin from a validated bilateral lifted
    grasp state.

    At this stage this class provides:
        - MuJoCo model/data ownership
        - raw simulation reset/stepping
        - deterministic construction of the validated lifted state

    RL observations, actions, rewards, and episode termination
    will be added separately.
    """

    def __init__(
        self,
        model_path=None,
        target_position=None,
        success_distance=None,
        action_mode="joint",
        progress_weight=PROGRESS_WEIGHT,
    ):
        if action_mode not in ("joint", "cartesian"):
            raise ValueError(
                "action_mode must be 'joint' or 'cartesian'."
            )

        self.action_mode = action_mode
        self.progress_weight = float(progress_weight)

        if (
            not np.isfinite(self.progress_weight)
            or self.progress_weight <= 0.0
        ):
            raise ValueError(
                "progress_weight must be finite and positive."
            )

        self.action_dim_per_agent = (
            ACTION_DIM_PER_AGENT
            if action_mode == "joint"
            else CARTESIAN_ACTION_DIM
        )

        if target_position is None:
            target_position = TRANSPORT_TARGET_POSITION

        self.target_position = np.asarray(
            target_position,
            dtype=np.float64,
        ).copy()

        if self.target_position.shape != (3,):
            raise ValueError(
                "target_position must have shape (3,)."
            )

        if not np.all(np.isfinite(self.target_position)):
            raise ValueError(
                "target_position must contain finite values."
            )

        if success_distance is None:
            success_distance = SUCCESS_DISTANCE_M

        self.success_distance = float(success_distance)

        if (
            not np.isfinite(self.success_distance)
            or self.success_distance <= 0.0
        ):
            raise ValueError(
                "success_distance must be finite and positive."
            )
        
        if model_path is None:
            model_path = DEFAULT_MODEL_PATH

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"MuJoCo model not found: {self.model_path}"
            )

        self.model = mujoco.MjModel.from_xml_path(
            str(self.model_path)
        )

        self.data = mujoco.MjData(self.model)

        self.controller = ClassicalController(
            self.model,
            self.data,
        )

        self.left_joint_names = (
            self.controller.left_joint_names
        )

        self.right_joint_names = (
            self.controller.right_joint_names
        )

        self.left_dof_ids = (
            self._get_joint_dof_ids(
                self.left_joint_names
            )
        )

        self.right_dof_ids = (
            self._get_joint_dof_ids(
                self.right_joint_names
            )
        )

        self.left_ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            LEFT_EE_SITE_NAME,
        )

        self.right_ee_site_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_SITE,
            RIGHT_EE_SITE_NAME,
        )

        self.object_body_id = mujoco.mj_name2id(
            self.model,
            mujoco.mjtObj.mjOBJ_BODY,
            "transport_object",
        )

        if self.left_ee_site_id < 0:
            raise RuntimeError(
                "Left end-effector site was not found."
            )

        if self.right_ee_site_id < 0:
            raise RuntimeError(
                "Right end-effector site was not found."
            )

        if self.object_body_id < 0:
            raise RuntimeError(
                "Transport object body was not found."
            )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        # ----------------------------------------------------
        # Validated RL initial-state cache
        # ----------------------------------------------------

        self._lifted_state_cache = None

        # Joint-position targets maintained between RL steps.
        self._left_action_target = None
        self._right_action_target = None
        self._episode_step_count = 0
        self._previous_target_distance = None
        self._episode_done = False

    @property
    def timestep(self):
        return float(
            self.model.opt.timestep
        )

    @property
    def simulation_time(self):
        return float(
            self.data.time
        )

    def reset_simulation(self):
        """
        Reset MuJoCo to the XML default state.

        This is a low-level simulation reset, not the final
        MARL episode reset.
        """

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def step_physics(self, num_steps=1):
        """
        Advance raw MuJoCo physics.

        This is not yet the MARL step(action) method.
        """

        if num_steps < 1:
            raise ValueError(
                "num_steps must be >= 1"
            )

        for _ in range(num_steps):
            mujoco.mj_step(
                self.model,
                self.data,
            )

    def _get_joint_dof_ids(self, joint_names):
        """
        Return the MuJoCo DOF addresses corresponding to
        the supplied one-DOF arm joints.
        """

        dof_ids = []

        for joint_name in joint_names:
            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:
                raise RuntimeError(
                    f"Joint was not found: {joint_name}"
                )

            dof_ids.append(
                int(
                    self.model.jnt_dofadr[
                        joint_id
                    ]
                )
            )

        return np.asarray(
            dof_ids,
            dtype=int,
        )

    def _get_nominal_ee_rotations(self):
        """
        Return the validated nominal Panda end-effector
        orientations used by the classical baseline.
        """

        self.controller.set_arm_qpos(
            INITIAL_ARM_Q,
            INITIAL_ARM_Q,
        )

        self.controller.open_grippers()

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        left_rotation = (
            self.data.site_xmat[
                self.left_ee_site_id
            ]
            .reshape(3, 3)
            .copy()
        )

        right_rotation = (
            self.data.site_xmat[
                self.right_ee_site_id
            ]
            .reshape(3, 3)
            .copy()
        )

        return (
            left_rotation,
            right_rotation,
        )

    def _solve_lift_initialization_ik(self):
        """
        Solve the pregrasp, grasp, and lift configurations
        required to construct the RL initial state.
        """

        (
            left_target_rotation,
            right_target_rotation,
        ) = self._get_nominal_ee_rotations()

        left_pregrasp_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.left_joint_names,
            site_name=LEFT_EE_SITE_NAME,
            target_position=LEFT_PREGRASP_POSITION,
            target_rotation=left_target_rotation,
            initial_q=INITIAL_ARM_Q,
        )

        right_pregrasp_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.right_joint_names,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=RIGHT_PREGRASP_POSITION,
            target_rotation=right_target_rotation,
            initial_q=INITIAL_ARM_Q,
        )

        left_grasp_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.left_joint_names,
            site_name=LEFT_EE_SITE_NAME,
            target_position=LEFT_GRASP_POSITION,
            target_rotation=left_target_rotation,
            initial_q=left_pregrasp_q,
        )

        right_grasp_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.right_joint_names,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=RIGHT_GRASP_POSITION,
            target_rotation=right_target_rotation,
            initial_q=right_pregrasp_q,
        )

        left_lift_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.left_joint_names,
            site_name=LEFT_EE_SITE_NAME,
            target_position=LEFT_LIFT_POSITION,
            target_rotation=left_target_rotation,
            initial_q=left_grasp_q,
        )

        right_lift_q, _, _ = solve_pose_ik(
            model=self.model,
            data=self.data,
            joint_names=self.right_joint_names,
            site_name=RIGHT_EE_SITE_NAME,
            target_position=RIGHT_LIFT_POSITION,
            target_rotation=right_target_rotation,
            initial_q=right_grasp_q,
        )

        return {
            "left_pregrasp_q": left_pregrasp_q.copy(),
            "right_pregrasp_q": right_pregrasp_q.copy(),
            "left_grasp_q": left_grasp_q.copy(),
            "right_grasp_q": right_grasp_q.copy(),
            "left_lift_q": left_lift_q.copy(),
            "right_lift_q": right_lift_q.copy(),
        }

    def reset_to_lifted_state(self):
        """
        Deterministically construct the validated bilateral
        grasp-and-lift state.

        This deliberately executes the full initialization
        sequence for correctness. A fast state-restoration
        reset will be added only after this version has been
        validated.
        """

        self.reset_simulation()

        trajectories = (
            self._solve_lift_initialization_ik()
        )

        # ----------------------------------------------------
        # Initial open configuration
        # ----------------------------------------------------

        self.controller.hold_pose(
            INITIAL_ARM_Q,
            INITIAL_ARM_Q,
            INITIAL_OPEN_HOLD_STEPS,
            gripper_ctrl=OPEN_GRIPPER_CTRL,
        )

        # ----------------------------------------------------
        # Approach pregrasp
        # ----------------------------------------------------

        self.controller.execute_joint_trajectory(
            INITIAL_ARM_Q,
            INITIAL_ARM_Q,
            trajectories["left_pregrasp_q"],
            trajectories["right_pregrasp_q"],
            APPROACH_PREGRASP_STEPS,
            gripper_ctrl=OPEN_GRIPPER_CTRL,
        )

        self.controller.hold_pose(
            trajectories["left_pregrasp_q"],
            trajectories["right_pregrasp_q"],
            PREGRASP_HOLD_STEPS,
            gripper_ctrl=OPEN_GRIPPER_CTRL,
        )

        # ----------------------------------------------------
        # Approach grasp
        # ----------------------------------------------------

        self.controller.execute_joint_trajectory(
            trajectories["left_pregrasp_q"],
            trajectories["right_pregrasp_q"],
            trajectories["left_grasp_q"],
            trajectories["right_grasp_q"],
            APPROACH_GRASP_STEPS,
            gripper_ctrl=OPEN_GRIPPER_CTRL,
        )

        self.controller.hold_pose(
            trajectories["left_grasp_q"],
            trajectories["right_grasp_q"],
            PRE_CLOSE_HOLD_STEPS,
            gripper_ctrl=OPEN_GRIPPER_CTRL,
        )

        # ----------------------------------------------------
        # Close grippers and establish bilateral grasp
        # ----------------------------------------------------

        self.controller.hold_pose(
            trajectories["left_grasp_q"],
            trajectories["right_grasp_q"],
            GRIPPER_CLOSE_STEPS,
            gripper_ctrl=CLOSE_GRIPPER_CTRL,
        )

        self.controller.hold_pose(
            trajectories["left_grasp_q"],
            trajectories["right_grasp_q"],
            POST_GRASP_HOLD_STEPS,
            gripper_ctrl=CLOSE_GRIPPER_CTRL,
        )

        # ----------------------------------------------------
        # Lift
        # ----------------------------------------------------

        self.controller.execute_joint_trajectory(
            trajectories["left_grasp_q"],
            trajectories["right_grasp_q"],
            trajectories["left_lift_q"],
            trajectories["right_lift_q"],
            LIFT_STEPS,
            gripper_ctrl=CLOSE_GRIPPER_CTRL,
        )

        # ----------------------------------------------------
        # Stabilize before handing control to RL
        # ----------------------------------------------------

        self.controller.hold_pose(
            trajectories["left_lift_q"],
            trajectories["right_lift_q"],
            PRE_TRANSPORT_HOLD_STEPS,
            gripper_ctrl=CLOSE_GRIPPER_CTRL,
        )

        mujoco.mj_forward(
            self.model,
            self.data,
        )

        return trajectories

    def _capture_lifted_state(self):
        """
        Capture the complete dynamic state required to restore
        the validated lifted cooperative grasp.

        The cache is kept in memory and is used to avoid
        replaying the 12,000-step classical initialization at
        every RL episode reset.
        """

        self._lifted_state_cache = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "ctrl": self.data.ctrl.copy(),
            "time": float(self.data.time),
        }

        if self.model.na > 0:
            self._lifted_state_cache[
                "act"
            ] = self.data.act.copy()
        else:
            self._lifted_state_cache[
                "act"
            ] = None

    def _restore_lifted_state(self):
        """
        Restore the previously captured validated lifted state.
        """

        if self._lifted_state_cache is None:
            raise RuntimeError(
                "Lifted-state cache has not been created."
            )

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[:] = (
            self._lifted_state_cache["qpos"]
        )

        self.data.qvel[:] = (
            self._lifted_state_cache["qvel"]
        )

        self.data.ctrl[:] = (
            self._lifted_state_cache["ctrl"]
        )

        if (
            self.model.na > 0
            and self._lifted_state_cache["act"]
            is not None
        ):
            self.data.act[:] = (
                self._lifted_state_cache["act"]
            )

        # RL episodes start at t = 0 even though the cached
        # state was generated after the classical initialization.
        self.data.time = 0.0

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def reset(self):
        """
        Reset the cooperative transport RL task.

        On the first call, construct the validated lifted state
        using the classical initialization and cache it.

        On subsequent calls, restore the cached state directly.

        Observations will be returned by this method after the
        observation interface is implemented.
        """

        if self._lifted_state_cache is None:
            self.reset_to_lifted_state()
            self._capture_lifted_state()

        self._restore_lifted_state()
        self._left_action_target = (
            self.controller.get_left_arm_qpos().copy()
        )

        self._right_action_target = (
            self.controller.get_right_arm_qpos().copy()
        )

        self._episode_step_count = 0
        self._episode_done = False

        self._previous_target_distance = float(
            np.linalg.norm(
                self.get_transport_target_error()
            )
        )

        return self.get_observations()

    def get_grasp_contacts(self):
        """
        Count contacts between each arm's finger geometries
        and the transported object.

        Geometry IDs are validated for the current MuJoCo model.
        These should later be resolved by geometry names.
        """
        left_finger_geoms = set(range(68, 84))
        right_finger_geoms = set(range(149, 165))
        object_geoms = set(range(165, 170))

        left_contacts = 0
        right_contacts = 0

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            pair = {
                int(contact.geom1),
                int(contact.geom2),
            }

            if pair & object_geoms and pair & left_finger_geoms:
                left_contacts += 1

            if pair & object_geoms and pair & right_finger_geoms:
                right_contacts += 1

        return left_contacts, right_contacts

    def get_object_state(self):
        """
        Return object pose and velocity information.

        Returns
        -------
        position : ndarray, shape (3,)
        quaternion : ndarray, shape (4,)
            MuJoCo quaternion ordering: [w, x, y, z].
        linear_velocity : ndarray, shape (3,)
        angular_velocity : ndarray, shape (3,)
        """

        position = (
            self.data.xpos[
                self.object_body_id
            ].copy()
        )

        quaternion = (
            self.data.xquat[
                self.object_body_id
            ].copy()
        )

        angular_velocity = (
            self.data.cvel[
                self.object_body_id,
                0:3,
            ].copy()
        )

        linear_velocity = (
            self.data.cvel[
                self.object_body_id,
                3:6,
            ].copy()
        )

        return (
            position,
            quaternion,
            linear_velocity,
            angular_velocity,
        )

    def get_transport_target_error(self):
        """
        Return the world-frame vector from the current object
        position to the suspended transport target.
        """

        (
            object_position,
            _,
            _,
            _,
        ) = self.get_object_state()

        return (
            self.target_position
            - object_position
        )
    
    def get_local_observation(self, agent):
        """
        Build the decentralized observation for one arm.

        Parameters
        ----------
        agent : str
            Either "left" or "right".

        Returns
        -------
        observation : ndarray, shape (33,)
        """

        if agent == "left":
            joint_q = (
                self.controller.get_left_arm_qpos()
            )

            joint_qvel = (
                self.data.qvel[
                    self.left_dof_ids
                ].copy()
            )

            ee_position = (
                self.data.site_xpos[
                    self.left_ee_site_id
                ].copy()
            )

        elif agent == "right":
            joint_q = (
                self.controller.get_right_arm_qpos()
            )

            joint_qvel = (
                self.data.qvel[
                    self.right_dof_ids
                ].copy()
            )

            ee_position = (
                self.data.site_xpos[
                    self.right_ee_site_id
                ].copy()
            )

        else:
            raise ValueError(
                "agent must be either 'left' or 'right'"
            )

        (
            object_position,
            object_quaternion,
            object_linear_velocity,
            object_angular_velocity,
        ) = self.get_object_state()

        target_error = (
            self.get_transport_target_error()
        )

        observation = np.concatenate([
            joint_q,
            joint_qvel,
            ee_position,
            object_position,
            object_quaternion,
            object_linear_velocity,
            object_angular_velocity,
            target_error,
        ]).astype(
            np.float32
        )

        if observation.shape != (33,):
            raise RuntimeError(
                "Unexpected local observation shape: "
                f"{observation.shape}"
            )

        return observation

    def get_observations(self):
        """
        Return decentralized observations for both agents.
        """

        return {
            "left": self.get_local_observation(
                "left"
            ),
            "right": self.get_local_observation(
                "right"
            ),
        }

    def get_global_state(self):
        """
        Build the centralized state used by the MAPPO critic.

        The actors do not receive this state.
        """

        left_q = (
            self.controller.get_left_arm_qpos()
        )

        right_q = (
            self.controller.get_right_arm_qpos()
        )

        left_qvel = (
            self.data.qvel[
                self.left_dof_ids
            ].copy()
        )

        right_qvel = (
            self.data.qvel[
                self.right_dof_ids
            ].copy()
        )

        left_ee_position = (
            self.data.site_xpos[
                self.left_ee_site_id
            ].copy()
        )

        right_ee_position = (
            self.data.site_xpos[
                self.right_ee_site_id
            ].copy()
        )

        (
            object_position,
            object_quaternion,
            object_linear_velocity,
            object_angular_velocity,
        ) = self.get_object_state()

        target_error = (
            self.get_transport_target_error()
        )

        global_state = np.concatenate([
            left_q,
            left_qvel,
            right_q,
            right_qvel,
            left_ee_position,
            right_ee_position,
            object_position,
            object_quaternion,
            object_linear_velocity,
            object_angular_velocity,
            target_error,
        ]).astype(
            np.float32
        )

        if global_state.shape != (50,):
            raise RuntimeError(
                "Unexpected global-state shape: "
                f"{global_state.shape}"
            )

        return global_state

    def apply_actions(self, actions):
        """
        Apply one simultaneous decentralized action from
        each Panda arm.

        Parameters
        ----------
        actions : dict
            {
                "left": array of shape (7,),
                "right": array of shape (7,)
            }

        Each action component is normalized to [-1, 1].

        This is the action/physics interface only.
        Reward and episode termination will be added later.
        """

        if (
            self._left_action_target is None
            or self._right_action_target is None
        ):
            raise RuntimeError(
                "Call env.reset() before applying RL actions."
            )

        if not isinstance(actions, dict):
            raise TypeError(
                "actions must be a dictionary."
            )

        if set(actions.keys()) != {"left", "right"}:
            raise ValueError(
                "actions must contain exactly "
                "'left' and 'right'."
            )

        processed_actions = {}

        for agent in ("left", "right"):
            action = np.asarray(
                actions[agent],
                dtype=np.float64,
            )

            if action.shape != (self.action_dim_per_agent,):
                raise ValueError(
                    f"{agent} action must have shape "
                    f"({self.action_dim_per_agent},)."
                )

            if not np.all(np.isfinite(action)):
                raise ValueError(
                    f"{agent} action contains NaN or infinity."
                )

            processed_actions[agent] = np.clip(
                action,
                -1.0,
                1.0,
            )
        if self.action_mode == "cartesian":
            cartesian_actions = processed_actions

            processed_actions = {
                "left": calculate_cartesian_action(
                    self,
                    self.left_ee_site_id,
                    self.left_dof_ids,
                    cartesian_actions["left"]
                    * CARTESIAN_ACTION_SCALE_M,
                ),
                "right": calculate_cartesian_action(
                    self,
                    self.right_ee_site_id,
                    self.right_dof_ids,
                    cartesian_actions["right"]
                    * CARTESIAN_ACTION_SCALE_M,
                ),
            }

        # Compute both new targets before changing either.
        new_left_target = (
            self._left_action_target
            + ACTION_SCALE_RAD
            * processed_actions["left"]
        )

        new_right_target = (
            self._right_action_target
            + ACTION_SCALE_RAD
            * processed_actions["right"]
        )

        # Respect the actuator's configured position-control
        # range. Both targets are updated symmetrically.
        left_ranges = self.model.actuator_ctrlrange[
            self.controller.left_arm_actuator_ids
        ]

        right_ranges = self.model.actuator_ctrlrange[
            self.controller.right_arm_actuator_ids
        ]

        new_left_target = np.clip(
            new_left_target,
            left_ranges[:, 0],
            left_ranges[:, 1],
        )

        new_right_target = np.clip(
            new_right_target,
            right_ranges[:, 0],
            right_ranges[:, 1],
        )

        self._left_action_target = new_left_target
        self._right_action_target = new_right_target

        # Keep both grippers closed throughout transport.
        self.controller.set_grippers(
            CLOSE_GRIPPER_CTRL
        )

        # Maintain the same low-level controller used by
        # the validated classical baseline.
        for _ in range(PHYSICS_STEPS_PER_ACTION):
            self.controller.step(
                self._left_action_target,
                self._right_action_target,
            )

        return self.get_observations()

    def _compute_transport_reward(self, actions):
        """
        Compute shared team reward and episode status.
        """

        (
            object_position,
            object_quaternion,
            linear_velocity,
            angular_velocity,
        ) = self.get_object_state()

        distance = float(
            np.linalg.norm(
                self.target_position - object_position
            )
        )

        # MuJoCo quaternion order is [w, x, y, z].
        # This measures deviation from the initial upright
        # world orientation, independent of quaternion sign.
        q_w = float(
            np.clip(
                abs(object_quaternion[0]),
                0.0,
                1.0,
            )
        )

        orientation_error = float(
            2.0 * np.arccos(q_w)
        )

        linear_speed = float(
            np.linalg.norm(linear_velocity)
        )

        angular_speed = float(
            np.linalg.norm(angular_velocity)
        )

        progress = (
            self._previous_target_distance - distance
        )

        action_cost = float(
            np.mean(
                np.concatenate([
                    np.asarray(actions["left"]),
                    np.asarray(actions["right"]),
                ]) ** 2
            )
        )

        reward = (
            self.progress_weight * progress
            - DISTANCE_WEIGHT * distance
            - ORIENTATION_WEIGHT * orientation_error
            - ANGULAR_VELOCITY_WEIGHT * angular_speed
            - ACTION_WEIGHT * action_cost
        )

        left_contacts, right_contacts = self.get_grasp_contacts()

        left_grasp_contact = left_contacts > 0
        right_grasp_contact = right_contacts > 0

        cooperative_grasp = (
            left_grasp_contact and right_grasp_contact
        )

        missing_grasp_count = (
            int(not left_grasp_contact)
            + int(not right_grasp_contact)
        )

        grasp_loss_penalty = (
            GRASP_LOSS_PENALTY * missing_grasp_count
        )
        reward -= grasp_loss_penalty

        success = (
            distance <= self.success_distance
            and linear_speed <= SUCCESS_LINEAR_SPEED_M_S
            and angular_speed <= SUCCESS_ANGULAR_SPEED_RAD_S
            and orientation_error <= SUCCESS_ORIENTATION_RAD
            and cooperative_grasp
        )

        dropped = (
            object_position[2] < DROP_HEIGHT_M
        )

        if success:
            reward += SUCCESS_BONUS

        if dropped:
            reward -= DROP_PENALTY

        terminated = bool(success or dropped)

        truncated = bool(
            self._episode_step_count >= MAX_EPISODE_STEPS
            and not terminated
        )

        self._previous_target_distance = distance

        info = {
            "distance_to_target": distance,
            "progress": float(progress),
            "orientation_error_rad": orientation_error,
            "linear_speed": linear_speed,
            "angular_speed": angular_speed,
            "success": bool(success),
            "dropped": bool(dropped),
            "episode_step": self._episode_step_count,
            "left_grasp_contact": bool(left_grasp_contact),
            "right_grasp_contact": bool(right_grasp_contact),
            "cooperative_grasp": bool(cooperative_grasp),
            "left_contact_count": int(left_contacts),
            "right_contact_count": int(right_contacts),
            "missing_grasp_count": int(missing_grasp_count),
            "grasp_loss_penalty": float(grasp_loss_penalty),
        }

        return (
            float(reward),
            terminated,
            truncated,
            info,
        )

    def step(self, actions):
        """
        Advance one multi-agent RL step.
        """

        if self._episode_done:
            raise RuntimeError(
                "Episode finished. Call reset() first."
            )

        if self._previous_target_distance is None:
            raise RuntimeError(
                "Call reset() before step()."
            )

        observations = self.apply_actions(actions)

        self._episode_step_count += 1

        (
            team_reward,
            terminated,
            truncated,
            info,
        ) = self._compute_transport_reward(actions)

        self._episode_done = terminated or truncated

        rewards = {
            "left": team_reward,
            "right": team_reward,
        }

        return (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        )