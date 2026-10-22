from pathlib import Path

import mujoco
import numpy as np

from src.controllers.classical_controller import ClassicalController
from src.controllers.ik_controller import solve_pose_ik


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

    def __init__(self, model_path=None):
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
            TRANSPORT_TARGET_POSITION
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