from dataclasses import dataclass
from enum import Enum, auto

import mujoco
import numpy as np

from src.controllers.gravity_compensation import (
    get_joint_dof_ids,
    apply_bias_compensation,
)


class ClassicalPhase(Enum):
    """
    High-level phases of the classical cooperative
    manipulation controller.
    """

    APPROACH = auto()
    GRASP = auto()
    LIFT = auto()
    TRANSPORT = auto()
    PLACE = auto()
    RELEASE = auto()
    RETREAT = auto()
    SUCCESS = auto()
    FAILURE = auto()


@dataclass
class ClassicalControllerConfig:
    """
    Configuration shared by the validated classical
    cooperative manipulation controller.
    """

    open_gripper_ctrl: float = 255.0
    close_gripper_ctrl: float = 0.0

    left_joint_prefix: str = "left_joint"
    right_joint_prefix: str = "right_joint"

    left_actuator_prefix: str = "left_actuator"
    right_actuator_prefix: str = "right_actuator"

    left_gripper_actuator_name: str = "left_actuator8"
    right_gripper_actuator_name: str = "right_actuator8"


class ClassicalController:
    """
    Low-level reusable controller for the cooperative
    dual-Panda classical baseline.

    This class intentionally contains only validated,
    generic low-level control functionality.

    High-level task sequencing will be added separately
    after the core controller has been validated.
    """

    def __init__(
        self,
        model,
        data,
        config=None,
    ):
        self.model = model
        self.data = data

        if config is None:
            config = ClassicalControllerConfig()

        self.config = config

        self.phase = ClassicalPhase.APPROACH


        # ----------------------------------------------------
        # Arm joint names
        # ----------------------------------------------------

        self.left_joint_names = [
            f"{self.config.left_joint_prefix}{i}"
            for i in range(1, 8)
        ]

        self.right_joint_names = [
            f"{self.config.right_joint_prefix}{i}"
            for i in range(1, 8)
        ]


        # ----------------------------------------------------
        # QPOS addresses
        # ----------------------------------------------------

        self.left_qpos_ids = self._get_qpos_ids(
            self.left_joint_names
        )

        self.right_qpos_ids = self._get_qpos_ids(
            self.right_joint_names
        )


        # ----------------------------------------------------
        # Actuator IDs
        #
        # Only the first seven actuators belong to the
        # Panda arm joints. actuator8 is the gripper.
        # ----------------------------------------------------

        self.left_arm_actuator_ids = (
            self._get_arm_actuator_ids(
                self.config.left_actuator_prefix
            )
        )

        self.right_arm_actuator_ids = (
            self._get_arm_actuator_ids(
                self.config.right_actuator_prefix
            )
        )


        # ----------------------------------------------------
        # Gripper actuator IDs
        # ----------------------------------------------------

        self.left_gripper_actuator_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.config.left_gripper_actuator_name,
            )
        )

        self.right_gripper_actuator_id = (
            mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                self.config.right_gripper_actuator_name,
            )
        )


        if self.left_gripper_actuator_id < 0:
            raise ValueError(
                "Left gripper actuator was not found."
            )

        if self.right_gripper_actuator_id < 0:
            raise ValueError(
                "Right gripper actuator was not found."
            )


        # ----------------------------------------------------
        # Generalized velocity DOFs used for MuJoCo
        # bias-force feed-forward compensation.
        # ----------------------------------------------------

        self.left_dof_ids = get_joint_dof_ids(
            self.model,
            self.left_arm_actuator_ids,
        )

        self.right_dof_ids = get_joint_dof_ids(
            self.model,
            self.right_arm_actuator_ids,
        )


    # ========================================================
    # Model lookup helpers
    # ========================================================

    def _get_qpos_ids(
        self,
        joint_names,
    ):
        qpos_ids = []

        for joint_name in joint_names:

            joint_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_JOINT,
                joint_name,
            )

            if joint_id < 0:
                raise ValueError(
                    f"Joint not found: {joint_name}"
                )

            qpos_ids.append(
                int(
                    self.model.jnt_qposadr[
                        joint_id
                    ]
                )
            )

        return qpos_ids


    def _get_arm_actuator_ids(
        self,
        prefix,
    ):
        actuator_ids = []

        for actuator_id in range(
            self.model.nu
        ):

            actuator_name = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_ACTUATOR,
                actuator_id,
            )

            if actuator_name is None:
                continue

            if not actuator_name.startswith(
                prefix
            ):
                continue

            # actuator8 is the Panda gripper.
            # The classical arm controller handles only
            # actuator1 ... actuator7 here.
            if actuator_name.endswith(
                "actuator8"
            ):
                continue

            actuator_ids.append(
                actuator_id
            )

        if len(actuator_ids) != 7:
            raise ValueError(
                f"Expected 7 arm actuators for prefix "
                f"'{prefix}', found {len(actuator_ids)}."
            )

        return actuator_ids


    # ========================================================
    # Joint-state helpers
    # ========================================================

    def set_arm_qpos(
        self,
        left_q,
        right_q,
    ):
        self._set_qpos(
            self.left_qpos_ids,
            left_q,
        )

        self._set_qpos(
            self.right_qpos_ids,
            right_q,
        )


    def _set_qpos(
        self,
        qpos_ids,
        q,
    ):
        if len(q) != 7:
            raise ValueError(
                "Expected 7 arm joint values."
            )

        for qpos_id, value in zip(
            qpos_ids,
            q,
        ):
            self.data.qpos[
                qpos_id
            ] = value


    def get_left_arm_qpos(
        self,
    ):
        return self._get_arm_qpos(
            self.left_qpos_ids
        )


    def get_right_arm_qpos(
        self,
    ):
        return self._get_arm_qpos(
            self.right_qpos_ids
        )


    def _get_arm_qpos(
        self,
        qpos_ids,
    ):
        return np.array([
            self.data.qpos[
                qpos_id
            ]
            for qpos_id in qpos_ids
        ])


    # ========================================================
    # Actuator commands
    # ========================================================

    def command_arms(
        self,
        left_q,
        right_q,
    ):
        self._command_arm(
            self.left_arm_actuator_ids,
            left_q,
        )

        self._command_arm(
            self.right_arm_actuator_ids,
            right_q,
        )


    def _command_arm(
        self,
        actuator_ids,
        q,
    ):
        if len(q) != 7:
            raise ValueError(
                "Expected 7 arm joint targets."
            )

        for actuator_id, target in zip(
            actuator_ids,
            q,
        ):
            self.data.ctrl[
                actuator_id
            ] = target


    def set_grippers(
        self,
        ctrl,
    ):
        self.data.ctrl[
            self.left_gripper_actuator_id
        ] = ctrl

        self.data.ctrl[
            self.right_gripper_actuator_id
        ] = ctrl


    def open_grippers(
        self,
    ):
        self.set_grippers(
            self.config.open_gripper_ctrl
        )


    def close_grippers(
        self,
    ):
        self.set_grippers(
            self.config.close_gripper_ctrl
        )


    # ========================================================
    # Bias compensation / physics stepping
    # ========================================================

    def apply_arm_bias_compensation(
        self,
    ):
        """
        Apply MuJoCo generalized bias-force feed-forward
        compensation to both Panda arms.

        This preserves the behaviour already validated in
        the reference classical manipulation script.
        """

        self.data.qfrc_applied[:] = 0.0

        apply_bias_compensation(
            self.model,
            self.data,
            self.left_dof_ids,
        )

        apply_bias_compensation(
            self.model,
            self.data,
            self.right_dof_ids,
        )


    def step(
        self,
        left_q,
        right_q,
    ):
        """
        Execute one MuJoCo physics step while maintaining
        the desired joint targets of both arms.
        """

        self.command_arms(
            left_q,
            right_q,
        )

        self.apply_arm_bias_compensation()

        mujoco.mj_step(
            self.model,
            self.data,
        )


    # ========================================================
    # Generic trajectory helpers
    # ========================================================

    @staticmethod
    def smoothstep(
        alpha,
    ):
        """
        Cubic smoothstep with zero velocity at both ends.
        """

        alpha = float(
            np.clip(
                alpha,
                0.0,
                1.0,
            )
        )

        return (
            alpha
            * alpha
            * (
                3.0
                - 2.0 * alpha
            )
        )


    @staticmethod
    def orientation_error_deg(
        target_rotation,
        achieved_rotation,
    ):
        """
        Return the magnitude of the relative orientation
        error between two 3x3 rotation matrices.
        """

        rotation_difference = (
            target_rotation.T
            @ achieved_rotation
        )

        cos_angle = np.clip(
            (
                np.trace(
                    rotation_difference
                )
                - 1.0
            )
            / 2.0,
            -1.0,
            1.0,
        )

        return float(
            np.degrees(
                np.arccos(
                    cos_angle
                )
            )
        )


    @staticmethod
    def interpolate_joint_targets(
        start_q,
        target_q,
        alpha,
    ):
        """
        Smoothly interpolate between two joint-space
        configurations.
        """

        interpolation_alpha = (
            ClassicalController.smoothstep(
                alpha
            )
        )

        return (
            start_q
            + interpolation_alpha
            * (
                target_q
                - start_q
            )
        )


    # ========================================================
    # Phase bookkeeping
    # ========================================================

    def set_phase(
        self,
        phase,
    ):
        if not isinstance(
            phase,
            ClassicalPhase,
        ):
            raise TypeError(
                "phase must be a ClassicalPhase."
            )

        self.phase = phase


    def get_phase(
        self,
    ):
        return self.phase
