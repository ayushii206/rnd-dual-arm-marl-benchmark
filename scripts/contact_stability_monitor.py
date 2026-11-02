"""Physics-step collision and stability diagnostics for dual-arm transport."""

from collections import Counter
import numpy as np
import mujoco


class ContactStabilityMonitor:
    def __init__(self, env):
        self.env = env
        self.physics_steps = 0
        self.contact_points = Counter()
        self.affected_steps = Counter()
        self.max_tilt_deg = 0.0
        self.total_tilt_deg = 0.0
        self.both_grasp_steps = 0
        self.max_orientation_deg = 0.0
        self.max_angular_speed = 0.0
        self.missing_left_steps = 0
        self.missing_right_steps = 0

    @staticmethod
    def group(geom):
        if geom < 3:
            return "environment"
        if geom < 84:
            return "left_robot"
        if geom < 165:
            return "right_robot"
        return "object"

    @staticmethod
    def contact_category(g1, g2):
        a, b = ContactStabilityMonitor.group(g1), ContactStabilityMonitor.group(g2)

        if "object" in (a, b):
            robot_geom = g1 if a != "object" else g2
            other = a if a != "object" else b

            if other == "environment":
                return "object_environment"

            if 68 <= robot_geom <= 83 or 149 <= robot_geom <= 164:
                return "intentional_finger_object"

            if robot_geom in (67, 148):
                return "palm_object"

            return "other_robot_object"

        if a == b and a in ("left_robot", "right_robot"):
            return "self_contact"

        if {a, b} == {"left_robot", "right_robot"}:
            return "inter_arm_contact"

        if "environment" in (a, b) and "robot" in (a + b):
            return "robot_environment_contact"

        return "other_contact"

    def record(self):
        env = self.env
        self.physics_steps += 1
        seen = set()

        for i in range(env.data.ncon):
            contact = env.data.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            category = self.contact_category(g1, g2)

            self.contact_points[category] += 1
            seen.add(category)

        for category in seen:
            self.affected_steps[category] += 1

        left, right = env.get_grasp_contacts()

        if left > 0 and right > 0:
            self.both_grasp_steps += 1

        if left == 0:
            self.missing_left_steps += 1
        if right == 0:
            self.missing_right_steps += 1

        _, q, _, angular_velocity = env.get_object_state()

        rotation = np.zeros(9, dtype=np.float64)
        mujoco.mju_quat2Mat(rotation, q)
        up = rotation.reshape(3, 3)[:, 2]

        tilt = np.degrees(
            np.arccos(np.clip(up[2], -1.0, 1.0))
        )
        orientation = np.degrees(
            2.0 * np.arccos(np.clip(abs(q[0]), 0.0, 1.0))
        )

        self.total_tilt_deg += float(tilt)
        self.max_tilt_deg = max(self.max_tilt_deg, float(tilt))
        self.max_orientation_deg = max(
            self.max_orientation_deg, float(orientation)
        )
        self.max_angular_speed = max(
            self.max_angular_speed,
            float(np.linalg.norm(angular_velocity))
        )

    def summary(self):
        return {
            "physics_steps": self.physics_steps,
            "contact_points": dict(self.contact_points),
            "affected_steps": dict(self.affected_steps),
            "max_tilt_deg": self.max_tilt_deg,
            "mean_tilt_deg": (
                self.total_tilt_deg / self.physics_steps
                if self.physics_steps else 0.0
            ),
            "both_grasp_contact_pct": (
                100.0 * self.both_grasp_steps / self.physics_steps
                if self.physics_steps else 0.0
            ),
            "max_orientation_deg": self.max_orientation_deg,
            "max_angular_speed_rad_s": self.max_angular_speed,
            "missing_left_steps": self.missing_left_steps,
            "missing_right_steps": self.missing_right_steps,
        }
