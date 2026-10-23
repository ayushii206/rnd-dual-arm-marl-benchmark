import argparse
import csv
from pathlib import Path

import numpy as np

from src.envs.dual_arm_transport_env import DualArmTransportEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--kp", type=float, default=10.0)
    parser.add_argument("--max-action", type=float, default=0.5)
    args = parser.parse_args()

    if not np.isfinite(args.kp) or args.kp <= 0:
        raise ValueError("kp must be finite and positive")
    if not 0 < args.max_action <= 1:
        raise ValueError("max-action must be in (0, 1]")

    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"Output already exists: {output}")

    env = DualArmTransportEnv(
        target_position=np.array([0.05, 0.0, 0.58]),
        success_distance=0.01,
        action_mode="cartesian",
        progress_weight=100.0,
    )

    env.reset()
    initial_position = env.get_object_state()[0].copy()
    records = []
    episode_return = 0.0

    for step in range(1, 501):
        position = env.get_object_state()[0].copy()
        error = env.target_position - position
        action = np.clip(
            args.kp * error, -args.max_action, args.max_action
        )

        actions = {
            "left": action.copy(),
            "right": action.copy(),
        }

        _, rewards, terminated, truncated, info = env.step(actions)
        episode_return += float(rewards["left"])
        left_contacts, right_contacts = env.get_grasp_contacts()
        current_position = env.get_object_state()[0].copy()

        records.append({
            "step": step,
            "distance_mm": float(info["distance_to_target"]) * 1000,
            "forward_displacement_mm": float(
                current_position[0] - initial_position[0]
            ) * 1000,
            "orientation_error_deg": float(
                np.degrees(info["orientation_error_rad"])
            ),
            "linear_speed_m_s": float(info["linear_speed"]),
            "angular_speed_rad_s": float(info["angular_speed"]),
            "left_contacts": int(left_contacts),
            "right_contacts": int(right_contacts),
            "success": bool(info["success"]),
            "dropped": bool(info["dropped"]),
            "truncated": bool(truncated),
            "episode_return": episode_return,
        })

        if terminated or truncated:
            break

    output.parent.mkdir(parents=True, exist_ok=True)

    # Exclusive creation prevents overwriting existing results.
    with output.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=records[0].keys())
        writer.writeheader()
        writer.writerows(records)

    final = records[-1]
    print("CLASSICAL CARTESIAN FEEDBACK")
    print("Steps:", final["step"])
    print("Success:", final["success"])
    print("Dropped:", final["dropped"])
    print("Final distance (mm):", final["distance_mm"])
    print("Final orientation error (deg):",
          final["orientation_error_deg"])
    print("Saved:", output)


if __name__ == "__main__":
    main()
