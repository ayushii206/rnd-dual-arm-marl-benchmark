"""Record one deterministic MAPPO transport trajectory."""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.spatial.transform import Rotation

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.mappo_networks import MAPPOActor


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--output",
        default="results/diagnostics/mappo_r4_deterministic_trace.csv",
    )
    parser.add_argument(
        "--stochastic",
        action="store_true",
        help="Sample actions instead of using the policy mean.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for action sampling.",
    )
    args = parser.parse_args()


    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing diagnostic: {output_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint["algorithm"] != "MAPPO":
        raise ValueError("Checkpoint is not MAPPO.")

    if checkpoint["action_mode"] != "cartesian":
        raise ValueError("Expected Cartesian action mode.")

    if checkpoint["action_dim"] != 3:
        raise ValueError("Expected 3D Cartesian actions.")

    target_position = np.asarray(
        checkpoint["target_position"], dtype=float
    )

    env = DualArmTransportEnv(
        target_position=target_position,
        success_distance=checkpoint["success_distance"],
        action_mode=checkpoint["action_mode"],
    )

    observations = env.reset()
    initial_position = env.get_object_state()[0].copy()
    initial_quaternion = env.get_object_state()[1].copy()

    def to_rotation(quaternion):
        # MuJoCo [w, x, y, z] -> SciPy [x, y, z, w]
        return Rotation.from_quat(
            np.asarray(quaternion)[[1, 2, 3, 0]]
        )

    initial_rotation = to_rotation(initial_quaternion)

    actors = {}

    for name in ("left", "right"):
        obs_dim = len(observations[name])

        actor = MAPPOActor(
            action_dim=checkpoint["action_dim"],
        )

        actor.load_state_dict(checkpoint["actors"][name])
        actor.eval()
        actors[name] = actor

    # Match original evaluator: seed immediately before sampling.
    torch.manual_seed(args.seed)

    records = []

    while True:
        actions = {}

        for name in ("left", "right"):
            obs_tensor = torch.as_tensor(
                observations[name],
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                if args.stochastic:
                    action, _, _ = actors[name].sample_action(
                        obs_tensor
                    )
                else:
                    action = torch.tanh(
                        actors[name].actor(obs_tensor)
                    )

            actions[name] = (
                action.squeeze(0).cpu().numpy()
            )

        (
            observations,
            rewards,
            terminated,
            truncated,
            info,
        ) = env.step(actions)

        object_position, object_quaternion, _, _ = env.get_object_state()

        relative_rotation = (
            initial_rotation.inv() * to_rotation(object_quaternion)
        )
        roll, pitch, yaw = relative_rotation.as_euler(
            "xyz", degrees=True
        )

        record = {
            "step": int(info["episode_step"]),
            "distance_mm": (
                float(info["distance_to_target"]) * 1000
            ),
            "object_x": float(object_position[0]),
            "object_y": float(object_position[1]),
            "object_z": float(object_position[2]),
            "forward_displacement_mm": (
                float(object_position[0] - initial_position[0])
                * 1000
            ),
            "linear_speed": float(info["linear_speed"]),
            "angular_speed": float(info["angular_speed"]),
            "orientation_error_rad": float(
                info["orientation_error_rad"]
            ),
            "left_action_norm": float(
                np.linalg.norm(actions["left"])
            ),
            "right_action_norm": float(
                np.linalg.norm(actions["right"])
            ),
            "relative_roll_deg": float(roll),
            "relative_pitch_deg": float(pitch),
            "relative_yaw_deg": float(yaw),
            "left_action_z": float(actions["left"][2]),
            "right_action_z": float(actions["right"][2]),
            "vertical_action_difference": float(
                actions["left"][2] - actions["right"][2]
            ),
            "left_action_x": float(actions["left"][0]),
            "right_action_x": float(actions["right"][0]),
            "left_contact": bool(info["left_grasp_contact"]),
            "right_contact": bool(info["right_grasp_contact"]),
            "success": bool(info["success"]),
            "dropped": bool(info["dropped"]),
        }

        records.append(record)

        if terminated or truncated:
            break

    closest = min(records, key=lambda r: r["distance_mm"])
    final = records[-1]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with output_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=records[0].keys(),
        )
        writer.writeheader()
        writer.writerows(records)

    print(
        "\nMAPPO "
        + ("STOCHASTIC" if args.stochastic else "DETERMINISTIC")
        + " TRAJECTORY"
    )
    print("=" * 55)
    print("Checkpoint:", checkpoint_path)
    print("Steps:", len(records))
    print("Success:", final["success"])
    print(
        "Closest distance:",
        f'{closest["distance_mm"]:.2f} mm',
        "at step",
        closest["step"],
    )
    print(
        "Final distance:",
        f'{final["distance_mm"]:.2f} mm',
    )
    print(
        "Final forward displacement:",
        f'{final["forward_displacement_mm"]:.2f} mm',
    )
    print(
        "Final linear speed:",
        f'{final["linear_speed"]:.5f} m/s',
    )
    print(
        "Final orientation error:",
        f'{final["orientation_error_rad"]:.5f} rad',
    )

    print("\nTRAJECTORY SAMPLES")
    print(
        "Step | Distance mm | Forward mm | "
        "Left action | Right action"
    )

    for record in records:
        if record["step"] % 50 == 0 or record is closest:
            print(
                f'{record["step"]:4d} | '
                f'{record["distance_mm"]:11.2f} | '
                f'{record["forward_displacement_mm"]:10.2f} | '
                f'{record["left_action_norm"]:11.4f} | '
                f'{record["right_action_norm"]:12.4f}'
            )

    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()
