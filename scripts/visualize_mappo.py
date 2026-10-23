from pathlib import Path
import argparse
import sys
import time

import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.mappo_networks import MAPPOActor
from src.envs.dual_arm_transport_env import DualArmTransportEnv

AGENTS = ("left", "right")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/mappo/mappo_cartesian_r4_10k_5cm.pt",
    )
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(1)

    checkpoint_path = PROJECT_ROOT / args.checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if checkpoint.get("algorithm") != "MAPPO":
        raise ValueError("Checkpoint is not MAPPO.")

    if (
        checkpoint["action_mode"] != "cartesian"
        or checkpoint["action_dim"] != 3
    ):
        raise ValueError("Unexpected action configuration.")

    actors = {
        name: MAPPOActor(action_dim=3)
        for name in AGENTS
    }

    for name in AGENTS:
        actors[name].load_state_dict(
            checkpoint["actors"][name]
        )
        actors[name].eval()

    target_position = np.asarray(
        checkpoint["target_position"],
        dtype=np.float64,
    )

    env = DualArmTransportEnv(
        target_position=target_position,
        success_distance=checkpoint["success_distance"],
        action_mode=checkpoint["action_mode"],
    )

    observations = env.reset()
    torch.manual_seed(args.seed)

    print("MAPPO visualization")
    print("Checkpoint:", checkpoint_path)
    print("Seed:", args.seed)
    print(
        "Mode:",
        "deterministic" if args.deterministic else "stochastic",
    )
    print("Opening MuJoCo viewer...")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        viewer.sync()
        time.sleep(1.5)

        for step in range(1, 501):
            if not viewer.is_running():
                print("Viewer closed.")
                break

            actions = {}

            for name in AGENTS:
                obs_tensor = torch.as_tensor(
                    observations[name],
                    dtype=torch.float32,
                ).unsqueeze(0)

                with torch.no_grad():
                    if args.deterministic:
                        action = torch.tanh(
                            actors[name].actor(obs_tensor)
                        )
                    else:
                        action, _, _ = (
                            actors[name].sample_action(obs_tensor)
                        )

                actions[name] = (
                    action.squeeze(0).cpu().numpy()
                )

            start = time.perf_counter()

            (
                observations,
                rewards,
                terminated,
                truncated,
                info,
            ) = env.step(actions)

            viewer.sync()

            if step % 50 == 0 or terminated or truncated:
                print(
                    f"Step {step:3d} | "
                    f"Distance: {info['distance_to_target'] * 1000:.2f} mm | "
                    f"Success: {info['success']}"
                )

            if terminated or truncated:
                print("\nEpisode finished.")
                print("Steps:", step)
                print("Success:", info["success"])
                print("Dropped:", info["dropped"])
                print(
                    "Final distance:",
                    f"{info['distance_to_target'] * 1000:.2f} mm",
                )
                time.sleep(3)
                break

            elapsed = time.perf_counter() - start
            time.sleep(max(0.0, 0.04 - elapsed))


if __name__ == "__main__":
    main()
