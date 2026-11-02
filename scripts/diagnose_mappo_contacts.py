"""Read-only deterministic MAPPO contact and stability diagnostic."""

import numpy as np
import torch

from src.envs.dual_arm_transport_env import DualArmTransportEnv
from src.agents.mappo_networks import MAPPOActor
from scripts.contact_stability_monitor import ContactStabilityMonitor


def main():
    torch.set_num_threads(1)

    checkpoint_path = "checkpoints/mappo/mappo_r4_seed456_10k.pt"
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["algorithm"] == "MAPPO"
    assert checkpoint["action_mode"] == "cartesian"
    assert checkpoint["action_dim"] == 3

    actors = {}

    for name in ("left", "right"):
        actor = MAPPOActor(action_dim=3)
        actor.load_state_dict(checkpoint["actors"][name])
        actor.eval()
        actors[name] = actor

    env = DualArmTransportEnv(
        target_position=np.asarray(
            checkpoint["target_position"],
            dtype=np.float64,
        ),
        success_distance=checkpoint["success_distance"],
        action_mode="cartesian",
    )

    observations = env.reset()
    monitor = ContactStabilityMonitor(env)

    original_step = env.controller.step

    def monitored_step(left_q, right_q):
        original_step(left_q, right_q)
        monitor.record()

    env.controller.step = monitored_step

    torch.manual_seed(100)

    for rl_step in range(1, 501):
        actions = {}

        for name in ("left", "right"):
            observation = torch.as_tensor(
                observations[name],
                dtype=torch.float32,
            ).unsqueeze(0)

            with torch.no_grad():
                action = torch.tanh(
                    actors[name].actor(observation)
                )

            actions[name] = action.squeeze(0).cpu().numpy()

        observations, rewards, terminated, truncated, info = (
            env.step(actions)
        )

        if terminated or truncated:
            break

    summary = monitor.summary()

    print("\n=== MAPPO CONTACT AND STABILITY DIAGNOSTIC ===")
    print("Checkpoint:", checkpoint_path)
    print("RL steps:", rl_step)
    print("Physics steps:", summary["physics_steps"])
    print("Success:", bool(info["success"]))
    print(
        "Final distance (mm):",
        round(1000 * float(info["distance_to_target"]), 3),
    )

    print("\nContact points:")
    for category, count in sorted(
        summary["contact_points"].items()
    ):
        print(f"  {category}: {count}")

    print("\nPhysics steps with contacts:")
    for category, count in sorted(
        summary["affected_steps"].items()
    ):
        print(f"  {category}: {count}")

    print("\nStability:")
    print(
        "  Maximum tilt (degrees):",
        round(summary["max_tilt_deg"], 4),
    )
    print(
        "  Maximum orientation error (degrees):",
        round(summary["max_orientation_deg"], 4),
    )
    print(
        "  Maximum angular speed (rad/s):",
        round(summary["max_angular_speed_rad_s"], 5),
    )
    print(
        "  Physics steps without left grasp contact:",
        summary["missing_left_steps"],
    )
    print(
        "  Physics steps without right grasp contact:",
        summary["missing_right_steps"],
    )


if __name__ == "__main__":
    main()
