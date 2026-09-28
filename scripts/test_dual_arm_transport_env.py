from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.envs.dual_arm_transport_env import DualArmTransportEnv


def main():
    print("=" * 70)
    print("DUAL-ARM TRANSPORT ENVIRONMENT SMOKE TEST")
    print("=" * 70)

    env = DualArmTransportEnv()

    print()
    print("Model:")
    print("  path:", env.model_path)
    print("  nq:", env.model.nq)
    print("  nv:", env.model.nv)
    print("  nu:", env.model.nu)
    print("  timestep [s]:", env.timestep)

    print()
    print("Initial simulation time [s]:", env.simulation_time)

    env.step_physics(num_steps=10)

    print(
        "Simulation time after 10 physics steps [s]:",
        env.simulation_time,
    )

    expected_time = 10 * env.timestep

    if abs(env.simulation_time - expected_time) > 1e-12:
        raise RuntimeError(
            "Physics stepping produced an unexpected simulation time."
        )

    env.reset_simulation()

    print(
        "Simulation time after reset [s]:",
        env.simulation_time,
    )

    if abs(env.simulation_time) > 1e-12:
        raise RuntimeError(
            "Simulation time did not return to zero after reset."
        )

    print()
    print("=" * 70)
    print("DUAL-ARM TRANSPORT ENVIRONMENT: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()