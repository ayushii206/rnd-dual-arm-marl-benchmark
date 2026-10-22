from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.envs.dual_arm_transport_env import (
    DualArmTransportEnv,
    PHYSICS_STEPS_PER_ACTION,
)


def main():
    print("=" * 70)
    print("RL ACTION INTERFACE TEST")
    print("=" * 70)

    env = DualArmTransportEnv()
    env.reset()

    initial_position = env.get_object_state()[0]

    zero_actions = {
        "left": np.zeros(7),
        "right": np.zeros(7),
    }

    print()
    print("Initial object position:", initial_position)

    num_action_steps = 100

    for _ in range(num_action_steps):
        observations = env.apply_actions(
            zero_actions
        )

    final_position = env.get_object_state()[0]

    displacement = float(
        np.linalg.norm(
            final_position - initial_position
        )
    )

    expected_simulation_time = (
        num_action_steps
        * PHYSICS_STEPS_PER_ACTION
        * env.timestep
    )

    print()
    print("Final object position:", final_position)
    print("Object displacement [m]:", displacement)
    print("Simulation time [s]:", env.simulation_time)

    dimensions_correct = (
        observations["left"].shape == (33,)
        and observations["right"].shape == (33,)
    )

    time_correct = np.isclose(
        env.simulation_time,
        expected_simulation_time,
        atol=1e-10,
    )

    finite = all(
        np.all(np.isfinite(obs))
        for obs in observations.values()
    )

    # This is an initial engineering stability threshold,
    # not a final benchmark success criterion.
    stable = displacement <= 0.005

    print()
    print("Checks:")
    print("  observation dimensions:", dimensions_correct)
    print("  simulation time:", time_correct)
    print("  finite observations:", finite)
    print("  displacement <= 5 mm:", stable)

    passed = all([
        dimensions_correct,
        time_correct,
        finite,
        stable,
    ])

    print()
    print("=" * 70)

    if passed:
        print("RL ACTION INTERFACE: PASS")
    else:
        print("RL ACTION INTERFACE: FAIL")
        raise RuntimeError(
            "RL action interface test failed."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()
