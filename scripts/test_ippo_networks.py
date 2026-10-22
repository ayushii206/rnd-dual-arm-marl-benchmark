
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.ippo_networks import IPPOActorCritic


def main():
    torch.manual_seed(42)

    left_agent = IPPOActorCritic()
    right_agent = IPPOActorCritic()

    observations = torch.randn(4, 33)

    actions, raw_actions, log_probs, values = (
        left_agent.sample_action(observations)
    )

    assert actions.shape == (4, 7)
    assert raw_actions.shape == (4, 7)
    assert log_probs.shape == (4,)
    assert values.shape == (4,)

    assert torch.all(actions >= -1.0)
    assert torch.all(actions <= 1.0)

    new_log_probs, new_values, entropy = (
        left_agent.evaluate_actions(
            observations,
            raw_actions,
        )
    )

    assert torch.allclose(
        log_probs, new_log_probs, atol=1e-5
    )
    assert new_values.shape == (4,)
    assert entropy.shape == (4,)

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()

    # Independent networks must not share parameters.
    left_params = list(left_agent.parameters())
    right_params = list(right_agent.parameters())

    assert all(
        a.data_ptr() != b.data_ptr()
        for a, b in zip(left_params, right_params)
    )

    print("=" * 70)
    print("IPPO NETWORK TEST")
    print("=" * 70)
    print("Actions:", tuple(actions.shape))
    print("Log probabilities:", tuple(log_probs.shape))
    print("Values:", tuple(values.shape))
    print("Actions bounded:", bool(
        torch.all(actions.abs() <= 1.0)
    ))
    print("Log probabilities consistent:", bool(
        torch.allclose(log_probs, new_log_probs, atol=1e-5)
    ))
    print("Independent parameters: True")
    print("=" * 70)
    print("IPPO NETWORKS: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()

