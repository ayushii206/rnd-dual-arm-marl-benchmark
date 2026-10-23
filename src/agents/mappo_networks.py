import torch
import torch.nn as nn

from src.agents.ippo_networks import (
    IPPOActorCritic,
    build_mlp,
)

GLOBAL_STATE_DIM = 50


class MAPPOActor(IPPOActorCritic):
    """
    Decentralized actor using a 33D local observation.

    Inherits the tanh-squashed Gaussian policy from IPPO.
    The inherited local critic is not used by MAPPO.
    """

    def __init__(self, action_dim=3):
        super().__init__(action_dim=action_dim)

        # Remove the inherited local critic.
        self.critic = nn.Identity()

    def sample_action(self, observations):
        distribution = self._distribution(observations)
        raw_actions = distribution.sample()
        actions = torch.tanh(raw_actions)
        log_probs = self._squashed_log_prob(
            distribution, raw_actions
        )
        return actions, raw_actions, log_probs

    def evaluate_actions(self, observations, raw_actions):
        distribution = self._distribution(observations)
        log_probs = self._squashed_log_prob(
            distribution, raw_actions
        )
        entropy = distribution.entropy().sum(dim=-1)
        return log_probs, entropy


class MAPPOCentralizedCritic(nn.Module):
    """One shared value function for the cooperative team."""

    def __init__(self):
        super().__init__()
        self.network = build_mlp(GLOBAL_STATE_DIM, 1)

    def forward(self, global_states):
        return self.network(global_states).squeeze(-1)
