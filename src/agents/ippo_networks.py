
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


OBS_DIM = 33
ACTION_DIM = 7
HIDDEN_DIM = 128


def build_mlp(input_dim, output_dim):
    return nn.Sequential(
        nn.Linear(input_dim, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
        nn.Tanh(),
        nn.Linear(HIDDEN_DIM, output_dim),
    )


class IPPOActorCritic(nn.Module):
    """
    Independent PPO actor and critic for one Panda arm.

    Actor: 33-D local observation -> 7-D continuous action.
    Critic: 33-D local observation -> scalar value.

    Actions use a tanh-squashed Gaussian distribution.
    """

    def __init__(self):
        super().__init__()

        self.actor = build_mlp(OBS_DIM, ACTION_DIM)
        self.critic = build_mlp(OBS_DIM, 1)

        self.log_std = nn.Parameter(
            torch.full((ACTION_DIM,), -1.0)
        )

    def _distribution(self, observations):
        mean = self.actor(observations)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    @staticmethod
    def _squashed_log_prob(distribution, raw_actions):
        """
        Log probability including tanh Jacobian correction.
        """
        correction = (
            2.0 * (
                np.log(2.0)
                - raw_actions
                - nn.functional.softplus(-2.0 * raw_actions)
            )
        )

        return (
            distribution.log_prob(raw_actions) - correction
        ).sum(dim=-1)

    def sample_action(self, observations):
        distribution = self._distribution(observations)

        raw_actions = distribution.sample()
        actions = torch.tanh(raw_actions)

        log_probs = self._squashed_log_prob(
            distribution,
            raw_actions,
        )

        values = self.critic(
            observations
        ).squeeze(-1)

        return actions, raw_actions, log_probs, values

    def evaluate_actions(self, observations, raw_actions):
        distribution = self._distribution(observations)

        log_probs = self._squashed_log_prob(
            distribution,
            raw_actions,
        )

        values = self.critic(
            observations
        ).squeeze(-1)

        # Entropy of the unsquashed Gaussian is a useful
        # exploration proxy, not the exact squashed entropy.
        entropy_proxy = distribution.entropy().sum(dim=-1)

        return log_probs, values, entropy_proxy

