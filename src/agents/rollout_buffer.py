
import numpy as np
import torch


class RolloutBuffer:
    """
    Stores on-policy experiences for one IPPO agent.

    Each agent maintains its own buffer, even when both
    agents receive the same cooperative team reward.
    """

    def __init__(
        self,
        observation_dim=33,
        action_dim=7,
        gamma=0.99,
        gae_lambda=0.95,
    ):
        self.observation_dim = observation_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clear()

    def clear(self):
        self.observations = []
        self.raw_actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.next_values = []
        self.terminated = []
        self.truncated = []

    def add(
        self,
        observation,
        raw_action,
        log_prob,
        reward,
        value,
        next_value,
        terminated,
        truncated,
    ):
        observation = np.asarray(
            observation, dtype=np.float32
        ).copy()

        raw_action = np.asarray(
            raw_action, dtype=np.float32
        ).copy()

        if observation.shape != (
            self.observation_dim,
        ):
            raise ValueError(
                "Incorrect observation shape."
            )

        if raw_action.shape != (self.action_dim,):
            raise ValueError(
                "Incorrect action shape."
            )

        numeric_values = np.array(
            [log_prob, reward, value, next_value],
            dtype=np.float64,
        )

        if not (
            np.isfinite(observation).all()
            and np.isfinite(raw_action).all()
            and np.isfinite(numeric_values).all()
        ):
            raise ValueError(
                "Rollout contains non-finite values."
            )

        self.observations.append(observation)
        self.raw_actions.append(raw_action)
        self.log_probs.append(float(log_prob))
        self.rewards.append(float(reward))
        self.values.append(float(value))
        self.next_values.append(float(next_value))
        self.terminated.append(bool(terminated))
        self.truncated.append(bool(truncated))

    def compute_advantages(self):
        """
        Generalized Advantage Estimation.

        True termination:
            No value bootstrap.

        Time-limit truncation:
            Bootstrap from the next observation value,
            but do not connect to the next episode.

        Rollout ending mid-episode:
            Bootstrap from the last next-state value.
        """
        n = len(self.rewards)

        if n == 0:
            raise ValueError("Rollout buffer is empty.")

        advantages = np.zeros(
            n, dtype=np.float32
        )

        gae = 0.0

        for t in reversed(range(n)):
            terminated = self.terminated[t]
            truncated = self.truncated[t]

            bootstrap_mask = (
                0.0 if terminated else 1.0
            )

            continuation_mask = (
                0.0
                if terminated or truncated
                else 1.0
            )

            delta = (
                self.rewards[t]
                + self.gamma
                * bootstrap_mask
                * self.next_values[t]
                - self.values[t]
            )

            gae = (
                delta
                + self.gamma
                * self.gae_lambda
                * continuation_mask
                * gae
            )

            advantages[t] = gae

        returns = (
            advantages
            + np.asarray(
                self.values, dtype=np.float32
            )
        )

        return advantages, returns

    def as_tensors(self, device="cpu"):
        advantages, returns = (
            self.compute_advantages()
        )

        return {
            "observations": torch.as_tensor(
                np.stack(self.observations),
                dtype=torch.float32,
                device=device,
            ),
            "raw_actions": torch.as_tensor(
                np.stack(self.raw_actions),
                dtype=torch.float32,
                device=device,
            ),
            "old_log_probs": torch.as_tensor(
                self.log_probs,
                dtype=torch.float32,
                device=device,
            ),
            "advantages": torch.as_tensor(
                advantages,
                dtype=torch.float32,
                device=device,
            ),
            "returns": torch.as_tensor(
                returns,
                dtype=torch.float32,
                device=device,
            ),
        }

    def __len__(self):
        return len(self.rewards)

