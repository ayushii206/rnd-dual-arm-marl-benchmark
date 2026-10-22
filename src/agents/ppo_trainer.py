
import torch
import torch.nn.functional as F


class PPOTrainer:
    """
    PPO optimizer for one independent actor-critic network.

    Expects a RolloutBuffer containing observations,
    raw Gaussian actions, old log probabilities,
    advantages and returns.
    """

    def __init__(
        self,
        agent,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        update_epochs=4,
        minibatch_size=64,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.agent = agent.to(self.device)

        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.optimizer = torch.optim.Adam(
            self.agent.parameters(),
            lr=learning_rate,
            eps=1e-5,
        )

    def update(self, rollout_buffer):
        if len(rollout_buffer) == 0:
            raise ValueError(
                "Cannot update PPO using an empty buffer."
            )

        batch = rollout_buffer.as_tensors(
            device=self.device
        )

        observations = batch["observations"]
        raw_actions = batch["raw_actions"]
        old_log_probs = batch["old_log_probs"]
        advantages = batch["advantages"]
        returns = batch["returns"]

        # Normalize advantages across the rollout.
        if advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False) + 1e-8
            )

        batch_size = observations.shape[0]

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_proxy": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "updates": 0,
        }

        for _ in range(self.update_epochs):
            indices = torch.randperm(
                batch_size,
                device=self.device,
            )

            for start in range(
                0, batch_size, self.minibatch_size
            ):
                minibatch_indices = indices[
                    start:start + self.minibatch_size
                ]

                new_log_probs, values, entropy = (
                    self.agent.evaluate_actions(
                        observations[minibatch_indices],
                        raw_actions[minibatch_indices],
                    )
                )

                old_log_probs_mb = old_log_probs[
                    minibatch_indices
                ]

                advantages_mb = advantages[
                    minibatch_indices
                ]

                returns_mb = returns[
                    minibatch_indices
                ]

                log_ratio = (
                    new_log_probs - old_log_probs_mb
                )
                ratio = torch.exp(log_ratio)

                unclipped_objective = (
                    ratio * advantages_mb
                )

                clipped_objective = (
                    torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    ) * advantages_mb
                )

                policy_loss = -torch.min(
                    unclipped_objective,
                    clipped_objective,
                ).mean()

                value_loss = F.mse_loss(
                    values,
                    returns_mb,
                )

                entropy_proxy = entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient
                    * entropy_proxy
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite PPO loss detected."
                    )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    self.agent.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                with torch.no_grad():
                    approx_kl = (
                        (ratio - 1.0) - log_ratio
                    ).mean()

                    clip_fraction = (
                        (
                            (ratio - 1.0).abs()
                            > self.clip_epsilon
                        ).float().mean()
                    )

                metrics["policy_loss"] += (
                    policy_loss.item()
                )
                metrics["value_loss"] += (
                    value_loss.item()
                )
                metrics["entropy_proxy"] += (
                    entropy_proxy.item()
                )
                metrics["approx_kl"] += (
                    approx_kl.item()
                )
                metrics["clip_fraction"] += (
                    clip_fraction.item()
                )
                metrics["updates"] += 1

        count = metrics["updates"]

        for key in (
            "policy_loss",
            "value_loss",
            "entropy_proxy",
            "approx_kl",
            "clip_fraction",
        ):
            metrics[key] /= count

        return metrics

