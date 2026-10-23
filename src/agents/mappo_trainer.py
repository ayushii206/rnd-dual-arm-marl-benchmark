import torch
import torch.nn.functional as F


class MAPPOTrainer:
    AGENTS = ("left", "right")

    def __init__(
        self,
        actors,
        critic,
        learning_rate=3e-4,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        entropy_coefficient=0.01,
        max_grad_norm=0.5,
        update_epochs=2,
        minibatch_size=32,
        device="cpu",
    ):
        self.device = torch.device(device)
        self.actors = {
            name: actors[name].to(self.device)
            for name in self.AGENTS
        }
        self.critic = critic.to(self.device)

        self.clip_epsilon = clip_epsilon
        self.value_coefficient = value_coefficient
        self.entropy_coefficient = entropy_coefficient
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.minibatch_size = minibatch_size

        self.optimizer = torch.optim.Adam(
            [
                parameter
                for actor in self.actors.values()
                for parameter in actor.parameters()
            ] + list(self.critic.parameters()),
            lr=learning_rate,
            eps=1e-5,
        )

    def update(self, rollout_buffer):
        if len(rollout_buffer) == 0:
            raise ValueError("Cannot update an empty MAPPO buffer.")

        batch = rollout_buffer.as_tensors(
            device=self.device
        )

        global_states = batch["global_states"]
        returns = batch["returns"]
        advantages = batch["advantages"]

        if advantages.numel() > 1:
            advantages = (
                advantages - advantages.mean()
            ) / (
                advantages.std(unbiased=False) + 1e-8
            )

        batch_size = global_states.shape[0]

        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy_proxy": 0.0,
            "approx_kl": 0.0,
            "updates": 0,
        }

        for _ in range(self.update_epochs):
            indices = torch.randperm(
                batch_size, device=self.device
            )

            for start in range(
                0, batch_size, self.minibatch_size
            ):
                idx = indices[
                    start:start + self.minibatch_size
                ]

                policy_losses = []
                entropies = []
                kls = []

                for name in self.AGENTS:
                    new_log_probs, entropy = (
                        self.actors[name].evaluate_actions(
                            batch[f"{name}_observations"][idx],
                            batch[f"{name}_raw_actions"][idx],
                        )
                    )

                    old_log_probs = (
                        batch[f"{name}_old_log_probs"][idx]
                    )

                    log_ratio = (
                        new_log_probs - old_log_probs
                    )
                    ratio = torch.exp(log_ratio)

                    unclipped = ratio * advantages[idx]
                    clipped = torch.clamp(
                        ratio,
                        1.0 - self.clip_epsilon,
                        1.0 + self.clip_epsilon,
                    ) * advantages[idx]

                    policy_losses.append(
                        -torch.min(
                            unclipped, clipped
                        ).mean()
                    )

                    entropies.append(entropy.mean())

                    with torch.no_grad():
                        kls.append(
                            ((ratio - 1.0) - log_ratio).mean()
                        )

                # Average actor losses to keep gradient
                # magnitudes comparable across agents.
                policy_loss = torch.stack(
                    policy_losses
                ).mean()

                entropy_proxy = torch.stack(
                    entropies
                ).mean()

                values = self.critic(global_states[idx])
                value_loss = F.mse_loss(
                    values, returns[idx]
                )

                loss = (
                    policy_loss
                    + self.value_coefficient * value_loss
                    - self.entropy_coefficient * entropy_proxy
                )

                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite MAPPO loss detected."
                    )

                self.optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    [
                        parameter
                        for actor in self.actors.values()
                        for parameter in actor.parameters()
                    ] + list(self.critic.parameters()),
                    self.max_grad_norm,
                )

                self.optimizer.step()

                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy_proxy"] += entropy_proxy.item()
                metrics["approx_kl"] += torch.stack(
                    kls
                ).mean().item()
                metrics["updates"] += 1

        for key in (
            "policy_loss",
            "value_loss",
            "entropy_proxy",
            "approx_kl",
        ):
            metrics[key] /= metrics["updates"]

        return metrics
