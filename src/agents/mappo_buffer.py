import numpy as np
import torch

from src.agents.rollout_buffer import RolloutBuffer


class MAPPORolloutBuffer(RolloutBuffer):
    """
    Shared team rollout for MAPPO.

    Stores local observations/actions for both actors,
    global states for the centralized critic, and
    one shared reward/advantage sequence.
    """

    AGENTS = ("left", "right")

    def __init__(self, action_dim=3):
        super().__init__(
            observation_dim=50,
            action_dim=action_dim,
        )
        self.local_observations = {
            name: [] for name in self.AGENTS
        }
        self.agent_raw_actions = {
            name: [] for name in self.AGENTS
        }
        self.agent_log_probs = {
            name: [] for name in self.AGENTS
        }

    def add(
        self,
        global_state,
        observations,
        raw_actions,
        log_probs,
        reward,
        value,
        next_value,
        terminated,
        truncated,
    ):
        global_state = np.asarray(
            global_state, dtype=np.float32
        )

        if global_state.shape != (50,):
            raise ValueError("Incorrect global-state shape.")

        validated = {}

        for name in self.AGENTS:
            obs = np.asarray(
                observations[name], dtype=np.float32
            )
            action = np.asarray(
                raw_actions[name], dtype=np.float32
            )
            log_prob = float(log_probs[name])

            if obs.shape != (33,):
                raise ValueError("Incorrect local observation shape.")

            if action.shape != (self.action_dim,):
                raise ValueError("Incorrect agent action shape.")

            if not (
                np.isfinite(obs).all()
                and np.isfinite(action).all()
                and np.isfinite(log_prob)
            ):
                raise ValueError("Non-finite actor rollout data.")

            validated[name] = (
                obs.copy(), action.copy(), log_prob
            )

        super().add(
            observation=global_state,
            raw_action=np.zeros(
                self.action_dim, dtype=np.float32
            ),
            log_prob=0.0,
            reward=reward,
            value=value,
            next_value=next_value,
            terminated=terminated,
            truncated=truncated,
        )

        for name in self.AGENTS:
            obs, action, log_prob = validated[name]
            self.local_observations[name].append(obs)
            self.agent_raw_actions[name].append(action)
            self.agent_log_probs[name].append(log_prob)

    def as_tensors(self, device="cpu"):
        batch = super().as_tensors(device=device)

        # The inherited observations hold global states.
        batch["global_states"] = batch.pop("observations")
        batch.pop("raw_actions")
        batch.pop("old_log_probs")

        for name in self.AGENTS:
            batch[f"{name}_observations"] = torch.as_tensor(
                np.stack(self.local_observations[name]),
                dtype=torch.float32,
                device=device,
            )
            batch[f"{name}_raw_actions"] = torch.as_tensor(
                np.stack(self.agent_raw_actions[name]),
                dtype=torch.float32,
                device=device,
            )
            batch[f"{name}_old_log_probs"] = torch.as_tensor(
                self.agent_log_probs[name],
                dtype=torch.float32,
                device=device,
            )

        return batch
