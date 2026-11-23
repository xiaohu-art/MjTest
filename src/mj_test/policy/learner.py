"""Actor-critic policy learning in the world model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from torch import nn

from mj_test.envs.imagination_env import ImaginationEnv
from mj_test.utils.optim import CapturedUpdate, build_optimizers, optimize_step


def gae_returns(
    values: torch.Tensor,
    last_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Computes the GAE returns of a time-major rollout.

    Args:
        values: Critic values of shape (T, B, 1).
        last_value: Critic value after the last step, shape (B, 1).
        rewards: Rewards of shape (T, B, 1).
        dones: Termination probabilities of shape (T, B, 1).
        gamma: Discount factor.
        lam: GAE lambda.

    Returns:
        ``A_t + V(s_t)`` of shape (T, B, 1).
    """
    returns = torch.empty_like(values)
    advantage = 0.0
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns


def actor_objective(
    rewards: torch.Tensor,
    dones: torch.Tensor,
    last_value: torch.Tensor,
    gamma: float,
    survival_masking: bool,
) -> torch.Tensor:
    """Computes the discounted imagined return with a critic bootstrap.

    Args:
        rewards: Rewards of shape (T, B).
        dones: Termination probabilities of shape (T, B).
        last_value: Critic value after the last step, shape (B,).
        gamma: Discount factor.
        survival_masking: Whether each reward and the bootstrap are weighted by the
            survival probability ``S_t``. Without it, ``S_t`` is 1 for the rewards
            and ``1 - dones[-1]`` for the bootstrap.

    Returns:
        ``sum_t gamma^t S_t r_t + gamma^T S_T V`` of shape (B,), where ``S_t`` is the
        product of ``1 - dones`` over the steps before ``t``.
    """
    horizon = rewards.shape[0]
    ret, survival = 0.0, 1.0
    for t in range(horizon):
        ret = ret + gamma**t * survival * rewards[t]
        if survival_masking:
            survival = survival * (1.0 - dones[t])
    alive = survival if survival_masking else 1.0 - dones[-1]
    return ret + gamma**horizon * alive * last_value


class ActorCriticPolicyLearner:
    """Trains the actor on imagined returns and fits the critic with GAE.

    Args:
        actor: The actor.
        critic: The critic.
        imagination: Env that rolls the actor out in the world model.
        cfg: The ``runner.policy`` section of the config.
    """

    def __init__(
        self,
        actor: nn.Module,
        critic: nn.Module,
        imagination: ImaginationEnv,
        cfg: DictConfig,
    ) -> None:
        self.actor = actor
        self.critic = critic
        self.imagination = imagination
        self.cfg = cfg
        self._actor_params = list(actor.parameters())
        self._critic_params = list(critic.parameters())
        device = self._actor_params[0].device
        self.actor_optimizers = build_optimizers(
            [actor], cfg.lr, betas=(0.9, cfg.beta2), device=device
        )
        self.critic_optimizers = build_optimizers([critic], cfg.lr, device=device)
        self.v_max = imagination.max_step_reward / (1.0 - cfg.gamma)
        self._step = CapturedUpdate(self._update_batch)

    def update(self, starts: Iterable[TensorDict]) -> dict[str, torch.Tensor]:
        """Runs one actor and critic update per batch of rollout starts.

        Args:
            starts: Real transitions of batch size (B,) to start rollouts from.

        Returns:
            Statistics averaged over the batches and keyed ``Policy/<name>``.
        """
        self.actor.train()
        logs: dict[str, list[torch.Tensor]] = {}
        for start in starts:
            for name, value in self._step(start).items():
                logs.setdefault(name, []).append(value)
        return {f"Policy/{name}": torch.stack(v).mean() for name, v in logs.items()}

    def _update_batch(self, start: TensorDict) -> dict[str, torch.Tensor]:
        """Takes one actor step on an imagined rollout, then fits the critic to it."""
        cfg = self.cfg
        self.critic.requires_grad_(False)
        steps, last_obs = self._rollout(start)
        last_value = self.critic(last_obs["critic"]).clamp(max=self.v_max).squeeze(-1)
        ret = actor_objective(
            steps["reward"], steps["done"], last_value, cfg.gamma, cfg.survival_masking
        )
        grad_norm = optimize_step(
            self.actor_optimizers,
            -ret.mean(),
            self._actor_params,
            cfg.max_grad_norm,
            cfg.grad_clip_value,
        )
        steps, last_obs = steps.detach(), last_obs.detach()
        return {
            "return": ret.mean().detach(),
            "reward": steps["reward"].mean() / self.imagination.dt,
            "survival": (1.0 - steps["done"]).prod(0).mean(),
            "actor_grad_norm": grad_norm,
            "value_loss": self._fit_critic(steps, last_obs),
        }

    def _rollout(self, start: TensorDict) -> tuple[TensorDict, TensorDict]:
        """Rolls the actor out in the imagination for ``horizon`` steps.

        Returns:
            The steps of batch size (horizon, B), holding the ``critic`` observations
            before each step, the ``reward`` and the termination probability
            ``done``, and the observations after the last step.
        """
        obs = self.imagination.reset(
            start["gen"], start["sim_ctx"], start["time_steps"], start["prev_action"]
        )
        steps = []
        for _ in range(self.cfg.horizon):
            action = self.actor(obs)
            next_obs, reward, term_prob, _, _ = self.imagination.step(action)
            step = {"critic": obs["critic"], "reward": reward, "done": term_prob}
            steps.append(TensorDict(step, batch_size=reward.shape))
            obs = next_obs
        return torch.stack(steps), obs

    def _fit_critic(self, steps: TensorDict, last_obs: TensorDict) -> torch.Tensor:
        """Fits the critic to the GAE returns of a detached rollout."""
        cfg = self.cfg
        with torch.no_grad():
            values = self.critic(steps["critic"])
            last_value = self.critic(last_obs["critic"])
        returns = gae_returns(
            values,
            last_value,
            steps["reward"].unsqueeze(-1),
            steps["done"].unsqueeze(-1),
            cfg.gamma,
            cfg.lam,
        ).clamp(max=self.v_max)
        obs, returns = steps["critic"].flatten(0, 1), returns.flatten(0, 1)
        num_rows = len(returns) // cfg.critic_minibatches * cfg.critic_minibatches
        perm = torch.randperm(num_rows, device=returns.device)
        self.critic.requires_grad_(True)
        losses = []
        for _ in range(cfg.critic_epochs):
            for idx in perm.view(cfg.critic_minibatches, -1):
                loss = (returns[idx] - self.critic(obs[idx])).square().mean()
                optimize_step(
                    self.critic_optimizers, loss, self._critic_params, cfg.max_grad_norm
                )
                losses.append(loss.detach())
        return torch.stack(losses).mean()
