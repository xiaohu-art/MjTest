"""Runner configuration of the G1 motion tracking task."""

from mj_test.policy.learner import ActorCriticPolicyLearnerCfg
from mj_test.runner.runner import RunnerCfg


def make_rl_cfg() -> RunnerCfg:
    """Builds the runner configuration.

    Returns:
        The runner configuration.
    """
    return RunnerCfg(policy=ActorCriticPolicyLearnerCfg(batch_size=2048))
