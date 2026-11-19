from mjlab.tasks.registry import register_mjlab_task

from mj_test.env_cfg import make_env_cfg
from mj_test.rl_cfg import make_rl_cfg
from mj_test.runner.runner import Runner

register_mjlab_task(
    task_id="Mjtest-Tracking-G1",
    env_cfg=make_env_cfg(),
    play_env_cfg=make_env_cfg(play=True),
    rl_cfg=make_rl_cfg(),
    runner_cls=Runner,
)
