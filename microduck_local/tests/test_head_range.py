"""A walker can be trained under the gaze head poses (roadmap 4c revisit,
2026-09-07): `MicroduckWalkEnv(head_cmd_ranges=...)` samples head commands
from the given ranges, the default is the contract's keep-alive range bit
for bit, and `train-walk --head-range` / `distill --head-range` carry it."""

import argparse

import numpy as np

from microduck_local import contract as C
from microduck_local.walk_env import MicroduckWalkEnv


def test_head_command_ranges_default_to_the_contract_and_widen_on_request():
    env = MicroduckWalkEnv(seed=3)
    assert env.head_cmd_ranges == C.HEAD_CMD_RANGES
    env.reset(seed=3)
    assert all(lo <= v <= hi for v, (lo, hi) in zip(env.head_cmd, C.HEAD_CMD_RANGES))
    gaze = ((-0.75, 0.05), (-0.05, 0.8), (-1.4, 1.4), (-0.015, 0.015))
    env2 = MicroduckWalkEnv(seed=3, head_cmd_ranges=gaze)
    seen = []
    for k in range(40):
        env2.reset(seed=10 + k)
        seen.append(env2.head_cmd.copy())
    seen = np.array(seen)
    assert seen[:, 0].min() < -0.4 and seen[:, 1].max() > 0.5              # the gaze poses are sampled
    assert all((seen[:, i] >= gaze[i][0] - 1e-6).all() and (seen[:, i] <= gaze[i][1] + 1e-6).all() for i in range(4))


def test_train_walk_reads_head_range_off_the_command_line():
    from microduck_local.train import env_kwargs_from_args
    ns = argparse.Namespace(no_domain_rand=False, no_obs_noise=False, actuator=None,
                            head_range="-0.75,0.05,-0.05,0.8,-1.4,1.4,-0.015,0.015")
    kw = env_kwargs_from_args(ns)
    assert kw["head_cmd_ranges"] == ((-0.75, 0.05), (-0.05, 0.8), (-1.4, 1.4), (-0.015, 0.015))
    assert "head_cmd_ranges" not in env_kwargs_from_args(argparse.Namespace(no_domain_rand=False, no_obs_noise=False, actuator=None, head_range=None))
