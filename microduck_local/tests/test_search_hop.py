"""The feasibility search must reset physics, and must not certify a null drop."""
import numpy as np

from microduck_local.search_hop import make_env, rollout
from microduck_local.behaviors import HOP_LOAD_POSE, HOP_THRUST_POSE, HOP_READY_POSE


def test_replay_is_deterministic_and_null_is_not_a_hop():
    env = make_env()
    x = np.r_[HOP_LOAD_POSE, HOP_THRUST_POSE, HOP_READY_POSE, .24, .16]
    try:
        first = rollout(env, x, 0)
        rollout(env, x, 1)
        assert rollout(env, x, 0) == first
        null = rollout(env, x, 0, null=True)
        assert not null['success'] and not null['valid_flight']
        assert not null['certified_landing']
        assert env.bam is not None and env.action_delay
    finally:
        env.close()
