"""The detector-dropout knob has to reach the detector in a MEASUREMENT env.

`MICRODUCK_BALL_DROPOUT` models the NPU missing a detection while the ball is
in frame. It used to be gated on `env.obs_noise` alongside the bearing jitter,
and the two envs built to measure it — `eval-find-ball` and `render-rollout` —
both pin `obs_noise=False`. So `--env MICRODUCK_BALL_DROPOUT=0.1` produced a
battery byte-identical to the run without it: the knob looked measured and was
inert. docs/roadmap.md section 2's dropout item was unrunnable as written.

The lesson is the FOV item's, one section earlier in that file: verify the knob
MOVES the thing it names before believing any sweep over it. These tests are
that verification, kept.
"""

import numpy as np

from microduck_local.behaviors import BehaviorEnv, _ball_place, _ball_sense


def _ball_env(seed=0, **kw):
    env = BehaviorEnv("find_ball", obs_noise=False, domain_rand=False,
                      action_delay=False, random_yaw=False, seed=seed, **kw)
    env.reset(seed=seed)
    return env


def _reported_share(dropout: str, seed: int = 0, n: int = 400) -> float:
    """Share of forced detector updates that REPORT the ball, with it parked
    dead ahead at 1 m — always genuinely in frame, so anything missing is the
    dropout.

    `env._ball_det[2]` is the detector's report; `env._ball_seen` is the
    geometric truth, which a dropout must never touch (that distinction is
    what the battery's `in frame` column measures, and it is the column
    docs/roadmap.md's dropout item is about)."""
    env = _ball_env(seed=seed,
                    spawn_overrides={"MICRODUCK_BALL_DROPOUT": dropout,
                                     "MICRODUCK_BALL_EVENT_RATE": "0"})
    _ball_place(env, 1.0, 0.0)
    reported = 0
    for _ in range(n):
        _ball_sense(env, force=True)
        assert env._ball_seen, "the ball is parked in frame; truth must not move"
        reported += int(float(np.asarray(env._ball_det)[2]) > 0.5)
    return reported / n


def test_dropout_reaches_the_detector_with_obs_noise_off():
    """The battery's own env (obs_noise=False) must see the knob."""
    assert _reported_share("0.0") == 1.0, "no dropout: every update reports"
    got = _reported_share("0.5")
    # Binomial over 400 forced updates at p=0.5: ~0.5 +- 0.08 at 3 sigma.
    assert 0.35 < got < 0.65, f"dropout 0.5 left {got:.0%} of updates seen"


def test_dropout_default_is_off_so_every_recorded_number_still_holds():
    """Nothing in the recipe or the curriculum turns it on: the default keeps
    every measurement in docs/roadmap.md and the policy READMEs comparable."""
    from microduck_local.behaviors import _BALL_KNOBS, BEHAVIORS

    assert float(_BALL_KNOBS["MICRODUCK_BALL_DROPOUT"]) == 0.0
    for stage in BEHAVIORS["find_ball"].curriculum:
        assert "MICRODUCK_BALL_DROPOUT" not in stage.env


def test_dropout_only_drops_reports_it_never_invents_one():
    """A missed detection reads as LOST, never as a ball somewhere else: the
    held report is what the memory slot is for, and a dropout must not write a
    fresh bearing."""
    env = _ball_env(seed=3,
                    spawn_overrides={"MICRODUCK_BALL_DROPOUT": "1.0",
                                     "MICRODUCK_BALL_EVENT_RATE": "0"})
    _ball_place(env, 1.0, 0.0)
    for _ in range(20):
        _ball_sense(env, force=True)
        assert env._ball_seen, "the geometry is untouched: the ball IS in frame"
        assert float(np.asarray(env._ball_det)[2]) == 0.0, "…and unreported"
    assert float(np.asarray(env._ball_det)[0]) == 0.0
