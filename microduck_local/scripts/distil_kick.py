"""Distil a BLIND kick into a SENSED kick's observation (roadmap 12as
follow-up (2), route b, 2026-09-10).

Why this exists
---------------
12as measured the sensed pair in play and found the left foot is a NUDGE:
34.5 % whiff against the vendored left's 8.5 %, and its connected touches
travel 0.19 m against 1.13 m. The diagnosis in that item is 12b's own
finding, one recipe later: "a random swing at a ball spread over 12 x 12 cm
is paid a little everywhere and the optimiser settles on the nudge". 12b
fixed it by WARM-STARTING the box from the vendored point strike — and that
route is the one `behaviors/lastmetre.py` cannot take, for a reason the
recipe's own comment records: the vendored kick's `VecNormalize` was fitted
with obs[51:55] carrying keep-alive noise (std 0.009-0.029 at a 2M count),
so a real bearing of 1.0 would enter the network at ~34 and the running
statistics would need another 2M steps to notice.

Behaviour cloning goes round that. The teacher's action is a function of
PROPRIOCEPTION only — it is blind, it cannot read those four slots — so
fitting a fresh network to (sensed observation -> teacher action) gives:

  * the vendored strike's input->output map, over the whole box and the
    whole gaze range, and
  * a `VecNormalize` whose statistics are the SENSED distribution's, because
    they are computed from the observations the sensed recipe actually
    produces, and
  * four slots that carry no information about the target in the fitting
    set, so the clone starts out ignoring them exactly as the teacher does.

That is precisely the warm start 12b had and 12as could not build, and PPO
on the recipe's own rungs then has to ADD slot-reading to a strike instead
of discovering a strike inside a box-wide pay.

The teacher is fed the observation the ARENA hands it — the four slots
ZEROED (`world/arena.py::_skill_cmd`, "the kick's observation carries an
all-zero command"), which is the distribution the vendored 8.5 % whiff was
measured under — while the student is fitted on the same step's SENSED
observation. One rollout, two views of it.

Fidelity is the thing to watch, not the loss curve: the distill note
(`distill.py`'s own table) measures correlation(action MSE, fall rate) =
+0.93 on the walker and defaults to 250 x 120 for that reason. The fit
itself is `distill.fit` unchanged — same optimiser, same critic head on the
teacher's own discounted returns, same log_std scaling — because the only
thing that differs here is where the transitions come from.

    cd microduck_local
    uv run python scripts/distil_kick.py --recipe kick_left_sensed \
        --teacher policies/kick/kick_left.onnx \
        --run-name lastmetre-left-distil-rung0 --episodes 800 --epochs 120

then the recipe's normal rungs on top of it:

    uv run train-behavior kick_left_sensed --init-from runs/lastmetre-left-distil-rung0 ...

It writes `model.zip` + `vecnormalize.pkl` (what `--init-from` wants),
`policy.onnx` (what every bench and the arena want) and `behavior.json`
(what the /train board wants).
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from microduck_local.behaviors import BEHAVIORS
from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.train import RUNS_DIR

# The four HEAD command slots the sensed recipe writes the ball into. The
# teacher never saw anything but keep-alive noise there and the arena hands
# it zeros, so this is the slice we blank for the teacher's view.
SENSED_SLOTS = slice(51, 55)


def collect(recipe: str, teacher: str, episodes: int, seed: int = 0,
            gamma: float = 0.99, weights: dict[str, float] | None = None,
            spawn: dict[str, str] | None = None):
    """Roll the BLIND teacher out inside the SENSED recipe's env.

    Returns (sensed obs, teacher action, discounted return, mean of the
    seen-slot over the sample) — the first three are exactly what
    `distill.fit` wants.

    The env is built the way `train_behavior.make_env` builds it for a trick
    (domain_rand off, random_yaw off, the recipe's own clip), so the states
    the clone is fitted on are the states the fine-tune will visit.
    """
    import onnxruntime as ort

    b = BEHAVIORS[recipe]
    sess = ort.InferenceSession(teacher, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    env = BehaviorEnv(recipe, weight_overrides=weights or None,
                      spawn_overrides=spawn or None, seed=seed,
                      max_episode_s=b.episode_s, domain_rand=False, random_yaw=False)
    obs_buf: list[np.ndarray] = []
    act_buf: list[np.ndarray] = []
    ret_buf: list[np.ndarray] = []
    seen: list[float] = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        rewards: list[float] = []
        for _ in range(10_000):
            obs = np.asarray(obs, dtype=np.float32)
            blind = obs.copy()
            blind[SENSED_SLOTS] = 0.0        # the arena's own all-zero command block
            act = sess.run(None, {inp: blind[None]})[0][0].astype(np.float32)
            obs_buf.append(obs.copy())
            act_buf.append(act)
            seen.append(float(obs[53]))
            obs, rew, term, trunc, _ = env.step(act)
            rewards.append(float(rew))
            if term or trunc:
                break
        g = 0.0
        tail = np.empty(len(rewards), np.float32)
        for i in range(len(rewards) - 1, -1, -1):
            g = rewards[i] + gamma * g
            tail[i] = g
        ret_buf.append(tail)
    return (np.asarray(obs_buf, np.float32), np.asarray(act_buf, np.float32),
            np.concatenate(ret_buf) if ret_buf else np.zeros(0, np.float32),
            float(np.mean(seen)) if seen else 0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", required=True, choices=sorted(BEHAVIORS),
                    help="the SENSED recipe whose env (and therefore whose "
                         "observation) the clone is fitted in")
    ap.add_argument("--teacher", required=True,
                    help="the BLIND kick ONNX to clone, e.g. policies/kick/kick_left.onnx")
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--episodes", type=int, default=800,
                    help="teacher episodes to fit on. A kick clip is 2 s (100 control "
                         "steps) against the walker clone's 500, so the walker's 250 "
                         "episodes would be a fifth of its transitions")
    ap.add_argument("--epochs", type=int, default=120,
                    help="fitting epochs — the distill note's measured budget "
                         "(correlation(action MSE, fall rate) = +0.93)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weights-json", default=None,
                    help="reward-weight overrides, for the CRITIC's returns only "
                         "(the actor targets are the teacher's actions). Pass the "
                         "same scorecard the fine-tune will run under")
    ap.add_argument("--spawn-json", default=None,
                    help="spawn overrides (a rung's env dict) to collect under; "
                         "default is the recipe's own finished world")
    ap.add_argument("--title", default=None)
    ap.add_argument("--description", default=None)
    ap.add_argument("--group", default=None)
    a = ap.parse_args()

    weights = json.loads(a.weights_json) if a.weights_json else {}
    spawn = json.loads(a.spawn_json) if a.spawn_json else {}
    out = RUNS_DIR / a.run_name
    out.mkdir(parents=True, exist_ok=True)

    print(f"collecting {a.episodes} episodes of {a.teacher} inside {a.recipe} ...", flush=True)
    obs, act, ret, seen = collect(a.recipe, a.teacher, a.episodes, seed=a.seed,
                                  weights=weights, spawn=spawn)
    print(f"  {len(obs)} transitions; teacher |action| {np.abs(act).mean():.3f}; "
          f"return {ret.mean():.1f} +- {ret.std():.1f}")
    # The positive control, printed rather than assumed (the bench's rule):
    # a sensed env that is not sensing reads 0 % here, and so does a blind one.
    print(f"  slots: seen {100 * seen:.0f}% of steps, |bearing| mean "
          f"{np.abs(obs[:, 51]).mean():.3f}, range mean {obs[:, 52].mean():.3f}, "
          f"conf mean {obs[:, 54].mean():.3f}")

    # The fit is `distill.fit` unchanged: the actor on the teacher's actions,
    # the critic head on its own discounted returns, the exploration std
    # scaled to the cloned action size. Nothing about it is walker-specific —
    # the obs and action spaces are the same 61 / 14 contract — so reusing it
    # is what keeps this script's only new idea the COLLECTION above.
    from microduck_local.distill import fit
    mse = fit(obs, act, out, epochs=a.epochs, seed=a.seed, returns=ret)
    print(f"  fit mse {mse:.5f} rad^2")

    from microduck_local.export_onnx import export
    export(out, out / "policy.onnx")

    card = {
        "behavior": a.recipe, "steps": 0, "weights": weights,
        "symmetry_coef": 0.0, "desired_kl": None, "net_arch": "512,256,128",
        "shared_trunk": False, "n_epochs": 5, "clip": None,
        "distil": {"teacher": a.teacher, "episodes": a.episodes,
                   "epochs": a.epochs, "transitions": int(len(obs)),
                   "action_mse": mse, "seen_share": seen},
    }
    if a.title:
        card["title"] = a.title
    if a.description:
        card["description"] = a.description
    if a.group:
        card["group"] = a.group
    (out / "behavior.json").write_text(json.dumps(card, indent=2))
    print(f"done: {out}")


if __name__ == "__main__":
    main()
