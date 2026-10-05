"""`eval_handover`: judge a sensed kick on the states PLAY hands it, not on
the states it was drawn from (roadmap 12aw).

    cd microduck_local
    uv run python scripts/eval_handover.py --buffer .cache/dsc/handover-test.jsonl \
        .cache/dsc/runs/12au-*-s*

Every policy is rolled under TWO spawns and both numbers are printed, because
either alone can be read the wrong way:

  * `play`     — a handover replayed out of the buffer (mode "full"): the
                 whole recorded state, the deployment distribution. THE
                 PRIMARY. The buffer passed here must be recorded on seeds the
                 training buffer did not use, or this is train-on-test.
  * `drawn`    — the recipe's own windows, untouched. The regression check: an
                 arm that wins on `play` by forgetting the world the incumbent
                 was judged in has not obviously won anything.

**Registered before looking** (AGENTS.md, "pick the primary before you look"):
the primary is CONNECT RATE on `play`, higher is better, and an arm is only
credited if its mean over training seeds clears baseline's by more than the
seed-to-seed spread of the two arms. Fall rate on `play` is a guardrail, not a
second chance at a positive: a connect-rate win bought with more falls is
reported as such. Three training seeds is thin — the seed spread is printed so
the reader can see what it can and cannot resolve.

A `fell` here is the env's own criterion (`FALL_GRAVITY_Z`, tilted past 70°)
read off the running model, never a method that might not exist — the first
draft of this called `env.fallen()` behind a `hasattr` and reported 0% falls
for every arm, which is AGENTS.md verification rule 0 wearing a new hat.
"""

from __future__ import annotations

import argparse
import importlib
import json
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from microduck_local import contract as C  # noqa: E402

CONNECT_M = 0.10      # `probe_kick_line`'s whiff threshold, unchanged
STEPS = 100           # 2.0 s at 50 Hz — the clip kick_<side>_sensed ships with


def roll(recipe: str, policy: str, knobs: dict, episodes: int, seed0: int):
    for k in list(os.environ):
        if k.startswith("MICRODUCK_LM_REPLAY"):
            del os.environ[k]
    os.environ.update(knobs)
    import microduck_local.behaviors.lastmetre as lm
    importlib.reload(lm)
    from microduck_local.behaviors.env import BehaviorEnv
    from microduck_local.behaviors.kick import _kick_ball_ids
    from microduck_local.brain.brain_env import onnx_infer

    infer = onnx_infer(policy)
    env = BehaviorEnv(recipe, seed=5, max_episode_s=STEPS * C.CTRL_DT + 0.5,
                      domain_rand=False, random_yaw=True, obs_noise=False,
                      action_delay=False)
    down = np.array([0.0, 0.0, -1.0])
    travels, falls = [], 0
    for e in range(episodes):
        out = env.reset(seed=seed0 + e)
        obs = out[0] if isinstance(out, tuple) else out
        _, qadr, _ = _kick_ball_ids(env)
        b0 = (float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1]))
        fell = False
        for _ in range(STEPS):
            obs = env.step(infer(np.asarray(obs, np.float32).reshape(-1)))[0]
            rq = env._root_qpos
            gz = float(C.quat_rotate_inverse(env.data.qpos[rq + 3:rq + 7], down)[2])
            fell = fell or gz > env.FALL_GRAVITY_Z
        travels.append(math.dist(b0, (float(env.data.qpos[qadr]),
                                      float(env.data.qpos[qadr + 1]))))
        falls += fell
    t = np.array(travels)
    return dict(connect=float(np.mean(t >= CONNECT_M)), travel=float(np.median(t)),
                fall=falls / episodes, n=episodes)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", help="run dirs holding policy.onnx + behavior.json")
    ap.add_argument("--buffer", required=True, help="HELD-OUT handover buffer (kick_gym --out)")
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--seed0", type=int, default=7000)
    ap.add_argument("--recipe", default=None, help="default: each run's own behavior.json")
    ap.add_argument("--out", default=None, help="also write the rows as JSON")
    a = ap.parse_args()

    buf = os.path.abspath(a.buffer)
    rows = []
    for rd in sorted(a.runs):
        d = Path(rd)
        pol = d / "policy.onnx"
        if not pol.exists():
            print(f"  (skipping {d.name}: no policy.onnx)")
            continue
        recipe = a.recipe or json.loads((d / "behavior.json").read_text())["behavior"]
        play = roll(recipe, str(pol),
                    {"MICRODUCK_LM_REPLAY": buf, "MICRODUCK_LM_REPLAY_MODE": "full"},
                    a.episodes, a.seed0)
        drawn = roll(recipe, str(pol), {}, a.episodes, a.seed0)
        rows.append(dict(run=d.name, play=play, drawn=drawn))
        print(f"{d.name:26} play connect {play['connect']:5.1%} travel {play['travel']:.3f} "
              f"falls {play['fall']:5.1%}   |   drawn connect {drawn['connect']:5.1%} "
              f"travel {drawn['travel']:.3f} falls {drawn['fall']:5.1%}")

    # Group by arm (the run name's middle field), and print the seed spread —
    # a mean over three training seeds is not a number without it.
    print()
    arms: dict[str, list] = {}
    for r in rows:
        # The arm is everything between the experiment prefix and the seed
        # suffix, NOT the second dash-field: `12au-neck-baseline-s0` has the
        # arm "neck-baseline", and taking field 1 pooled all nine gaze-fixed
        # runs under "neck" and printed one row where there should have been
        # three. A run name is dashed data, so parse it from the END.
        parts = r["run"].split("-")
        arm = "-".join(parts[1:-1]) if len(parts) > 2 and parts[-1].startswith("s") else r["run"]
        arms.setdefault(arm or r["run"], []).append(r)
    hdr = f"{'arm':10} {'seeds':>5}  {'PLAY connect':>22}  {'play falls':>18}  {'drawn connect':>16}"
    print(hdr)
    print("-" * len(hdr))
    for name, rs in arms.items():
        pc = np.array([r["play"]["connect"] for r in rs])
        pf = np.array([r["play"]["fall"] for r in rs])
        dc = np.array([r["drawn"]["connect"] for r in rs])
        print(f"{name:10} {len(rs):5}  {pc.mean():7.1%} (spread {pc.max()-pc.min():4.1%})  "
              f"{pf.mean():6.1%} ({pf.max()-pf.min():4.1%})  {dc.mean():7.1%} ({dc.max()-dc.min():4.1%})")
    print("\nprimary = PLAY connect. An arm is credited only if its mean clears "
          "baseline's by more\nthan the two arms' seed spreads; otherwise this is NO RESULT "
          "and the seeds that\nwould settle it are the honest output.\n")
    if a.out:
        Path(a.out).write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
