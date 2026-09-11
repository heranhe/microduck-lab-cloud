"""How much of a sensed kick policy's ACTION the four head slots carry
(roadmap 12as follow-up F/G, 2026-09-10).

    cd microduck_local
    uv run python scripts/probe_slot_sensitivity.py --recipe kick_left_sensed \
        scratch=runs/lastmetre-left-distil-v1/policy.onnx \
        from-scratch=runs/lastmetre-left-v1/policy.onnx

A sensed recipe writes the duck's own ball track into `obs[51:55]`
(`behaviors/lastmetre._lm_sense`: bearing, range, seen, confidence). Whether
a trained policy actually READS those four numbers is not a reward curve and
not a bench row — a policy that ignores them entirely still connects, because
the vendored blind kick does. The blindfold bench row
(`grid_kick_bench_sensed.py`'s `label:kick_left=path`) answers it in OUTCOMES,
one bit per cell; this answers it in the POLICY, one number per step:

    mean |a(obs) - a(obs with obs[51:55] = 0)|   as a share of   mean |a(obs)|

taken over the recipe's own finished world. The rollout is driven by the TRUE
action throughout — the blanked action is a counterfactual query on the same
step, never stepped — so the trajectory is the policy's own and the two
actions are compared on identical states.

Read it as a share, not as radians: 12as's from-scratch left arm reads ~95 %
(the slots are most of what it does) and follow-up F's distilled clone ~5 %
(the vendored swing with a perturbation on top). The blanked value 0.0 is the
all-zero command block the arena hands a BLIND skill (`world/arena._skill_cmd`)
and the value `distil_kick.py` shows its teacher, so "blanked" here is exactly
"this policy run blind".

Positive control, printed and not assumed: `seen` (the share of steps whose
slots actually carried a sighting) and the mean |slot| vector. A probe
accidentally run against a blind recipe reads seen ~0 and a near-zero delta
for ANY policy, which is the failure this column catches.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from microduck_local.behaviors.env import BehaviorEnv
from microduck_local.brain.brain_env import onnx_infer

SLOTS = slice(51, 55)          # the four HEAD command slots the recipe writes


def measure(path: str, recipe: str, steps: int, seed: int,
            domain_rand: bool, obs_noise: bool) -> dict:
    infer = onnx_infer(path)
    env = BehaviorEnv(recipe, seed=seed, domain_rand=domain_rand,
                      obs_noise=obs_noise, action_delay=False)
    obs, _ = env.reset(seed=seed)
    d_sum = a_sum = 0.0
    seen = 0.0
    slot_abs = np.zeros(4)
    eps = 0
    n = 0
    while n < steps:
        a = np.asarray(infer(obs), dtype=np.float64)
        blind = np.array(obs, dtype=np.float32, copy=True)
        blind[SLOTS] = 0.0
        b = np.asarray(infer(blind), dtype=np.float64)
        d_sum += float(np.abs(a - b).mean())
        a_sum += float(np.abs(a).mean())
        seen += 1.0 if float(obs[53]) > 0.5 else 0.0
        slot_abs += np.abs(np.asarray(obs[SLOTS], dtype=np.float64))
        n += 1
        obs, _, term, trunc, _ = env.step(a.astype(np.float32))
        if term or trunc:
            eps += 1
            obs, _ = env.reset()
    return {"path": path, "recipe": recipe, "steps": n, "episodes": eps,
            "d": d_sum / n, "a": a_sum / n, "share": d_sum / max(a_sum, 1e-12),
            "seen": seen / n, "slots": slot_abs / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recipe", default="kick_left_sensed",
                    help="behavior id whose env every arm is driven through "
                         "(override per arm with label:recipe=path)")
    ap.add_argument("--steps", type=int, default=10_000, help="control steps an arm")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--domain-rand", action="store_true")
    ap.add_argument("--obs-noise", action="store_true")
    ap.add_argument("policies", nargs="+", metavar="LABEL[:RECIPE]=PATH")
    a = ap.parse_args()
    print(f"blanking obs[51:55] over {a.steps} steps of the recipe's own world "
          f"(seed {a.seed}, domain_rand {a.domain_rand}, obs_noise {a.obs_noise})")
    print(f"  {'arm':<28}{'env':<20}{'|dA| rad':>10}{'|A| rad':>10}{'share':>8}"
          f"{'seen':>8}{'eps':>6}  mean |slots| psi/r/seen/conf")
    for spec in a.policies:
        label, path = spec.split("=", 1)
        recipe = a.recipe
        if ":" in label:
            label, recipe = label.split(":", 1)
        if not Path(path).exists():
            raise SystemExit(f"no such policy: {path}")
        r = measure(path, recipe, a.steps, a.seed, a.domain_rand, a.obs_noise)
        s = " ".join(f"{v:.2f}" for v in r["slots"])
        print(f"  {label:<28}{recipe:<20}{r['d']:>10.4f}{r['a']:>10.4f}"
              f"{r['share']:>7.1%}{r['seen']:>8.1%}{r['episodes']:>6}  {s}")


if __name__ == "__main__":
    main()
