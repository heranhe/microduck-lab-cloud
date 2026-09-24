#!/usr/bin/env bash
# The get-up physics ladder (roadmap B.1 / bead mdl-0ad), run end to end.
#
#   cd microduck_local && ./scripts/train_getup_ladder.sh [ENVS] [SEED]
#
# Five rungs, each warm-started from the one before (--init-from), each
# under its own stage knobs from `behaviors/getup.py`'s `curriculum`. The
# reward is IDENTICAL in every rung — only the tilt window (how far from
# upright the duck wakes up), the actuator model and the fall mix move.
# That is the ladder contract AGENTS.md sets and tests/test_behaviors.py
# enforces; this script only executes it, so the recipe stays the single
# source of truth for what a rung is.
#
# Why a ladder at all: the control run `getup-flat-scratch` trains the LAST
# rung's config from scratch and does not find a stand. The reward is not
# the problem — the same reward scores the shipped alpha_stand.onnx at
# 13.2/step against that run's 3.3 — it is that flat-on-the-floor rollouts
# never contain a stand to learn the value of.
set -euo pipefail
cd "$(dirname "$0")/.."

ENVS="${1:-12}"
SEED="${2:-3}"
GROUP=getup-ladder

# `uv run python`, not the system python: the recipe this reads the
# stages out of lives in the project venv (a bare python3 here failed
# with ModuleNotFoundError after the run it was queued behind).
uv run python - "$ENVS" "$SEED" "$GROUP" <<'PY'
import json
import subprocess
import sys

from microduck_local.behaviors import BEHAVIORS

envs, seed, group = sys.argv[1], sys.argv[2], sys.argv[3]
stages = BEHAVIORS["getup"].curriculum
prev = None
for i, st in enumerate(stages, 1):
    run = f"getup-l{i}"
    cmd = ["uv", "run", "train-behavior", "getup",
           "--envs", envs, "--steps", str(st.steps), "--seed", seed,
           "--run-name", run, "--group", group,
           "--title", f"Get-up ladder {i}/{len(stages)} — {st.label}",
           "--description", st.detail]
    if prev:
        cmd += ["--init-from", f"runs/{prev}"]
    print(f"\n=== rung {i}/{len(stages)}: {st.label}  ({st.steps:,} steps)\n"
          f"    knobs {json.dumps(st.env)}\n", flush=True)
    subprocess.run(cmd, check=True, env={**__import__("os").environ, **st.env})
    prev = run
print(f"\nladder done — the tip is runs/{prev}/policy.onnx\n"
      f"bench it:  uv run python scripts/bench_getup.py --policy runs/{prev}/policy.onnx")
PY
