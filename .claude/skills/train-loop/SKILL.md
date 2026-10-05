---
name: train-loop
description: >-
  Run the CLOSED training loop on a MOSS/duck policy without a human having to
  ask "is it done yet": launch through the lab, block until the run finishes,
  print the standard report and the per-term reward budget, then adjust and
  relaunch. Use whenever a training run is started and its result matters.
  Trigger on: "train it again", "when it's done check the results", "tune the
  rules and retrain", "let me know when training finishes", "is it still
  training".
---

# train-loop — launch, wait, measure, adjust, relaunch

A human should not have to poll an agent for "is it done". Run the waiter in
the BACKGROUND: it exits when the run ends, which re-invokes the agent with
the report already computed.

```bash
uv run python scripts/watch_training.py 127.0.0.1:8788 40
```

It blocks on `GET /teach/status`, then exports the finished run (never judge a
raw checkpoint — `export-walk` bakes the normalizer in) and prints:

* `scripts/pick_report.py` — picks, can displacement, base command, chassis
  turn, floor dragging, torque saturation, and the wrist-vs-axis SLOPE with
  its p-value.
* `scripts/reward_budget.py` — what each reward term is worth per episode.

## The three rules this loop exists to enforce

**1. Evaluate a policy under the flags it TRAINED with.** `eval_env_kwargs(run)`
reads them off `run.json` — what the policy SAW (`OBS_FLAGS`) and what it was
ALLOWED TO DO (`DYN_FLAGS`). Getting this wrong does not look like an error, it
looks like a RESULT:

| the mistake | what it reported | the truth |
|---|---|---|
| attitude slots filled for an axis-blind policy | 0/12 picks | 10/12 |
| `publish_size` omitted from the guard | 3 grips | 63 |
| `base_lock` omitted from the guard | 23.7 deg of chassis turn | 0.3 deg |

All three happened on 2026-09-25, and the last two happened AFTER a guard was
written — because the guard covered some flags and read as covering all of
them. A new flag that changes the observation goes in `OBS_FLAGS`; one that
changes the dynamics goes in `DYN_FLAGS`.

**2. Read the reward budget after ANY change to what a term measures.**
Retargeting the progress term from the chassis to the gripper without
rescaling dropped it from ~+18 per episode to +1.47, and two full runs trained
with almost no approach shaping before anyone noticed. Changing what a term
MEASURES changes its MAGNITUDE.

**3. Check a new knob's reachable set before believing in it.** Print the share
of ticks it fires on. A rung calibrated under different semantics from where it
was applied fired on 1 of 40 episodes instead of the predicted 15%.

## Adjusting, then relaunching

Knobs reach the env as `MICRODUCK_MOSS_*` in the `/teach` request's `env`, and
`Body.train_env_kwargs` copies them into `run.json` so the run records what it
trained on. **Add every new knob there**, or the run is unreproducible and the
loop above cannot evaluate it correctly.

```bash
curl -s -X POST http://127.0.0.1:8788/teach -H 'Content-Type: application/json' \
  -d '{"text":"pick up the can","robot":"moss","initFrom":"<donor>",
       "env":{"MICRODUCK_MOSS_PICK_RUNG":"2","MICRODUCK_MOSS_BASE_LOCK":"1"},
       "steps":800000,"weights":{}}'
```

Restart the lab BEFORE launching if any env module changed (the stage previews
in-process; the trainer is a subprocess), and confirm the trainee is really on
stage — see the two sections in `microduck_local/AGENTS.md`.

**Warm starting into a NEW observation slot needs the normalizer reseeded**
first: a donor trained with a slot dead carries variance ~1e-10 there, so a
real value arrives as a z-score in the thousands and the policy stops working.
Copy the donor run, overwrite `obs_rms.mean/var` for those slots with the
MEASURED distribution, warm-start from the copy.
