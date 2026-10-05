# MOSS pick + stow — handoff, 2026-09-25

Read this, then `microduck_local/AGENTS.md` (the measurement-discipline and
lab-visibility sections) before touching anything.

## What is running right now

**Nothing.** The last run (`teach-moss_stow-67c5f6`, the staged retract)
finished and FAILED — see "The fold" below. The lab is idle at
`127.0.0.1:8788`; watch it at `localhost:63317` with **no `?lab=` parameter**.

```bash
cd microduck_local
uv run python scripts/watch_training.py 127.0.0.1:8788 24   # run in BACKGROUND
```

That blocks until the run ends, then exports and prints the right report for
the task. **Use it** — do not poll by hand, and do not make the human ask
whether training is done.

## The two shipped policies

| leg | run | state |
|---|---|---|
| pick | `teach-moss_pick-ad9876` | **88%** over six object shapes (ball/block/can/squat 96-100%, **card 8/21**), base locked, no spinning, no floor dragging |
| stow | `teach-moss_stow-ccb305` | delivered 14/24 over six shapes, **bin contact 65.8% -> 0.0%**, fold incomplete |

Both are marked `--pick` in their `record.json`, which is what the lab palette
reads.

## The three things that are NOT solved

1. **The flat CARD** — 8/21 on pick, 0/11 on stow. A 2-6 mm sheet needs a
   different grasp (under an edge, not across a body). Distinct skill, not
   tuning.
2. **The stow SEAT caps delivery at ~14/24.** It places the object standing on
   the floor at its own half-height, which suits a 330 ml can and leaves a
   card 45 mm below the jaws and a TALL can 385 mm away — never in the
   gripper. Measured. No policy can fix it; the seat has to change (seat
   without gravity, or close the jaw before it falls).
3. **The fold to rest.** Solved as a BEHAVIOUR (see below) and wired into the
   brain, but never learned. The staged-retract run is the first honest
   attempt at learning it.

## The fold: the whole story, so nobody repeats it

The tuck pose is collision-free and stable (0.013 rad drift when held). The
post-delivery pose is fine too. **The bin sits between them** — commanding the
arm straight to tuck and HOLDING it 600 steps leaves it jammed on `bin_x1`,
0.65 rad short. `brain/tidy_moss.py`'s tuck state had recorded this for months
("jams 0.74 rad short and rides 75 mm proud") without anyone finding the cause.

So there is no monotone path, which is why **none of these worked**:

| approach | result |
|---|---|
| 3M steps of RL | plateaued at 2.66 rad, 0/24 home |
| more reward (retract 10, bonus 40) | REGRESSED: delivered 14 -> 9, bin contact 0.1% -> 10.8% |
| ten hand-built paths (waypoint searches, task-space lift, three-phase, joint orders) | 0-4/24 |
| behaviour cloning, multi-modal teacher | 1/28 |
| behaviour cloning, deterministic teacher | 0/28 (fwd 178 -> 126 mm) |
| DAgger, no aggregation | 0/32, delivery collapsed 17 -> 6 |
| DAgger, aggregated | 0/32, delivery 11/32 |

**What DOES work**: `moss.RETRACT_WAYPOINT`, found by a one-waypoint planner
(`scripts/plan_retract.py`) run against 14 real post-delivery poses — it
connects **14/14** to the tuck by collision-free segments. Flown in physics it
folds the arm home **14/24** at a 0.154 rad residual, bringing forward reach to
114 mm (the tuck's own value) against 178 mm direct. It is wired into
`brain/tidy_moss.py`'s tuck state and into the staged retract reward.

**Why training failed**: the reward pointed through a wall. Paying progress
straight at the tuck asks the policy to close a distance it cannot close.

**The staged fix was tried and also failed** (`teach-moss_stow-67c5f6`):
splitting the reward into two reachable legs through the waypoint gave
delivered 11/24 (down from 14), arm home 0/24, and episodes collapsing from
305 steps to 46 — it stops early rather than folding.

**So the training route is closed.** RL, more reward, ten hand-built paths,
behaviour cloning at two fidelities, DAgger with and without aggregation, and
a staged reachable objective have all failed. **The fold stays SCRIPTED** in
`brain/tidy_moss.py`, which routes through `RETRACT_WAYPOINT` and folds the
arm home 14/24 at a 0.154 rad residual — the same standing as the mission
loop's scripted `search`. Do not spend more runs on this; spend them on the
CARD and the SEAT, which are the two ceilings that actually bind.

## Tools built today — use them

* `scripts/watch_training.py` — block until a run ends, then report. Task-aware.
* `scripts/pick_report.py` / `scripts/stow_report.py` — the standard reports.
* `scripts/reward_budget.py` — **run after ANY change to what a reward term
  measures.** A retarget once dropped a term from +18 to +1.47 per episode and
  two runs trained on a dead signal before anyone noticed.
* `scripts/plan_retract.py` — the planner and `UNIVERSAL_WAYPOINT`.
* `scripts/distill_retract.py`, `scripts/dagger_retract.py` — the imitation
  attempts, kept as the record of what did not work.
* `.claude/skills/train-loop` — the closed loop, with the three rules.

## Traps that cost hours today

* **Evaluate a policy under the flags it TRAINED with** —
  `moss_env.eval_env_kwargs(run)` reads them off `run.json`. Getting it wrong
  does not look like a bug, it looks like a finding: 0/12 instead of 10/12,
  3 grips instead of 63, 23.7 deg of chassis turn on a robot that cannot turn.
* **A knob must be read from the environment INSIDE `train_env_kwargs`**, not
  as a module constant. 15 knobs were import-time constants, so the lab's
  in-process preview showed a plain can with a free base while the trainer ran
  six shapes with the base locked — every run for a whole day.
* **A recorded flag can be ignored.** `prop_variety` was in `run.json` and the
  stow env never read it (its `reset` overrides the pick's). 2M steps trained
  on one can while the record claimed six shapes.
* **Check what a filter MISSES.** A `ps | grep python` found 7 blender-mcp
  processes; `pgrep -f` found 53.
* **Frames.** `_can_base()` moves with the robot: "the can was pushed 12.6 cm
  into the chassis" was the robot driving forward. Same error inverted on an
  arm-silhouette check.
* **Circular quantities.** A linear regression of wrist angle on object axis
  produced EIGHT runs of "the wrist never aims" — false. Measured properly
  (alignment at the grasp) the policy goes 0.49 -> 0.88. Do not fit a line to
  angles that wrap.

## Machine

Load is not the training. An Android emulator (285%) and RobloxStudio (123%)
were each up over a day; 53 idle `blender-mcp` servers hold 1.8 GB. Training
itself is ~205% of 18 cores.

## Uncommitted

104 files. Nothing is committed — that is the next housekeeping job, and the
repo's rule is to verify the INDEX (`scripts/check_staged_python.py`) before
committing, because a green test suite has sat on top of a broken index here.

## Addendum 2026-09-26 — the fold, three more runs, and bin clutter

* `teach-moss_stow-71087f` (the retract DRILL) never ran: `_reset_retract_drill`
  never set `_last_contact`, every worker died on step 1, and `/teach/status`
  said "training" for six hours. Fixed; `test_the_retract_drill_reset_sets_
  everything_the_stow_reset_does` compares the two resets' attribute sets and
  was seen to fail on the planted bug.
* `bd60ff` (drill, warm): learned LEG 1 (waypoint 12/12 — the shipped stow 0/12),
  never leg 2. `4be5a4` (+50% starts AT the waypoint, warm): sat at the
  waypoint. Cause: every stow run since ccb305 shares one frozen `obs_rms`, and
  the drill's joint angles sit at 13-25 sigma under it.
* `moss_fold` recipe (`teach-moss_fold-92301e-s1`, from scratch, own
  normaliser, z <= 3): still 0/24. ep_rew plateaued at -105 — what sitting
  still earns — and no training episode ever reached home.
* **The scripted two-segment fold, flown through the env: 24/24 home at 0.13
  rad** from the drill's starts; learned leg 1 + scripted leg 2 is also 24/24.
  The fold stays scripted. OPEN: the brain's version is 14/24 — either its
  flying or real post-delivery poses outside the drill's spread.
* `stow_report.py` now evaluates at the rung the run TRAINED on (it always used
  rung 0 — the note's 14/24 was a rung-0 number; rung 2 happens to give 14/24
  too) and prints a FOLD-ALONE section.
* **Bin clutter** (`MICRODUCK_MOSS_BIN_CLUTTER=N`, 0..N `sample_prop` items
  dropped in the bin per episode; off = bit-identical replay). The shipped stow
  delivers 68/120 with 0-3 items against 71/120 empty — no measurable effect;
  the seat still binds. Next, pending a decision: a drop point chosen from the
  arm camera (scripted choice in the target slots, vs a contract change).

### The fold IS learnable — `teach-moss_fold-5d4a30` (picked), later on 2026-09-26

The "training route is closed" verdict above was wrong; four things were in the
way, each measured:

1. **The finish was never sampled** (92301e: 0 episodes reached home in 2M).
   Reverse curriculum — 70% of drill episodes start ALONG the clear path,
   weighted toward home (`retract_drill_path`) — first learned fold, afd698.
2. **The drill's start poses were not the deployment's**: 40/40 from the
   Gaussian, but the live handoff failed. Starts now come from a bank of 174
   REAL post-delivery poses (`data/moss_handoff_poses.npy`, ccb305; held-out
   set beside it).
3. **The stow's fresh object fix arrived in slots 24-26 at |z| ~2000** under a
   fold trained with them always zero: 0/38 -> 36/38 with them zeroed. The
   drill now starts with that fix and lets it go stale, as a handoff does.
4. **A pinned jaw stores a whip**: commands keep advancing while the arm is
   blocked, then release at 10-14 rad/s — learned AND scripted fold alike.
   `cmd_leash=0.08` (command within 0.08 rad of the measured position) ends it.

Result, live stow -> fold on unseen seeds 400-600 with the leash: home 58/58
(0-6 clutter) and 59/59 (empty), arm contact 0.1% / 0.0%, worst joint
1.20 rad/s, 0 of 200 episodes over 1.5 rad/s. **To deploy**: `brain/tidy_moss.py`
must load this leg for its tuck state AND apply the same leash to the servo
goals — neither is wired yet. `train.py` now passes donor-dead obs slots
through on warm starts; `watch_training.py` takes a run-name prefix for
concurrent jobs.

### Wired into the brain — `brain/tidy_moss.py` tuck state, `teach-moss_fold-81875e`

The `tuck` state runs the learned fold whenever `SHIPPED_RUNS["fold"]` is on
disk (override: `MICRODUCK_MOSS_FOLD_POLICY`; the scripted waypoint ramp is
the fallback): seeded from the MEASURED arm, stepped at 25 Hz (not per 50 Hz
world tick — that doubles its speed), every goal leashed to 0.08 rad of the
measured joint (`fold_leash_rad`), done at the fold tolerance or
`fold_policy_s`. `test_the_brain_folds_with_the_learned_leg_at_25hz_on_a_leash`
fails on both planted regressions.

The brain tucks from three places, not one — after a delivery, a failed stow
and a failed pick — so the shipped leg was fine-tuned on 92 real tuck-entry
poses (yard seeds 10-21, `data/moss_tuck_entry_poses_v2.npy`). Yard seeds
0-2, 22 tucks: home 19/22, arm-bin contact 1.2%, fold's own worst 1.56 rad/s.
The SCRIPTED tuck it replaces: 21/22, 4.0%, and 11.5-11.9 rad/s on every
tuck. Open: the ~2.9 rad/s spikes at tuck ENTRY are momentum from the state
before (the scripted stow ramps fast), not the fold; and the lab backend
still holds the old brain until it restarts.

Trap found on the way: an `initFrom` fine-tune of a STAGED recipe gets the
recipe's last-stage env merged OVER the request's `env` (viz_server
`_stage_env`), so a knob in both silently takes the recipe's value. Put the
knob in the recipe, and read run.json before trusting a launch.
