# Roadmap — what to run next, and what would settle it

A working list, not a plan of record. Each item carries the command to run and
the **number that decides it**, because this project's history is full of
things that looked right in a reward curve and were wrong on screen
(`microduck_local/AGENTS.md`, "Verification discipline").

Convention: `[ ]` not started · `[~]` running · `[x]` done, with the answer
written back into the item. Keep the answers — a negative result that took an
hour is worth as much as the positive one, and this file is where the next
person finds out it was already tried.

---

## Now: the 🔎 `find_ball` brain

Context: `find_ball` is a scan-and-track behavior that aims the duck at a
ball — the eyes the ball-blind kick and ground-pick policies never had. It was
prototyped and trained end to end in a 4-core cloud container at ~1.2k
steps/s (~8M steps, ~2.5 h across six warm-started stages). Everything below
wants a real machine: an M-series Mac runs the same recipe at ~12× that, so a
stage that took 25 minutes there takes ~2 minutes here.

The recipe, the slot layout, and the measured results are documented in
`microduck_local/README.md` ("🔎 `find_ball`"); the shipped export and its
lineage are in `microduck_local/policies/find_ball/`.

### 0. Verify the branch on real hardware — DONE (2026-09-03, M-series Mac)

- [x] **The test suite is green — confirmed, no bisect needed.**
      `cd microduck_local && uv run --with pytest pytest tests/`
      → **411 passed, 1 skipped, 0 failed in 37 s** on the M-series Mac
      (2026-09-03). All 12 golden bit-parity tests (`test_step_perf_parity`,
      `test_bam_perf_parity`, the `test_symmetry` drift case) pass here, which
      settles it: the cloud failures were environment drift (a Mesa/BLAS-level
      float difference), not this branch. The suite is 412 passed with the
      `body_aimed` term and its test added below.
- [x] **The battery reproduces.** `uv run eval-find-ball policies/find_ball/policy.onnx --episodes 40`
      → **it does**, and closely (Mac, 2026-09-03; cloud numbers in brackets):

      | bucket | n | found | t_first med | in frame | centred | fell |
      |---|---:|---:|---:|---:|---:|---:|
      | front | 10 | 100% [100%] | 0.03 s [0.03] | 100% | 98% | 0 [0] |
      | side | 20 | 90% [85%] | 0.61 s [0.60] | 77% | 62% | 0 [0] |
      | back | 10 | 60% [60%] | 0.94 s [0.94] | 32% | 0% | 2 [2] |
      | all | 40 | 85% | 0.43 s | 72% | 56% | 2 |

      Only the side bucket moved (85% → 90%, i.e. one episode). Note the
      battery takes **2.2 s** here — it is free, run it on everything.
      Worth recording for every item below: this Mac trains at **16-27k
      steps/s** against the container's ~1.2k, so the full 8M-step chain is
      ~7 minutes, not 2.5 hours.
- [x] **Looked at it — and item 1 is visible in the frames.** `uv run
      render-rollout --policy policies/find_ball/policy.onnx --out /tmp/rr-fb --episodes 4`
      → it stands cleanly the whole time (`trunk_z` 0.113-0.116 against the
      0.120 stand reference, `floor:none`, both feet 98-99%, 0 reversals —
      no collapsed-crouch or cycling failure here). What the sheet shows is
      the gaze-policy problem, plainly: in ep3 the **feet never move for
      10 s** while the head is visibly cranked round, and the true body
      bearing goes p+98° → p+75° in the first second and then sits at
      **p+60° for the last 8 s** with the detector holding `x+0.21`. Aim
      streak 0 steps in every episode — the handoff never comes close.
      Confirms the item-1 trace independently on this machine.
- [x] **It works in the lab — first browser run of the ball path, and it
      is correct.** `cp -r microduck_local/policies/find_ball
      microduck_local/runs/find_ball`, `uv run duck-lab --port 8789
      runs/find_ball`, viewer on `?lab=127.0.0.1:8789`, then drop
      `run:find_ball` on a duck from the 🧠 palette.
      → the orange ball draws next to the duck, and the label carries the
      spawn note exactly as specified (`↻ ball -61° 0.7m blind`,
      `↻ ball +87° 0.9m prior` — both the prior and blind variants). The
      marker follows ball events live. Assigned duck: 0.00 m/s, 1 fall,
      r̄ 6.6-8.9 over 16 s.

      **Rough edge found on the way (pre-existing, not this branch):** a run
      dir passed as a `duck-lab` CLI positional (`duck-lab runs/find_ball`)
      is NOT recognised as a trick duck, because `build_ducks` never sets
      `policy_id` for it (`viz_server.py:897`) and `is_trick_duck` bails on
      anything without a `run:` prefix (`viz_server.py:2141`). The lab then
      sends this ball-brain a 0.9 m/s **walk command**, which is pure
      out-of-distribution noise to it: **1058 falls and r̄ -3.2** in ~4
      minutes. Dropping the identical run dir from the palette sets
      `policy_id="run:find_ball"`, commands go to zero, and the same policy
      is immediately healthy. So the documented path works and the CLI path
      silently does not — the fix is one argument
      (`add(p.name, ..., policy_id=f"run:{p.name}")` for dirs under
      `RUNS_DIR`), and it is exactly the failure mode the `is_trick_duck`
      docstring says it exists to prevent.

### 1. The open problem: the body never turns — `find_ball` is a gaze policy

**Sharpened by the handoff work (see item 4, which is done): the duck aims its
HEAD at the ball and leaves its body where it was.** Traced over a full
8 s episode with the ball 15° off at 1.2 m: it centres the camera perfectly
(bearing +0.11, elevation 0.00, in frame 100% of steps) using **21° of head
yaw**, while the body bearing to the ball stays at 18–20° and drifts slightly
*further* away. It never squares up, so it never satisfies the kick handoff.

This is one problem with the back-bucket weakness, not two. The head does the
eyes-on job alone and for free, while turning the body costs steps, smoothness
penalties, and fall risk — so the policy takes the cheap option. `face_the_ball`
is paid while the ball is seen, but its tight layer (std 0.4 rad) still pays
~2/3 at 19° off, so the last 20° has almost no gradient behind it; and
`turn_to_belief` is gated to fire only while the ball is *out* of frame, so
once the head finds the ball nothing pays for the body to catch up.


**FIRST, AND IT RE-BASELINES EVERYTHING BELOW: most of this was
under-training, not mis-pricing.** Before A/B-ing any fix, the *unchanged*
recipe was retrained on the Mac — the declared 3-stage curriculum via the
lab's `/teach` (which is the only path that chains stages; `train-behavior`
does not), 8M steps total, ~7 minutes, `body_aimed` pinned to 0 so it is
literally the shipped recipe. Run `teach-find_ball-5f89d9`. Against the
shipped stage-5 export, on 40 static-ball episodes:

| | shipped s5 | s4 baseline | **control (same recipe, Mac, 8M)** |
|---|---:|---:|---:|
| head yaw while centred, front | 21.1° | 18.7° | **13.9°** |
| head yaw while centred, all | 40.8° | 41.1° | **21.1°** |
| body bearing turned out, all | 23.7° | 17.9° | **57.6°** |
| handoff fired (front/side/back) | 40/10/0% | 50/0/0% | **60/15/10%** |
| falls / 40 | 2 | 0 | **0** |
| back-bucket in frame / centred | 32% / 0% | — | **54% / 52%** |
| median time to first sight, all | 0.43 s | — | **0.15 s** |

So the shipped export was **not** a converged instance of its own recipe: the
same terms, trained straight through the declared curriculum, turn 2.4× more
body, halve the head yaw, drop the falls to zero and go from *never* centring
a back-bucket ball to centring it half the time. Any A/B run against the
shipped ONNX would have credited a reward change with all of that.

**But item 1's symptom survives retraining**, which is why the fixes below are
still worth running: rendered (`/tmp/rr-ctrl`), the control visibly steps its
body round — ball at p+70° is p+20° by 1.4 s, and after a ball event throws it
to p−119° it turns all the way back — and then **parks 18–20° off and leaves
the last stretch to the neck**, aim streak 0, handoff never fired in that
episode. 13.9° of head yaw on a front start is under the 0.25 rad gate; 24.4°
on a side start is not. The remaining problem is precisely "the last 20°".

Two process notes for whoever runs the fixes:

- **The shipped ONNX cannot be warm-started from.** `policies/find_ball/` has
  no `model.zip` / `vecnormalize.pkl` (the container was ephemeral), so
  `--init-from runs/find_ball` in the item below cannot work as written. Every
  arm has to retrain the chain, which on this machine is fine (~7 min).
- **The lab always trains at seed 0** (`TrainingJob` never passes `--seed`), so
  two `/teach` arms are seed-matched for free — but a *second* training seed,
  which `AGENTS.md` wants before crediting a small effect, needs the CLI and a
  hand-chained curriculum.

Three candidate fixes, cheapest first — **A/B them, do not stack them**:

- [x] **Price the state you actually want — SHIPPED AS `body_aimed`, and it
      is the fix.** Added to `behaviors/ball.py` as
      `exp(-(bx²+by²)/0.25²) × exp(-head_yaw²/0.3²)` while seen, weight 2.0,
      locked by `test_find_ball_body_aimed_pays_the_body_not_the_neck`. Both
      factors are detector output + one joint encoder, so it prices exactly
      what `_ball_handoff_due` gates on and needs no privileged state.
      A/B: run `teach-find_ball-3fc099` vs the control `teach-find_ball-5f89d9`
      — identical recipe, identical 3-stage curriculum, 8M steps, same seed
      (the lab always trains seed 0), the **only** difference the weight.

      | 40 static-ball episodes | control (0.0) | **body_aimed 2.0** |
      |---|---:|---:|
      | head yaw while centred, front / side / back | 13.9 / 24.4 / 23.0° | **4.3 / 6.2 / 9.1°** |
      | head yaw while centred, all | 21.1° | **6.4°** |
      | final body bearing, all | 32.4° | **9.3°** |
      | body bearing turned out, all | 57.6° | **80.7°** |
      | **handoff fired, front / side / back** | 60 / 15 / 10% | **100 / 95 / 70%** |
      | handoff fired, all | 25% | **90%** |
      | battery: found, all | 85% | **98%** |
      | battery: back-bucket found | 60% | **100%** |
      | battery: in frame / centred, all | 78% / 76% | **85% / 81%** |
      | **battery: falls / 40** | **0** | **4** |

      → **decide-on met, and not marginally**: head yaw 6.2° on a side start
      against the 14° (0.25 rad) gate, and the handoff fires on **95% of side
      starts** against 15%. Rendered and read (`/tmp/rr-treat-solo`), so this
      is not a reward-sum claim: from a ball at **p+98°** the body is at
      **p+11° by 0.72 s** and holds p+9…p+12 square-on, head straight, for the
      remaining 8 s — aim streaks of **7.2 s and 7.8 s** on the −80° and +98°
      side starts, where the control never held one at all. `trunk_z` 0.115-0.117,
      `floor:none`, 0 reversals.

      **The cost is falls: 0 → 4 per 40** (1 side, 3 back) — it commits to big
      turns now and sometimes topples doing one. Nothing else regressed.
      The weight-down arm (`teach-find_ball-0ad2b0`, `body_aimed` 1.0, same
      chain) says it is a genuine trade, not a mispricing artifact — every
      column moves monotonically with the weight:

      | body_aimed | head yaw, all | handoff, all | battery found | back in frame | falls / 40 |
      |---:|---:|---:|---:|---:|---:|
      | 0.0 (control) | 21.1° | 25% | 85% | 54% | **0** |
      | 1.0 | 10.8° | 78% | 82% | 44% | 2 |
      | 2.0 | **6.4°** | **90%** | **98%** | **80%** | 4 |

      Halving the weight halves the falls and gives up most of the win, so
      there is no free setting on this axis — the falls are the duck
      committing to turns it did not attempt before, and 2.0 is the
      recommended ship. If they have to go, the honest levers are more steps
      or a stability term, not a smaller `body_aimed`. Caveat worth keeping:
      one training seed per arm (the lab pins seed 0), and `AGENTS.md` wants a
      second seed before crediting a small effect — the head-yaw and handoff
      columns are far too big to be seed noise, the ±2 falls plausibly are.

      **Do not read the `--handoff` render as find_ball falling.** With
      `--handoff ball_kick_right`, all 4 episodes fired the handoff (0.72-2.32 s,
      side starts included — that answers item 4's re-render) and all 4 then
      fell within ~0.5 s of the switch. The same 4 seeds with no handoff fell
      **zero** times. That is the kick toppling a duck it was handed at
      0.8-1.9 m, exactly as item 4 predicted; it is an argument for the
      approach behavior, not evidence against this term.
- [x] **Tighten `face_the_ball`'s tight layer** (std 0.4 → 0.2 rad) —
      **SHIPPED.** `_BALL_FACE_TIGHT_STD` is 0.2 in the recipe, and
      `policies/find_ball/policy.onnx` is this arm's export. It beats
      `body_aimed` on the trade that matters. The std is now the named constant `_BALL_FACE_TIGHT_STD`
      (`behaviors/ball.py`) precisely so this A/B is one edit. Arm
      `teach-find_ball-c60e89`, `body_aimed` pinned to 0 so nothing is
      stacked. The feared failure — a steeper Gaussian producing a policy
      that fights to hold an exact pose — did not appear: 0 reversals, and
      rendered it holds a clean square-up.
- [x] **Ungate `turn_to_belief`** — **negative result, and it is the wiggle
      the item predicted.** Arm `teach-find_ball-961dfd` (`body_aimed` 0,
      `_BALL_TURN_GATED_TO_LOST = False`). It is worse than the *untouched
      control* on every axis: handoff 53% vs the control's 38% but with
      **7 falls / 60** against 0, and `psi_turned` collapses to **19.1°**
      (control 42.4°, fix 2 52.1°) while the final bearing gets *worse*
      (70.9° vs 47.6°). Rendered at 50 fps, the mechanism is visible in one
      episode: from a −80° start it turns in to **p−15° by 2.9 s** and then
      **drifts back out to p−28° and parks there** for the last 5 s. A
      yaw-rate pay that never switches off makes *arriving* worth nothing, so
      the policy keeps some bearing in hand to turn back toward. The gate is
      correct; the constant stays `True` and this is why.

### The three fixes, side by side

Every arm is the same 3-stage curriculum, 8M steps, seed 0, via `/teach`;
they differ in exactly one thing each. **Read the events-on block, not the
battery block, when judging falls** — `eval-find-ball` pins
`MICRODUCK_BALL_EVENT_RATE=0`, and the two regimes disagree about which arm
is safest (fix 2 shows 0 falls in the battery and does fall under events).
The recipe trains, and `render-rollout` runs, with events at 0.33.

Battery FINDING table, 40 static-ball episodes (`uv run eval-find-ball`):

| arm | found | in frame | centred | falls |
|---|---:|---:|---:|---:|
| shipped s5 | 85% | 72% | 56% | 2 |
| control (unchanged recipe) | 85% | 78% | 76% | 0 |
| fix 1 `body_aimed` 1.0 | 82% | 71% | 69% | 2 |
| fix 1 `body_aimed` 2.0 | **98%** | **85%** | **81%** | 4 |
| **fix 2 face std 0.2** | 95% | 79% | 76% | 0 |
| fix 3 turn ungated | 85% | 79% | 77% | 3 |

AIMING table, 60 episodes with **ball events on** — the honest regime
(`uv run eval-find-ball <onnx> --episodes 60 --events 0.33`):

| arm | head yaw \| centred | handoff fired | **falls / 60** |
|---|---:|---:|---:|
| shipped s5 | 44.8° | 15% | 2 |
| control (unchanged recipe) | 25.0° | 38% | **0** |
| fix 1 `body_aimed` 1.0 | 13.3° | 68% | 4 |
| fix 1 `body_aimed` 2.0 | **8.1°** | **83%** | **10** |
| **fix 2 face std 0.2** | 18.6° | 68% | **1** |
| fix 3 turn ungated | 20.4° | 53% | 7 |

**Fix 2 is shipped** (`_BALL_FACE_TIGHT_STD` 0.2, and the arm's export
promoted to `policies/find_ball/`). It nearly doubles the handoff rate over
the control (38% → 68%) for one extra fall in sixty, and it is a one-constant
change to a term that already exists. `body_aimed` 2.0 is the better *aimer*
by a distance — it is the only arm that puts head yaw under the gate and the
only one that holds multi-second aim streaks — but 10 falls / 60 is a 17%
fall rate, and this file's own rule is that falls are the veto. `body_aimed`
is therefore left in the recipe **at weight 0**: measured, tested, documented,
one edit away. The thing that would settle it is whether those falls survive
more steps or a stability term, which is the item below, not a fourth term.

**Left in the tree:** fix 2 shipped (`_BALL_FACE_TIGHT_STD = 0.2`); fix 3
rejected and its gate kept (`_BALL_TURN_GATED_TO_LOST = True`); fix 1 present
but unpriced (`body_aimed` weight 0). `policies/find_ball/policy.onnx` is the
fix-2 export — and unlike its cloud-trained predecessor it has a real SB3
checkpoint behind it, at `runs/teach-find_ball-c60e89-s3/` on the machine that
trained it (4.6 MB, not in git: `runs/` is gitignored and this repo does not
ship raw checkpoints). If warm-startability should survive a machine, that is
the decision to make. All five trained chains are under `runs/`
(`5f89d9` control, `0ad2b0` / `3fc099` fix 1 at 1.0 / 2.0, `c60e89` fix 2,
`961dfd` fix 3). The aim probe that produced these numbers is now
**part of `eval-find-ball`** (an AIMING table beside the existing FINDING one,
plus `--events`), not a second command: the two share one rollout loop, so the
battery and the aim columns cannot drift apart, and every future measurement
gets these columns for free. Locked by `tests/test_eval_find_ball.py`.

**Caveat on all of it:** one training seed per arm. The lab pins `--seed 0`
(`TrainingJob` never passes one), so the arms are seed-matched to each other
for free, but `AGENTS.md` wants a second seed before crediting a small
effect. The head-yaw and handoff columns move far too much to be seed noise;
the fall counts (0 / 1 / 4 / 7 / 10) are small enough that only the extremes
are safe to lean on.

Also still open, and probably the same root cause:

- [~] **Train the full circle longer.** The command as written **cannot run**:
      `--init-from runs/find_ball` needs `model.zip` + `vecnormalize.pkl` and
      `policies/find_ball/` has only the ONNX. Answered instead by the control
      arm above — the same recipe trained straight through its own curriculum
      for 8M steps.
      → **half met.** Back-bucket found **60%**, so the ≥ 80% bar is missed;
      but **0 falls / 40** and **0 falls / 60 with events on**, and the
      back-bucket goes from 32%/0% in frame/centred to 54%/52%. The bar is met
      by two other arms — fix 1 at 2.0 (back 100% found, 4 and 10 falls) and
      fix 2 (back 100% found, 0 and 1 falls) — so **fix 2 is the first arm to
      satisfy both halves of this item at once**, on the battery at least.
      Still open: whether fix 2's single events-on fall goes away with more
      steps.
- [ ] **A/B the turn term** at weight 0 vs 1.0, same warm start, same seed, 2M
      steps each (`--weights-json` or the viewer's sliders).
      → still open, but **re-scope it before running it**: the premise was
      that stage 5's `turn_to_belief` bought back-bucket found rate at the
      cost of the chain's only falls. The control arm shows the term is not
      what was doing either — with the identical recipe trained through, back
      found stays 60% and the falls go to **zero**. So most of what stage 5
      looked like it bought (and cost) was under-training. Worth running as a
      clean 0-vs-1.0 A/B on the Mac chain, where it is 14 minutes.
- [ ] **If falls persist:** drop the weight to 0.5 before adding anything new.
      A recipe that needs a fourth term to survive its third is usually
      mispriced, not underspecified.

### 2. Sim2real realism (cheap, run each as a 1M-step fine-tune)

The whole chain trained under `actuator="xml"` with no domain randomization —
a prototype, not a robot policy. These are the knobs that decide whether the
behavior survives contact with reality.

- [ ] **BAM servos.** `MICRODUCK_ACTUATOR=bam` fine-tune.
      → **decide on:** side-bucket median time-to-first-sight stays under 1 s.
      BAM's real current limit slows head yaw, and head yaw *is* the search.
- [ ] **Detector dropout.** `MICRODUCK_BALL_DROPOUT=0.1` fine-tune, then run
      the battery with dropout still on.
      → **decide on:** in-frame share drops by no more than a few points. The
      real NPU detector will miss frames; the brain should not lose the ball
      when it does.
- [x] **FOV sensitivity — NOT a blocker, and the reason is worth knowing.**
      `eval-find-ball` grew the `--env KEY=VALUE` this item always assumed it
      had. Swept far wider than the item asked (40 static-ball episodes each,
      shipped brain):

      | HFOV | 24° | 30° | 40° | **48°** | 56° | 70° | 90° |
      |---|---|---|---|---|---|---|---|
      | found | 95% | 95% | 95% | 95% | 95% | 95% | 95% |
      | in frame | 71% | 77% | 78% | 79% | 79% | 81% | 82% |
      | handoff | 82% | 80% | 80% | 80% | 80% | 78% | 85% |
      | head yaw \| centred | 12.7° | 13.9° | 14.0° | 14.4° | 14.6° | 14.1° | 14.0° |
      | falls | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

      → **decide-on says ship**: found rate does not move AT ALL across a
      3.75× range, and in-frame share degrades gracefully rather than
      swinging. The placeholder IMX219 intrinsics do not have to be measured
      before this goes near hardware.

      The knob was verified to actually reach the detector before believing
      the flatness (`half_h` moves, and a ball 22° off is LOST at 40° and SEEN
      at 56°) — a flat sweep is only good news if the thing swept.

      **Why it is flat, because this is a design property worth keeping:** the
      detector reports `bx = angle / half_h` — a bearing NORMALIZED by the
      field of view — so the policy's control loop works in frame-relative
      units and the geometry cancels. FOV only sets how wide the "seen" window
      is in angle, and a head sweep covers the circle regardless.

      **What this does NOT clear, and it is the sharper version of the same
      worry:** the sweep moves the camera and the normalizer together. The
      dangerous case on hardware is a MISMATCH — a real 56° camera whose
      daemon divides by an assumed 48° — which is a constant gain error on
      bx/by, not a FOV change, and is untested. The contract to write down is
      "bearing normalized by the camera's ACTUAL horizontal/vertical FOV";
      get that right and, per the table, the precise value stops mattering.

- [x] **The camera is not an IMX219, and VFOV — not HFOV — is the axis that
      matters.** The datasheet (2026-09-03) is 1920x1080 on 2.75 µm pixels,
      1/2.9 in: **16:9**, 5.28 x 2.97 mm active, 6.06 mm diagonal, up to
      90 fps. `ball.py` derived its `48° x 62°` pair from a 4:3 IMX219, and a
      4:3 assumption is the one thing a 16:9 sensor definitely breaks — in
      portrait the tall/wide angular ratio is **1.778**, so VFOV should be
      ~77° against that 48°, not 62°. The absolute pair still cannot be
      derived without the **lens focal length**, which is the one number
      nobody has written down (portrait reference: f=2.8 mm → 56 x 87°,
      f=3.6 mm → 45 x 73°, f=4.0 mm → 41 x 67°).

      VFOV swept on the shipped brain, 60 episodes at `--events 0.33`:

      | VFOV | 40° | 50° | **62°** | 77° | 90° |
      |---|---|---|---|---|---|
      | found | 95% | 97% | 97% | 97% | 97% |
      | in frame | 58% | 62% | 66% | 69% | 71% |
      | t_first med | 0.58 s | 0.27 s | 0.24 s | 0.21 s | 0.20 s |
      | falls | 1 | 1 | 1 | 3 | 3 |

      Unlike HFOV (flat), VFOV has a real gradient — 13 points of in-frame
      share — which is what you would expect of a behavior built around
      nodding down for near-floor balls.

      **Orientation matters more than the sensor swap**, and it is the thing
      to confirm on the hardware:

      | mount | found | in frame | centred | handoff | t_handoff |
      |---|---|---|---|---|---|
      | current placeholder 48 x 62 | 97% | 66% | 62% | 68% | 2.08 s |
      | **16:9 PORTRAIT 48 x 77** | 97% | **69%** | 63% | **72%** | 2.10 s |
      | 16:9 LANDSCAPE 77 x 48 | 93% | 60% | 56% | 65% | 3.30 s |

      **The focal length arrived (2026-09-04) and it is a fisheye: EFL 2.9 mm,
      H 116° / V 60° / D 142.2°, max DFOV 165°.** Those are the sensor's own
      axes; mounted rotated 90° they swap into the robot's frame as **60°
      across, 116° up** — far wider than the 48 x 62 placeholder, and tall,
      which is the right shape for a duck hunting a floor ball. The knobs are
      now these values.

      | config | found | in frame | centred | handoff | falls |
      |---|---|---|---|---|---|
      | placeholder 48 x 62 | 97% | 66% | 62% | 68% | 1 |
      | **real, PORTRAIT 60 x 116** | 98% | **75%** | **66%** | **72%** | 5 |
      | real, LANDSCAPE 116 x 60 | 97% | 65% | 61% | 67% | 2 |

      Portrait is better at finding across three eval seeds (+6 to +9 points
      of in-frame share); landscape is worse than the placeholder. **Mount it
      portrait.**

      The projection needed no rework, which is worth knowing before someone
      "fixes" it: `_ball_sense` divides an ANGLE by the half-FOV angle, an
      equidistant f-theta mapping — exactly what a fisheye does. A rectilinear
      (tan/tan) model would have been badly wrong at 58°.

      **Two consequences that are not about numbers:**

      - **The near-floor nod is obsolete.** A ball at the closest spawn
        distance (0.3 m) sits 35.6° below a level gaze: outside the old 31°
        half-VFOV, comfortably inside the real 58°. The whole spawn window is
        now visible with the head level, so the recipe's nose-down pitch band
        (`_BALL_PITCH_DOWN`) covers a case the hardware does not have. Locked
        by `test_find_ball_near_balls_no_longer_need_a_nod` so a narrower lens
        or a landscape remount fails loudly.
      - **The aim tolerances are FOV-RELATIVE, not angular — and that is a
        design bug the wider camera exposed.** `eyes_on_ball`'s stds and the
        handoff gate are all in normalized-bearing units, so widening the
        camera silently loosened them:

        | | placeholder 48 x 62 | real 60 x 116 |
        |---|---|---|
        | "centred" (tight layer) | ±6.0° across, **±7.8° up** | ±7.5° across, **±14.5° up** |
        | kick handoff gate | ±6.0°, ±7.8° | ±7.5°, **±14.5°** |

        The vertical tolerance nearly DOUBLED without anyone editing a reward.
        This matters on hardware too: the daemon runs the same gate, and it
        will now hand the kick a ball 14.5° off vertically.

- [ ] **Retraining at the real FOV made it WORSE — negative result, and the
      tolerance rescale above is the leading suspect.** Run
      `teach-find_ball-c06bbb` (same recipe, 8M steps, only the FOV knobs
      changed). Against the shipped brain, both evaluated in the real-FOV env
      at `--events 0.33`, three eval seeds:

      | seed | old brain (trained 48 x 62) | new brain (trained 60 x 116) |
      |---|---|---|
      | 123 | 98% found, 75% in frame, **5** falls | 88%, 58%, **12** |
      | 200 | 100%, 66%, **1** | 93%, 63%, **9** |
      | 300 | 97%, 64%, **2** | 85%, 58%, **14** |

      Rendered, it is bimodal rather than uniformly bad: two of four episodes
      hold 7.24 s aim streaks at 98-99% in frame, one falls at 1.26 s, one
      holds the ball 24% of the time. So the skill is there and something is
      destabilising it.
      → **done, and the diagnosis was right about the AIM and incomplete about
      the FALLS.** The tolerances are now in degrees (`_BALL_AIM_DEG` 7,
      `_BALL_EYES_TIGHT_DEG` 7, `_BALL_EYES_WIDE_DEG` 16, chosen to reproduce
      what the 48x62 placeholder happened to mean), the battery's `centred`
      column is measured the same way, and the recipe retrained as
      `teach-find_ball-72af49`. Three eval seeds, real-FOV env,
      `--events 0.33`, all three brains scored with the new angular metric:

      | | old brain (48x62) | FOV retrain (normalized) | **angular tolerances** |
      |---|---|---|---|
      | in frame | 75 / 66 / 64% | 58 / 63 / 58% | **86 / 79 / 79%** |
      | centred | 56 / 54 / 54% | 45 / 49 / 45% | **74 / 69 / 70%** |
      | worst t_first | 6.9 / 7.6 / 7.9 s | 7.1 / 5.7 / 4.7 s | **1.1 / 4.6 / 1.7 s** |
      | found | 98 / 100 / 97% | 88 / 93 / 85% | 98 / 92 / 92% |
      | **falls** | **5 / 1 / 2** | 12 / 9 / 14 | 5 / 8 / 11 |

      The aim recovered and then some — best of all three on in-frame, centred
      and worst-case first sight, by a wide margin. **The falls did not.** They
      improved on the FOV retrain (12/9/14 -> 5/8/11) but remain well above
      the old brain's 5/1/2, so the tolerance rescale was *a* cause of the
      regression and not the only one.

      **Keep the angular tolerances regardless of the falls.** They fix a real
      defect that has nothing to do with any policy: with tolerances in
      normalized bearing, changing the camera silently changes what "centred"
      means, what the reward pays, and what the kick handoff accepts — on
      hardware as well as in sim. A recipe whose meaning moves when a lens
      changes is wrong even when it happens to score well.

- [ ] **The remaining falls are an over-aggressive belief-driven turn, and
      that is a stability problem, not an aiming one.** Rendered
      (`/tmp/rr-ang`), 5 of 6 episodes are excellent — in frame 88-99%,
      centred 74-98%, aim streaks to 7.26 s. The one fall is legible and is
      the same shape as the other arms' back-start falls: from a ball at
      p+173° the duck turns ~57° in 0.7 s while `trunk_z` sinks 0.127 ->
      0.066, tilt climbs 0° -> 41°, and it goes single-footed at 0.76 s. It
      never sees the ball at all, so this is purely the belief-driven turn,
      executed as **a one-footed leaning pivot instead of steps**.
      → **decide on:** back-bucket falls at or below the old brain's ~2/60 at
      `--events 0.33`, without losing the in-frame/centred gain above.
      Candidates, cheapest first, and **not by adding a fourth term** (this
      file's own rule — a recipe that needs a new term to survive its last one
      is mispriced).

      **`step_dont_skid` was the obvious candidate and it is NOT the lever.**
      Swept as a weight override through `/teach` (no code change), three eval
      seeds each at `--events 0.33`:

      | `step_dont_skid` | falls / 60 | in frame | centred |
      |---|---|---|---|
      | 0.5 (`teach-find_ball-2a07a0`) | 18 / 12 / 18 | 66 / 68 / 70% | 53 / 55 / 56% |
      | **1.0 — the recipe's own value** | **5 / 8 / 11** | **86 / 79 / 79%** | **74 / 69 / 70%** |
      | 2.0 (`teach-find_ball-e92868`) | 12 / 18 / 17 | 80 / 73 / 74% | 61 / 53 / 57% |

      Both directions are worse on **every** axis, so 1.0 is at or near a local
      optimum and the falls come from somewhere else. Raising it was reasoned
      backwards, and the mistake is worth keeping: the term is a **positive pay
      for a foot being AIRBORNE while rotating** (0.5 per foot inside a
      0.06-0.35 s window), so a bigger weight buys *more* one-footed turning —
      which is precisely the fall mode it was supposed to suppress. It declines
      to pay a sustained lean only because the lean outlasts the window; it
      never charges for one.

      **Revised hypothesis, and it survives everything measured so far: the
      wide camera removed an implicit posture constraint.** Every arm trained
      at the real 60x116 FOV falls more (5-18 per 60) than the brain trained
      at the narrow 48x62 placeholder (5 / 1 / 2), regardless of tolerance
      units or skid weight. With a 31° half-VFOV, tilting far enough to topple
      LOST the ball, so the centring pay was quietly buying uprightness; at
      58° the duck can lean hard and keep the ball in frame, and nothing in the
      recipe replaces what the narrow lens used to enforce.
      **`stay_upright` swept too, and it does not control the falls either.**
      Weight overrides through `/teach`, three eval seeds each at
      `--events 0.33`:

      | `stay_upright` | falls / 60 | in frame | centred | t_first med |
      |---|---|---|---|---|
      | **1.5 — the recipe's own value** | 5 / 8 / 11 | **86 / 79 / 79%** | **74 / 69 / 70%** | **0.18 / 0.22 / 0.34 s** |
      | 2.0 (`teach-find_ball-17d994`) | **21 / 21 / 22** | 71 / 76 / 72% | 62 / 67 / 62% | 0.24 / 0.26 / 0.30 s |
      | 3.0 (`teach-find_ball-60be16`) | **0 / 0 / 0** | 52 / 52 / 49% | 46 / 47 / 45% | 0.88 / 1.84 / 2.31 s |
      | 5.0 (`teach-find_ball-02e577`) | 8 / 4 / 6 | 74 / 75 / 69% | 57 / 58 / 53% | 0.36 / 0.66 / 0.74 s |

      Falls go 5-11 -> 21-22 -> 0 -> 4-8 as the weight rises monotonically.
      That is not a function of the weight in any usable sense, and the one
      setting that does eliminate falls (3.0) buys it by **abandoning the
      task**: in-frame share collapses to ~50% and median time to first sight
      goes from 0.2 s to 0.9-2.3 s. It stops falling by stopping turning.

      **`1.5` stays.** Nothing in the sweep beats it on a single aim column.

      **Training here is bit-for-bit deterministic, which is what makes that
      table interpretable — and it also kills the obvious excuse.** Re-running
      the 1.5 configuration from scratch (`teach-find_ball-38d788`) reproduces
      `teach-find_ball-72af49` EXACTLY: same found rate, in-frame, centred and
      fall count on all three eval seeds. So none of the swings above are
      run-to-run noise; they are exact single-sample draws, and the map from
      this weight to the fall count is simply not smooth. Two consequences:

      - A weight sweep at one training seed cannot find a "good weight" for
        falls, because neighbouring weights land in qualitatively different
        basins. More sweeping is not the answer.
      - Determinism is a *reproducibility* guarantee, not a *generalization*
        one. Every arm in this file is one draw from seed 0 (the lab pins it);
        a second seed would need the CLI and a hand-chained curriculum.

- [ ] **The falls want a structural fix, and there is a specific candidate
      this codebase has already used once.** `_upright` pays
      `exp(-sin²(tilt)/0.05)` — a std of ~12.9° of tilt, so it is worth 0.55
      at 10°, 0.10 at 20° and 0.007 at 30°. Past ~25° there is essentially no
      pay left and therefore **no gradient pulling the duck back**: the term
      prices being upright but not *recovering*, so once the lean is committed
      the fall is free. That is exactly the failure `face_the_ball`'s docstring
      records for its own std-1.5 Gaussian ("paid 0.04 with the ball straight
      behind and sloped nowhere the policy was"), and the fix there was a
      raised cosine that slopes all the way round.
      → **tried, and it is Pareto-DOMINATED.** `_upright_wide` (in the catalog,
      `_upright_term(w, wide=True)`) is the two-layer shape `eyes_on_ball` and
      `head_up` use: `0.5*exp(-s/0.5) + 0.5*exp(-s/0.05)`, worth 0.44 at 20°
      of tilt, 0.31 at 30 and 0.21 at 41 where the narrow term pays 0.10,
      0.007 and 0.0002. It is **opt-in**, because `_upright` gates `one_leg`'s
      hold and SCALES backflip's brake and two of imitate's terms — widening
      it globally would silently re-price four other behaviors.

      A/B on find_ball (`teach-find_ball-fe8c25`) did cut the falls and did
      cost the aim: 5/4/5 against 5/8/11, in-frame 66/61/54% against
      86/79/79%. Rendered, 4/4 upright with no falls — but the aim streak is
      **zero in three of four episodes**: it keeps the ball and stops squaring
      up. Nothing adopts it; `test_no_recipe_has_adopted_the_wide_upright`
      locks that and points here.

### The falls/aim frontier — why three sweeps in a row "failed"

Every arm trained at the real FOV, mean over three eval seeds at
`--events 0.33`:

| arm | falls / 60 | in frame | centred | on the frontier |
|---|---:|---:|---:|---|
| `stay_upright` 3.0 | **0.0** | 51.0% | 46.0% | yes |
| **old brain (trained at the NARROW 48x62)** | **2.7** | 68.3% | 54.7% | **yes** |
| wide `_upright` | 4.7 | 60.3% | 50.0% | no — dominated |
| `stay_upright` 5.0 | 6.0 | 72.7% | 56.0% | yes |
| **angular tolerances, recipe as it stands** | 8.0 | **81.3%** | **71.0%** | **yes** |
| `step_dont_skid` 2.0 | 15.7 | 75.7% | 57.0% | no |
| `step_dont_skid` 0.5 | 16.0 | 68.0% | 54.7% | no |
| `stay_upright` 2.0 | 21.3 | 73.0% | 63.7% | no |

The three sweeps did not fail to find a good weight. They found that **falls
and aim are in direct tension in this recipe, and every reward lever tried
moves ALONG that frontier rather than pushing it out**: 0 falls at 51%
in-frame, 2.7 at 68%, 6.0 at 73%, 8.0 at 81%. Pick a point; there is no
setting that gets both.

The most informative row is the second. The brain trained at the WRONG,
narrow camera sits on the frontier and beats every wide-FOV-trained arm at
its fall count — because at a 31° half-VFOV, leaning far enough to topple
LOST THE BALL, so the centring pay was buying uprightness for free. The real
lens removed that coupling, and no reward term has replaced it.

- [ ] **Push the frontier out by changing the WORLD, not the pay.** Three
      reward sweeps is enough evidence that this is not a pricing problem, and
      it is this file's own lesson: "if rollouts never contain the skill you
      are paying for, ladder the physics, not the reward" (`AGENTS.md`). The
      skill missing here is **recovering from a committed lean**, and no
      episode ever starts in one — the duck only ever reaches 30-40° of tilt
      on its way to the floor, where the value function has nothing to learn
      from. Give it a spawn family that starts already leaning 20-40°, the
      reverse-curriculum pattern `backflip` and `headstand` use, so recovery
      is practised rather than hoped for.
      → **built and A/B'd. It teaches real recovery, it does NOT move the
      frontier, and it turned up the physical reason why.**
      `_ball_spawn_leaning` starts 20-40° tipped about a random horizontal
      axis and still tipping (`MICRODUCK_BALL_LEAN_LO_DEG` / `_HI_DEG` /
      `_RATE`), using the backflip spawn's attitude-aware clearance so nothing
      wedges. Validated before training: 22.5% of resets, tilt 20.1-39.9°,
      **zero** terminating on step 1 — and the shipped brain survives only
      **1/30** of them, which is the gap stated as a number.

      Trained at a 0.25 mix (`teach-find_ball-3c1b2e`) and evaluated on the
      STANDARD spawn distribution (`--env MICRODUCK_SPAWN_FAMILY_PROBS=0.0`,
      or the battery silently tests a different thing):
      **falls 8.0 → 2.7 per 60, in-frame 81.3% → 68.0%.** That is
      (2.7, 68.0) — the old brain's point, reached for the first time by a
      policy trained on the RIGHT camera, and still on the frontier rather
      than outside it. Decide-on not met.

      **The recovery skill is real, and only visible when split by angle.** A
      first pass over the whole 20-40° band read 18% vs 15% and looked like
      nothing; the band is wide enough to average a real effect away:

      | spawned tilt | 5-10° | 10-15° | 15-20° | 20-25° | 30-35° |
      |---|---|---|---|---|---|
      | baseline survival | **87%** | 80% | 50% | 20% | **0%** |
      | lean-trained | 77% | 77% | **67%** | **47%** | **17%** |

      It roughly doubles survival at 20-25° and takes 30-35° from zero, paying
      a little near-upright robustness for it.

- [x] **Why the frontier is real: the fall line is ~20-25° of tilt, not 70°.**
      `FALL_GRAVITY_Z` terminates at 70°, which is nowhere near where this
      robot is actually lost — even the lean-trained brain only saves 17% of
      30-35° leans, and neither arm saves anything past ~40°. So a big body
      turn that produces a 30° lean is already a fall; the policy cannot
      "recover better", it can only not get there. **That is the frontier's
      physical root**, and it explains why three reward sweeps and a spawn
      curriculum all slid along the same line: every one of them was trading
      turn commitment for tilt, because tilt past the fall line is
      unrecoverable by construction.

      Consequence for what to try next: stop looking for a reward or a spawn
      that buys both. The remaining honest options are (a) **pick a point** —
      or (b) **change how the duck turns**, so a big turn does not produce a
      30° lean in the first place. (b) is a gait/technique problem — stepping
      round versus pivoting — and belongs to locomotion, not to this recipe's
      reward. It is also the one thing here that could move the frontier
      rather than slide along it.

- [x] **Point picked and SHIPPED: the lean-trained arm** (`teach-find_ball-3c1b2e`
      → `policies/find_ball/`), with the spawn family on at 0.25.

      Calling it "the safe point" undersold it. It loses in-frame share and
      **wins the deliverable**, on all three eval seeds:

      | | aim-heavy (`72af49`) | **shipped** (`3c1b2e`) |
      |---|---:|---:|
      | in frame | **86 / 79 / 79%** | 63 / 70 / 71% |
      | **handoff fired** | 85 / 77 / 82% | **92 / 87 / 93%** |
      | head yaw \| centred | 15.9 / 14.2 / 11.9° | **9.5 / 11.5 / 9.5°** |
      | falls / 60 | 5 / 8 / 11 | **2 / 3 / 3** |

      In-frame share is a means; the kick handoff is the end, and a brain that
      holds a ball in frame while never squaring up is precisely the failure
      item 1 opened with. 92% handoff at 9.5° of head yaw, under the gate,
      with a third of the falls. Rendered before shipping: 3 of 4 episodes hold
      7.5 s / 2.6 s / 7.4 s aim streaks, the fourth falls on a +173° back start.

      **`eval-find-ball` now pins `MICRODUCK_SPAWN_FAMILY_PROBS=0.0`.** The
      recipe trains with leaning starts; the battery must not fire them, or it
      silently stops being the test every number in this file was taken with.
      Pass `--env MICRODUCK_SPAWN_FAMILY_PROBS=0.25` to measure the training
      distribution on purpose. Locked by a test — this is the same class of
      trap as measuring `centred` in normalized bearing across two cameras.

      **Left in the tree, deliberately inconsistent:** the knobs are the real
      camera's, and `policies/find_ball/policy.onnx` is still the brain
      trained at 48 x 62 — because it is *better in the real-FOV env* than the
      one trained there. Fidelity beats consistency here: modelling the wrong
      camera would have hidden all of the above. One training seed per arm, as
      always.

- [x] **Detector rate is the real compute requirement: ≥ 10 Hz.** The sensor's
      90 fps is ~9x more than this pipeline can consume, so frame rate is not
      where compute should go; the NPU's detection rate is. Swept
      `MICRODUCK_BALL_DETECT_EVERY` against the 50 Hz control loop, 60
      episodes at `--events 0.33`:

      | detector | 50 Hz | 25 Hz (modelled) | 17 Hz | **10 Hz** | 6 Hz | 4 Hz |
      |---|---|---|---|---|---|---|
      | in frame | 65% | 66% | 66% | 64% | 51% | 19% |
      | centred | 62% | 62% | 60% | 60% | **23%** | **5%** |
      | handoff | 80% | 68% | 72% | 75% | **13%** | **0%** |
      | falls | 1 | 1 | 1 | 1 | 4 | 16 |

      → **50 / 25 / 17 / 10 Hz are indistinguishable, and it falls off a cliff
      between 10 and 6** — *as measured then, before the stale-pose fix below.
      The cliff turned out to be an artifact and the requirement is now ~4 Hz;
      this table is what an uncompensated detector costs.*

      **CORRECTED — it IS the tidy pipeline's stale-pose bug, and the same
      cure works.** An earlier revision of this item concluded the opposite.
      That conclusion came from a bug in the compensation, not from the data.

      find_ball holds `det[0]/det[1]` — a bearing measured off the camera at
      capture — unchanged while the head keeps sweeping, so the policy acts on
      a pose the duck has already left. The error tracks the cliff exactly:

      | detector | 25 Hz | 17 Hz | **10 Hz** | **6 Hz** | 4 Hz |
      |---|---|---|---|---|---|
      | stale bearing, mean / p95 | 0.4° / 2.0° | 0.6° / 3.1° | 1.1° / **5.0°** | 5.5° / **20.9°** | 31° / 79° |

      against an aim tolerance of 7°. Rotating the held report by the head's
      own rotation since capture — head encoders and the gyro, no new sensor,
      the same ingredients `brain/tidy.py`'s `stale_fix` used — **removes the
      cliff outright.** Shipped brain, unretrained, 60 episodes at
      `--events 0.33`:

      | rate | handoff / falls, fix OFF | fix ON |
      |---|---|---|
      | 25 Hz | 92% / 2 | 90% / 2 |
      | 10 Hz | 88% / 2 | 88% / 3 |
      | **6 Hz** | **35%** / 3 | **83%** / 5 |
      | **4 Hz** | **2%** / **16** | **80%** / **4** |

      → **the requirement drops from ≥ 10 Hz to ~4 Hz**, for about two points
      of handoff at the nominal rate. Worth taking: the NPU's real throughput
      is the least certain number in the pipeline. `MICRODUCK_BALL_STALE_FIX`
      is ON by default.

      **Two mistakes made getting here, both worth keeping.** The signs were
      guessed first, which made the error ten times worse; they are measured
      now (d(bx)/d(cam yaw) = +1/half_h, d(by)/d(cam pitch) = −1/half_v). Then
      the correction was written onto `det` in place while measuring rotation
      from capture, so every held step re-added the whole accumulated rotation
      — ~2.5× over-correction at 10 Hz and worse below. That version degraded
      every rate and read convincingly as "the compensation does not work",
      which is exactly what produced the wrong conclusion. It corrects from
      the captured bearing now, and a test locks the linear-growth property.

      **A FRESH chain trained with it on is worse; a WARM START from the
      shipped brain is the best result in this file.** Both were run, and the
      difference between them is the finding:

      | | fresh chain (`38ffd2`) | shipped + fix, no retrain | **warm start (`f31a4f`)** |
      |---|---:|---:|---:|
      | found | 90% | 95% | **100%** |
      | in frame | 72% | 63% | **74%** |
      | handoff | 77% | 90% | **93%** |
      | falls / 60 | **25** | 2 | **1** |

      A fresh chain rediscovers the whole behavior against the corrected signal
      and lands where every aim-heavy arm lands — more tracking, far more falls.
      Warm-starting keeps the low-fall policy the leaning-spawn curriculum
      bought and only adapts it to the new observation, which is why it gains
      tracking **without** paying for it. It also holds up as the detector
      slows: handoff 97% at 10 Hz, 92% at 6, 90% at 4.

      That makes it the **first change in this file to move off the falls/aim
      frontier rather than along it** — and the second is worth naming too: the
      other one was also not a reward (the leaning spawns). Both were changes
      to the world or to what the policy is told, not to what it is paid.

      Shipped: `policies/find_ball/` is `teach-find_ball-f31a4f`.

      **The trap in this table is worth its own line:** `found` stays 97% at
      EVERY rate, including the 4 Hz column where the handoff never fires and
      the duck falls in 16 of 60 episodes. "Ever saw the ball" is satisfied
      eventually by any detector. Only the AIMING columns see the collapse —
      one more reason the battery grew them.

### 3. Search speed itself — the actual ask

### Time spent LOST — the strongest signal found so far, and it says ship the blind recipe

Chasing the one column that tracked performance in the sweep audit above.
"Steps spent lost" conflates two failures, so split them: **search** (steps
before the ball is EVER seen) and **tracking** (steps lost after the first
sight — dropping a ball you already had). 40 episodes, no prior:

With a STATIC ball, tracking is essentially solved for every arm (0.0-3.7%
lost after first sight) — so with events off, "time lost" is **search, almost
entirely**. With the recipe's own events (0.33) the two separate:

| arm | search steps | tracking lost | drops/ep | regain med / p90 | never regained |
|---|---:|---:|---:|---:|---:|
| shipped s5 (cloud) | 3689 | 26.0% | 0.82 | 0.76s / 4.80s | 12 |
| control | 3013 | 18.7% | 0.82 | 0.52s / 2.32s | 9 |
| fix1 `body_aimed` 2.0 | 1943 | 27.0% | 0.82 | 1.41s / 3.38s | 15 |
| fix2 (shipped) | 2313 | 17.2% | 0.95 | **0.27s** / 0.71s | 14 |
| fix3 turn ungated | 2010 | 32.7% | 0.72 | 0.49s / 3.66s | 17 |
| **blind-trained** | **735** | **8.5%** | 1.07 | 0.45s / 1.11s | **3** |

The blind-trained arm searches 3x faster than the shipped brain and loses the
ball half as often, and it is the cleanest A/B on this page: it is the fix2
recipe with **one knob changed**, `MICRODUCK_BALL_PRIOR_PROB` 0.7 -> 0.0,
same curriculum, same 8M steps, same seed.

**Head to head in the deployment regime** (60 episodes, events 0.33):

| brain | prior at eval | found | in frame | centred | head yaw | handoff | falls | worst t_first |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| fix2 (shipped) | none | 97% | 68% | 63% | 19.4° | 73% | 1 | 6.50 s |
| fix2 (shipped) | on | 95% | 66% | 62% | 19.1° | 68% | 1 | 7.72 s |
| blind-trained | none | **100%** | **83%** | **74%** | **15.3°** | **77%** | 1 | 5.78 s |
| **blind-trained** | **on** | **100%** | **85%** | **78%** | **14.2°** | **78%** | 1 | **1.16 s** |

It dominates every column in both prior conditions, back-bucket found goes
94% -> 100%, and the side bucket's worst-case time to first sight collapses
from **6.30 s to 0.58 s** — that is the wrong-side tail this section is about,
essentially gone.

**This overturns the previous section's recommendation, and the error is worth
naming: those conclusions were measured with `--events 0`.** At events 0 the
shipped brain looked like the better aimer (93% handoff vs 68%); with the
events the recipe actually trains on, the ordering flips (73% vs 77%). That is
the second time in this file the static-ball battery has pointed the wrong way.
**Judge find_ball at `--events 0.33`.** Full stop.

Rendered before believing it (`/tmp/rr-blind`): 4/4 episodes upright, in frame
86-98%. From a ball at **p+173°** — directly behind — it turns ~180° by
**1.44 s**, holds the bearing within a few degrees for the rest of the episode,
and re-acquires cleanly after a mid-episode ball event. The shipped fix2 brain
FELL on that same seed.

**Why, and it is the project's own lesson:** with `PRIOR_PROB=0.7`, 70% of
training episodes started by handing the duck the answer, so a real
search-from-nothing was under-practised — the same shape as the headstand
lesson in `AGENTS.md` ("if rollouts never contain the skill you are paying
for, change the world, not the pay"), except here the world was too *easy*
rather than too hard. Removing the prior makes every episode a real search.
Note what it does NOT do: the blind-trained brain still uses a prior perfectly
well when handed one — better than without — so this is about what it
PRACTISED, not what it can read.

- [ ] **Ship it?** The case is strong and one-knob, but it rests on a **single
      training seed** (the lab pins `--seed 0`), and it would mean replacing
      `policies/find_ball/` a second time and flipping a default this file
      currently justifies at 0.7. Effect sizes are far larger than the
      run-to-run variance seen elsewhere here (in frame 68 -> 83, worst
      t_first 7.72 s -> 1.16 s), which is the argument for; one seed is the
      argument for a confirmation run first.
      → **decide on:** a second blind-trained chain reproducing in-frame ≥ 80%
      and back-bucket found ≥ 95% at `--events 0.33`.


**Why this section now has a mechanism, not just a hunch (measured
2026-09-03).** "Which way should it look first?" has a fixed answer: with
nothing known the belief slot is seeded `_BALL_NO_PRIOR_MEM = +pi/2`, "sweep
left first". Splitting the battery by which side the ball was really on shows
what that convention costs, on the shipped brain, blind episodes, balls
outside the initial view (n=66):

| ball side | found | t_first mean | t_first median |
|---|---:|---:|---:|
| LEFT (with the convention) | 100% | **0.35 s** | 0.18 s |
| RIGHT (against it) | 97% | **2.48 s** | 2.75 s |

A **7x mean penalty** for being on the wrong side; 17 of 40 right-side balls
took over 2 s against 2 of 40 on the left. Note what this is NOT: it is not
fixable by choosing the side better. Spawns are uniform in bearing, so no
observable information distinguishes left from right, and the convention
exists precisely because a symmetric obs leaves the mean action with nothing
to learn (AGENTS.md). The wrong guess is unavoidable half the time; what is
tunable is what it COSTS, which is ~2/3 of a sweep period. That makes
SCAN_PERIOD the dominant term in mean time-to-first-sight for half of all
blind episodes, and it is why the item below is now the most valuable one
here rather than a nice-to-have.

**Unexplained, and probably the same bug as the prior item below:** force the
prior ON and the asymmetry REVERSES — left 2.29 s / 91% found, right 1.61 s /
97%. If the convention were the whole story, a seeded belief should make the
two sides symmetric, not swap them. Worth understanding before tuning either.

- [ ] **Faster sweep.** `MICRODUCK_BALL_SCAN_PERIOD=2.5` (from 4.0) fine-tune.
      → **decide on:** median time-to-first-sight on side and back, weighed
      against falls and centred share. A faster sweep that loses the ball on
      the way past is not faster. **Split the result by ball side** (above):
      the number this is really moving is the 2.48 s wrong-side mean, and a
      whole-battery median will hide it behind the 0.35 s right-side half.
- [x] **Is the prior worth producing? NO — and an ORACLE prior is worse than
      none, which rules out "the prior is just too noisy".** 60 episodes each
      on the shipped brain:

      | belief seeding | found | t_first med | in frame | centred | falls |
      |---|---:|---:|---:|---:|---:|
      | **blind (convention only)** | **98%** | 0.36 s | **85%** | **84%** | **1** |
      | prior, ORACLE (noise 0) | 93% | 0.36 s | 74% | 71% | 2 |
      | prior, noise 0.3 rad | 92% | 0.24 s | 73% | 70% | 2 |
      | prior, noise 0.6 rad (the recipe) | 93% | 0.22 s | 75% | 72% | 2 |
      | prior, noise 1.2 rad | 97% | 0.34 s | 77% | 74% | 2 |

      → **decide-on met: the daemon does not need to synthesize a prior**, and
      the slot can carry the sweep convention alone. Note the shape: prior
      ACCURACY is irrelevant (93 / 92 / 93 / 97% found across a 4x noise
      range) while prior PRESENCE costs ~11 points of in-frame share. A belief
      pointing exactly at the ball still loses. So this is not a bad estimate,
      it is the mechanism: a seeded belief buys a marginally faster first
      sight (0.22 s vs 0.36 s) and then keeps the ball worse.

      **Why, from probing the exported ONNX directly** (feed one obs, vary one
      slot, read the action — the method that found the original
      symmetric-obs bug). The belief slot is NOT ignored and NOT sign-flipped:
      with the ball lost, head yaw responds monotonically and correctly across
      the whole range. But the response is badly **asymmetric in magnitude**,
      and that is what produces every oddity here:

      | belief slot | −1.0 | −0.45 | −0.25 | **0.0** | +0.15 | +0.25 | +0.45 | +1.0 |
      |---|---:|---:|---:|---:|---:|---:|---:|---:|
      | head-yaw action | −3.75 | −2.10 | −1.45 | **−0.39** | +0.23 | +0.57 | +1.09 | +2.13 |

      A rightward belief commands a **2.6× stronger** turn than the mirrored
      leftward one, and a NEUTRAL belief already commands −0.39 (rightward).
      Two consequences, both of which had been observed and unexplained:

      - The no-prior convention (+0.15) sits in the flattest part of that
        curve — it produces a limp +0.23 — which is why blind episodes find
        left-side balls in 0.35 s and right-side ones in 2.48 s (section 3's
        table above).
      - Force a prior on and the asymmetry **reverses** (left 2.29 s, right
        1.61 s), because now the slot carries a real bearing and rightward
        ones pull much harder. That was the unexplained reversal noted above;
        this is the explanation.

      **RETRACTED — that 2.6× is an artifact of the probe, not a property of
      the policy.** `vecnormalize` says slot 54 has **std 0.11** against a
      nominal range of ±1, so sweeping the slot to ±1 probes the network at
      **±8 sigma**: inputs it has effectively never seen. Three checks, all
      negative:

      - The baked normalizer is symmetric — `mean[54]` is -0.004..+0.011
        across all five arms, and ±1 maps to ±8.2..8.8 sigma with a
        |left|/|right| ratio of 0.95-1.02. The normalizer is not doing it.
      - Within ±2 sigma the head-yaw response is monotonic with a **correctly
        signed positive slope in every arm** (+0.31 to +0.71 per sigma). What
        looked like asymmetry at the extremes is a large constant offset,
        and that offset is just *where the scan clock happens to be pointing*
        — the probe had frozen obs[59]/obs[60] at one phase.
      - Swept across a full clock cycle instead, the belief slot shifts the
        sweep's **centre** while leaving its **span** roughly unchanged, which
        is exactly what a belief is supposed to do.

      Measured behaviourally instead — head yaw over 40 blind episodes on
      every step the ball is not seen — the arms DO sweep lopsidedly, and it
      turns out not to matter:

      | arm | steps lost | span | L/R balance | found |
      |---|---:|---:|---:|---:|
      | shipped s5 (cloud) | 4737 | 192° | 0.87 | 82% |
      | control | 2749 | 177° | **0.99** | 95% |
      | fix1 `body_aimed` 2.0 | 2255 | 90° | 0.44 | 98% |
      | **fix2 (shipped)** | 2173 | 181° | 0.66 | **100%** |
      | fix3 turn ungated | 2974 | 153° | 2.43 | 75% |
      | blind-trained | **769** | 159° | 0.62 | **100%** |

      Balance ranges 0.44-2.43 and predicts nothing: the control has the most
      symmetric sweep of any arm (0.99) and is not the best finder, while fix2
      is lopsided (0.66) and finds everything. **The column that tracks
      performance is steps spent LOST** — 4.8% of steps for the blind-trained
      arm against 30% for the old cloud export — which is re-acquisition and
      tracking, not search symmetry. Sweep span does not explain it either
      (blind-trained sweeps NARROWER than fix2 and finds better).

      So: the belief slot is read correctly, the sweep asymmetry is real but
      inert, and the only left/right effect with a measured cost is the plain
      one in section 3 — the convention decides which way it looks FIRST, so
      left balls come in at 0.18 s and right balls wait for the sweep.

      Method note worth keeping, since this cost two wrong hypotheses: before
      reading anything off a slot sweep, check that slot's std in
      `vecnormalize.pkl` and stay inside ±2 sigma, and vary the scan clock
      rather than freezing it — a frozen clock turns "what phase am I at" into
      a fake constant bias.

      Caveat on method: sweeping slot 54 while holding bx and head yaw fixed
      leaves the manifold for the SEEN case (while the ball is visible the
      slot carries the body-frame bearing, which is tied to bx and head yaw by
      geometry), so only the ball-LOST rows above are trustworthy. They are
      the rows that matter — the belief exists for when the ball is lost.

      **CORRECTION, from the blind-TRAINED arm this queued**
      (`MICRODUCK_BALL_PRIOR_PROB=0` through the whole curriculum, run
      `teach-find_ball-31f14b`). Everything above is measured on a brain that
      saw a prior in 70% of its training episodes, and **it does not
      generalize**: fed a prior, the blind-trained brain gets BETTER, not
      worse. The effect is an interaction between how a brain trained and what
      it is fed at run time — not a fact about priors. 60 episodes per cell:

      | | evaluated BLIND | evaluated WITH prior |
      |---|---|---|
      | **trained with prior** (the shipped brain) | 98% found · 85% in frame · 84% centred · **93% handoff** · 1 fall | 93% · 75% · 72% · 75% handoff · 2 falls |
      | **trained blind** | 100% · 94% · 86% · 68% handoff · 0 falls | **100% · 95% · 92%** · 73% handoff · **0 falls** (worst-case first sight **0.98 s**, against 6.30 s fed nothing) |

      Neither arm dominates, and the two halves of the task come apart:

      - **Best FINDER: blind-trained, fed a prior** — 95% in frame, 92%
        centred, no falls, and a worst case of 0.98 s where every other cell
        has a 6-8 s tail. Training without the prior makes a brain that
        searches better and then *uses* a prior well when handed one.
      - **Best AIMER: the shipped prior-trained brain, fed nothing** — 93% of
        episodes reach the kick handoff, against 68-75% in all three other
        cells. Aiming is the deliverable, so this is the cell that matters
        most today.

      So the section's decide-on ("does the daemon need to synthesize a
      prior?") splits by which brain ships. **For the brain in
      `policies/find_ball/`, no — feed it the sweep convention and it aims
      better** (93% vs 75%). The stronger claim, that a prior is dead weight
      in general, is WRONG and the row above is the counterexample.

      `MICRODUCK_BALL_PRIOR_PROB` is left at its 0.7 default: nothing here
      justifies changing what the recipe trains, and the two candidate
      recipes trade finding against aiming rather than one beating the other.
      The experiment worth running next is the obvious missing cell —
      blind-trained, then judged on the handoff after more steps — since the
      only thing it is clearly worse at is the one thing this behavior is for.

### 4. The soccer handoff — the demo that proves the point

`find_ball` aims; `ball_kick_right.onnx` kicks but is blind. One clip of find →
square up → kick is the whole argument for this behavior.

- [x] **Handoff condition — done.** It lives on the behavior
      (`Behavior.handoff_fn`), so `render-rollout --handoff` and the lab's
      showcase duck ask one implementation instead of two copies kept in step
      by a comment. Fires on: detector reports the ball centred (|bx|, |by| <
      0.25) **and** the head straight ahead (|head_yaw| < 0.25 rad), held
      0.5 s. Head centred plus head aligned means the *body* is pointing at
      the ball, and both halves are detector output plus joint encoders — the
      daemon can run the same test, no privileged state. `handoff_policy`
      names the kick; `handoff_recenter=False` keeps the lab from spinning
      away the turn the duck just made.
- [x] **Rendered it, and it found the real problem.** The gate is correct and
      the policy does not meet it: an episode with the ball centred 98% of the
      time never handed off, because the duck aims with 21° of head yaw and
      leaves its body put (item 1). The two episodes that *did* fire were
      near-frontal starts, and the kick toppled the duck within 0.4 s in
      both — expected, since it was handed a ball 0.9–1.3 m away.
- [x] **Re-rendered after item 1 — the handoff fires from a side start, and
      the kick does not connect.** Both halves of the prediction confirmed, on
      the `body_aimed` 2.0 arm with
      `--handoff ../microduck/policies/ball_kick_right.onnx`:
      **all 4 of 4 episodes fired** (t = 0.72, 0.72, 1.30, 2.32 s), including
      the **+70° and +98° side starts** that never fired before — against 2 of
      4 near-frontal-only on the shipped brain. And all 4 then **fell within
      ~0.5 s of the switch**. The same 4 seeds rendered with no handoff fell
      **zero** times, so this is the kick toppling a duck handed a ball at
      0.8-1.9 m, not the aimer failing — exactly the "asserts *aimed*, never
      *in range*" gap. The aiming half of the soccer demo is done; the clip
      needs the approach behavior below.
- [ ] **(Stretch) Approach.** Walking to the ball is a `forward_cmd` locomotion
      task steered by the bearing slot — a different recipe, not a stage of
      this one. Range needs the detector to report box size (distance ≈
      focal × real diameter / box height), which the sim's fake detector does
      not yet produce; adding it is a prerequisite, not an afterthought.

### 5. Lab ergonomics, while the above trains

- [ ] **Teach a `find_ball` chain from the browser** (`/teach`, or the 🎓 panel)
      and use the `watch-training` skill mid-stage.
      → **decide on:** do the three stage descriptions read correctly in the
      viewer's stage inspector, and does the ball marker follow a teleport
      without visible lag at 25 Hz? The stage narration and the marker stream
      have only ever been exercised headlessly.

---

## Track 12 — does the carry need its own walker? (sim-roadmap 12.5) — CLOSED NEGATIVE (2026-09-04)

Context: the playroom's pick-up stack is the shipped `alpha_walking` walker,
the shipped `alpha_ground_pick.onnx` as a hard-swapped skill, and the
hand-written `tidy` state machine (`brain/tidy.py`). Nothing learned in this
repo tidies — every brain under `microduck_local/brains/` is a `follow`
brain, and `train-brain` has no tidy task. So "a better model for pick-up"
means one of the open Track 12 items, and 12.5, a carry-walk reflex, is the
one with a ready-made benchmark (`eval-tidy`) and a walker slot to drop it
into (`eval-tidy --walker`, `trace-tidy --walker`, `walker-facts --walker`).

What is already known, and it cuts against the item:

- `docs/sim-roadmap.md` records 12.5 as **not needed** — "the shipped walker
  carries a 20 g block". The heaviest toy IS 20 g (`PICKABLE_KINDS`: brick
  2.5 g, block 20 g, sock 20 g); upstream's carry DR is 10–40 g.
- The one traced fall study (README, the `backoff_back_s` A/B, 80 seeds) put
  the falls in **`approach`** and at the basket rim — the 6 cm rim sits under
  the ToF guard until the last 0.26 m — not in `carry`. Falls per run rise
  0.31 → 0.56 → 0.75 across ideal → datasheet → hostile odometry, which is
  a heading/route effect, not a payload effect, unless step 1 says otherwise.
- **Locomotion training here is a clone, not PPO**: from-scratch PPO stands
  still (0.001 m/s), `uv run distill` clones the walker to 0 falls in ~1 min,
  and PPO on top of the clone is a resolved bad trade over 14 paired seeds
  (fall rate +0.44). A carry variant would have to be a clone with a
  payload in the loop, and it would have to clear the same bar: judged by
  `select-run` (falls as a rejection floor), never `ep_rew`.

So the item was a measurement first and a training run only if the
measurement earned it. It did not: **the answer is below, and the next
lever for the tidy loop is the basket rim, not the walker.** The scratch
tally that produced the table is gone; its durable version is in
`eval-tidy` itself (2026-09-04): every result row carries
`falls_by_state`, `falls_laden` / `falls_unladen` and `m_laden` /
`m_unladen` (true trunk metres, the respawn teleport excluded), each seed
line prints the split (`falls 1 (deliver 1 · 1 laden)`), and the summary
pools the states and prints laden vs unladen falls per 100 m. Rows in an
older `--out` file still resume and still count in falls/run; the pool
says how many seeds it covered. So the table below is re-measured by
every battery from now on — `eval-tidy --seeds 16 --seconds 300 --odom hostile`
is the command, and a laden rate above the unladen one is what would
reopen this item.

- [x] **Step 1 — count the falls by state and by `holding` — DONE, and it
      says no: the walker does not fall because it is carrying.** Ran the
      benchmark loop (`eval-tidy`'s own `run_one` shape, via `trace_tidy.senses_of`)
      with a per-step tally of state, `holding`, true trunk metres and each
      fall, **64 seeds × 300 s × 6 toys per odometry preset**, ~5 s a seed
      on the M-series Mac (2026-09-04, upstream models at the pinned shas):

      | odometry | tidied | falls/run | laden falls / m | unladen falls / m | laden per 100 m | unladen per 100 m |
      |---|---:|---:|---:|---:|---:|---:|
      | ideal | 0.875 | 0.42 | **0** / 844 | 27 / 1770 | **0.00** | 1.53 ± 0.29 |
      | datasheet | 0.888 | 0.39 | **0** / 873 | 25 / 1751 | **0.00** | 1.43 ± 0.29 |
      | hostile | 0.818 | 0.53 | 5 / 886 | 29 / 1730 | 0.56 ± 0.25 | 1.68 ± 0.31 |

      **Zero falls in 1.7 km of laden walking** under ideal and datasheet
      drift, and under hostile drift the laden rate is a third of the
      unladen one — every one of those five was in `deliver`, i.e. the
      blind final leg at the basket rim, not the carry. The falls live at
      the basket: by state, pooled over the three presets (81 falls),
      `backoff` 29, `drop` 27, `deliver` 5 — **75% within a metre of the
      rim** — then `approach` 13, `explore` 8, `scan` 3, `blind` 1. (A
      `drop` fall counts as unladen because the weld is already released;
      it is the same rim event.) The tidied fractions reproduce the README
      benchmark's 0.89 / 0.84 / 0.79 within the ±0.03 run-to-run band.

      **Verdict: the shipped walker carries every toy in the playroom
      (2.5–20 g) without a single payload-attributable fall.** 12.5 stays
      closed, now with a number behind it rather than a 20 g block. The
      fall budget is the basket rim, which is 12.6/12.7 work in the brain
      (the release stance and the retreat), not a reflex.
      Original plan for this step:
      `eval-tidy` records one `falls` integer per seed; `trace-tidy` prints
      each fall with its state and `holding`. Run the trace over the
      benchmark seeds under the drift preset where falls are highest:
      `for s in $(seq 0 15); do uv run trace-tidy --seed $s --odom datasheet --seconds 300 --history 0; done`
      and tally `=== FALL` lines by `state` and by `holding`. (Cleaner: add
      a per-state fall count and metres-walked-while-holding to the
      `eval-tidy` result row, so the 16-seed benchmark answers this every
      time it runs — done, see above.)
      **The number:** falls per carried metre vs falls per unladen metre.
      If the laden rate is not above the unladen rate at n ≥ 16 seeds,
      **close this item negative** — the walker carries the payload and the
      falls belong to the route/rim work, which is a brain fix (12.4/12.7),
      not a reflex.
- [x] **Step 2 — NOT RUN, step 1 closed it** (0 laden falls; a payload
      clone has nothing to fix). Kept for the day the toys get heavier
      than 20 g or the carry pose changes:
      Add a held-mass domain randomisation to `walk_env.py` next to the
      trunk-mass DR (a 10–40 g mass on the head/jaw body, matching upstream,
      plus the held head pose the tidy brain commands during carry), then
      `uv run distill --teacher ../microduck/policies/alpha_walking.onnx --run-name carry-clone`
      with that DR on, `uv run export-walk runs/carry-clone`, and judge it
      by `uv run select-run` before anything else — a clone that falls
      unladen is rejected there, whatever it does laden. Do NOT fine-tune
      with PPO on top unless the clone itself fails laden; that route is
      the recorded bad trade above. Note the clone learns the teacher's
      mapping, not a new gait: the bet is that the teacher's DR already
      covers 20 g and the clone only has to reproduce it under the held
      pose, so if step 1 said the walker falls laden, expect this to
      confirm the failure rather than fix it, and the fix moves to the GPU
      stack (an mjlab cfg with mouth-payload DR, the sim2real route).
      **The number:** `eval-tidy --seeds 16 --seconds 300 --odom datasheet --walker runs/carry-clone/policy.onnx`
      paired against the shipped walker on the same seeds — falls/run
      (must not rise; the effect that replicated in the 80-seed study was
      +0.35 falls at p = 0.002, so that is the scale a real change shows at)
      and toys in the basket (the 16-seed benchmark sits at 4.94–5.31 of 6;
      the run-to-run band is about ±0.3 toys, so a win has to clear that).
      Then **render and look** (`render-rollout --policy runs/carry-clone/policy.onnx --behavior stand`
      and a `trace-tidy --walker` run) before claiming anything.
- [x] **NOT RUN, same reason** — the carry-speed cap only mattered if
      the carry fell, and it does not. `carry` walks at the brain's normal twist;
      a `TidyParams` carry-speed cap is a one-line A/B on the same
      `eval-tidy` seeds and needs no training. Run it before step 2.

## Track 12 — the basket rim: where the tidy loop's falls are — MECHANISM FOUND, fix measured, ships off (2026-09-04)

Follow-on to the carry item above, which put 75% of the tidy loop's falls
within a metre of the basket. Traced six of them (`trace-tidy --history 2.5`)
and they are ONE event: the duck stops at the rim, releases, and within a
second pitches nose-down over the rim — three of the "backoff" falls were
already pitching during the drop. Then instrumented every drop over 2 × 64
seeds (716 drops, ideal + datasheet odometry): trunk distance at the stop,
MuJoCo contacts between duck and basket, peak forward gravity, landing.

- [x] **What separates the 35 drops that toppled (4.9%) from the 681 that
      did not.** Both toes on the basket wall (33/35 left, 32/35 right;
      clean drops 18% / 30%), a stop a centimetre closer (true trunk distance
      median 0.225 vs 0.237), and 3 cm of forward creep during the stand
      (min distance 0.196 vs 0.226). By stop distance: **0 of 230 toppled at
      ≥ 0.24, 18 of 431 in 0.225–0.24, 16 of 53 in 0.21–0.225.** The
      release is not the push — `beak: open` only drops the weld, no servo
      moves — and 34 of the 35 toppled drops had already landed the toy IN.
      So a topple costs a respawn (spawn point, odometry reset, a fresh
      search) and not a toy.
- [x] **Move the stop out? NO — toys.** 64 paired datasheet seeds, the
      baseline arm reproducing the README benchmark (5.33 of 6, 0.39 falls):

      | `basket_reach` | tidied | falls/run | topple % | landed in % | Δtoys [95% CI] | Δfalls [95% CI] |
      |---|---:|---:|---:|---:|---|---|
      | 0.20 | 0.911 | 2.06 | 30.0 | 92.6 | +0.14 [−0.08, +0.36] | +1.67 [+1.31, +2.03] |
      | 0.21 | 0.893 | 1.20 | 15.8 | 95.3 | +0.03 [−0.17, +0.23] | +0.81 [+0.53, +1.09] |
      | **0.22** | 0.888 | 0.39 | 5.2 | 94.2 | — | — |
      | 0.23 | 0.794 | 0.16 | 1.3 | 82.0 | −0.56 [−0.80, −0.31] | −0.23 [−0.42, −0.06] |
      | 0.24 | 0.638 | 0.14 | 0.8 | 64.4 | −1.50 [−1.88, −1.14] | −0.25 [−0.41, −0.09] |
      | 0.25 | 0.474 | 0.11 | 0.0 | 48.0 | −2.48 [−2.81, −2.17] | −0.28 [−0.45, −0.11] |

      A centimetre out removes most of the topples and drops half a toy on
      the rim; a centimetre in buys nothing measurable and triples the
      falls. **0.22 is the knee and stays.** `neck_reach` 0.6 (pitch the neck
      back to push the tip in) is worse at every stop: 68% landed at 0.22,
      38% at 0.24, 20% at 0.25 — the earlier "ships OFF" verdict, now with
      the mechanism (it is the landing, not the stance).
- [x] **Leave the rim faster? NO.** Two new `TidyParams`, both defaulting
      to today's behaviour: `backoff_clear_s` (a `gait.back_up` reverse before
      the sidestep, so the toes come off the wall first) and `drop_s` (the
      stand after the release, was a hard-coded 0.6). Same 64 seeds:

      | arm | tidied | falls/run | topple % | landed % | Δtoys [95% CI] | Δfalls [95% CI] |
      |---|---:|---:|---:|---:|---|---|
      | clear 0.3 s | 0.833 | 0.34 | 4.2 | 90.1 | −0.33 [−0.53, −0.11] | −0.05 [−0.20, +0.11] |
      | clear 0.5 s | 0.852 | 0.36 | 4.2 | 90.6 | −0.22 [−0.48, +0.03] | −0.03 [−0.20, +0.14] |
      | clear 0.8 s | 0.859 | 0.36 | 4.5 | 93.5 | −0.17 [−0.41, +0.05] | −0.03 [−0.19, +0.12] |
      | drop 0.2 s | 0.844 | 0.34 | 4.2 | 92.4 | −0.27 [−0.52, −0.02] | −0.05 [−0.22, +0.12] |
      | drop 0.2 + clear 0.5 | 0.872 | 0.66 | 7.6 | 94.1 | −0.09 [−0.31, +0.11] | **+0.27 [+0.08, +0.45]** |
      | drop 0.2 + clear 0.8 | 0.870 | 0.38 | 5.1 | 94.9 | −0.11 [−0.33, +0.09] | −0.02 [−0.22, +0.17] |

      The topple rate does not move (it is decided in the stop stride,
      before any of this runs), the reverse costs landings, and a short
      stand plus a short reverse nearly doubles the falls. Both parameters
      stay at their defaults; the numbers are in `tidy.py` next to them.
- [x] **The stop stride — DONE, and it was not the stride: it is the
      approach ANGLE against a square basket.** Probed the walker on a flat
      floor (30 gait phases, four stop commands): from the blind-leg speed
      the toes reach **0.045 m** past the trunk at the stop (p10–p90
      0.040–0.048), the trunk coasts 4 mm, and there is no creep — a 0.22
      stop puts the near wall 0.064 m ahead, 1.5 cm clear, every time. A
      reverse pulse or a dead-band −0.3 at the stop changes nothing useful;
      a 0.15 taper lands the toes 7 mm further. So the loop's toes-on-the-
      wall came from somewhere else, and the 716 instrumented drops said
      where: the basket is square and the brain stops on the distance to
      its CENTRE. **No drop within 15° of a wall's normal toppled (0 of
      291); 34 of 35 topples came in at 20° or more.** The perpendicular
      distance to the wall plane splits them cleanly — 26 falls in 165
      drops under 0.20 m, 2 in 366 at 0.22 m or more — while landings stay
      at 93–97% out to 0.24 m.
- [x] **Two fixes from that, both measured, both shipping OFF.** Same 64
      paired seeds, datasheet and ideal odometry (`TidyParams`
      `basket_square`, `basket_normal`, `basket_normal_out`,
      `basket_normal_skip_deg`; the numbers sit next to them in `tidy.py`):

      | arm | topple % (ds / ideal) | landed % | Δtoys ds [95% CI] | Δtoys ideal [95% CI] | Δfalls ideal |
      |---|---:|---:|---|---|---|
      | stop on the wall plane | 1.3 / 0.5 | 62 / 59 | −1.69 [−2.00, −1.38] | −1.66 [−2.02, −1.30] | −0.31 |
      | stage on the normal, out 0.15 | 1.2 / 0.0 | 94 / 93 | −0.47 [−0.69, −0.27] | −0.31 [−0.53, −0.09] | −0.23 |
      | stage on the normal, out 0 | 1.5 / 0.0 | 95 / 95 | −0.22 [−0.42, +0.00] | −0.33 [−0.55, −0.12] | −0.27 |
      | … skip the detour under 15° | 1.5 / 0.6 | 95 / 93 | −0.22 [−0.41, −0.03] | −0.17 [−0.41, +0.06] | −0.30 |
      | … skip the detour under 25° | 3.7 / 0.6 | 96 / 95 | −0.06 [−0.23, +0.11] | −0.11 [−0.30, +0.08] | −0.28 |

      Stopping on the wall plane proves the coupling: toes off the wall
      means the beak (0.08 m out, 0.045 m of toe, 3.5 cm of geometry that
      cos θ eats) is no longer inside it — landings crater. Staging on the
      wall's normal removes the topple as predicted and cuts falls by 0.3
      a run on ideal odometry, and **never pays on toys**. Where the time
      goes was measured too: not the detour (deliver seconds per drop
      14.1 → 14.5) but the NEXT approach, +6 s a run — a duck that
      delivered along a normal backs off into a different place and
      heading, and its next route is longer, the same heading effect the
      `backoff_back_s` study found. The benchmark is time-bound (1 of 64
      runs finishes early) and 0.39 falls a run cost ~0.15 toys, so the
      fix would have to be free, and it is not; under drift the 25° arm's
      other falls also rise (0.39 → 0.52).
- [ ] **If the rim topple ever has to go** (a real basket, where a fall
      costs more than 20 s): `basket_normal=1, basket_normal_skip_deg=25`
      is the arm, and the thing to fix on top of it is the back-off's
      end-heading after a normal-axis delivery, so the next approach does
      not pay for it — measure `approach` seconds a run, which is where the
      +6 s went. (A first version that entered `aim` at the staging point
      lost 7 points of landings because the standing look was no longer
      taken at 0.42 m; the hand-back to the ordinary servo-and-aim fixed
      it. Worth knowing before touching the deliver leg again.)

## Track 4 — positional soccer: teams by colorway, the right goal, role brains — OPEN (2026-09-05)

Context: the ask is a game that *looks* like soccer — ducks that know which
team they are on, score on the far goal and not their own, clear the ball up
the pitch or hold the goal when it is near, and in 2v2 / 3v3 split into a
defender, a mid and a striker instead of six ducks on one ball. What the
code already does is more than the ask assumed and worse than it looks, so
read this before designing anything:

- **Teams and goals exist, and every duck attacks the right mouth.**
  `Duck.team` is in the scenario contract, `make_pitch` makes `left`
  (spawns at −x, attacks +x) and `right`, teammates share a blackboard
  (`brain/team.py`: one attacker by predicted time-to-ball with hysteresis,
  the rest support), and a goal restarts play from the spawns.
  `World.goal_for` hands each `chase` brain the mouth **its spawn heading
  faces** — not the nearest one — and `Chase._plan` lays the kick line at
  that goal. "They score on whichever goal is closest" is not what the code
  does. What it does is subtler:
- **Over half of all kicks are aimed AWAY from the goal the kicker
  attacks.** `_plan` aims at the goal only when that costs under `aim_max`
  = 1.05 rad of detour from the line of sight; otherwise it kicks along the
  line of sight, and `push_beyond` is ∞ so there is no dribble or clear mode
  at all. A duck that reaches the ball from the goal side — which the
  support geometry guarantees, since a supporter stands 0.7 m goal-side of
  the ball (`support_back`) and walks in from there when it becomes the
  attacker — kicks it toward its own goal. The baseline in item 0 is the
  number.
- **The benchmark cannot see an own goal.** `World.goals` is keyed by
  MOUTH (`left` / `right`), `eval-pitch` reports mouths, the /sim
  scoreboard shows mouths. A ball the sky team puts into its own net and
  one the cream team scores are the same row. `PitchMetrics` knows who last
  had the ball (`_holder`) and the World knows *when* the last kick was
  (`last_kick_t`) but not whose; nobody reads either at the goal tick.
- **The editor has no notion of a team.** `SimEditor` offers
  `wander / follow / tidy / script` — not `chase` — has no team field, and
  its `ScenarioDuck` type lacks `team` and `odom` (they survive a save only
  because the draft is a JSON deep copy). A team's goal is inferred from
  spawn yaw, so a hand-placed roster with one duck facing the other way
  makes `PitchMetrics` raise `ValueError` *after* the world is swapped in
  (`world_server.load_world` builds the metrics outside its `try`: the load
  answers 500 and the scoreboard keeps the previous world's metrics object).
- **Every duck is the same colour.** The viewer builds ONE template
  geometry from `GET /scene` (the single-robot walk scene) and reuses it for
  every duck. The composed world does carry per-duck materials
  (`d0/left_shell_material`, verified on the pinned MuJoCo), but nothing
  writes them and nothing streams a colour.
- **No brain can tell a teammate from an opponent by sight.** A `duck`
  detection carries `name` (the duck id — privileged, and no brain reads it;
  `tidy` reads toy names only). Mates are known only from the board's Wi-Fi
  poses; opponents are anonymous `duck` tracks — which is why
  `mate_keepout` measured off (12 of 13 traced falls were beside an
  OPPONENT the board does not carry).
- **The learned striker loses** (sim-roadmap 4.4: it never reaches the
  ball). Learned role brains are downstream of that, not a way round it.
- **The shipped colorways** (Pollen press kit): Cream `#f7e6cb` (orange
  trim and beak), Graphite `#6c6a68` (yellow trim), Lavender `#bfa9cf`
  (yellow trim), Sky `#a9dbe8` (orange trim). Two ducks of one colorway
  cannot be told apart on hardware either, which is the honest reason to
  make **team = colorway**.

The order below is deliberate: the benchmark first (an own goal has to be
countable before a rule that prevents one can be judged), the contract and
editor second (so the page can *show* a team and a role), scripted roles
third (cheap, over the existing chase brain, judged on the new numbers),
perception fourth, learning last. Every "measured" line above about brain
ideas ends the same way in `microduck_local/README.md` ("Where the soccer
track actually stands"): found on four seeds, gone on twelve fresh ones. So
each item here names its number and its seed count before its code.

### 0. Baseline — DONE (2026-09-05): what the pitch does today

The `eval-pitch` loop with two things added: the kick line read off the
brain at the moment it fires (`Chase._hunt_u`, cos against the direction
from the ball to the goal it attacks), and the team credited at each goal —
by the last kick within `KICK_GOAL_S` = 4 s, else by `PitchMetrics`'
possession rule (last team within 0.25 m).
`microduck_local/scripts/probe_own_goals.py` (run it from `microduck_local`:
`uv run python scripts/probe_own_goals.py <seed> 300 <per_side>`); the
fields become `eval-pitch`'s in 1.1. Four seeds × 300 s, shipped `chase`
both sides, seeds 0–3:

| roster | goals (kicked in / walked in) | own goals | kicks | aimed away by the PLAN | sent back by the BALL | falls |
|---|---:|---:|---:|---:|---:|---:|
| 1v1 | 6 (0 / 6) | 2 (+2 unplaced) | 26 | **14 (54%)** | 12 (46%) | 4 |
| 2v2 | 8 (1 / 7) | **6 (+2 unplaced)** | 27 | **14 (52%)** | 18 (67%) | 16 |

Own goals here are the shipped instrument's (item 1.1: the kicker inside 4 s,
else the last team on the ball inside 4 s, else unplaced); the probe's looser
"whoever last held it, however long ago" gave 3 and 8. Both say the same
thing about 2v2.

Read it the playbook's way: fourteen goals is an event count that resolves
nothing; **28 back-kicks out of 53 kicks** is the finding, and it is the
mechanism, not the score. Two more things the rows say. Every goal but one
was *walked* in — a duck at its own line shoving the ball over it, credited
by the possession rule to the team standing there, which is how 8 of 8 in
2v2 are own goals: today's own goal is a supporter's or a blocked attacker's
stumble at its own mouth, and a defender parked ON that line (3.2) is the
duck most likely to make one, so 3.2 is judged on `ownGoals` first. And the
run is deterministic in the seed — two runs of the probe matched to the
tick — so a paired A/B pairs exactly. A brain that never kicks toward its
own goal is the cheapest change on this list, and the metric that judges it
costs about as many seeds as `kicks` (62 for a 25% shift), not goals' 146.
Cost of a seed on this Mac: 300 s of 1v1 in ~10 s, of 2v2 in ~27 s, eight
in parallel — a 24-seed 2v2 battery is under two minutes at `--jobs 8`.

**The floor under this table had no rolling resistance (found 2026-09-06).**
The ball geom carried upstream's `friction="0.5 0.005 0.0001"` on a
condim-3 contact, and MuJoCo applies the rolling coefficient only on
condim 6 — so a ball a duck barely brushed rolled until a wall stopped it
(measured on an open floor: 12 m from a 0.2 m/s nudge, 84 m from a kick,
never stopping). Upstream's `ball.xml` has the same silent bug. Fixed:
condim 6 and `Ball.rolling` (0.002 — a short carpet; a 0.2 m/s nudge stops
in 0.25 m, a walked-into ball at 0.45 m/s in 0.7 m, a 1.4 m/s kick in
3.5 m; the table of floors is on the dataclass, `tests/test_world.py`
guards it). The same four seeds re-run on the new floor:

| roster | floor | goals (kicked / walked in) | own goals | kicks | back-kicks | falls |
|---|---|---:|---:|---:|---:|---:|
| 1v1 | frictionless (above) | 6 (0 / 6) | 2 (+2 unplaced) | 26 | 12 (46%) | 4 |
| 1v1 | rolling 0.002 | 1 (1 / 0) | 1 | 14 | 6 (43%) | 4 |
| 2v2 | frictionless (above) | 8 (1 / 7) | 6 (+2 unplaced) | 27 | 18 (67%) | 16 |
| 2v2 | rolling 0.002 | 1 (1 / 0) | 0 | 11 | 4 (36%) | 19 |

Read it against this track's premises. *Every walked-in goal was the
frictionless ball* — a shove at the halfway line that used to trickle
over a goal line now stops a stride later — and the own goals went with
them (7 of 8 in 2v2 were exactly that shove at the duck's own mouth), so
"a defender parked on its own line is the duck most likely to score an
own goal" (3.2) was a property of the old floor, not of the defender.
Kicks halve (a ball that stops near the kicker is re-acquired less often
than one that comes back off the boards), and the back-kick PROPORTION
holds at 36–43% on 25 kicks, so 2.x (the aim clamp) is still the
mechanism to fix; goals on a real floor are now rarer still (2 in eight
runs), which makes `goals` even less of a metric than 1.5 already said.
Every soccer number older than this paragraph — README, brain
docstrings, the `ChaseParams` measurements — was taken on the
frictionless ball; `ball_decel` (tracker prediction) moves from 0.04 (the
tracker's noise on a ball that never slowed) to 0.3 (the first two
seconds of a kick on the new floor, where the decay is speed-proportional:
1.4, 1.08, 0.75, 0.47, 0.28 m/s second by second).

### 1. A benchmark that can see an own goal, a back-kick and a pile-up

- [x] **1.1 Goals per TEAM: for, against, own — DONE (2026-09-05).**
      `World.last_kick_duck` and `World.goal_credit_duck` (filled on exactly
      the test the World's own kicked/walked-in split uses, so the two cannot
      disagree), and `PitchMetrics` turns them into `goalsFor`, `goalsAgainst`,
      `ownGoals` per team plus a run-scalar `goalsUnattributed`. `goalsFor` and
      `goalsAgainst` need no credit at all — the mouth a team attacks is known
      — which is why they are split out from `ownGoals`, the only one that
      does. `eval-pitch` prints the ledger per team; 8 tests in
      `tests/test_pitch_metrics.py` drive it on hand-built states. Baseline
      reproduced through it (the table in item 0). The original: `World` records the team
      of the last kick (`last_kick_team`, beside `last_kick_t`) and
      `PitchMetrics` attributes each goal at the tick it happens: the
      kicking team if a kick started within `KICK_GOAL_S`, else the last
      holder; own = the credited team's attacked mouth ≠ the mouth scored
      on. Row fields `goalsFor`, `goalsAgainst`, `ownGoals` per team;
      `eval-pitch` prints them beside the mouths (the mouth keys stay — old
      rows and `side_reading` read them). The /sim scoreboard shows the
      per-team counts. Test in `tests/test_pitch_metrics.py`: push the
      ball over a line with each team as last holder and as last kicker.
      `uv run eval-pitch --seeds 4 --seconds 300` → the four seeds above
      reproduce (the loop is deterministic in the seed: the probe's two
      runs matched to the tick).
- [x] **1.2 Kick direction, measured off the BALL — DONE, and it agrees with
      the plan (2026-09-05).** `kickCount`, `kicksBack` and `kickCarry` per
      team, off each duck's own kick window (`WorldDuck.skill_t0`, not the
      World's single last-kick stamp — two ducks kicking on the same step
      would have collapsed into one), settled `CARRY_S` later or at the goal
      line if a goal lands first. The decide-on, answered: **28 of 53 kicks
      back by the plan line, 30 of 53 by the ball** over the same 4+4 seeds.
      Aggregate agreement; per roster the two differ by 2 kicks (1v1) and 4
      (2v2), which is noise at this size. So the aim rule (3.1) is the
      mechanism, and line-up scatter is neither shown nor ruled out — it needs
      the 24-seed battery, where it will show up as the ball number staying
      high after the plan number falls. The original: Per
      kick, the ball's signed displacement toward the kicker's goal over
      the 2 s after the swing (`CARRY_S`) and its angle to the goal line;
      `kicksBack` per team = kicks whose 2 s displacement is toward the
      kicker's own goal. This is playbook rule 6 — the plan's `_hunt_u` says
      where the brain *meant* the ball to go, the ball says where it went,
      and only the second one can judge a line-up. → **decide on:** the
      baseline's 54% by the plan line against the ball's own number; if
      they differ by more than the noise, the line-up is scattering shots
      and item 3.1 is only half the story.
- [x] **1.3 Shape — DONE (2026-09-05).** `ballOwnHalf`, `spread`, `crowd`
      (`CROWD_R` 0.5 m) and `depth` per team at the control tick, `None` for
      the two a one-duck team cannot have. Baseline 2v2, 4 seeds: crowd
      **9% / 17%**, spread 0.62 / 0.70 m, depth 1.30 / 1.37 m of a 1.7 m half
      — i.e. nobody stays back at all, which is what item 3.2 is for.
      `ballOwnHalf` is printed per team and never averaged: over the pair it
      is 60 s/min by construction. The original: Per team at the control tick: `ballOwnHalf` (s/min
      the ball is in the team's own half), `spread` (mean pairwise distance
      between teammates), `crowd` (fraction of ticks with two teammates
      inside 0.5 m of the ball — the 24.5% the README quotes from a trace,
      made a row field), `depth` (the deepest teammate's distance from its
      own goal line). These are what "pile up" and "somebody defends"
      mean as numbers; goals cannot say either.
- [x] **1.4 A roster A/B harness — DONE (2026-09-05).** `eval-striker` takes
      a roster a side: `--left "chase+defender,chase" --right chase`, one entry
      a duck (a brain kind with an optional `+role`), a single entry covering
      the side. It writes the roster onto the scenario's ducks and builds every
      brain through the same `brain_kwargs` the lab uses, so the harness and
      the page cannot drift. `side_reading` is per TEAM and now carries the
      ledger (own goals, back-kicks, crowd); the byte-for-byte pin against
      `eval-pitch` still holds when both sides are plain `chase`. The original: `eval_striker` already swaps the LEFT
      duck's brain (`--left striker:v1`); generalise it to a per-duck spec
      per side — `--left chase,chase --right defender,striker` — with
      `--out/--tag` resume and a paired summary (per-seed wins on
      `ownGoals`, signed `ballProgress` per team, `crowd`, falls). This is
      how every role brain below is measured: against today's roster, same
      seeds, one side changed. Keep `eval_striker`'s byte-for-byte pin
      against `eval_pitch` when both sides are `chase`.
- [x] **1.5 Power — DONE (2026-09-05), and it changes which metric judges
      what.** 24 seeds × 300 s of 2v2, shipped roster, `runs/t4-base24-2v2.jsonl`.
      CV of the per-run total and the seeds an arm needs to resolve a 25%
      shift at p<0.05 / 80% power:

      | metric | CV | seeds | events over the 24 |
      |---|---:|---:|---:|
      | `depth` | 0.07 | **1** | — |
      | `possession` | 0.18 | **8** | — |
      | `spread` | 0.21 | **11** | — |
      | `ballAdvance` | 0.36 | 32 | — |
      | `crowd` | 0.39 | 38 | — |
      | `kickCount` | 0.53 | 72 | 183 kicks |
      | falls | 0.65 | 106 | 64 falls |
      | goals | 0.74 | 136 | 36 goals |
      | `kicksBack` (per run) | 0.78 | 151 | 90 back |
      | `ownGoals` | 1.18 | **347** | 19 own |

      Three things follow, and they set how every item below is judged.
      **`ownGoals` cannot be a judge** — 19 events over 24 seeds and a CV
      worse than goals', so it is reported and never decided on. **The shape
      metrics are the cheap instruments for a POSITIONAL change**: `depth` at
      one seed, `spread` at eleven, `crowd` at 38, against goals' 136 — which
      is exactly what a defender or a striker moves. And **`kicksBack` must
      be read as a PROPORTION, not a per-run count**: 90 of 183 kicks is a
      binomial with 183 events, and halving 49% needs ~9 seeds of it, where
      the per-run mean needs 151. That is the playbook's rule 1 with teeth —
      the same measurement is cheap or hopeless depending on whether you
      count runs or events.
      `ballOwnHalf` is 60 s/min over the pair by construction (CV 0.00 as the
      mean over teams): a per-TEAM reading only, for an asymmetric matchup.

### 2. Team = colorway — in the contract, the world, the stream, the editor

- [x] **2.1 Contract — DONE (2026-09-05), and it closed an old trap.**
      `TEAM_COLORWAYS` (cream, graphite, lavender, sky, each with its trim
      colour), `Duck.team` restricted to them with `left`/`right` mapping
      forward so saved scenes still load, `Duck.role` from `ROLES`, and
      `Scenario.attacks` declaring the mouth a team attacks. `make_pitch` was
      cream (−x, attacks +x) v sky — it is cream v LAVENDER since 2.5. The trap: the teams used to be called
      after the two SIDES, which are the same two words the World writes its
      goal MOUTHS under — the ambiguity `eval_pitch` carries a standing
      warning paragraph about. A cream duck and the +x mouth cannot be
      confused, so that class of misreading is now impossible rather than
      documented. A team facing both goals is refused by `validate_scenario`
      with the duck's name in the message, at PUT time. The original: `Duck.team` ∈ {`cream`, `graphite`, `lavender`,
      `sky`} or null; `validate_scenario` maps the legacy `left` → `cream`
      and `right` → `sky` so every saved scene still loads, and `make_pitch`
      emits cream (attacks +x) v sky. A scenario-level
      `attacks: {team: "left"|"right"}` says which mouth a team attacks;
      absent, it comes from the spawn heading as today — and validation
      refuses a team whose ducks face both mouths with a `ScenarioError`
      naming the duck, at PUT time, instead of the 500-after-swap above.
      `eval-pitch` rows keep working (the metric dicts are keyed by team
      name; `_fmt` sorts them).
- [x] **2.2 The world paints it — DONE (2026-09-05).** `compose.paint_team`
      writes the colorway into that duck's own `*_shell_material` (both body
      halves, both head halves) and its trim into the beak, feet and ankles
      (2.5 widened both lists to every printed part) —
      `MjSpec.attach` prefixes materials per duck, so it is a write to one
      duck and no other. It returns how many it painted so a caller can assert
      9 rather than discover an upstream CAD rename by seeing nothing; the
      test does. `duck_info` streams `team` and `role`. This is what
      `render-striker` and any MuJoCo view of a world show; the browser draws
      from the single-robot scene and tints itself (2.3). The original: After `compose`, write the colorway's
      shell rgba into the duck's `*_shell_material` (left, right, top and
      bottom head) and the trim colour (orange for cream / sky, yellow for
      graphite / lavender) into `foot_*`, `ankle_*`, `jaw_material` —
      per-duck materials exist, so it is `model.mat_rgba[...]` and nothing
      else. `duck_info` streams `team` (and the swatch); the physics is
      unchanged (a colour is not a mass), which `tests/test_arena.py`'s
      step-for-step lock against the walk env must keep saying.
- [x] **2.3 The viewer paints it — DONE (2026-09-05).**
      `buildBodyGeometries(scene, team)` bakes the colorway into the vertex
      colours, and `SimDucks` builds ONE set per colorway (at most four,
      whatever the roster), not one per duck. The scoreboard is per team with
      swatches, and it shows `goalsFor` rather than the mouth counts — the
      mouths moved into its tooltip — plus own goals, back-kicks as `n/N` and
      the crowd fraction, red when a team kicks backwards more often than not.
      The inspector shows the selected duck's team and role. Verified on
      `pitch-2v2`: cream v sky on the pitch, the sky duck visibly blue in
      d0's head camera, `d2 · alpha_walking · sky` in the inspector. The tint
      itself is locked by `lib/duckskin.test.ts` — under that lighting a
      screenshot cannot tell a cream shell from a white one. (Cream v sky is
      cream v lavender since 2.5, where the colorway also grew to cover every
      printed part rather than the four body shells alone.) The original: `buildBodyGeometries(scene, colorway)`
      recolours geoms by material name at build time and is cached per
      colorway (four at most, so still one geometry set per colorway, not
      per duck); `<Duck>` takes the colorway from `SimDuck.team`; the
      label carries a swatch; the editor's spawn rings take the team's
      colour instead of amber. The scoreboard reads `goalsFor / against /
      own` per team under the team's swatch, with the mouth counts in the
      tooltip where they belong. → **check:** eight ducks on screen still
      hold the WebGL context (the README's pitfall) — four geometry sets,
      not eight.
- [x] **2.4 The editor — DONE (2026-09-05).** Per duck: a team select (the
      four colorways), a role select (disabled until it has a team), and a
      brain select built from `GET /world`'s registry — which is how `chase`
      became reachable at all; the hard-coded four never listed the brain the
      pitch actually runs. A "make a pitch" button sets the goals, drops a
      ball on the spot and splits the ducks by the half they already stand in,
      so the scene it produces is legal the moment it is toggled. A duck
      placed on a pitch joins the team of its own half facing the right way
      (`lib/sim.test.ts`), and the spawn rings on the floor wear the team's
      colour. `ScenarioDuck` gained `team`, `role` and `odom`. The original: Per duck: a team select (the four colorways or
      none) and a role select (item 3.5); the brain select built from
      `world.brains` (every registry kind, `chase` included) instead of the
      hard-coded four; `ScenarioDuck` gains `team`, `role`, `odom`. A
      "make it a pitch" control sets `goal_width` and the teams' mouths. A
      duck placed on a pitch defaults to the team whose half it stands in.
      `npm test` covers `applyFloorClick` for the default; the server test
      covers the refusal in 2.1.
- [x] **2.5 Cream v LAVENDER, and one colour per duck — DONE (2026-09-05).**
      (a) The default pair is now `PITCH_TEAMS = ("cream", "lavender")`, one
      constant that `make_pitch`, `LEGACY_TEAMS` and the editor's "make a
      pitch" all read, so it changes in one place and cannot drift between the
      two repos' halves of the UX. Cream against sky was two pale COOL shells
      that a person watching a 3v3 had to squint at; purple is the furthest of
      the four ships from cream in hue while still being a colorway you could
      actually print.

      (b) A colorway now owns EVERY printed part, in two colours: shells are
      the head, trunk, legs and hips, trim is the beak, feet, ankles and
      soles. Servos, PCBs, bearings and the lens are left alone — they are the
      same on every duck, as on the robot.

      The finding, and it is the only reason the material lists are long: the
      MJCF materials are OnShape export appearances, and four PRINTED parts
      carried colours no colorway ever claimed — a teal thigh plate and shoe
      rim (`upper_leg_rigidity_plate`, `sole_*` at #89dad3), a pale-blue hip
      (`yaw_roll_motion`), and a pink soft mouth (`jaw_soft`,
      `soft_mouth_top`, the SAME pink on all four colorways). Painting only
      the four body shells hid it. The first attempt at this item made it
      worse: it gave the legs a deeper cast of the shell on the theory that
      the shells are the parts that hold still and a darker swinging limb
      reads through motion. That is true in the abstract and wrong on the
      duck — it made five colours instead of four, and a screenshot of a
      lavender duck read as a patchwork rather than a printed shell. **The
      table looked fine; only the render showed it** (the README's rule, one
      more time). Now every shell material lands on one colour and every trim
      material on one, asserted per pitch team in `test_world.py` and in
      `lib/duckskin.test.ts` against the vertex colours the browser draws.

      (c) Team CHIPS stay the shell colour. A two-tone shell-over-trim chip
      was tried and reverted the same way: the trim is shared between
      colorway PAIRS (cream and sky are both orange, graphite and lavender
      both yellow), so at 12 px the trim half swamped the chip and cream and
      lavender became the same amber block — the chip lost the one
      distinction it exists for. Caught by looking at the /sim scoreboard,
      not by a test.

- [x] **2.6 The score panels minimize — DONE (2026-09-05).** Both score
      overlays (`Pitch` and `Tidy score`) grew the same —/+ in their title
      bar, on `B`, persisted as `simScoreOpen`. They sit top-left over the
      near half of the room, and the pitch one is the tallest panel on the
      page once the per-minute table is showing — 8 rows over the ground the
      ducks are actually playing on.

      Collapsed keeps the HEADLINE and drops the table: the score line with
      its team chips on a pitch, `n / N in the basket` on a playroom. A
      scoreboard whose minimize costs you the score is not worth pressing;
      what covers the room is the table under it. The kickoff banner also
      survives a collapse — it is one line and it is live state you want at
      the moment it appears. The head-camera inset re-measures its dock every
      frame (`CamInset`'s `belowRef`), so it slides up on its own.

      The four title-bar toggles (inspector, controls, and the two score
      panels) are now one `PanelToggle` — they were three copies of the same
      twelve inline style properties. The head camera's button is deliberately
      NOT this one: it floats over the video with its own backing rather than
      sitting in a title row.

### 3. Scripted positional play — the brains that make it look like soccer

> **2026-09-08 correction (code review).** The striker's post was on the ball's SIDE for the team attacking −x: `_hold_target` chose the side in the pitch frame and applied it along the lane's left normal, whose sense flips with the attack direction. Every roles battery in this section compared two different strikers (one posting off the ball, one on it). Fixed in `controllers.py` and locked by `test_the_striker_posts_off_the_balls_side_for_both_attack_directions`; the roles numbers here predate it and want a fresh block before they are quoted again.

Scripted first, and over the existing `Chase`, not beside it: every role
is the same state machine with a different *target* and a different rule
for when to attack. The README's record is that nothing new at the brain
tier has survived a fresh-seed confirmation; a role is a change to where a
duck STANDS, which is the one thing the shape metrics in 1.3 can see
directly.

- [x] **3.1 Never kick toward your own goal — the clamp is MEASURED and it
      works (2026-09-05).** `ChaseParams.aim_mode`: when the goal is more
      than `aim_max` round the ball, `"los"` (shipped) gives up and kicks
      along the line of sight, `"clamp"` kicks at the edge of the cone on
      the goal's side, `"goal"` aims at the goal whatever the walk-round
      costs. Inside the cone all three agree, so this touches only the case
      the shipped rule gives up on.

      24 paired seeds × 300 s of 2v2, `scripts/compare_pitch.py`
      (`runs/t4-base24-2v2.jsonl` v `runs/t4-clamp24-2v2.jsonl`):

      | | los | clamp | |
      |---|---:|---:|---|
      | **kicks sent back** | **90 of 183 (49%)** | **64 of 178 (36%)** | **−13 pts, p = 0.011 on the events** |
      | goals | 1.50 | 1.75 | +0.25 ± 0.53, p = 0.33 |
      | falls | 2.67 | 2.50 | −0.17 ± 1.08, p = 0.75 |
      | own goals | 19 | 17 | (19 events: cannot resolve — item 1.5) |
      | possession | 23.99 | 23.54 | −0.45 ± 2.32, p = 0.69 |
      | ballAdvance | 0.775 | 0.786 | +0.011 ± 0.147, p = 0.87 |
      | ballProgress | −0.043 | −0.065 | −0.022 ± 0.185, p = 0.81 |

      **The mechanism is confirmed and the play is unchanged** — the same
      shape as the attacker-handover result, and stated the same way. Falls
      did not rise, which was the risk. `spread` came out −0.071 m at
      p = 0.047; nine metrics were read, so that is the p one expects by
      chance and it is not claimed.

      Back-kicks fell by a quarter, not the half the item asked for, and the
      geometry says why: the clamp helps only when the line of sight is
      within `aim_max` of somewhere useful. A duck standing squarely between
      the ball and the goal it attacks has its own goal straight down that
      line, and ±60° of it is still 120° from the target.

      **So the third arm was run — always at the goal, whatever the
      walk-round costs — and it re-earns the 2026 verdict with an instrument
      that can see why.** Same 24 seeds:

      | | los | goal | |
      |---|---:|---:|---|
      | kicks sent back | 90 of 183 (49%) | **26 of 97 (27%)** | −22 pts, p = 0.0003 |
      | **kicks taken** | **183** | **97** | the walk-round costs half of them |
      | **ballProgress** | **−0.043** | **−0.301** | **−0.258 ± 0.213, p = 0.012, worse on 17 of 24** |
      | goals | 1.50 | 1.67 | +0.17 ± 0.70, p = 0.62 |
      | falls | 2.67 | 2.38 | −0.29 ± 1.07, p = 0.57 |

      It aims best and plays worst: signed progress is the metric churn
      cannot inflate, and it says the ball ends up nearer the ducks' OWN
      goals. The mechanism is visible in the kick count — a duck walking a
      long arc round the ball is in possession the whole way and shoves the
      ball backwards as it goes, which is the "a walk-around crossed walls
      and the other duck" objection from the first form, now measured on
      something better than four seeds of goals.

      **`clamp` ships, `goal` ships off with its numbers.** The clear half
      of this item (a `clear_zone` mode near our own line) is NOT built and
      is now less attractive: the remaining back-kicks are the walk-round
      case, and the walk-round is what just measured worse.

      **CONFIRMED on 24 seeds nobody had run (100–123), which no brain-tier
      soccer change in this repo had managed before.** Fresh block: back-kicks
      **121 of 236 (51%) → 45 of 146 (31%), p = 0.0001**, against the
      discovery block's 49% → 36%, p = 0.011. Pooled over all **48 paired
      seeds**: **211 of 419 (50%) → 109 of 324 (34%), −16.7 points,
      p < 0.0001** — and every other metric flat over the 48 (goals +0.02
      p = 0.93, possession +0.03 p = 0.97, advance −0.026 p = 0.57, signed
      progress +0.008 p = 0.89, crowd −0.006 p = 0.49).

      Two honest caveats, both from reading the blocks separately.
      **The kick count did not replicate as flat**: 183 → 178 on the
      discovery seeds, 236 → 146 on the fresh ones, pooling to a real
      **−23% in touches** (419 → 324). Aiming better costs kicks, and the
      fresh block says more than the first one did. **And falls looked bad
      on the fresh block alone** (+0.75 a run, p = 0.052, against −0.17
      p = 0.75 on the discovery seeds) and **do not resolve pooled**
      (+0.29 ± 0.66, p = 0.37, 236 fall events) — which is exactly why a
      single block is not a result, in either direction.
      `aim_mode` now defaults to `"clamp"`.
- [x] **3.2 / 3.3 Defender and striker — BUILT AND MEASURED (2026-09-05).
      The pile-up is gone, and that is the whole of what resolves.** A role
      is a POST, not a new state machine: the same `_support` servo walks to
      `Chase._hold_target` and faces the ball. The defender holds the line
      from its own goal to the ball, `defend_depth` = 0.5 m out, never over
      the halfway line; the striker holds `strike_ahead` = 0.8 m up-pitch of
      the ball and `strike_side` = 0.4 m OFF the kick line, on the side the
      ball is not on — the offset being the whole difference from the
      poacher, which stood on the line and reversed on fresh seeds. A role
      also owns a THIRD (`brain/team.ROLE_ZONES`) and may only take the ball
      inside it; the gate is on the shared board, not per duck, because a
      duck that gates itself leaves the ball to nobody.

      `eval-striker --left "chase+defender,chase+striker" --right chase`,
      reading the ROLES side, on the shipped brain (aim_mode `clamp`), **24
      paired seeds and then 24 fresh ones** — 48 in all, `runs/t5-*.jsonl`:

      | | plain | roles | pooled Δ (48) | p | blocks |
      |---|---:|---:|---:|---:|---|
      | **spread** (distance between the pair) | 0.68 m | **1.63 m** | **+0.945 ± 0.076** | **< 0.001** | up on **48 of 48 seeds** |
      | **crowd** (two of ours within 0.5 m of the ball) | 13.0% | **1.8%** | **−0.112 ± 0.020** | **< 0.001** | both, p < 0.001 each |
      | **depth** (deepest duck, from its own line) | 1.43 m | **0.80 m** | **−0.628 ± 0.124** | **< 0.001** | both, p < 0.001 each |
      | **falls** | 1.46 a run | **0.75** | **−0.708 ± 0.417** | **0.001** | both (p = 0.027, p = 0.009); **70 → 36 events** |
      | possession | 10.8 s/min | 8.3 | −2.50 ± 1.07 | < 0.001 | both |
      | ball in our own half | 28.0 s/min | 33.0 | +5.05 ± 4.08 | 0.013 | fresh only |
      | ballAdvance | 0.373 | 0.309 | −0.064 ± 0.066 | 0.051 | discovery only |
      | ballProgress (signed) | −0.069 | −0.004 | +0.066 ± 0.095 | 0.17 | neither |
      | goals scored | 0.73 a run | 0.94 | +0.21 ± 0.38 | 0.27 | **−0.21 then +0.63: it did not replicate** |
      | own goals | 15 | 17 | — | — | 32 events; cannot resolve |
      | kicks sent back | 61 of 179 (34%) | 36 of 124 (29%) | — | 0.35 | neither |

      **Claimed, and confirmed on seeds it was not found on: the ducks stop
      piling onto the ball, somebody stays back, and they fall half as
      often.** Spread moves by twelve times its interval and is up on every
      one of 48 seeds; crowd falls from 13% of the run to under 2%; the
      deepest duck sits 0.80 m from its own line instead of 1.43 m up the
      pitch. Falls go 70 → 36 events, p = 0.001 pooled and under 0.03 in
      each block separately — **a larger and better-replicated fall
      reduction than the bump-stand rule this repo had to withdraw**, and
      the mechanism is visible in the same table: most falls here are
      duck-on-duck, and there is far less crowd to fall into.

      **Not claimed: anything about the score.** Goals went −0.21 on the
      discovery block and +0.63 (p = 0.045) on the fresh one and pool to
      +0.21, p = 0.27 — a textbook non-replication, and had only the second
      block been run it would have shipped as "roles score more". Advance
      points down (p = 0.051) on the same pattern in reverse. Own goals and
      back-kicks move nothing.

      **The cost is real and replicated: the roles side holds the ball 23%
      less** (possession −2.50 s/min) **and keeps the ball in its own half
      5 s/min longer.** Two ducks standing at posts are two ducks not
      chasing. Whether that trade is worth making is a question about goals,
      and item 1.5 says goals need 136 seeds to answer it — so the honest
      position is that positional play here buys shape and safety, at a
      price in possession, with the score unresolved.

      **Rendered and read** (`scripts/render_pitch.py`, 60 s of seed 0 as a
      12-frame contact sheet), because a table cannot tell a defender
      holding its post from one stuck against the boards. It holds its post:
      through the last 20 s the striker is on the ball at the far end
      (0.27–0.54 m, `lineup`) while the defender sits 2.7 m away at
      depth 0.46–0.55 m, which is the geometry the item asked for. **And the
      middle-third hole is visible in the same sheet**: at t = 27 s and
      t = 33 s the depth reading jumps to 1.78–1.80 m with the defender in
      `lineup` — the ball was at midfield, no role owns that third, the gate
      fell back to "everybody may", and the defender left its post to go for
      it. Both halves of 3.5's zone note, in pictures.

- [x] **3.3 Striker** — measured with the defender, above (the roster arm is
      defender + striker against two plain chase brains, so the pair is what
      was tested). Whether a striker alone pays is a separate arm and has
      not been run.
- [x] **3.4 Midfielder, and the full 3v3 — MEASURED (2026-09-05), and it is
      the strongest result on this track.** The midfielder holds the middle
      third on the ball's side, and with all three roles filled every third
      of the pitch has an owner, so the zone gate is live everywhere (the
      2v2 hole in 3.5 does not exist here).

      `--left "chase+defender,chase+midfielder,chase+striker" --right chase`,
      3v3, reading the roles side, 24 paired seeds and then 24 fresh ones
      (`runs/t6-*.jsonl`):

      | | plain | roles | pooled Δ (48) | p | blocks |
      |---|---:|---:|---:|---:|---|
      | **crowd** | 22.4% | **3.5%** | **−0.189 ± 0.023** | **< 0.001** | both; better on **48 of 48** |
      | **spread** | 0.70 m | **1.56 m** | **+0.860 ± 0.049** | **< 0.001** | both; **48 of 48** |
      | **depth** | 1.29 m | **0.57 m** | **−0.718 ± 0.129** | **< 0.001** | both |
      | **falls** | 1.96 a run | **0.85** | **−1.104 ± 0.358** | **< 0.001** | both; **94 → 41 events** |
      | **ballProgress (signed)** | **−0.206** | **−0.048** | **+0.157 ± 0.093** | **0.001** | both in direction; better on 32 of 48 |
      | ballAdvance | 0.372 | 0.257 | −0.115 ± 0.062 | < 0.001 | both |
      | possession | 12.4 s/min | 6.5 | −5.92 ± 1.06 | < 0.001 | both |
      | goals | 0.60 a run | 0.56 | −0.04 ± 0.25 | 0.74 | neither |
      | own goals | 16 | 5 | — | — | 21 events; cannot resolve |

      **Everything 2v2 showed, larger — and one thing 2v2 could not show.**
      Six ducks on one ball was the worst case in this repo (the README's
      "falls per duck climb with the roster"), and roles cut the crowding
      from 22% of the run to 3.5% and the falls from 94 events to 41.

      The new thing is **signed `ballProgress`, the metric churn cannot
      inflate**: the plain 3v3 roster carries the ball toward its OWN goal
      at −0.206 m/min and the roles roster very nearly does not (−0.048),
      **+0.157, p = 0.001**, and the direction holds in both blocks. Read it
      with `ballAdvance`, which goes DOWN by 0.115: advance keeps only the
      forward part and is inflated by churn, and with 56% fewer kicks
      (151 → 66) there is far less churn to keep. Together they say the ball
      moves less and goes less wrong — which is what a team that stops
      scrambling for it should look like, and is the pattern `eval_pitch`'s
      docstring says to read these two for.

      The cost is the same one, doubled: **possession halves** (12.4 → 6.5
      s/min). Goals are flat and, at 56 events over 48 seeds, could not have
      said anything either way.
- [x] **3.5 Roles in the contract, on the board, in the inspector — DONE
      (2026-09-05).** `Duck.role` validated against `ROLES` (and refused
      without a team); `Team.jobs` + `Team.half_x` + `Team.attack_sign`
      filled by `brain_kwargs` from the scenario, so every teammate computes
      the same candidate set; `Team.payload` carries the jobs and
      `inputs.chase.job` reaches the /sim inspector beside the dynamic
      attack/support (`d0 · alpha_walking · cream defender`). Static roles
      only, as planned — the dynamic swap is the churn the board's
      hysteresis exists to stop, and it is its own item.

      **One thing about the zones, worth the next person's time.**
      `ROLE_ZONES` splits the pitch in THIRDS, so on a 2v2 of defender +
      striker the middle third belongs to nobody and `candidates` falls back
      to "everybody may" there. Reading the code that looked like it might
      gut the gate, so it was measured rather than assumed (120 s of the
      roster, ball position by third): **own 14%, middle 25%, final 61%** —
      the gate is live for three quarters of the run and idle for the other
      quarter, which is a real hole but not the whole rule. A roster with no
      midfielder should split at the halfway line instead. Easy, unmeasured,
      not in.
      The original:
      `Duck.role` ∈ {`defender`, `midfielder`, `striker`} or null (null =
      today's dynamic attacker / support). `Team.roles` reads the roster;
      `payload()` carries each duck's role; the chase brain's `inputs.chase.role`
      already reaches the inspector, so the page shows it for free; the
      brain picker gains nothing — a role is a property of the duck on
      the pitch, not a brain kind. Static roles first: dynamic swapping is
      exactly the churn the board's hysteresis was built to stop, and "the
      defender is nearest, swap" is its own measured item afterwards.

### 4. Telling a teammate from an opponent by colour — perception honesty

- [x] **4.1 The sim detector reports a colorway — DONE (2026-09-05).**
      `Target.color` (the World fills it from the duck's team),
      `Detection.color`, and two knobs in every noise preset:
      `color_range` (beyond it the classifier gives up) and `color_p`
      (inside it, how often it is right) — datasheet 1.0 m / 95%, hostile
      0.6 m / 75%. **A wrong answer is ANOTHER COLORWAY, not "unknown"**,
      because a softmax always answers, and a brain that treats
      "not my colour" as "opponent" has to survive that. `Track` keeps a
      VOTE over its hits, seeded at birth: at the hostile rate one look is
      a coin, and a brain that yields to a teammate on a coin is worse than
      one that ignores colour.

      **One part of this item was NOT done: `name` still carries the sim
      id.** The reason recorded here was wrong, and is corrected in place
      (2026-09-06). It said the `Tracker` associates detections to tracks by
      name, so dropping it would move every soccer number. `_associate`
      matches on class, a bearing gate and a range gate; the name is only
      voted onto the track (`Track.names`), and `Chase` never reads it —
      the only track attribute the soccer brain reads is `color`.
      **Measured, not read** (`scripts/probe_name.py`, the name stripped
      between detector and brain): 6 seeds x 120 s of 2v2, named tracks
      1 278 → 0, and the runs are **bit-for-bit identical** on 6 of 6 —
      duck tracks kept, kicks and falls all unchanged. Soccer does not
      depend on the sim id. What DOES is the tidy brain: `Tidy._trusted`
      is literally `bool(det.name)`, and `memory` / `given_up` are keyed by
      name. So this is a Track 12 change, not a soccer battery.
- [x] **4.2 Brains use it — BUILT, MEASURED, SHIPS OFF.** `Chase`
      gains `use_color` and `opp_keepout`: with the sense on, a duck gives a
      STRANGER its own keep-out radius and keeps the standard 0.40 m for a
      teammate, on the reasoning that the team board already coordinates
      teammates and nothing coordinates an opponent. `_is_mate` reads the
      track's vote, and unknown counts as an opponent — the cost of treating
      a teammate as a stranger is a wasted metre, the cost of the reverse is
      walking into one.
      **MEASURED, AND IT FAILED ITS CONFIRMATION — the knob ships off.**
      `MICRODUCK_CHASE="use_color=1,opp_keepout=0.55"` against a matched
      baseline, 3v3, 24 seeds and then 24 fresh ones (`runs/t7-*.jsonl`):

      | `crowd` | plain | colour | Δ | p | better on |
      |---|---:|---:|---:|---:|---|
      | seeds 0–23 | 26.5% | 19.7% | −0.068 | **0.002** | 16 of 24 |
      | **seeds 100–123** | **21.4%** | **20.5%** | **−0.009** | **0.593** | **9 of 24** |
      | pooled (48) | 23.9% | 20.1% | −0.038 | 0.008 | **25 of 48** |

      The pooled p is 0.008 and the pooled win rate is 25 of 48 — a coin.
      That combination IS the finding: the whole effect lives in the
      discovery block, which is the poacher's shape and the bump-stand
      rule's shape, and the rule that catches all three is this repo's
      third. Nothing else resolves either (falls 181 → 160, p = 0.27; goals
      +0.31, p = 0.30; advance −0.000; back-kicks 37% → 41%), and the one
      thing that does replicate is the COST: possession −2.16 s/min,
      p = 0.032.

      So a duck that gives strangers more room does not measurably crowd
      less, and it does hold the ball less. `use_color` and `opp_keepout`
      ship at 0. **The sense itself is not what failed** — the classifier
      and the track vote are exercised by tests and cost nothing at 0 — and
      it is now available for a rule that has a better idea what to do with
      it. Note also what this arm was up against: item 3.4's roles take the
      same metric from 22.4% to 3.5%, five times further, and they were
      confirmed. Standing off an opponent is a much weaker lever than
      standing somewhere useful in the first place.

      The defender marking the nearest opponent rather than shadowing the
      ball is NOT built — the posts already emptied the crowd, so marking
      would have to beat THAT rather than the old roster, which is a
      different and harder question.
- [x] **4.3 Goal sensing — MEASURED (2026-09-05), and the known-pitch
      assumption does NOT hold at the datasheet preset.**
      `scripts/probe_odom_goal.py`, 3 seeds × 300 s of 2v2 with the ducks
      actually playing, sampling each duck's odometry error once a second
      and turning the heading half of it into the miss it causes at the goal
      line from where that duck really stands:

      | odom | position error (med / 95%) | heading error (med / 95%) | miss at the goal (med / 95%) | over the 0.35 m half-width |
      |---|---|---|---|---:|
      | ideal | 0.000 / 0.000 m | 0.00° / 0.00° | 0.000 / 0.000 m | 0% |
      | **datasheet** | 0.171 / 1.397 m | **12.2° / 58.5°** | **0.348 / 1.870 m** | **50%** |
      | hostile | 0.634 / 3.636 m | 38.1° / 157.6° | 0.690 / 2.915 m | 68% |

      At `datasheet` the median miss is 0.348 m against a half-width of
      0.35 m: **half the time the duck's belief about where its goal is
      would put the shot outside the posts.** The dominant term is heading,
      not position — a gyro bias of 0.3°/s integrated over the ~200 s
      between kickoffs, and the kickoff's `_odom_reset` is the only thing
      re-anchoring it.

      **And the same drift breaks the TEAM's shared frame, which is the
      thing the blackboard is built on.** `brain/team.py` passes "the ball is
      at (x, y)" and "I am at (x, y, yaw)" in each duck's own odometry frame,
      and those frames are the same frame only for as long as nobody has
      drifted. Measured on the same runs — the distance between two
      teammates' position errors, i.e. how far apart their frames have
      wandered:

      | odom | teammates disagree about a point (median) |
      |---|---:|
      | ideal | 0.000 m |
      | **datasheet** | **0.456 m** |
      | hostile | 0.714 m |

      Half a metre is wider than the goal mouth's half-width and about
      thirteen ball diameters: at `datasheet`, one duck's "the ball is here"
      is not a place its teammate can act on. The board's own hysteresis and
      cost arithmetic are unaffected (they compare times, not places), but
      `Team.ball()` — the fix a supporter walks to and a blind duck is costed
      against — is. Nothing in the shipped brain notices. The fixes are the
      same two as for the goal: re-anchor on something both can see, or carry
      a per-duck frame offset the way `brain/mapping.py`'s loop closure
      already does for one duck's own map.

      Three consequences worth writing down. **(1) Every soccer number in
      this repo is measured at `ideal` odometry** (`make_pitch`'s default),
      so none of them is affected — and none of them is evidence about a
      robot either. **(2) A goal detector is no longer speculative**: it is
      the only listed option that re-anchors heading without a goal to walk
      to, and 4.1's colour-aware detector is the same machinery. **(3) The
      cheaper fix is re-anchoring**, which the pitch already does at every
      kickoff — a duck that could re-anchor on any landmark it sees would
      not need a goal class at all. Measure a bias-estimating odometry
      before building either: the presets are assumptions, and the docstring
      of `OdomNoise` says so.

### 4b. Does a duck know where its own kick sends the ball? — MEASURED (2026-09-05)

It did not, and chasing that question found something larger. The brain lays a
kick line `u`, stands square to it, hunts along it afterwards and tells its
teammates about it — and the ball leaves the foot at an angle to all of that.
`scripts/probe_kick_line.py` measures the angle in PLAY (the ball's travel over
`CARRY_S` after each swing) rather than on a bench, over **237 kicks**:

| foot | measured in play | 95% CI | the bench map said |
|---|---:|---:|---:|
| left | **+23.6°** | [+13.1, +34.1] | +21.6° |
| right | **−28.7°** | [−33.8, −23.6] | −11.0° |

The bench was right about the left foot and **18° wrong about the right** — it
swept a ball across a standing duck's foot at the sweet spot, and in play the
ball is nowhere near it (below). `kick_exit_*` carry the in-play numbers now.

- [x] **Rotating the STANCE to cancel the angle — REFUTED, with the
      mechanism.** `kick_deflect_*` set to the measured values, 24 paired
      seeds of 2v2: goals **2.08 → 1.38** (p = 0.045), ballAdvance
      **0.839 → 0.655** (p = 0.023), kicks **151 → 78**, and the aim error it
      was supposed to remove got **worse**, 45.4° → 67.3° mean absolute. The
      deflection is a function of where the ball sits relative to the foot
      (15°/cm near 2 cm, 4.5°/cm at 4–8 cm), so rotating the stance moves the
      ball to a different part of that function and produces a different,
      larger deflection — the right foot went from −27.3° to −48.6° off the
      body. **A fixed rotation cannot cancel an offset that its own rotation
      changes.**
- [x] **Using it as KNOWLEDGE — neutral, ships on.** `hunt_exit`: hunt along
      the true exit line and publish THAT to the board (`Team.publish_kick`).
      24 paired seeds and 24 fresh: nothing resolves on either block or
      pooled — goals +0.125 (p = 0.49), falls −0.083 (p = 0.77), possession
      −0.514 (p = 0.32), advance +0.023 (p = 0.48), signed progress −0.026
      (p = 0.54), back-kicks 34% → 37% (p = 0.45). A real null, not a dead
      path: the arms differ seed by seed. It ships on because the alternative
      is knowingly walking along a line the ball never took, and the search
      behind the hunt finds the ball anyway. **No performance claim.**

#### And then the probe was asked where the ball actually WAS — THE DUCK NEVER KICKS THE BALL PROPERLY

Over 191 kicks (24 seeds × 300 s of 2v2), the ball's position relative to the
kicking body at the instant of the swing:

| | measured | what the kick needs |
|---|---:|---:|
| ball ahead of the trunk | **0.238 m** (IQR 0.197–0.299) | 0.06–0.10 m |
| ball to the side | **0.141 m** (IQR 0.077–0.248) | 0.04–0.08 m |
| **on the sweet spot** | **0 of 191** | — |
| whiffed (ball moved < 10 cm) | **18%** | — |
| ball drift since the spot was planned | **0.220 m** (90th 0.400) | — |
| age of the plan being swung at | **3.26 s** | — |

**Not one kick in 191 had the ball where the kick policy was measured to send
it 1.6–2.3 m.** Every "kick" in this benchmark is a glancing contact at a
quarter of a metre — which is why the exit angle scatters 40°, why nearly a
fifth of swings move the ball less than 10 cm, and why the shipped kick
distance (17.7 cm of ball travel in the 2 s after a swing, README) is a tenth
of what the bench measured.

That single fact re-reads most of this track. **The kick map, the two-stage
line-up, `lineup_lat`, the aim clamp and the deflection compensation were all
arguing about the direction of a shot that was never struck.** It also
explains why they behaved the way they did: the clamp helped because it only
changes which way a glancing contact goes, and the deflection compensation
hurt because it moved a body whose foot was not on the ball anyway.

The cause is in the last two rows, and it is not precision — it is
**staleness**. The line-up plans a spot from a sighting at `refresh_min`
(0.35 m) or further, walks blind (the level camera loses a floor ball inside
~0.3 m), and swings 3.3 s later at a ball that has moved 0.22 m. That is the
whole of the 0.16 m shortfall. It is also why every arm that made the line-up
LONGER lost: **a longer line-up is a staler plan.**

- [x] **Two of the three ways out are now measured, and both fail.**
      *Arrive sooner*: `lineup_range` 0.6 → 0.35 does nothing (plan age
      3.26 → 3.44 s, whiff 18% → 23%); `lineup_s` 4.0 → 1.5 does exactly
      what it says — plan age 3.26 → 2.05 s, drift 0.220 → 0.151 m, the
      SIDE offset 0.141 → 0.093 m — and still leaves the ball 0.228 m ahead
      and **0% on the sweet spot**, at the cost of 61% of the kicks
      (191 → 74). *Aim where it will be*: `spot_lead` swept 0 / 0.5 / 1.0 /
      2.0 s, **0% on the sweet spot in every arm** and the whiff rate rising
      18% → 24%. The reason is the same blindness: the track's velocity is
      differenced from SIGHTINGS, which stop at `refresh_min`, so the
      prediction is extrapolated from data exactly as old as the plan it is
      meant to rescue. **You cannot predict your way out of not looking.**

      Three aim-side fixes have now died on this (`kick_deflect_*`, the
      line-up arms, and the lead), which is what turns the question into a
      question about the HEAD — below.
- [ ] **The remaining way out: stop going blind.** (1) Arrive sooner —
      `lineup_range` 0.6 → 0.35 or `lineup_s` 4.0 → 1.5, measuring now and
      judged on the on-spot fraction and the whiff rate, never on goals.
      (2) Aim where the ball WILL be — the tracker's `predict` with the decel
      model, currently off (`predict_s` = 0), which is the only option that
      addresses drift rather than avoiding it. (3) Refuse to swing at a stale
      plan — cheap, but the blind zone guarantees the sighting is old, so it
      risks starving the duck of kicks entirely.
      → **decide on:** the on-spot fraction (0 of 191 today) and the whiff
      rate (18%). Both are per-kick events with ~150–190 of them a battery,
      so they resolve where goals cannot. Only then the ledger.

#### 4c. The head — the blind radius IS a choice, and un-choosing it does not help (MEASURED, 2026-09-05)

Watching the ducks, the repo owner said they hunt for a ball that is under
their feet, that the neck looks straight when it could clearly bend further,
and that the head comes back UP as they walk in to kick. **All three are
true, all three are now measured, and fixing them changes nothing the kick
can feel.** Two new probes carry the numbers: `scripts/probe_head_pitch.py`
(the command swept on the shipped walker, and the blind radius read off the
real `Detector` on a real composed `World` at each pose) and
`scripts/probe_gaze.py` (what the head is doing, and how long the duck has
been blind, at each swing in play).

**The command against the camera.** Standing, the head-pitch slot buys
0.79 rad of camera depression per unit of command, linearly, until cmd 1.25
— where the `head_pitch` JOINT hits its +1.571 MJCF limit and the camera
stops at 1.17 rad (67°). The walker tracks the command all the way there and
stays upright, so **the limit is the joint, not the policy**, and
`head_down` = 0.6 (0.65 rad, 37°) was half of what is available. Walking,
the same slot costs forward speed: −11.7% at 0.6, −19.6% at 1.0, −28.2% at
1.25.

**The neck slot is real, free, and was never commanded.** `head_pose_cmd[0]`
is `neck_pitch`; `Chase.step` emitted `(0.0, gaze, 0.0, 0.0)`, so the brain
has never moved it. Positive looks UP, so a downward gaze is a NEGATIVE neck
command, worth 0.43 rad/unit standing. Depression is additive across the two
slots — and the price is not:

| camera depression | through the head slot | through both slots |
|---|---|---|
| 59° | cmd +1.00, **−19.6%** speed | neck −0.30 head +0.60, **+4.4%** |
| 69° | cmd +1.25, **−28.2%** | neck −0.60 head +0.60, **−4.9%** |
| 72° | not reachable (joint) | neck −0.40 head +0.80, **−1.1%** |
| 80° | not reachable | neck −0.75 head +0.75, −24.0% |

(walking at 0.30, four headings, steady over seconds 2–6). Standing the
depression peaks at **1.47 rad (84°)** at neck −1.0 / head +1.0 and then
comes back UP as the head joint saturates and the neck keeps folding.

**The blind radius per pose**, nearest floor ball the real detector still
reports, standing, ground distance trunk→ball — with the far edge beside it,
because pitching down trades the horizon for the feet:

| head pose | camera | nearest ball | visible out to |
|---|---:|---:|---:|
| level | 0.19 rad (11°) | **0.37 m** | 0.90+ m |
| head 0.6 (shipped clamp) | 0.65 (37°) | 0.18 m | 0.77 m |
| head 0.9 | 0.90 (52°) | 0.12 m | 0.36 m |
| head 1.25 (joint stop) | 1.17 (67°) | 0.08 m | 0.21 m |
| neck −1.0 head +1.0 | 1.46 (84°) | **0.05 m** | 0.15 m |

So "the level camera loses a floor ball inside ~0.3 m" is right: 0.37 m of
ground distance, which is 0.35 m of the slant `range_est` that `refresh_min`
is compared against. `refresh_min` = 0.35 is **exactly the level camera's
blind radius**, and a pitched head beats it by 20–30 cm.

**What the ducks were actually doing** (195 kicks, 24 seeds × 300 s of 2v2):
the head-pitch command at the swing is **0.000** at the median; the 3.5 s
run-up is head-UP 55% of the time and standing still 49%; the duck has not
seen the ball for **1.48 s and 0.175 m of walking** when it fires, 16 of 195
kicks never saw it at all in the whole run-up, and **81%** of the duck-steps
spent with the ball truly inside 0.40 m are steps in which the detector is
reporting nothing. Every one of the owner's three observations, as a number.

- [x] **Hold the gaze through the settle — mechanism confirmed, result null.**
      `ChaseParams.gaze_still` keeps the head down while the duck is standing
      still (a turn in place still takes it level: the walker cannot turn
      head-down, 0.2 rad in 5 s against 3.1), and `gaze_neck` routes the
      command across both slots. With both on: head-up 55% → 37%, blind at
      the swing 1.48 s / 0.175 m → **0.14 s / 0.006 m**, never-saw-it kicks
      16 of 195 → **1 of 189**. Then, over **48 paired seeds** (two blocks of
      24, the second fresh), 360 baseline kicks against 339: on the sweet
      spot 0/360 → 4/339 (p = 0.055, **all four in the first block**), whiff
      69/360 = 19.2% → 64/339 = 18.9% (p = 1.00), ball-ahead / spot-to-ball /
      plan age / drift all flat. The play ledger over 24 seeds of 2v2 is flat
      on goals, falls, possession, signed progress, spread, crowd and depth,
      with `ballAdvance` +0.115 (p = 0.048) while signed `ballProgress` is
      −0.003 (p = 0.98) — churn, per the reading rule. **Ships off.**
- [x] **Through the head slot ALONE it is worse, and that replicates.**
      `gaze_still` at `gaze_neck` = 0 takes the whiff rate 19.2% → **26.4%**
      (84 of 318, p = 0.027) in BOTH blocks (27.9%, 25.1%); with the neck
      carrying the same depression the cost disappears (18.9%). The bench
      predicted it: the head slot costs 12–28% of forward speed at these
      depressions and the split costs about nothing. **If the gaze is ever
      turned on, it must go through the neck.**
- [x] **Why it cannot win, measured.** On the kick spot the ball is
      `kick_ahead` 0.08 m ahead and `kick_side` 0.06 m to the side — **37°
      off the nose, and the camera's horizontal HALF-field is 31°**. The last
      centimetres of a line-up are unseeable at any pitch. What a held gaze
      recovers is the run-IN (at 0.20 m ahead the same side offset is 17°),
      and the spot has already been planned by then. The blindness was real;
      it was not what the kick was waiting for.
- [x] **`refresh_min` 0.35 → 0.20 is the one arm that survived fresh seeds
      — and it is a trade, not a win.** Pooled over 48 paired seeds, 150
      kicks against 360: ball ahead 0.244 → **0.182 m** (p = 7e-11),
      spot-to-ball 0.297 → **0.215** (p = 1e-12), plan age 3.32 → **2.30 s**
      (p = 2e-19), drift 0.224 → **0.151** (p = 7e-8), near the sweet spot
      1/360 → 9/150 (p = 0.0001). On the ledger, **possession 21.6 → 26.5
      s/min (p = 0.0003, better on 19 of 24 seeds)** and goals 39 → 47
      (p = 0.34) — with signed `ballProgress` FLAT (−0.050, p = 0.66) and
      **58% of the touches gone** (193 kicks → 84). The duck re-plans
      instead of swinging: possession bought with touches, the ball no
      further forward. Ships at 0.35, with the numbers.
- [ ] **What is left, in order.** (1) `gaze_yaw` — wired, unit-tested,
      **never run in a battery**: the only thing that can reach a ball 37°
      off the nose, and the ToF is in the HEAD, so a yawed head points the
      bumper sideways (which is how `look_aim` died). Unknown, not measured
      off. (2) The endpoint is a GEOMETRY problem, not a sensing one: with
      `kick_side` = 0.06 at `kick_ahead` = 0.08 the ball is outside the lens
      at the moment that matters, so a kick spot the duck can watch itself
      arrive on would need a different offset — or a kick that fires on the
      ToF rather than on the plan. (3) `refresh_min` with something that
      stops the re-plan dithering (freeze the FOOT choice once laid, re-plan
      only the position) — that is the specific failure the first
      `refresh_min` attempt recorded, and it would keep the placement gain
      without paying 58% of the touches.

**Revisited 2026-09-07 (the owner, watching: "they still miss the ball a
lot, they don't look fully down at their feet, maybe a scanning head").
All measured, 3v3 with roles, 24 seeds × 300 s, every arm forked with its
control on one package copy; the whiff line from `probe_kick_line.py`
(2v2, 24 seeds).** The joints were never the limit (84° down with neck +
head); the shipped brain looks 37° down through the head slot alone and
never commands the neck. Looking down THROUGH THE NECK while walking in
(`gaze_neck=1`, the split that costs no walking speed), level again for
the swing because the gaze drops when the duck stops:

| | shipped | neck gaze 0.64 | neck gaze 0.45 | neck gaze 0.64, bearing to 1.4 | + 0.8 rad sweep | sweep 0.6 alone |
|---|---|---|---|---|---|---|
| ball in view | 25.4% | 26.5% | 26.6% | **27.3%** (p=0.084) | 26.4% | 25.9% |
| losses over 2 s | 29% | 27% (t=−1.9) | **26%** (t=−3.0) | 26% (t=−2.2) | 28% | 27% |
| whiffs at the swing (2v2) | 66% | **47%** (p=0.043) | — | — | — | — |
| possession s/min | 16.9 | 15.5 (p=0.15) | 14.7 (p=0.059) | 16.1 (p=0.45) | 16.4 | **13.4 (p=0.009)** |
| ball progress | 0.084 | 0.046 | 0.080 | 0.082 | 0.118 | 0.046 |
| goals in 24 runs | 6 | 1 | 6 | 5 | 7 | 2 |
| falls in 24 runs | 1 | 2 | **7 (p=0.044)** | 7 (p=0.097) | 6 | 4 |

Three things in it. **Sight improves every time** — two points of view,
the long losses down by a tenth, whiffs at the swing cut by a third with
the head level at the swing in both arms (head +0.40 / neck +0.21, the
kick's requirement) and the plan 0.6 s fresher. **The ball does not
follow.** The 0.64 gaze trades the far view for the near one (losses per
run +5 to +12%, shorter ones); the widest-bearing arm is the best of them
and ledger-neutral. **And walking head-down falls**: 1 fall in 24 runs
shipped against 7, 7, 6 and 2 in the head-down arms — a walker trained
with the head near HOME is a walker that stumbles with it pitched, which
is the same limit item 7 found in the kick, one policy over. The search
SWEEP (0.6 rad, under the obstacle-sense threshold) costs possession
outright (−3.6 s/min, p=0.009): a duck that sweeps its head while
searching walks less. Also found: `gaze_bearing_max` never reached the
walking gaze (a hard-coded 0.6 rad) — fixed, default unchanged.

**What would actually improve their ability to look at the ball**, in the
order it pays: (1) the head-down kick retrain (the patch, a GPU run) —
then the gaze can be HELD through the swing, which is where 81% of the
blind close-ball steps are; (2) a walker retrained with the gaze poses in
its head-command range, so walking head-down stops costing falls; (3) the
replacement camera module the detector spec records (60° vertical
against the assumed 48°), which widens the near/far trade every arm above
ran into. Nothing here ships as a default; `gaze_neck=1, head_down=0.64,
gaze_bearing_max=1.4` is the configuration to re-run the day (1) or (2)
lands.

#### 4e. The search — the freeze is a flinch, and it is load-bearing (MEASURED, 2026-09-05)

Watching the ducks, the repo owner said they hunt for the ball by stopping and
looking down instead of keeping their momentum, and asked why the head cannot
sweep while walking. `scripts/probe_search.py` measures all of it.

**The stop is bigger and stranger than the constants suggest.** Search is
27.4 s of a 300 s run per duck (9.1%), and **61% of that is frozen** — zero
twist, gaze down at 0.22 m. The `0.6 / 1.5` duty cycle predicts 40%, and the
gap is the finding: `_search_t0` resets on every ENTRY into search, so **every
search opens with the 0.6 s freeze**. The median search is 0.60 s — exactly
the dip — **48% of searches are nothing but the opening freeze**, and 78%
never reach a second one. It is a flinch each time the ball leaves view
(~33 a run a duck), not a pause in a long hunt.

**And the dip does not look.** Detector frames binned by the state that
COMMANDED the head pose (read before the step, so the test is not circular):

| frames captured while | n | ball in frame | rate |
|---|---:|---:|---:|
| search, frozen in the dip | 15 646 | **11** | **0.07%** |
| search, walking the circle | 11 116 | 505 | 4.54% |
| not searching | 261 277 | 63 904 | 24.46% |

The dip takes 58% of search frames and returns 2.1% of the search's sightings
— **65× worse than simply walking on with the head level** (z = 26.2). The
comment beside it did not describe what it does; it has been corrected in
place.

- [x] **Removing the freeze is much worse, confirmed on fresh seeds.**
      `search_dip_s` = 0 gives back 19 s a run of standing still and costs
      **falls 121 → 195 over 48 seeds (+61%)**, kicks −30%, blocked seconds
      +5.9, for **no visibility gain at all** (+0.008 / +0.011, both
      p > 0.35). Standing is the one thing this walker does safely against
      another body, and the flinch fires exactly when the crowd is densest.
      Mis-commented, not mis-designed. (Dipping while WALKING is not
      available either: head-down the walker turns 0.2 rad in 5 s against
      3.1 level, measured twice in this repo.)
- [x] **The head sweep's old verdict was STALE, and it re-screens as a clean
      null.** It shipped off on "makes the body turn MORE, 5/5 seeds", whose
      recorded mechanism was the ToF being in the head. The clearance rule
      has since moved from sensor columns to bearings, and that coupling is
      dead: recomputing both rules on the same 1 439 904 frames, past
      0.70 rad of head yaw the old rule stops on **13.8%** of frames and the
      shipped one on **0.3%**. Re-run on 24 seeds, every field is inside the
      noise (spinFrac +0.007 p = 0.55, falls 57 → 56, possession +0.93).
      Free to leave off, free to turn on — not a knob with a reason any more.
- [x] **Head-yaw ball tracking is the one real effect and it is a trade.**
      `predict_s=1, head_yaw_when=always`: ball in view **+7.8 points on 44
      of 48 seeds** (p < 0.001, twice) against **falls +61%** (p < 0.001,
      p = 0.011), everything else flat. The cause is the flip side of the
      bearing fix — the sensor is still in the head, so a yawed head is now
      honestly blind rather than confidently wrong. `head_yaw_max` = 0.5 does
      not rescue it: the cap removes the falls and the visibility together.
      **And it does not touch the kick**: plan age 3.26 → 3.20 s, on-spot
      0% → 1%. The extra sight is at RANGE; the staleness is in the last
      0.35 m, which is 4c's geometry problem.
- [x] **A dead knob, caught by rule 0.** `head_yaw_when=always` ALONE changes
      nothing — three seeds came back bit-for-bit identical in every field.
      With `predict_s` = 0 the look target is `None` outside search/look, so
      the flag gates nothing. It is broken-not-null and needs `predict_s > 0`
      to mean anything; every arm above was run that way.
- [x] **The gate works, and head tracking SHIPS ON.** (2026-09-06.) The idea
      was to gate the head yaw on FORWARD CLEARANCE rather than cap its
      magnitude: in `Chase.step`, drop the look target when `ahead` is inside
      a margin, so the head leaves the walking line only while the bumper
      says the line is empty. That is `ChaseParams.yaw_clear`, shipped at
      **0.45** with `predict_s` = 1.0 and `head_yaw_when` = "always".

      Six arms, 300 s of 2v2 each: {baseline, tracking ungated, tracking
      gated} × {discovery seeds 0–23, fresh seeds 100–123}, paired on seed,
      48 seeds pooled. `scripts/probe_search.py --seeds 24 --seconds 300
      --per-side 2 --jobs 8`, arms in `runs/yawgate/`.

      | pooled, 48 paired seeds | ball in view | falls a run | median loss | kicks |
      |---|---|---|---|---|
      | gated vs baseline | **+8.0 pts** (p<0.0001, 42/48) | −0.06 (p=0.82) | **−0.27 s** (p=0.0002) | −0.44 (p=0.56) |
      | gated vs ungated | +0.2 pts (p=0.80) | **−1.60** (p<0.0001, 8/48) | −0.06 (p=0.43) | +0.92 (p=0.26) |
      | ungated vs baseline | +7.8 pts (p<0.0001, 44/48) | **+1.54** (p<0.0001) | −0.21 s (p=0.002) | −1.35 (p=0.07) |

      Read the middle row: the gate gives up **no** visibility and removes
      **all** of the falls. Possession, spread, crowd, depth, ballProgress,
      ballAdvance and spinFrac are flat in every contrast. Both blocks agree
      on every significant line, which is the bar that killed the colour
      keep-out. This is the second brain-tier change to survive a fresh-seed
      confirmation, after the aim clamp.

      **Why the cap failed and the gate did not** — the mechanism, from the
      ToF yaw bins over all 48 seeds. "Blind" is the head past 0.35 rad,
      where the bearing rule reports `+inf`; "obstacle ahead" is the old
      column rule, which does not care where the head points, saying
      something really is there:

      | | blind frames | of which, obstacle ahead | dangerous per 1000 |
      |---|---|---|---|
      | tracking off | 0.0% | – | 0.0 |
      | ungated | 16.7% | 13.0% | 21.8 |
      | gated 0.45 | 14.1% | 8.2% | **11.6** |

      The gate cuts total blind frames by 16% and the DANGEROUS ones by 47%.
      It is selective — the visibility lives in the harmless blind frames and
      the falls in the rest — which is exactly what a magnitude cap cannot
      do, because by magnitude the two live in the same frames.

      **Caveats, both worth keeping.** (1) The gate fails OPEN: `ahead` is
      `+inf` with no fresh ToF, so a dead sensor tracks the ball rather than
      freezing the head, and 8.2% of blind frames still have something ahead
      because the gate can only consult clearance the head can currently see.
      (2) `goals` moved +0.58 a run (p=0.019) on the fresh block and −0.21
      (p=0.51) on the discovery one — pooled p=0.36. That is what an
      underpowered metric looks like from the inside (4.1.5: 136 seeds), and
      it is the single most cherry-pickable number in this table. **No goals
      claim is made.**

      The 2×2 with the dip was not run: the dip re-screened as a clean null
      two items above, so it is not a factor to cross with.

### 5. Learned role brains — after 3 lands, and only if a learned striker can reach the ball

**Still gated, and item 3 landing does not open the gate.** The condition in
this heading was written before any of the above was measured, and it has
not been met: `striker-v1` loses to the scripted brain because it never
reaches the ball (possession 11.8 → 5.9 s/min, 3612 kicks at empty floor),
and nothing here has changed that. Item 3 makes the gate *harder*, not
easier — the scripted roles a learned brain would have to beat now hold
crowd at 3.5% and falls at 41 events over 48 seeds of 3v3, so a learned role
has a much better opponent than it did this morning. Do the approach first
(the roadmap's own note under 4.4: "a striker that reached the ball as often
as `Chase` does would be worth measuring; this one is not"), and only then
this.

- [ ] `StrikerEnv` with the role as an observation (a one-hot in the
      contract's reserved slots, plus the board's teammate poses in the
      body frame — the same eight-float pattern the striker's goal geometry
      used), scripted teammates and opponents as the world, reward = the
      per-team signed `ballProgress` the benchmark judges by minus an own
      goal (observable: the goal geometry slots). Every run named and
      described (`train-brain --title/--description --group soccer-roles`;
      add the group to `describe_brain.GROUPS`). The self-play ladder
      (sim-roadmap 4.5) after that. → **decide on:** `eval-striker`
      paired against the scripted role from item 3 on 24 fresh seeds —
      `possession` and advance-per-kick not below the scripted brain's,
      then the same numbers as 3.2 / 3.3. The striker-v1 lesson stands:
      an approach that cannot reach the ball is not a reward problem.

### What to build first, and what settles it

1 → 2 → 3.1 → 3.2 → 3.5 → 3.3 → 3.4 → 4 → 5. The first visible change is
2 (two colours on the pitch and a scoreboard that says who scored); the
first measured one is 3.1, and it has the cheapest number on the list:
back-kicks, 28 of 53 today, judged on 24 seeds in two minutes.

**That order was followed, and everything through 4.1 is done** (2026-09-05).
What is left, in the order it is worth doing:

1. ~~4.2's battery~~ — run, and it failed its confirmation; the knob ships
   at 0 and the sense stays for a better rule.
2. ~~**Zones that split at halfway for a roster with no midfielder.**~~
   **DONE (2026-09-05).** `zones_for` derives the split from the jobs that
   are present: a defender+striker pair owns the pitch at halfway (midfield
   is the striker's), thirds stay when a midfielder is on the roster.
   Lab builtins `pitch-2v2` / `pitch-3v3` stamp `formation_roles` (2:
   defender+striker, 3: defender+mid+striker); `eval-pitch` still calls
   `make_pitch` without `formation`, so the chase-vs-chase control is
   unchanged. Cover: a teammate `give_up_s` quicker than the zone owner may
   attack without a job rewrite. After a kick the board publishes the
   in-play exit line (`hunt_exit`, `Team.publish_kick`) only at kick-like
   speed. Locked by `tests/test_team.py` / `test_world.py` / `test_striker.py`.
   Skill: `.claude/skills/pitch-formation/SKILL.md`.
3. ~~**Gate the head yaw on forward clearance.**~~ **DONE (2026-09-06).**
   `yaw_clear` = 0.45 ships on with `predict_s` = 1.0 and `head_yaw_when` =
   "always". Head tracking buys +8.0 points of ball-in-view (p<0.0001, 42 of
   48 paired seeds) and 0.27 s off the median time the ball is lost, and the
   clearance gate removes the +1.54 falls a run it used to cost, giving up no
   visibility at all. Both seed blocks agree. The gate is SELECTIVE — it cuts
   blind frames by 16% but blind-with-something-ahead by 47% — which is why
   capping the yaw's magnitude had failed. Numbers and caveats in 4e.
4. **A striker that can reach the ball** (roadmap 4.4's own note). Every
   learned item is behind this one, and item 5 says why.
5. **`Detection.name` stops carrying the sim id** (4.1) — **re-scoped
   2026-09-06: it is not a soccer item at all.** It was listed here because
   4.1 recorded that the tracker associates on the name. It does not: it
   matches on class, bearing and range, and `Chase` never reads a track's
   name. Measured with `scripts/probe_name.py` (the name stripped between
   detector and brain): 6 seeds x 120 s of 2v2, named tracks 1 278 → 0,
   runs **bit-for-bit identical on 6 of 6**. The real dependency is the
   TIDY brain, where `_trusted` is `bool(det.name)` and both `memory` and
   `given_up` are keyed by it — so removing the free id is a Track 12
   design problem (what earns trust without an id?) and belongs there.
6. **The clear** (3.1). Deliberately not built: the clamp already aims as
   far up-pitch as the cone allows, and the case a clear would add — the
   walk-round — is the arm that measured worse.
7. ~~**Line-up precision**~~ (4b) — **CLOSED AS A CONTROL PROBLEM
   (2026-09-06). It got a coefficient, then every rule the coefficient
   allows was built and measured, and the answer is that the kick is a
   SENSING limit.** Read the closed list at the end before proposing
   anything here.

   Two arms had tried this and lost (`two_stage`, `lineup_lat`), both judged
   on goals, before the angle could be measured at all. It became measurable
   with `probe_kick_line.py`. Regressing aim error on where the ball
   actually was, over 462 kicks pooled from the 2026-09-06 gaze arms:

   | term | coefficient | p |
   |---|---|---|
   | **ball side offset** | **+1.90° per cm** | 1e-98 |
   | ball ahead offset | −0.13° per cm | 0.072 |
   | foot (left) | −16.2° | 2e-17 |
   | head yaw held at the swing | +0.03° per ° | 0.66 |

   R² = 0.56. **The sideways placement of the ball IS the kick error.** The
   sweet spot is 4–8 cm to the side; the observed median is 13.3 cm, so
   5–9 cm of avoidable offset is 10–17° of avoidable systematic error —
   comparable to the whole aim-clamp win, and available without touching aim.
   The residual scatter (33–49° of sd) is what is left after that.

   The head-yaw row is there because it is the trap: the raw correlation
   between head yaw at the swing and aim error is r = +0.34, p = 5e-15, and
   past 0.40 rad the mean error is +40.7°. It looks mechanical and it is
   entirely a proxy for the ball being off to the side. Control for the ball
   and it vanishes. Do not chase the head here.

   → **judge the next attempt on** median side offset at the swing (13.3 cm
   today) and on-spot % (1.7%), not on goals. `scripts/probe_kick_line.py`
   prints both, and its rows now carry `seed`, so arms can be paired.

   **And the specific thing to try, which is new.** Every previous attempt
   tried to REDUCE the offset — put the duck somewhere better. That is the
   staleness problem and it is already measured out: the duck reaches its
   planned spot to 1.4 cm, the spot is 0.285 m from the ball because the
   plan is 3.0 s old and the ball has drifted 0.21 m, and `refresh_min` 0.20
   fixes the staleness at the cost of 58% of the touches (confirmed on fresh
   seeds — read its note in `ChaseParams` before re-trying it).

   **A door that looked open and is now measured shut — read this before
   proposing it again.** The obvious use of a calibration is to compensate:
   rotate the stance (or the intended line) to pre-aim the shot by the
   expected deflection. `kick_deflect_*` is exactly that mechanism, and its
   old verdict was 8 seeds judged on goals, so it was re-run properly on
   24 paired seeds against the quantity it moves (`runs/deflect/`):

   | arm | kicks | mean absolute aim error |
   |---|---|---|
   | no compensation | 178 | 32.0° |
   | by the measured error (+13.7 / −6.0°) | 120 (−2.42/seed, p=0.004) | 44.7° (+13.1, p=0.0005) |
   | by the in-play map (+23.6 / −28.7°) | 80 (−3.82/seed, p<1e-4) | 43.0° (+11.4, p=0.037) |

   It does not halve the error, it DOUBLES it, and costs a third to a half
   of the touches. **The mechanism:** the kick spot is laid out in the
   rotated heading, so a rotation does not pre-aim the shot — it moves where
   the duck stands. Rotating the left stance +23.6° shifts the ball's
   departure off the body by +24.0°, essentially 1:1, and grows the side
   offset by +8.7 cm (p=0.0006). At +1.90°/cm that predicts +16.5° of extra
   error against +11.4 observed. **The coefficient predicts its own
   compensation failing.**

   So: **you cannot fix this kick by rotating anything.** Every rotation
   moves the side offset, and the side offset is the error. That kills the
   whole family — `kick_deflect_*`, a predicted-offset line rotation (which
   this roadmap proposed earlier the same day, and this refutes), and any
   other pre-aim. Two levers were left: the offset ITSELF (placement, which
   is the staleness problem above), and declining the shot.

   **The decline gate was built and it is measured off — but read WHY, it is
   not the idea that failed.** `ChaseParams.kick_side_max` drops the spot and
   re-approaches when the ball is too far to the side, refusing only on a
   fresh estimate so a stale ball still gets its swing. 24 paired seeds
   (`runs/decline/`), against 178 kicks / 137 effective / 23.0% whiffs:

   | arm | kicks | effective | whiff |
   |---|---|---|---|
   | gate off | 178 | 137 | 23.0% |
   | decline > 12 cm | 152 | 124 | 18.4% (p=0.059) |
   | decline > 9 cm | 133 (−1.88/seed, p=0.002) | 107 | 19.5% (p=0.11) |

   The per-foot bias LOOKS like it is coming out — left +13.7 → +7.1 → +4.1,
   right −6.0 → −0.6 → +1.6 — and that is the trap: every one of those
   intervals spans zero, the baseline's included. Paired per seed nothing
   improves, mean absolute error moves the wrong way, and touches fall. Rule
   6 again.

   **The diagnostic, and a correction.** The side offset of the kicks that
   SURVIVED the gate did not change (−0.002 m, p=0.83; +0.004 m, p=0.75).
   The obvious reading is that the estimate is junk and the gate refuses at
   random. That reading went into this item first and it is **wrong** —
   `scripts/probe_shot_gate.py` measures the signal directly, over 162 kicks
   on 24 seeds:

   | | |
   |---|---|
   | \|predicted side\| vs \|actual side\| | r = +0.48 (p = 7e-5) |
   | the estimate's own error | median 2.3 cm, 90th 8.3 cm |
   | as a gate at 0.12 m | refuses 31% of the swings it can see, and **88% of those really were wide** |
   | **swings it can see at all** | **34%** |

   The estimate is good. What it is not is AVAILABLE. The gate assesses a
   third of the population, refuses about a tenth of all swings, and cannot
   move a pooled statistic the other two thirds still dominate — while
   paying the full price in touches for the ones it does refuse. **Precision
   is fine; coverage is the problem**, and coverage is the camera's blind
   radius, not a threshold to retune.

   → **which leaves ONE lever, and it is not a brain rule.** Placement, aim
   compensation and shot selection have each now been measured out, and all
   three failed for the same underlying reason: the duck cannot see the ball
   when the swing is decided. Every route to seeing it has also been
   measured, and this is the place to record that the list is now closed:

   | route | verdict |
   |---|---|
   | pitch the head down (`gaze_still`, the search dip) | cannot reach — on the spot the ball is 37° off the nose, the camera's horizontal half-field is 31° |
   | yaw the head at it (`gaze_yaw`) | measured off 2026-09-06 — reaches the endpoint, fixes nothing, adds +12.4° of bias |
   | read it off the ToF (`tof_ball_m`) | measured off twice — redundant on 88–94% of the ticks it fires, and wrong 2 times in 3 in the blind case it exists for |
   | predict it (`predict_s`, `spot_lead`) | `spot_lead` measured off; the prediction exists on under half of swings and cannot price a shot |
   | re-plan later (`refresh_min`) | fixes the staleness, costs 58% of the touches, confirmed on fresh seeds |

   The camera sits 23 cm above the floor pointing forward, which puts a
   blind radius of 23 cm under the duck's nose at a level head and 6.9 cm
   fully dipped (`docs/camera-hardware.md` §3). The kick spot is inside it.
   Everything above is worth keeping because it says, with numbers, that the
   rules have been tried.

   **CORRECTED THE SAME EVENING — it is not a sensing limit after all, and
   the sensor the duck has can see the spot.** Three measurements, all on
   the one real camera, all on the pre-rolling-resistance floor:

   1. *The "any pitch" line in the table above was a level-camera rule.*
      Measured on the composed model: with the head 60° down, a ball on the
      kick spot (37° off the nose) sits at **16.5° camera bearing, −19.7°
      elevation** — inside the frustum — and nothing on the duck occludes
      it (`mj_ray` from the lens hits the ball first in every case). The
      neck-carried gaze reaches ~52° at `head_down`. The gaze was being
      REFUSED at exactly that ball by `GAZE_MAX_BEARING` = 0.6, whose
      comment said the target "is not in the picture at any head pitch".
      Now `ChaseParams.gaze_bearing_max`, corrected in place.
   2. *Looking down works.* `scripts/probe_shot_gate.py`, "swings the brain
      can see at all": shipped **34%**; `gaze_still=1, gaze_neck=1` **65%**,
      with the estimate's correlation to the true offset r = 0.48 → **0.95**
      and its median error 2.3 → **0.9 cm**. Lifting the azimuth cap to 1.4
      adds nothing further (64%) — the pitch was the lever, not the cap.
      And the ball moves where the whole track has been trying to put it:
      side offset at the swing **0.133 → 0.06 m** (the sweet spot is
      0.04–0.08), on-spot **1.7% → 18–21%**, drift since the plan
      0.213 → 0.06 m.
   3. *And then the kick whiffs.* Those same arms whiff **82–85%** against
      23%, on ~50 kicks a battery against 178. Benched directly (12 kicks
      each, ball on the left sweet spot ±1 cm, `kick_left`):

      | head at the swing | head joint | whiff | median travel |
      |---|---|---|---|
      | level | +0.39 rad | **0%** | 2.02 m at +23.9° off the body |
      | head down (cmd 0.6) | +0.97 rad | **100%** | 0.00 m |
      | head + neck down | +0.93 rad | **100%** | 0.00 m |

      The arena zeroes the kick's head COMMAND for its 0.5 s window, not
      its joints; the shipped `ball_kick_*.onnx` were trained from a level
      head and cannot swing from a pitched one. The +23.9° level result is
      the in-play left-foot exit angle (+23.6°, 4b) reproduced on a bench.

   So the true shape is: **the camera can see the spot; the gaze can put the
   ball on it; the KICK SKILL cannot swing from the pose that does it.** That
   is a training-distribution problem in `ball_kick_left/right` — the third
   different kind of limit this item has named, and the first that points at
   something buildable without hardware: either kicks trained with the head
   pitched (the honest fix, upstream's recipe), or a brain that raises the
   head to level in the last ~0.3 s of the settle and fires when the joint
   is back (`settle_head_level`, not built) — which costs a sighting again,
   but 0.3 s of a stationary ball is not 3 s of a rolling one. → judge on
   whiff and on-spot from `probe_kick_line.py`, re-baselined on the new
   floor (see below).

   **Baseline note.** Every number in Track 4 up to here was measured on a
   floor with NO rolling resistance — a bumped ball rolled until a wall
   stopped it. `Ball.rolling` (0.002, condim 6) landed in this checkout on
   the evening of 2026-09-06 from a parallel session, with `ball_decel`
   0.04 → 0.3 to match. Every kick and search battery needs a new baseline
   before anything below is compared to anything above; the paired
   comparisons above are internally consistent because each battery's
   workers forked before the physics landed.

   **The settle that raises the head — BUILT, and measured off on the new
   floor (2026-09-06, late).** `ChaseParams.settle_head_level`: for the
   last N seconds of the settle the gaze is dropped and the head commanded
   level, so the swing starts from the pose the kicks were trained in. Its
   lead was benched first: head+neck gazed down, then level for `lead`,
   then `kick_left` on the sweet spot —

   | lead | head joint at the swing | whiff |
   |---|---|---|
   | 0.0 s | +0.93 | 100% |
   | 0.1 s | +0.64 | 100% |
   | **0.2 s** | +0.45 | **0%** |
   | 0.3 s | +0.41 | 0% |

   In play (24 seeds, four arms forked on one tree state, TODAY'S floor):

   | arm | kicks | whiff | on-spot | ball ahead | effective kicks/seed vs shipped |
   |---|---|---|---|---|---|
   | shipped | 47 | 34% | 23% | 0.117 m | – |
   | gaze (still+neck) | 50 | 76% (p<0.001) | 6% | 0.138 | −0.79 (p=0.003) |
   | gaze + raise 0.3 s | 42 | 52% (p=0.08) | 10% | 0.163 | −0.46 (p=0.11) |
   | gaze + raise, settle 0.6 s | 39 | 44% (p=0.37) | 5% | 0.128 | −0.38 (p=0.25) |

   The raise does what it was built to do — with it the head-pitch joint at
   the swing is a median **+0.41 rad on all 42 kicks, none above 0.6** — and
   it recovers about half the gaze's whiffs. It does not beat shipped, and
   two things say why. First, **the floor fix did more for placement than
   any brain rule this session**: on the old floor shipped put 2% of kicks
   on the sweet spot with the ball drifting 0.213 m off a 3 s plan; on the
   new floor it puts **23%** there with 0.072 m of drift, because a ball
   that stops within a stride does not leave the plan behind. The gaze's
   whole advantage was drift, and the floor took it. Second, a residual the
   head does not explain: with the head level and the ball inside 10 cm,
   shipped whiffs **0 of 14** and the raise arm **4 of 12** — so the gaze
   line carries a second cost at the swing, not measured, most likely the
   neck joint (only head-pitch was recorded) or the posture the walker
   arrives in after a line-up walked head-down. `settle_head_level` ships
   at 0, with the mechanism confirmed and the lead benched, for the day the
   kick skill is retrained head-down and the gaze becomes worth holding.

   **The next lever, on the new floor, with its number.** Whiff by where
   the ball was ahead of the trunk at the swing, shipped brain:

   | ball ahead | 0–10 cm | 10–15 | 15–25 | >25 |
   |---|---|---|---|---|
   | whiff | **0%** (n=14) | 27% (15) | 71% (7) | 75% (8) |

   The kick reaches ~10 cm. `kick_ahead` = 0.08 plans the spot 8 cm behind
   the ball and the residual drift is 4–7 cm, so the median ball is 11.7 cm
   ahead at the swing and a third of them are past the reach. That is the
   whole whiff on this floor, and it is not a sensing problem. Both obvious
   levers were then measured the same evening, and both close:

   - *Decline on AHEAD* cannot fire. On this floor the shipped brain has a
     fresh ball estimate on **13%** of swings (8 of 62; was 34% on the old
     floor) — a ball that stops is last seen more than `predict_s` before
     the swing — so a gate on the predicted distance sees nothing.
   - *Plan the spot closer* bumps the ball. `kick_ahead` 0.06 and 0.05
     against 0.08, 24 seeds (`runs/ahead/`): the duck still reaches its
     spot (trunk-to-spot 0.015–0.016 m, flat) but the ball has moved since
     the plan — drift **0.042 → 0.108 m, +5.4 cm (p=0.002) and +7.0 cm
     (p=0.031)**, the only significant effect — and sits further out
     (spot-to-ball 0.135 → 0.165). Whiff 39 → 44/50%, on-spot 14 → 7/9%,
     side 0.059 → 0.096/0.108, none significant on ~70 kicks, all the
     wrong way. The walk-in's feet reach a ball 5–6 cm ahead before the
     settle does. **0.08 is the walk-in's minimum.**

   **And this is where measuring on this floor stops, deliberately.** The
   parallel session's world files (`compose.py`, `scenario.py`,
   `arena.py`) changed again at 19:36:29, between the sweep's second and
   third arms, so only shipped-vs-0.06 shares a tree state; three "shipped"
   forks tonight (19:10, 19:33, 19:46) gave 34%, 39% and 63% whiff on the
   same seeds as `compose.py` and `arena.py` kept moving under them. Every
   number in this item from "Baseline note" down is provisional until that
   session commits. Re-baseline then; the arms and the probes are all here
   to re-run — `probe_kick_line.py` now records the head joint, the neck
   joint and the trunk pitch at every swing.

   **The residual is the NECK, and the retrain is specified to the line.**
   With `settle_head_level` 0.3 the head joint is level at the swing
   (+0.41) but the neck is still low — +0.17 against shipped's +0.21, lower
   quartile +0.14 — and pooled over both arms with the ball inside 15 cm:

   | neck joint at the swing | n | whiff |
   |---|---|---|
   | +0.00 .. +0.15 | 12 | **83%** |
   | +0.15 .. +0.30 | 51 | 43% |

   Trunk pitch is flat (−0.004 vs −0.006) and is not it. The kick skill
   needs BOTH `neck_pitch` and `head_pitch` near HOME, and upstream's kick
   task (`microduck_ball_kick_env_cfg.py`) resets every joint within
   ±0.05 rad of HOME and pulls the neck home with `pose_stand_neck` — it
   never saw a gazing start. **The port is
   `docs/patches/microduck_rl-kick-head-down.patch`**: two reset events
   after `reset_robot_joints`, `head_pitch` offset (0, +0.60) and
   `neck_pitch` offset (−0.30, 0) — the gaze pose is head +0.95 / neck
   −0.05 against HOME +0.39 / +0.21 — leaving `pose_stand_neck` to pay the
   policy to bring the head home as it kicks. Validated: the patched cfg
   constructs on CPU with the events ordered `reset_robot_joints` → head →
   neck → `set_ground_state`, and `git apply --check` passes against the
   pinned `badc4e7`. It needs the GPU stack to train (AGENTS.md: the
   sim2real recipe is upstream's); nothing here can run it.

   **Retrained locally after all (2026-09-07, 10:00) — the premise below
   was wrong, and the kick trains here in four minutes.** The walk env
   indexes joints by address, so the ball's free joint changes nothing it
   assumes; only the keyframes were missing, and `contract.
   scene_walk_ball_xml()` supplies them (the walk scene rewritten beside
   symlinks to the upstream files, the ball included last, every keyframe
   padded by its seven qpos). `behaviors/kick.py` is upstream's recipe on
   that scene with the one thing the shipped kick never saw — the head and
   neck spawned across the gaze range. `train-behavior kick_right --steps
   2000000 --envs 12`: ~11k steps/s, 2M steps in ~4 minutes; both feet in
   under ten. `scripts/bench_kick_headdown.py`, 12 seeds a pose, 1.2 s from
   standing with the ball on the sweet spot:

> **2026-09-08 (code review).** Two corrections to the bench table below: the row labelled "the line-up gaze" is head 1.30 rad ABSOLUTE (an offset of +0.95 on the 0.349 home pitch), 0.35 rad past the gaze the brain actually holds — the +0.60 row (0.95 rad absolute, `head_down` 0.6) is the line-up gaze; the whiff verdict (0% everywhere) stands. And the sidecar exit angles the brain now reads for the local kicks (−0.16 / 0.0 rad) are BENCH numbers; the shipped kicks read +5°/−11° on the bench against +24°/−29° in play, so it was measured in play the same day (`probe_kick_line.py --ball-out-s 5`, 24 seeds × 300 s of 2v2 a block): the LEFT foot leaves at +15.0° off the body (52 kicks, SE 6.7°), not −9.2° — the sidecar now carries 0.26 rad, and a fresh block (seeds 100–123) with it took the left foot's systematic error against the intended line from +29.6° to −4.5° (SE 12°); pooled over both blocks the raw exit is +12° ± 6°. The RIGHT foot's bench 0.0 agreed with play (+2.1° then +9.9°, pooled +6° ± 3°, not decisive; kept). The scatter is the story either way: sd 40–70° a foot, and 44–51% of swings still whiff.

   | head pose at the swing | shipped right | **local right** | shipped left | **local left** |
   |---|---|---|---|---|
   | level | 33% whiff, 1.00 m | 0%, 1.01 m | 25%, 0.84 m | 0%, 1.12 m |
   | head +0.60 (the shipped gaze clamp) | 100%, 0 m | 0%, 1.06 m | 83%, 0 m | 0%, 1.22 m |
   | neck −0.30 / head +0.60 (the split) | 100% | 0%, 1.16 m | 100% | 0%, 1.25 m |
   | neck −0.25 / head +0.95 off home = 1.30 rad absolute (past the line-up gaze; the +0.60 row IS the line-up gaze — this row was mislabelled "the line-up gaze" until 2026-09-08) | 100% | 0%, 1.13 m | 100% | 0%, 1.28 m |
   | peak speed / when | 1.03 m/s at 0.15 s | 1.2–1.5 at 0.18 s | 0.88 at 0.18 s | 1.2–1.5 at 0.14 s |
   | exit off the body | −11° (in play −29°) | ~0° (sd 7–14) | +5° (in play +24°) | −9° (sd 4) |
   | falls in 60 swings | 0 | 0 | 0 | 0 |

   **Zero whiffs from every gaze pose, both feet**, the ball 1.0–1.3 m
   away at 1.2–1.5 m/s, no falls; the rendered rollout shows the duck
   start looking at its feet, plant the left foot, swing at 0.2 s and
   settle standing. The exits are near straight where the shipped kicks
   bend, so the brain's `kick_exit_left/right` (+23.6° / −28.7°, measured
   in play for the shipped pair) are set per kick when these run
   (`kick_exit_left=-0.16,kick_exit_right=0.0`), and the arena takes them
   with `MICRODUCK_SKILL_KICK_RIGHT/LEFT=runs/<run>/policy.onnx`. In play,
   with the gaze HELD through the swing (`gaze_still=1,gaze_neck=1`) —
   the configuration that put the ball in view at 0.14 s before the swing
   and whiffed on the shipped kick — was measured next (runs/localkick;
   2v2 kick probe, 24 seeds; 3v3 ledger with roles, 24 seeds; every arm
   forked with the others on one package copy):

   | at the swing (2v2) | shipped | shipped + gaze held | **local kicks** | **local + gaze held** |
   |---|---|---|---|---|
   | kicks | 64 | 56 | 45 | 50 |
   | whiff (< 10 cm of travel) | 66% | 73% | **42%** | **34%** |
   | …of the head-down swings | — | 84% of 38 | — | 37% of 38 |
   | on the sweet spot | 5% | 5% | **16%** | 6% |
   | ball drift since the plan | 0.165 m | 0.154 | **0.063** | 0.126 |
   | ball ahead of the trunk | 0.172 m | 0.180 | 0.127 | 0.164 |
   | aim error of the kicks that connected | 29° (sd 34) | 69° (sd 83) | 30° (sd 49) | 43° (sd 52) |

   | 3v3 ledger | shipped | local kicks | local + gaze held | shipped + gaze held |
   |---|---|---|---|---|
   | possession s/min | 16.94 | 16.98 | 17.10 | 16.39 |
   | ball progress | 0.084 | 0.072 (p=0.71) | **0.012 (p=0.023)** | 0.036 (p=0.12) |
   | goals / own goals | 6 / 1 | 6 / 2 | 5 / 1 | 5 / 2 |
   | back-kicks a run | 0.83 | **0.46** (p=0.12) | 0.75 | 0.71 |
   | kicks a run / falls | 3.33 / 1 | 2.63 / 2 | 2.88 / 4 | 3.04 / 3 |

   **The local kicks halve the whiff in play** (66 → 42%, two-proportion
   p≈0.01; 34% with the gaze held, p≈0.001) — the 42% that remain are
   the ball off the spot and moving, not the pose, as the bench's 0% from
   every pose says — land on the sweet spot three times as often, and
   leave the ledger where it was (possession, progress, goals flat;
   back-kicks fewer). **The gaze held through the swing still costs the
   ball**, with either kick: progress 0.084 → 0.012 (p=0.023) with the
   local kicks, 0.036 with the shipped; the kicks that connect head-down
   aim worse (43–69° against 29–30°) — the head-down line-up puts the
   ball somewhere the plan did not expect. So the kicks are worth
   having and the held gaze is still not, which is the same verdict item
   7 reached with one fewer reason.

   **Fresh block (seeds 100–123):** whiff 55% → 38%, on the sweet spot
   2% → 12%, drift 0.167 → 0.068 m, aim error 45° → 35°; **pooled over
   48 seeds, whiff 61% → 40% (p=0.0025).** The local kicks are the sim's
   default (13:05): vendored under `microduck_local/policies/kick/` with
   sidecars (`exit_rad`, the bench and in-play numbers), resolved by
   `World.skill_path` before the shipped Hub file, their exit angles
   reaching the brain through `brain_kwargs` (`World.kick_exits`) unless
   the command line names them; `MICRODUCK_SKILL_<NAME>` still wins. The
   /sim lab was restarted on this code the same minute — until then it had
   been a 00:49 process on `c9d1d43` with none of tonight's changes in it,
   which is why the owner saw nothing new. **`support_gaze` — the
   supporter looking at the ball, which is most of what the eye sees on
   /sim — measured a null** (3v3, local kicks, 24 seeds: ball in view
   24.1 → 24.2%, possession −1.6 s/min p=0.15, everything else flat) and
   stays off: a supporter stands 0.7–1.0 m off the ball, at the edge of
   `head_range`, so there is little for the gaze to add. These are local
   policies for the SIM; the robot's kick still ships from upstream (the
   patch), as AGENTS.md's sim2real rule requires.

   *The note as it stood before that, kept because its premise is what
   was measured wrong:* The local `train-behavior`
   has no kick behaviour, and cannot have one cheaply: `MicroduckWalkEnv`
   knows two scenes, neither with a ball, and upstream's `scene_ball.xml`
   adds a free joint that changes `nq` — every joint slice, keyframe and
   observation index the local env assumes. `distill` collects its
   observations in the walk env under walking commands, off-distribution
   for a standing kick. A local kick behaviour is a real track (ball
   scene + nq handling, distill-in-the-kick-env as the warm start, reward
   = ball speed along the heading + support foot planted + settle, the
   head/neck spawned across the gaze range) and its product would still
   need the upstream retrain to reach the robot. The patch is the part
   that reaches the robot.
8. ~~**`gaze_yaw`**~~ (4c) — **MEASURED OFF (2026-09-06).** Two corrections
   and a result, all measured.

   *It was a dead knob alone.* `gaze_yaw` only produces a non-zero yaw
   inside the `gaze_still` branch, and `_gaze_range` — the only thing it
   widens — is called from nowhere else. 24 000 duck-ticks over three seeds,
   4 157 in lineup/settle where it would apply: **0 differ** with it on. The
   same shape as the `head_yaw_when="always"` dead knob in 4e, caught the
   same way, before a battery was spent on it. "Default unknown" was too
   generous — with `gaze_still` off there was nothing to be unknown about.

   *Its blocker had a fix.* `gaze_still` was parked partly on the ToF hazard
   that 4e has since measured and gated, so `yaw_clear` now covers the gaze
   yaw too (proven bit-for-bit inert on the shipped brain). That made the
   real arm runnable: `gaze_still=1, gaze_neck=1, gaze_yaw=1`.

   *And it does not help.* Three arms x 24 seeds x 300 s of 2v2 on
   `scripts/probe_kick_line.py`, against the shipped 178 kicks / 23.0%
   whiffs / 1.7% on-spot:

   | arm | kicks | whiff | on-spot | spot-to-ball | plan age |
   |---|---|---|---|---|---|
   | shipped | 178 | 23.0% | 1.7% | 0.285 m | 3.03 s |
   | held gaze on the neck | 204 | 23.5% (p=0.91) | 0.5% | 0.315 m | 3.05 s |
   | + gaze yaw | 211 | 19.9% (p=0.45) | 0.5% | 0.307 m | 2.96 s |

   It reaches the 37° endpoint and nothing it was meant to fix moves.
   Paired per seed the absolute aim error is flat as well (p=0.93). What it
   DOES move is the systematic aim: **+12.4°, 95% CI [+6.2, +18.6]**,
   against a shipped brain whose interval spans zero. A systematic bias
   lands on every kick the same way, so that is a real cost for nothing.

9. **Head tracking does not bias the kick** — settled 2026-09-06 because it
   was a risk to what 4e shipped, and it is closed, not open. Median head
   yaw at the swing is 0.000 on the shipped brain (the duck is back on the
   line by then), mean aim error −0.1° with an interval spanning zero,
   against +2.4° with tracking off. Paired per seed every kick metric is
   flat: kicks p=0.68, whiffs p=0.64, |error| p=0.73, ball-ahead p=0.21.

10. ~~**A shared frame for the blackboard**~~ (4.3) — **DONE (2026-09-06,
   late): self-localisation from the goal posts.** At `datasheet` drift two
   teammates' frames wandered 0.456 m apart over a run, so "the ball is at
   (x, y)" stopped being a place the teammate could act on. Now
   `brain/localize.py`: a 200-particle filter over (x, y, yaw) in the
   odometry-at-spawn frame, moved by the odometry's own deltas, weighed by
   sightings of the four goal posts (a `post` detection class the sim's
   detector reports as fixed-position landmarks on the mouth line; on the
   robot a coloured-post class in duck_detect), nearest-post association,
   resample at half ESS, re-seeded on a respawn, one weighing per frame.
   Wired into `Chase` before anything reads the pose, and turned on by
   `brain_kwargs` for any duck whose odometry preset is not `ideal`.
   Measured with `scripts/probe_odom_goal.py`, 3 seeds × 300 s of 2v2,
   medians over the run:

   | preset | position error | yaw error | miss at goal line | over half-width | mates disagree |
   |---|---|---|---|---|---|
   | datasheet, raw | 0.215 m | 20.2° | 0.663 m | 68% | 0.295 m |
   | datasheet, localised | **0.072 m** | **2.3°** | **0.075 m** | **11%** | **0.078 m** |
   | hostile, raw | 0.706 m | 66.3° | 1.034 m | 80% | 0.958 m |
   | hostile, localised | **0.089 m** | **5.4°** | **0.158 m** | **30%** | **0.146 m** |

   A shot laid out in the localised frame misses the goal line by 7.5 cm
   at `datasheet` where it missed by 66 cm; two teammates now put the same
   ball 8 cm apart where they put it 30 cm apart. This is the metric that
   does NOT depend on the ball's floor, so it stands regardless of the
   parallel session's physics. At `ideal` the knob stays off and the
   shipped brain is bit for bit what it was (checked on package copies
   with the posts present: 0 of 24 000 ticks differ — post detections draw
   from their own random stream and the tracker ignores the class).
   Locked by `tests/test_localize.py` and `test_team.py`. What it opens:
   C.3 (a shared world model with a frame that means the same thing to
   both ducks) and passing (D.1), which were both waiting on this.

11. **Where a run's time actually goes — the game-flow budget (2026-09-07,
   evening).** Jonathan asked what could be improved about the soccer
   games. Before another knob, a clock was put on the whole run: a
   per-tick probe of every duck's state, the ball's zone (open / within
   0.20 m of the boards / a corner), stretches the ball sits still for
   5 s or more, and every line-up's exit. 8 seeds x 300 s, the lab's
   `pitch-3v3` and `pitch-2v2` rosters (formation roles on, as the /sim
   page runs them), on a package copy of the tree at `00e408d` plus the
   parallel session's uncommitted world files:

   | | 3v3 | 2v2 |
   |---|---|---|
   | ball at rest >= 5 s | **85%** of the run (254 s) | 84% |
   | …with a duck a median | 0.24 m from it | 0.24 m |
   | ball at the boards / in a corner | 51% / 22% | 37% / 28% |
   | kicks a run (whole pitch) | 1.9 | 2.4 |
   | kicks taken at the boards or a corner | **0 of 15** | 1 of 19 |
   | line-ups entered / settled | 439 / 16 (**96% abandoned**) | 507 / 24 |
   | a duck's time in `support` / `search` / `retreat` / `lineup` | 61 / 10 / 10 / 7% | 45 / 15 / 13 / 11% |

   Three things, in the order they were found, each with its lever
   measured on the same seeds (12 x 300 s of 3v3, arms forked on one
   package copy, paired per seed, then a fresh block of seeds 100-111
   for the one that ships):

   **(a) The cover role churned every tick — FIXED (`Team.candidates`).**
   `record-world`'s own doc names the symptom ("a chase duck cycling
   every 0.1 s is fighting a teammate") and the seed-1 events log had it:
   the midfielder and the striker swapping `support` <-> `hunt` at the
   control rate for 30 s straight with the ball dead between them. The
   board admits an off-zone teammate as cover only while it is
   `give_up_s` (2 s) quicker than the zone owner; the moment it holds the
   role and its cost jitters above that line it is no longer a
   *candidate*, and `attacker` moves the role at once (the "incumbent not
   live" path has no hysteresis), then the give-up path moves it back
   the next tick. Instrumented: 300 handovers a run, median spell 0.09 s,
   77% of spells under a second, 1968 of 3605 handovers by that path.
   The fix keeps the incumbent cover a candidate until an owner is
   `give_up_s` quicker, so the way back runs through the same
   `switch_s`/`hold_s` hysteresis as every other handover (a keeper is
   never kept out of its box). Discovery block / fresh block:

   | | base | fix | p |
   |---|---|---|---|
   | handovers a run | 300 / 289 | **46 / 44** | <0.001 |
   | spells under 1 s | 77% / 75% | **10% / 9%** | <0.001 |
   | median spell | 0.09 / 0.12 s | 9.7 / 10.5 s | <0.001 |
   | `search` s a duck | 29.7 / 28.5 | 22.8 / 22.7 | <0.001 both |
   | `lineup` s a duck | 21.3 / 24.0 | 26.8 / 26.9 | 0.003 / 0.11 |
   | line-ups lost to `support` | 96 / 103 | **12 / 16** | — |
   | possession, progress, advance, spread, depth, goals, kicks | flat | flat | all p > 0.3 |
   | crowd | 0.245 / 0.197 | 0.249 / 0.273 | 0.86 / 0.065 |
   | falls, 24 runs pooled | 1 | 7 | — |

   The ledger does not move (the ball is still dead, see (b)); the shape
   caution is real and stated: the cover now stays on the ball while the
   owner arrives, so two teammates are within 0.5 m of it a little more
   (crowd +0.08 on the fresh block, p=0.065) and falls went 1 -> 7 in 24
   runs — still 0.3 a run, and every one a duck near another duck. Ships,
   with `tests/test_team.py::test_cover_that_holds_the_role_leaves_through_the_hysteresis`.

   **(b) The ball at the boards is unkickable, and it is there 72% of the
   time.** With (a) in, the line-up exits were re-read by zone: every one
   of the 36 kicks in both rosters was taken in the open; at the boards
   and in the corners **every** line-up ran out the 4 s `lineup_s`
   timeout (search exits, 99% at age >= 4 s, 194 of 212 in 3v3). The
   kick spot is laid 8 cm behind the ball on the line to the goal with no
   regard for the boards: for a ball against the side wall that line
   tilts away from the wall, so the spot is *inside* it, and the walker
   stops at `tof_stop` and stands until the timeout, then searches, sees
   the ball, lines up again — 11-12 line-ups a minute of boards time,
   none of them a kick. Two levers, both measured against (a):

   | fix + … | dead-ball s | kicks a run | possession | progress | advance | goals | falls |
   |---|---|---|---|---|---|---|---|
   | (a) alone | 247 | 2.9 | 34.7 | 0.119 | 0.45 | 0.08 | 0.25 |
   | + kick ALONG the boards (`board_margin` 0.12) | 242 (p=0.44) | 4.2 (p=0.056) | 36.1 | 0.158 | 0.50 | 0.33 (p=0.056) | 0.25 |
   | + BALL OUT (World: at rest 5 s within 0.20 m of the boards -> placed 0.45 m in) | **176 (p<0.001)** | **7.8 (p<0.001)** | **42.8 (p<0.001)** | **0.32 (p=0.002)** | **1.02 (p<0.001)** | 0.42 (p=0.08) | 0.25 |

   The brain rule (a kick line along the side wall up the pitch, or
   along the end wall toward the middle, whenever the spot would land
   within `board_margin` of the boards; the foot chosen so the body
   stands on the open side) is the honest fix and it is not enough alone:
   kicks +1.3 a run at p=0.056, the dead-ball clock flat, crowd up. The
   ball-out rule is what a referee does on a walled table and it is
   worth 2.6x the kicks and a quarter of the dead time — but it changes
   the benchmark under every number in this track, so it goes in as a
   World knob (0 = off, bit for bit what was measured) and ON for the
   lab's pitch builtins, the way `getup_s` and the kickoff rule went in.
   Both were first kept as patches under `docs/patches/` because
   `arena.py` and `controllers.py` were open in the parallel session when
   this was measured; they landed later the same day (below) and the
   patches are gone. Back-kicks rise with the ball-out rule (0.8 ->
   2.4 a run) at the same PROPORTION of kicks (29% -> 31%): more play,
   not worse play.

   **Landed (2026-09-07, later the same day, "complete all the
   suggestions").** `World(ball_out_s=)` / `soccer_score["ballOuts"]`,
   `eval-pitch --ball-out-s` (0 by default: the benchmark's baseline is
   bit for bit what it was), and `world_server.PITCH_BALL_OUT_S = 5.0` for
   every pitch the lab builds — so `/sim` and `record-world` play under
   the rule and `eval-pitch` does not unless asked. `ChaseParams.
   board_margin` landed too, and was re-measured ON the ball-out floor,
   which is what the bead asked (12 seeds of 3v3, paired against the rule
   alone):

   | ball-out + … | kicks a run | dead-ball s | possession | progress | goals | falls |
   |---|---|---|---|---|---|---|
   | rule alone | 7.7 | 175 | 42.8 | 0.31 | 0.42 | 0.25 |
   | + boards line, margin 0.12 | 8.7 (p=0.38) | 179 | 41.6 | 0.28 | 0.42 | 0.08 |
   | + boards line, margin 0.08 | 7.9 (p=0.79) | 183 | 42.3 | 0.43 (p=0.21) | 0.83 (p=0.27) | 0.33 |

   A null both ways (and 0.12 could never fire for a ball actually
   against the wall: ball radius 0.035 + `kick_side` 0.06 puts the trunk
   0.095 m from it, so the 0.12 arm only ever re-laid spots for balls 8 cm
   or more off the boards). The referee's placement already takes the
   ball off the wall; a kick along it adds nothing on top. **Ships off**
   (`board_margin` 0), kept for a pitch without the rule, where it is the
   only thing that ever kicks a ball at the boards (+1.3 a run, p=0.056).
   `tests/test_ball_out.py` locks the rule's default, its placement, the
   lab's on-switch, and the along-the-boards spot at a margin.

   **Reviewed, and one of these numbers was wrong (2026-09-08).** A code
   review of the three commits above turned up eight findings; all are
   fixed, each with a test in `tests/test_ball_out.py` that fails on the
   code as it was. The one that mattered to the result: **the placement's
   own 0.45 m was being credited to a team as ball progress.**
   `PitchMetrics.tick` already excluded the goal recentre for exactly that
   reason ("that jump is not anybody's progress") and had no equivalent
   guard for a ball-out, so a ball parked on a team's own end board with one
   of its ducks inside `POSSESSION_R` booked **+0.400 m of `progress` and
   `advance`** on the tick the referee moved it — measured directly, not
   argued. `advance` is the metric this benchmark's own docstring says to
   judge a variant on, so the arms were re-run on the fixed metric (12 seeds
   of 3v3, the same seeds, one package copy and a flag rather than two
   forks):

   | ball-out arm | as first reported | on the fixed metric |
   |---|---|---|
   | ballAdvance | +0.627 | **+0.572** (p<0.001) |
   | ballProgress | +0.193 | **+0.196** (p=0.002) |
   | dead-ball s | −72.2 | −71.4 |
   | kicks a run | +4.75 | +4.83 |
   | kicksBack | +1.58 | +1.58 |

   So about **9% of the advance was the referee** and nothing else moved:
   most placements are at SIDE boards, where only y changes and `progress`
   reads x. The rule's verdict stands, on a number that is now the ducks'.
   The table above carries the corrected figures. The other finding worth
   naming here: the along-the-boards line aimed **across our own goal
   mouth** at our own end board (a ball at (−1.45, +0.40) aimed at −90° is
   in our net after 0.10 m), because the end-board branch read only the sign
   of `by` and not which end it was; it now clears away from the mouth at
   our end and across it at theirs. That knob ships off, but the +1.3
   kicks/run measured for it at 0.12 was taken with the own-goal line live.

   **And the review found something bigger than its own eight findings.**
   Verifying the fixes against the COMMITTED tree rather than the working
   one — the discipline the fifth finding was about — turned up that
   `development` **has not imported for at least twelve commits**. Four
   files do not parse in the committed history and never did:

   | file at HEAD | line | error |
   |---|---|---|
   | `world/arena.py` | 593 | a stray `self.kickoff_team = …` inside `kickoff()` |
   | `brain/team.py` | 560 | an indented block whose `if mates > 1:` header is gone |
   | `brain/controllers.py` | 2370 | `_on_the_line`'s `return` tail, its `def` gone |
   | `walk_env.py` | 385 | the BAM block, its `if` header gone |

   Every one of them is FINE in the working tree, which is why nothing
   caught it: the tests import the tree, the batteries run the tree, and
   `scripts/precommit.sh` compiles the tree. `git show HEAD:` gives an
   `IndentationError`, and a fresh clone of the branch cannot import the
   package. The mechanism is this repo's own shared-checkout rule turned
   against it — an anchored edit applied separately to the file and to its
   committed copy, so that only that hunk is staged, lands somewhere else
   when the two texts have drifted and strands a fragment. It stages
   cleanly, it diffs plausibly, and it never runs.

   The gate for it is **`scripts/check_staged_python.py`**, now the third
   step of `precommit.sh`: every `.py` in the INDEX must parse — what the
   commit will write, not HEAD and not the tree, which is also what lets
   the commit that FIXES a broken file through. `tests/test_staged_python.py`
   stages a file that is broken as committed and fine on disk, which is the
   exact shape of the twelve commits, and checks the gate catches it. The
   four files themselves are not repaired here: three of them need code
   that exists only in the parallel session's working tree, so the repair
   is that session's commit to make, and its staged index already holds
   the correct content for all four.

   **Three more, from the parallel session's read of the same code
   (2026-09-08).** All in the ball-out rule's own files, all fixed with a
   test in `tests/test_ball_out.py`:

   - **A battery killed mid-write could not resume.** `load_done` parsed
     every line, so the half-row this machine leaves when it reclaims a
     container took the whole file down. A truncated LAST line is now
     dropped with a note (that seed is simply re-run); a bad line anywhere
     else is fatal, because silently skipping rows would quietly shrink a
     battery.
   - **A duck lying where it fell held the ball.** Under `getup_s` a fallen
     duck lies on a zero command for as long as a get-up would cost, and it
     was still the nearest duck to the ball — so it took the possession
     clock and became the `_holder` credited with the ball's motion. A duck
     flat on the floor was being paid for whatever the other side did to
     the ball. `PitchMetrics.nearest` now skips a duck that is down.
   - **The placement could drop the ball inside a duck.** The duck lining
     up on the ball stands ~0.12 m from it, which is where the rule moves
     the ball; a ball placed inside a body interpenetrates and the solver
     flings both apart — the failure `_clear_of_persons` already exists for
     on the respawn path (the physics audit's "fling"). The placement now
     steps out to `ball_out_clear` (0.25 m) from any duck, still inside the
     boards.

   **(c) In the open, line-ups die to `avoid` in 0.4 s** — 263 of 508
   3v3 exits, with the ball 0.43 m away and another duck 0.33 m ahead,
   70% of them an OPPONENT (87% in 2v2). Two attackers meet at the ball,
   each turns away from the other, each re-lines-up: the duel the survey
   (C.4, bead mdl-23b) said needs a colour sense that survives contact
   range. Not attempted here; after (b) it is the largest remaining
   share of the dead clock (175 s of 300 with the ball-out rule).

   The first form of a lever was measured on the ball-out floor
   (`ChaseParams.lineup_keepout`, ships 0): on a line-up with the ball
   nearer than the other duck the keep-out shrinks to 0.25 m so the
   attacker finishes unless the other is about to touch. 12 seeds of 3v3
   against the rule + boards line: kicks 8.7 → 8.9 (p=0.85), dead ball
   −5 s (p=0.66), progress +0.08 (p=0.51), falls 1 → 2 — a null. The two
   attackers meet at the ball whatever the radius; what the duel needs is
   to know WHICH duck is at the ball (the colour sense, bead mdl-23b),
   not a smaller circle around it.

   **What this says about the track.** The ledger metrics all read
   "flat" for three sessions of brain knobs because the game they measure
   is 85% a stationary ball. The two cheap instruments that see it —
   dead-ball seconds and kicks a run — resolve at 12 seeds what goals
   need 136 for, and should be the first row of every soccer battery
   from here.

13. **The re-baseline: every positional knob, re-measured on a pitch where
   the ball moves (2026-09-08).** Item 11 changed the benchmark under the
   whole track: the ball-out rule took dead time 247 → 176 s and kicks 2.9
   → 7.8 a run, so every "null" and "worse" above was measured on a game
   that was 85% a stationary ball. This re-runs the positional knobs on
   the new floor. Method throughout: **3v3 with formation roles**, 300 s,
   `--ball-out-s 5` in every arm, all arms forked from ONE package copy of
   the tree at `b2ecd72` + the working files of that hour (so before item
   12a/12c's `kick_ahead_max` and `gaze_still` — the kick has since got
   *better*, which only sharpens the push verdict below), differing by a
   single `MICRODUCK_CHASE` name read back off the CONSTRUCTED brain;
   paired per seed, discovery block 0–23 then a fresh block 100–123 for
   anything that looked real. Judged on dead-ball seconds, kicks a run,
   possession and `ballAdvance`; goals and own goals reported and never
   judged; `kicksBack` as a proportion of kick events.
   (`scripts/probe_handover.py` shape, extended with a knob echo.)

   **(a) Push-first is dead — and it was the track's strongest shelved
   candidate.** `kick_select_push=1,kick_select_p_whiff=0.5` was recorded
   in §6 A.3 as three blocks in agreement (+1.97 s/min possession, +0.068
   progress) and held back only for the floor. Both blockers are gone, and
   on a pitch where the ball actually travels it reverses, hard, on both
   blocks:

   | push-first vs shipped, paired | discovery 0–23 | fresh 100–123 | pooled 48 |
   |---|---|---|---|
   | dead-ball s a run | +34.0 (p<0.001, 22/24) | +37.0 (p<0.001, 23/24) | **+35.5 (p<0.001, 45/48)** |
   | kicks a run | 9.04 → 0.42 | 8.17 → 0.21 | **8.60 → 0.31** |
   | ballAdvance | −0.284 (p=0.007) | −0.388 (p<0.001) | **−0.336 (p<0.001)** |
   | ballProgress | −0.124 (p=0.26) | −0.212 (p=0.025) | −0.168 (p=0.019) |
   | ball carried a kick | 1.01 → 0.04 m | 0.96 → 0.00 m | 0.99 → 0.02 m |
   | possession s/min | +4.37 (p<0.001) | +4.43 (p<0.001) | +4.40 (p<0.001) |
   | crowd | +0.070 (p=0.001) | +0.065 (p=0.011) | +0.068 (p<0.001) |
   | ballOuts a run | 4.92 → 2.67 | — | 4.71 → 2.77 (p<0.001) |
   | goals a run (reported) | 0.63 → 0.46 | — | 0.73 → 0.40 |

   The possession gain is real and worthless, and it is playbook rule 5
   ("ask what would inflate your metric") in one line: the pusher stands
   *on* the ball by construction, so walking it 0.64 m instead of kicking
   it 3 m books possession while the ball goes nowhere. On the dead pitch
   possession was the only instrument sensitive enough to move, so the
   knob measured as a win three times. With dead-ball seconds and
   `ballAdvance` in the row it is unmistakable: **the ball stops moving.**
   Looked at, not argued (`record-world pitch-3v3 --seed 3 --seconds 90`,
   both arms): shipped, the ball crosses the pitch — (0,0) → (−1.44,−0.13)
   → (−1.78,−0.06) → (−1.52,−1.15) → (+0.23,+0.61), six kicks in
   `events.txt`; push-first, it crawls (0,0) → (+0.12,+1.15) over 90 s and
   sits at one spot for 16 s at a time inside a six-duck scrum, **zero
   kicks**. **Ships off, and the standing instruction in
   `ChaseParams.kick_select_push`'s comment — "when that floor is
   committed, one fresh block on it; if it agrees, flip both on" — is
   hereby retired: the fresh block was run and it disagreed.** With it
   dies `defender_clears`, which is bit-for-bit inert without the push
   (`controllers.py` offers the push, and only the push, behind it); it is
   not re-measured because there is nothing left for it to gate.

   **(b) The supporter field survives, on `ballAdvance` and on the ball it
   carries.** `support_field` (with `field_mid_ahead=-0.5`) is already ON
   for a roster with a midfielder, so the arm is the knob turned OFF:

   | field OFF vs the shipped ON, paired | discovery | fresh | pooled 48 |
   |---|---|---|---|
   | ballAdvance | −0.239 (p=0.017) | −0.159 (p=0.078) | **−0.199 (p=0.003)** |
   | ball carried a kick | −0.398 (p=0.072) | −0.449 (p=0.009) | **−0.424 (p=0.002)** |
   | ballProgress | −0.188 (p=0.100) | −0.136 (p=0.223) | −0.162 (p=0.041) |
   | dead-ball s a run | +11.6 (p=0.035) | +1.8 (p=0.74) | +6.7 (p=0.087) |
   | kicks a run | −1.21 (p=0.032) | +0.04 (p=0.96) | −0.58 (p=0.26) |
   | possession s/min | +0.01 (p=0.99) | +1.40 (p=0.15) | +0.70 (p=0.31) |
   | crowd | +0.016 (p=0.49) | +0.042 (p=0.047) | +0.029 (p=0.064) |
   | falls a run | −0.042 | −0.375 (p=0.036) | **−0.208 (p=0.043)** |

   So the discovery block's dead-ball and kick-count gains **did not
   replicate** and are withdrawn; what did replicate is the ball
   measurement — with the field on, the ball advances 0.199 more and each
   kick carries it 0.42 m further, both resolving pooled, in the same
   direction on both blocks. The shape reading of 2026-09-07 also holds
   (crowd better with the field on). The cost is falls: 0.17 → 0.38 a run
   with the field on, p=0.043 over 48 runs — the supporter stands nearer
   the play. **Ships on, where it already ships** (a roster with a
   midfielder); no default changed.

   `field_mid_ahead` −0.5 vs 0 is a **null on the new floor**, 24 seeds:
   advance −0.134 (p=0.14), dead ball +4.7 (p=0.51), kicks −0.13
   (p=0.88), possession +0.14 (p=0.90), shape flat. The 2026-09-07
   possession win for holding the midfielder behind the ball (+2.1 s/min
   pooled) does not reproduce once the ball moves; the shipped −0.5 is
   kept because nothing argues against it, not because it earns its keep.

   **(c) The fused team ball is still off, and now for a duller reason.**
   `fuse_ball=1`, 24 seeds: dead ball +7.6 (p=0.20), kicks −0.83 (p=0.33),
   possession +1.64 (p=0.14), advance −0.138 (p=0.19), progress −0.105
   (p=0.34), shape flat, `kicksBack` 23% → 22% of kicks (p=0.90). Not the
   2026-09-07 catastrophe (own goals 0 → 6) and not a win either: a plain
   null on every instrument that can resolve at this size. Own goals went
   4 → 0 (p=0.032) which at 24 seeds on a metric needing 347 is a coin.
   **Stays off** — but the honest new statement is "no measured effect",
   not "worse in play".

   **(d) Opponents in the roll-out: still off.** `kick_select_opps=1`, 24
   seeds: dead ball +9.6 (p=0.096), kicks −0.92 (p=0.079), advance −0.065
   (p=0.52), possession −0.70 (p=0.48); spread +0.112 (p=0.017) and search
   +2.6 s a duck (p=0.021). Same shape as the 2026-09-06 verdict — it
   takes fewer kicks and leans worse on the clock — so the floor did not
   rescue it. **Ships off.**

   **(e) A keeper on 3v3 buys shape and costs nothing that resolves.**
   `keeper,midfielder,striker` against the shipped `defender,midfielder,
   striker`, same seeds 0–23 (the `ChaseParams` are identical; the roster
   is the arm): **depth 0.804 → 0.413 m (p<0.001, on 24 of 24)**, spread
   1.293 → 1.489 (p<0.001), crowd 0.337 → 0.276 (p=0.003) — the back line
   stays home, exactly what a keeper is for. The ledger: kicks +0.83
   (p=0.34), dead ball +7.4 (p=0.22), advance −0.090 (p=0.35), possession
   −1.71 (p=0.058), falls 0.21 → 0.42 (p=0.12), `kicksBack` 23% → 27% of
   kicks (p=0.28). **Needs more seeds** before it is a recommendation: the
   shape is unarguable and free, the possession lean is the one number
   that would make it a cost and it does not resolve. Not made a default.

   **What the re-baseline says.** One shelved "best result in the track"
   was an artefact of the dead pitch and is now measured off with 45 of 48
   seeds against it; one shipped knob survived on a different metric than
   the one it was sold on, with its discovery-block gains withdrawn; three
   more stayed off. The two instruments that did all the work are the ones
   item 11 named — dead-ball seconds and `ballAdvance` — and the metric
   that produced every wrong verdict on the old pitch is **possession**,
   which pays a duck for standing next to a ball it is not moving.

And one thing this track did NOT settle, which every item above kept
running into: **the score.** Goals need 136 seeds to move 25% and own goals
347 (1.5). Positional play buys shape, safety and a ball that goes less
wrong; whether it wins games is a question this benchmark cannot answer at
any sane cost, and saying so is the honest end of the track.

### 6. What the field does that this stack does not — a survey (2026-09-06)

Jonathan asked what a robot-soccer stack has that ours is missing, after the
basics (a brain, positions, tracking). This is a read of the RoboCup
literature — the Standard Platform League (NAO) and Humanoid League code
releases and symposium papers, plus the recent learned-soccer work on small
humanoids — mapped onto what is actually in this repo. Every item says what
we have, what the field does, what it would take here, and what number
would settle it. They are ordered by how directly each one addresses a
problem this track has already measured, not by how impressive it sounds.

One framing fact first. The RoboCup Humanoid League's smallest class,
KidSize, requires a robot **40–100 cm** tall; the Microduck is ~25 cm. This
is not a competition robot and never will be, so nothing below is about
rules or eligibility. It is about which ideas transfer. Most of them do,
because the NAO (58 cm, two cameras in the head, 25 DOF) has the same
problems this duck has — it just solved some of them a decade ago.

#### A. The kick — what the field does about the exact limit 4b/4c/item 7 hit

Item 7 closed with: the kick is a **sensing** limit. The camera sits 23 cm
up pointing forward, so the ball on the kick spot is inside a 23 cm blind
radius, and every rule for placing, aiming and choosing the swing failed for
that one reason. The field has met this limit and has four answers, and
this stack has none of them.

- [x] **A.1 A second, downward camera — built as an ABLATION, and it did its
      job: it proved the real camera is enough.** The NAO carries two
      identical cameras in the forehead: the top one pitched 1.2° down, the
      bottom one **39.7°** down, for one reason — to see the ground at the
      feet. The Microduck has one camera and is not getting a second, so
      `DetectorSpec.bottom_pitch_deg` (`MICRODUCK_CAMERA=bottom_pitch_deg=…`)
      exists only to ask the simulator whether the blind radius is what the
      kick was waiting on. Off by default, never a baseline, proven
      bit-for-bit inert at 0. The answer: **swings the brain can see 34% →
      97%**, estimate r = 0.48 → 0.96, median error 2.3 → 1.2 cm. So
      sensing was the limit — *at the shipped head pose.* The geometry it
      forced out is the useful part: the duck's lens is 25 cm up, half the
      NAO's, so the NAO's 39.7° reaches a floor ball at 20 cm but the 10 cm
      kick spot needs about **60°**, and nothing on the duck occludes it.
      Which means the head the robot HAS, pitched with the neck (~52°),
      already sees the spot — measured at 65% coverage on the real camera,
      and then the kick skill fails from that pose. The whole chain is in
      item 7's correction. Nothing more to build here; the ablation stays
      as the tool that settles "is it the sensor?" in one run.
- [ ] **A.2 In-walk kicks — kick inside the gait instead of stop, settle,
      swing.** B-Human's `WalkKickEngine` defines every kick as a set of
      relative ball positions converted into **walk step sizes**: a pre-step
      that does not touch the ball, then a kick step, interpolated inside one
      gait cycle, with `maxXDeviation`/`maxYDeviation` bounds that refuse a
      kick the ball has drifted out of and a `maxClipBeforeAbort` that aborts
      one the step cannot reach. NimbRo's 2023 AdultSize winner does the same
      with parametric waveform kicks blended into the walk. The advantage is
      exactly our failure mode: our duck plans a spot, walks to it, **stands
      for `settle_s` and swings at a plan that is 3.0 s old and 0.21 m stale**
      (4b). An in-walk kick has no settle and no separate kick policy; the
      decision is made on the last step, with the freshest sighting there is.
      Ours cannot do this today: the two shipped kicks are separate ONNX
      skills that run from standing. **What it would take:** a walking policy
      with a kick command channel — the 61-obs contract has zero-padded
      command slots for exactly this (AGENTS.md) — trained on the local
      harness with the ball in the curriculum, then ported to `microduck_rl`.
      A real training track, not a knob. → **what settles it:** plan age at
      the swing (3.0 s) and ball-drift-since-plan (0.21 m) from
      `probe_kick_line.py`, which an in-walk kick should cut to under a step.
- [x] **A.3 Choose the kick by simulating its outcomes — BUILT, CONFIRMED
      ON FRESH SEEDS, SHIPS ON (2026-09-07).** `brain/kickselect.py`, after
      Mellmann, Schlotter & Blum (RoboCup 2016): `_plan` lays a fan of kick
      lines inside the clamp's own `aim_max` window — so no line-up gets
      longer — offers BOTH feet on every line, rolls each candidate out 30
      times under the measured kick model (1.4 m/s ± 0.3, `ball_decel`, the
      per-foot exit angles ± 35° of scatter), labels each sample by where
      it stops (their mouth, our mouth, the field of play — the pitch is
      walled, there is no OUT), refuses any line with more than 10% of its
      samples in our own net, and takes the most likely to score, ties
      broken by a potential field over the pitch. Locked by
      `tests/test_kickselect.py`; inert when off (24 000 ticks, 0 differ).

      **What building it found.** The planner's foot rule takes the foot on
      the ball's side of the line, and that foot's exit angle bends the
      kick back toward the ball's own heading — a 60° clamp turn leaves an
      outcome ~30° off the line of sight. Facing our own mouth from 0.4 m,
      every line with the planner's foot put 50–83% of kicks in our own
      net; the OTHER foot on the edge line put in 7%. That is why
      `kicksBack` stalled at 34% after the aim clamp (3.1): the clamp
      turned the line and the foot turned it back. The foot is the lever,
      and the selector is what chooses it. Mellmann's zero own-goal
      tolerance also had to go — his kicks repeated to a few degrees, ours
      scatter 35°, and at zero tolerance the selector refused every line
      near our mouth and only ever fell back to the clamp; 0.10 admits the
      lines the clamp itself would take.

      **Measured**, `probe_kick_line.py` (rows now carry the ball, the goal
      and the verdict), 24 discovery + 24 fresh seeds, each block's arms
      forked on one tree state, every kick that moved the ball scored by
      the line it actually travelled:

      | | discovery 0–23 | fresh 100–123 | pooled 48 |
      |---|---|---|---|
      | kicks aimed AWAY from their goal | 25% → 5% (p=0.048) | 39% → 20% | **31% → 13%, p=0.029** |
      | kicks on a line through THEIR mouth | 9% → 18% | 4% → 24% | **7% → 21%, p=0.040** |
      | kicks on a line through OUR mouth | 6% → 0% | 4% → 4% | 5% → 2% (too rare) |
      | whiffs | 52% → 66% | 50% → 55% | 51% → 61% (p=0.14) |
      | back-kicks a seed, paired | | | −0.23, p=0.016 |
      | effective kicks a seed, paired | | | −0.17, p=0.41 |

      Both numbers it was built to move replicate in direction on the fresh
      block and resolve pooled; the whiff cost shrinks from 14 points to 5
      on fresh seeds and does not resolve. The second brain-tier change to
      ship on a fresh-seed confirmation, after the aim clamp. The absolute
      levels are on the parallel session's uncommitted floor; both arms of
      every block share it. Goals remain unjudgeable at this seed count
      (4.1.5), as always.
- [x] **A.4 Dribbling — MEASURED (2026-09-07): walking the ball beats
      kicking it on every ball measure, and it has one cost that the
      selector should own.** The brain's `push` mode (`push_beyond` = 0 makes
      every approach a walk through the ball, `push_behind` behind it) had
      never been measured against the kick. `probe_search.py`, 24 discovery
      + 24 fresh seeds of 2v2, each block's arms forked on one tree state,
      the shipped brain (kicks, with the selector on) against push-only:

      | paired per seed | discovery | fresh | pooled 48 |
      |---|---|---|---|
      | possession, s/min | +4.18 (p=0.001) | +2.12 (p=0.006) | **+3.15, p<0.001**, better on 38/48 |
      | ball advance | +0.063 | +0.097 (p=0.010) | **+0.080, p=0.003** |
      | signed ball progress | +0.079 (p=0.068) | +0.068 (p=0.086) | **+0.073, p=0.011** |
      | falls | −0.04 | 0.00 | −0.02 (p=0.81) |
      | goals for, a run | +0.13 | +0.42 (p=0.036) | +0.27 (p=0.079) |
      | **own goals, a run** | 0.00 | **+0.42 (p<0.001; 0 → 10)** | **+0.21, p=0.008** |
      | crowd | +0.07 (p=0.005) | +0.05 (p=0.058) | **+0.06, p=0.001** |

      Why it wins: a push has no settle, no 3 s stale plan, no head-down
      pose and no exit angle — the whole chain items 4b/4c/7 spent the
      session on — and on this floor a walked ball rolls 0.72 m, enough to
      matter. Why it cannot ship as it is: **a push has no aim.** The duck
      walks through the ball wherever it stands, and near its own mouth
      that is into its own net — ten own goals on the fresh block against
      none, an effect the discovery block happened not to show (2 v 2),
      which is exactly what the power table warned own goals would do. The
      kick's aim — the clamp, now the selector — is worth its whiffs only
      near our own goal. Ducks also bunch around a pushed ball (crowd +0.06).

      **Built the same night: the push as a selector action.** Benched
      first (10 walks at 0.45 m/s per side offset, deterministic per
      offset): a walked ball rolls 0.56–0.71 m and leaves at +17° dead
      ahead, ±12° at 4 cm off, ±45° at 8 cm off — a 30° spread across
      offsets, a fifth of a kick's reach, every walk touching. That is
      `kickselect.push_model`; the kick model also gained the kick's own
      whiff rate (`kick_select_p_whiff`, 50–61% on this floor), since a
      roll-out that assumes every swing connects rates the kick against a
      push that always does. Three findings, each measured on 24 seeds
      forked with shipped on one tree state:

      1. *Mellmann's rule chooses the push almost never.* "Most likely to
         score first" — and on a 3 m pitch a kick has SOME scoring chance
         nearly everywhere — so the arm came back as the shipped brain
         (kicks 2.79 → 2.12 a run, possession 14.2 → 14.0, progress 0.046
         → 0.024). A one-shot roll-out cannot see what the push is worth,
         which is tempo: a reliable 0.64 m every approach, no settle, no
         whiff.
      2. *The whiff term alone helps nothing* (possession 14.2 → 12.9,
         progress +0.016, goals 11 → 7): the ranking among kicks barely
         moves.
      3. *Push first unless a kick can shoot* (`kick_select_shoot` = 0.3:
         prefer a safe push unless some kick scores in ≥30% of its samples;
         a push that itself reaches the mouth scores too and wins outright)
         keeps most of push-only's gain and the selector's aim:

      | push-first vs shipped, paired | discovery | fresh | pooled 48 |
      |---|---|---|---|
      | signed ball progress | +0.077 (p=0.072) | +0.060 (p=0.095) | **+0.068, p=0.013**, better on 30/48 |
      | possession, s/min | +1.89 (p=0.15) | +2.05 (p=0.062) | **+1.97, p=0.020**, better on 31/48 |
      | ball advance | +0.030 | +0.076 (p=0.077) | +0.053 (p=0.091) |
      | falls / crowd / spread | flat | flat | flat (p=0.78 / 0.45 / 0.12) |
      | goals for, a run | 0.00 | +0.25 | +0.13 (p=0.46) |
      | own goals, a run | +0.04 (2 → 3) | +0.17 (0 → 4) | +0.10 (p=0.13; 2 → 7, was 2 → 12 push-only) |
      | kicks a run | 2.79 → 0.50 | 2.46 → 0.54 | it shoots only near their mouth |

      Same direction on both blocks for the two ball measures, both
      resolving pooled, no significant cost, and the own-goal cost halved
      by the filter but not gone — five extra events over 48 seeds on a
      metric that needs 347 (4.1.5).

      **Replicated a third time, in 3v3 with roles** (defender, midfielder,
      striker; `probe_search.py --roles`, 24 seeds, one tree state):
      possession **+3.68 s/min (p<0.001, 18/24)**, signed progress **+0.074
      (p=0.002, 19/24)**, advance +0.051 (p=0.051), falls flat, own goals
      0 → 1. And the cost that only a formation can show: **crowd 0.19 →
      0.30, spread 1.51 → 1.23 m, depth 0.57 → 0.77 m, all p<0.001.** Part
      of the crowd is the metric counting the pusher, who is within 0.5 m
      of the ball by definition; the depth and spread are real — a
      defender that gets the ball under push-first walks it up the pitch
      and leaves its post, because nothing tells a pusher to clear. That
      is D.2's problem, named below.

      **Not shipped tonight, for a physical reason.** Every number above is
      the push's roll on the parallel session's UNCOMMITTED floor; on the
      old floor a walked ball rolled to the boards exactly as a kick did,
      and the whole trade-off is that difference. `kick_select_push` and
      `kick_select_p_whiff` ship off with these numbers beside them. When
      that floor is committed: one fresh block on it, and if it agrees,
      flip both on — that is the strongest single candidate the new floor
      has produced, three blocks in agreement, and it answers the original
      ask ("kick it up the pitch, or carry it") with a measurement rather
      than a rule.
#### B. Things that are simply not modelled

- [ ] **B.1 A get-up.** Every RoboCup humanoid must recover from a fall
      unaided; the KidSize rules require it. DeepMind's OP3 soccer agent
      (Science Robotics 2024) trained a get-up as one of its two stage-1
      skills and distilled it in with KL regularisation gated on "is the
      agent upright"; the classical teams use keyframe sequences (B-Human
      "Fall Motions", 95% success rates reported). **Ours respawns.**
      `arena.py` teleports a fallen duck and increments `falls`; the shipped
      policies have no floor-to-stand (`alpha_sitstand` is sit↔stand). So
      every falls number in this track is a count of events that, on a
      robot, each cost ~10–20 s of a duck lying down. The `behaviors/` track
      has the pieces (poses, a physics ladder, `train-behavior`). → **what
      settles it:** a `getup` behaviour that stands from the two common fall
      poses (`open-loop-holds-topple` says which: the level squat, not the
      spawn fold), then an eval-pitch flag that replaces respawn with the
      get-up so falls cost time instead of nothing. Then `possession` moves
      for the right reason.

      **The second half was built first (2026-09-07, later): falls cost
      time.** `World.getup_s` (0 = the respawn as it always was) makes a
      fallen duck lie where it fell on a zero command, the fall counted
      once, and respawn only after `getup_s` — the stand-in for the get-up
      policy until `Mjlab-StandUp-Flat-MicroDuck` is trained (a GPU run,
      bead mdl-0ad; the smoke submission on 2026-09-07 was refused for
      want of HF Jobs credits). `eval-pitch --getup-s` and
      `probe_search.py --getup-s` carry it; locked in
      `tests/test_gamestate.py`; eval-pitch's chase arm byte-for-byte
      unchanged at 0. **Measured, and inert where it was measured:** 3v3
      with roles, 24 seeds × 300 s, 0 v 15 s forked on one package copy —
      **one fall in 24 runs**, 23 seeds bit-identical, every ledger number
      the same to three decimals. The shipped roster no longer falls (41
      falls over 48 seeds of 3v3 at item 3; 1 in 24 tonight), so a fall's
      price has nothing to price. It will matter on the day a
      configuration falls again (the fused-ball arm fell 8 times in 24,
      push-first 4) and on the day the get-up costs its real 10–20 s; the
      knob is there for both.

      **The first half is now built too, and the premise above is wrong
      twice (2026-09-08).** Building the bench for it —
      `microduck_local/scripts/bench_getup.py`, which spawns the duck on
      its back / front / side, lets the physics SETTLE it (1 s, servos
      holding what they fell in: unsettled, the duck is still arriving and
      a policy can cash its unspent potential energy — `alpha_stand`
      "recovered" from that in 0.4 s, which measures the arrival) and then
      scores the recipe's own four-gate stand — turned up the thing this
      item said did not exist:

      | policy | back | front | side | head up? |
      |---|---|---|---|---|
      | limp / zero (null control) | 0% | 0% | 0% | — |
      | `alpha_walking` | 0% | 0% | 0% | — |
      | `alpha_sitstand` | 0% | 0% | 0% | — |
      | `alpha_ground_pick` | 0% | 75% | 0% | 0% |
      | **`alpha_stand`** | **100%** | **100%** | **100%** | **100%** |
      | `getup-flat-scratch` (control, no ladder) | 0% | 83% | 0% | 0% |
      | **`getup-l2`** (this repo, 3.5M steps) | **100%** | **100%** | **100%** | **100%** |
      | `getup-l5` (ladder tip, 10M steps) | 100% | 100% | 100% | 0% |

      12 seeds a pose, honest BAM, observation noise + domain
      randomization + the action delay on, deterministic exported ONNX,
      rendered and read. **`alpha_stand.onnx` already does the get-up** —
      flat on its back to a full held stand in 0.48 s median, jaw back at
      0.234 m against the STAND keyframe's 0.233 — so "the shipped
      policies have no floor-to-stand" was never true, and
      `Mjlab-StandUp-Flat-MicroDuck` (bead mdl-0ad) is not what stands
      between this project and a get-up.

      **What IS missing is a controller switch, not a policy.** A duck that
      falls on the pitch is still being driven by the walker, and
      `alpha_walking` recovers 0 of 24. Under `getup_s` it lies on a zero
      command; without it, `arena.py` teleports. Handing a fallen duck to
      `alpha_stand` for ~1.5 s and clearing `down_until` when it stands
      would replace the teleport with a real recovery **today, with nothing
      trained**, and give `getup_s` a measured price (0.2–1.3 s, not the
      assumed 10–20 s) instead of a guessed one. That is the next move on
      this item, and it belongs in `world/arena.py`.

      **Trained here anyway, and it works** (`behaviors/getup.py`,
      `scripts/train_getup_ladder.sh`, ~35 min on a busy Mac). The control
      run settles the ladder's own justification: the LAST rung's config
      trained from scratch for 2M steps learns the front push-up (83%) and
      nothing from the back or the side, and it is not a reward problem —
      scored under this very recipe, `alpha_stand` earns 13.2/step against
      that run's 3.3. The reward is right; flat-on-the-floor rollouts just
      never contain a stand. Laddering the PHYSICS (tilt 20→115°, settle
      0→1 s, XML servos → honest BAM; identical terms in every rung, locked
      by `tests/test_behaviors.py`) closes it by rung TWO: `getup-l2` at
      3.5M steps recovers 36 of 36 and holds the stand unbroken for 20 s.
      A trap worth keeping: **rungs 3–5 keep the recovery and lose the
      head** — the head-up column goes 100% → 3% → 0% → 0% while the trunk
      stays at full height and perfectly upright, because the recipe prices
      head POSE (joint angles, weight 0.8) and not head HEIGHT, so once the
      starts get hard the head is a free counterweight. `getup-l2` is the
      artifact; a future revision should put head height inside the salary
      gate. Caveat as always: this is the local CPU harness under the BAM
      actuator model, a subset of upstream's DR — a get-up that survives
      here is a prototype, not a hardware claim.
- [x] **B.2 A goalkeeper — BUILT, and measured off in 2v2 (2026-09-07).**
      A fourth static role, `keeper`: its zone is the last fifth in front of
      its own mouth (`Team.ROLE_ZONES`, the field players share the rest as
      they would without one), the board never sends it after a loose ball
      — not as cover, not as the "everybody may" fallback — its post is
      `keeper_depth` (0.25 m) out on the ball-to-goal line clamped inside the
      posts (`Chase._hold_target`), and it blocks by default
      (`keeper_intercept_eta`, the 4d machinery, which the field players
      ship without). Declared per duck in a scenario; the editor's role menu
      lists it; not in `formation_roles`. Locked by `tests/test_team.py`.
      Rendered (`render_pitch.py`, keeper+striker v defender+striker, 20 s):
      the keeper holds its post the whole run while the striker plays.

      Measured with the threat probe (`scripts/probe_threat.py --roles`,
      which now stamps static jobs per side), 24 seeds × 300 s of 2v2, both
      arms forked on one tree state, keeper+striker against the formation
      control defender+striker:

      | | defender + striker | keeper + striker | |
      |---|---|---|---|
      | conceded threats | 10/30 = 33% | 11/32 = 34% | p = 0.93 |
      | danger clock, ball within 0.9 m of a mouth | 9.09 s/min | 17.5 s/min | **p = 0.031**, worse on 17 of 24 seeds |
      | danger clock, within 0.45 m | 1.11 s/min | 5.68 s/min | p = 0.10 |
      | nearest the ball got to the mouth | 0.416 m | 0.403 m | p = 0.84 |

      It does what it was built to do, and that is the cost: **a side of two
      cannot spare a duck to stand in goal.** With the striker alone on the
      field the ball lives in the keeper's box twice as long, and a keeper's
      clearing kick on this floor travels too little to get it out (item 7's
      kick). Same shape as the interception result: works, does not pay.
      What was NOT measured, and is where a keeper would earn its place: 3v3
      with a defender in front of it. The numbers above are on the parallel
      session's uncommitted floor, forked together, so they are paired but
      provisional like everything after item 7's baseline note.
- [x] **B.3 Game state — BUILT, measured, shipped on (2026-09-07).** Every
      league runs a GameController with `initial / ready / set / playing /
      penalized`; we had `kickoff_brains` (a reset) and a 1 s hold. Now the
      World is the controller: after a goal the side that CONCEDED kicks
      off (`World.kickoff_team`, from which mouth the ball crossed and who
      defends it), `game_state` runs set → kickoff → playing (the hold;
      then until the ball has left the spot by 0.1 m or 10 s have passed),
      `soccer_score` carries both and the /sim banner shows them.
      `kickoff_brains(brains, teams, world)` hands the message to the
      boards (`Team.kickoff` / `Team.waits`); a chase brain with
      `ChaseParams.kickoff_wait` stands off the other side's restart —
      every duck of the scoring side is a supporter with its post clipped
      into its own half and out of a 0.3 m centre circle (state "wait")
      until it, or the board, sees the ball leave the spot. The first
      kickoff is contested as before; nothing penalises a duck that
      crosses early. Locked by `tests/test_gamestate.py`; the search probe
      keeps a restart ledger. Not built: a *ready* walk to kickoff
      positions (the World teleports, so `depth` and `spread` at the
      restart are the spawn layout in every arm), penalties, kick-ins (the
      pitch is walled).

      Measured, 24 seeds × 300 s, rule off and on forked on one package
      copy:

      | | 1v1 off | 1v1 on | 3v3 roles off | 3v3 roles on |
      |---|---|---|---|---|
      | restarts | 12 | 12 | 3 | 2 |
      | wait, s a duck a run | 0 | 1.7 | 0 | 0.3 |
      | possession s/min | 8.62 | 8.58 (p=0.94) | 13.83 | 13.64 (p=0.28) |
      | goals / own goals | 12 / 3 | 12 / 2 | 3 / 0 | 2 / 0 |
      | back-kicks a run | 1.17 | 0.96 (p=0.16) | 0.71 | 0.75 |
      | everything else | flat | | 22 of 24 seeds bit-identical | |

      It fires and it costs nothing: the ball leaves the spot within a few
      seconds of a restart, so the wait is short, and no ledger number
      moves. What it buys is a game whose restart belongs to the side that
      conceded, as every league's does. `kickoff_wait` ships on.

#### C. The world model — what "tracking is working" leaves out

- [x] **C.1 A ball model with uncertainty — BUILT and CALIBRATED
      (2026-09-07).** Berlin United's selector (A.3) runs on a
      *multi-hypothesis extended Kalman filter* for the ball; B-Human's
      ball model carries covariance. Our `Track` was an exponentially-
      smoothed point with a velocity and **no covariance**. Now each hit
      carries `sig_meas`, the detector's DECLARED noise at that range
      (bearing σ × range, and the width-ranged distance's own fraction —
      range is radius / tan(width/2), so a 10% width error is a 10% range
      error; `TrackerParams.for_detector` reads the duck's detector preset
      off the scenario, and the default IS the datasheet, so no existing
      caller changed), and `vel_sig`, the scatter of the velocity samples;
      `Track.sigma(t)` grows them by the age of the hit. The chase brain
      reports `predicted_sigma` beside `predicted` and sends it with its
      claim (C.3). Locked by `tests/test_ball_sigma.py`.

      **Does it predict the error?** `scripts/probe_shot_gate.py` now
      samples every 25th live estimate against the true ball (2v2, 24
      seeds × 300 s, ~15 700 samples a block). The error of a 2-D estimate
      is *radial*, so a calibrated per-axis σ holds 39% of errors inside
      1σ and 86% inside 2σ — not 68 / 95, which is what the first table
      was read against. Discovery block (seeds 0–23), first draft (sensor
      σ shrunk by the polar smoothing's √(k/(2−k)), 0.15 m/s prior):

      | hit age | n | median σ | median err | in 1σ | in 2σ | r(σ, err) |
      |---|---|---|---|---|---|---|
      | 0.0–0.1 s | 6055 | 0.041 | 0.054 | 30% | 77% | 0.49 |
      | 0.1–0.3 s | 5927 | 0.050 | 0.057 | 35% | 83% | 0.46 |
      | 0.3–0.6 s | 2101 | 0.076 | 0.057 | 61% | 92% | 0.42 |
      | 0.6–1.0 s | 1647 | 0.125 | 0.058 | 79% | 92% | 0.33 |

      Two things in it: the smoothing does NOT reduce the placement error
      (the tracker's own `smooth` note says why — the body frame lags), and
      **the error does not grow with the age of the hit at all**: a ball
      not seen for a second has mostly not moved. Fresh block (seeds
      100–123) with the raw sensor σ and a 0.06 m/s prior:

      | hit age | n | median σ | median err | in 1σ | in 2σ | r(σ, err) |
      |---|---|---|---|---|---|---|
      | 0.0–0.1 s | 5654 | 0.060 | 0.054 | 51% | 96% | 0.40 |
      | 0.1–0.3 s | 6312 | 0.062 | 0.056 | 50% | 95% | 0.40 |
      | 0.3–0.6 s | 2078 | 0.075 | 0.057 | 62% | 94% | 0.48 |
      | 0.6–1.0 s | 1650 | 0.111 | 0.057 | 78% | 92% | 0.31 |
      | all | 15694 | 0.067 | 0.056 | 55% | 95% | 0.41 |

      **It predicts the error** (r 0.40–0.48 within every age bin, median σ
      within 10% of the median error on fresh hits) and it errs on the
      conservative side — 51–62% inside 1σ against a calibrated 39% for
      hits under 0.6 s, and the oldest bin still over-dispersed because a
      track WITH a velocity grows by its samples' scatter (0.1–0.2 m/s)
      while the ball itself is standing still. A conservative σ is the
      right side for a gate to err on; the two refinements that would
      tighten it — 0.8 × the sensor σ, and no growth by `vel_sig` inside a
      second — were each one probe run, **and were run (later the same
      night)**: alone on the discovery block (0.8×: fresh hits 37 / 40%
      inside 1σ, calibrated, old hits unchanged; no `vel_sig` growth: the
      old hits 38 / 56%, r 0.62–0.73), together on the fresh block:

      | hit age | median σ | median err | in 1σ | in 2σ | r(σ, err) |
      |---|---|---|---|---|---|
      | 0.0–0.1 s | 0.047 | 0.054 | 34% | 85% | 0.40 |
      | 0.1–0.3 s | 0.043 | 0.056 | 29% | 80% | 0.45 |
      | 0.3–0.6 s | 0.042 | 0.057 | 23% | 78% | 0.70 |
      | 0.6–1.0 s | 0.059 | 0.057 | 48% | 81% | 0.75 |
      | all | 0.047 | 0.056 | 32% | 82% | **0.57** |

      From a fifth too wide (55 / 95%) to a little too tight (32 / 82%),
      with the σ tracking the error far better (r 0.41 → 0.57; 0.70–0.75
      on older hits). Both ship (`TrackerParams.meas_scale` 0.8,
      `vel_sig_after_s` 1.0). What is left is the floor: a ball unseen for
      0.3–0.6 s is usually NEAR, so its range-proportional σ shrinks to
      0.042 while its error stays 0.057 — the near-ball error is the
      frame's age and the body's motion, not the range (the tracker's own
      `_place` note measured 5.5 cm inside 0.6 m). A 4 cm `meas_floor` is
      the next probe run. At the swing itself
      (10 of 51 kicks had a fresh estimate — 20%, item 7's coverage
      limit unchanged) σ 0.048 against a side error of 0.021, r 0.89.
      Also out of this: the shot-gate probe's own estimate-error figure
      (median 5.4–5.7 cm on this floor) is what the selector's `dir_sd`
      already prices at +1.90°/cm.
- [x] **C.2 Self-localisation from the pitch — DONE (2026-09-06, late).**
      Built as the goal-post particle filter in `brain/localize.py`, with a
      `post` detection class (fixed-position landmarks; the pitch's goal
      was a scored line with no geometry) and an auto-on rule for any duck
      whose odometry is declared to drift. Numbers and mechanism in item 10
      above: `datasheet` position error 0.215 → 0.072 m, yaw 20° → 2.3°,
      miss at the goal line 0.663 → 0.075 m; `hostile` 0.706 → 0.089 m. The
      one honest gap: the landmarks are the four posts and nothing else
      (no lines, no corners), so a duck facing the side boards for a long
      stretch is on dead reckoning until a post comes back into the 62°
      lens — the cloud reports its own spread (`Localizer.spread`) for a
      brain that wants to know.
- [x] **C.3 A shared world model, not a shared point — BUILT, closer in
      the probe, worse in play, ships off (2026-09-07).** SPL teams fuse
      teammates' ball estimates weighted by their covariances into a team
      ball; B-Human 2022 ("More Team Play with Less Communication") rebuilt
      the behaviour to play pass-oriented soccer while *sending fewer
      messages*. Our blackboard sent one point estimate a second with no
      confidence. Now every claim carries its sender's sigma (C.1), the
      frame is the localised one (C.2), and with `Team.fuse` /
      `ChaseParams.fuse_ball` the board's ball is the inverse-variance
      mean of every live sighting, each weighed by that sigma grown by
      the claim's age at the calibrated 0.06 m/s; `Team.ball_sigma` is the
      fused ball's own sigma. Locked by `tests/test_team_ball.py`.

      **Against the truth** (`scripts/probe_odom_goal.py`, 2v2, 8 seeds ×
      300 s, the board's ball sampled once a second):

      | odometry | n | freshest, median / 95th | fused, median / 95th | two saw it | fused closer, when two saw it | in 1σ / 2σ |
      |---|---|---|---|---|---|---|
      | ideal | 2377 | 0.043 / 0.253 m | 0.040 / 0.236 m | 23% | **70%** | 72% / 94% |
      | datasheet (localised) | 2323 | 0.098 / 0.883 m | 0.097 / 0.685 m | 27% | **61%** | 37% / 65% |

      Closer when two ducks see the ball (which is one sample in four),
      the same point otherwise; the 95th percentile is where it shows.

      **In play** (3v3 with roles, shipped kicks, 24 seeds × 300 s,
      freshest against fused, forked together on one package copy):

      | | freshest | fused | |
      |---|---|---|---|
      | goals, both mouths | 2 | 10 | p=0.004, more on 7 seeds, fewer on 0 |
      | own goals | 0 | 6 | **p=0.006**, 6 seeds / 0 |
      | ball advance / progress | 0.172 / 0.028 | 0.234 / 0.076 | p=0.025 / 0.16 |
      | crowd / spread | 0.177 / 1.514 | 0.220 / 1.420 | p=0.053 / 0.06 |
      | falls | 3 | 8 | p=0.16 |
      | possession | 13.64 | 14.65 | p=0.37 |
      | ball in view | 25.6% | 27.4% | p=0.10, better 19/24 |

      A more accurate point makes a worse game: the ball reaches both
      mouths more — six own goals against none — and the team compresses.
      The mechanism the numbers point at: the fusion keeps every claim
      inside 3 × `stale_s` (3 s), so when the ball MOVES the board's ball
      is pulled toward where teammates last saw it (a 3 s-old claim still
      carries ~10% of the weight), and a supporter or a defender walks to
      a point the ball has left; the freshest-sighting rule has no such
      lag. The probe cannot see this because it samples a mostly still
      ball. What would fix it — fuse only claims within `stale_s` of the
      freshest, or drop the fusion once the board's velocity is kick-like
      — is one more arm; `fuse_ball` ships off, and the sigma each claim
      now carries is there for whoever runs it.

      **That arm was run (later the same night): `Team.fuse_window`, only
      claims within 0.5 s of the freshest fused** (3v3 with roles on the
      then-current tree, 24 seeds, fused v freshest forked together): own
      goals 1 → 2 (p=0.57; the 3 s window's 0 → 6 is gone, which is the
      lag mechanism confirmed), goals 6 → 8, crowd / spread / possession
      flat, progress 0.084 → 0.028 (p=0.10), ball in view +1.3% (p=0.13).
      The harm is removed and nothing is gained: closer in the probe,
      nothing in the game. `fuse_window` defaults to 0.5 s so the fusion,
      if anyone turns it on, is the safe one; `fuse_ball` stays off.
- [ ] **C.4 An opponent model and a duel — the first half BUILT and
      measured (2026-09-07).** B-Human has a `Zweikampf` (one-on-one)
      behaviour; every stack tracks opponents as first-class objects. Ours
      had duck tracks, a colour vote that failed confirmation (4.2), and
      `avoid`/`blocked`/`yield`. Now `Chase._opponents` is the opponent
      model this stack can honestly have: every duck track seen within
      `lost_s` that the board does not own — not within 0.35 m of a
      teammate's own claim of its position, not our colour when the
      colour sense is on. The field (D.2) uses it to keep a supporter's
      pass lane open, and with `kick_select_opps` the selector (A.3)
      rolls every sample through it: a path that passes within 0.15 m of
      an opponent stops at its feet, BLOCKED, valued there less 0.5
      potential units. So a line through a body scores as what it is,
      and the selector turns the kick (or the push) off it. Locked in
      `tests/test_kickselect.py`.

      Measured, 3v3 with roles, 24 seeds × 300 s, four arms forked on one
      package copy (the discovery block):

      | | shipped kicks | kicks + opponents | | push-first + field | + opponents | |
      |---|---|---|---|---|---|---|
      | ball progress | 0.028 | **0.096** | **p=0.027**, better 15/24 | 0.096 | 0.133 | p=0.26 |
      | ball advance | 0.172 | 0.226 | p=0.057 | 0.198 | **0.258** | **p=0.041**, better 15/24 |
      | possession s/min | 13.64 | 15.32 | p=0.099 | 17.46 | 18.47 | p=0.25 |
      | kick carry, m a kick | 0.178 | 0.203 | (53 → 66 kicks) | 0.32 | 0.50 | (5 → 3 kicks) |
      | back-kicks a run | 0.75 | 0.54 | p=0.41 | 0 | 0 | |
      | crowd / spread / depth | 0.177 / 1.514 / 0.574 | 0.210 / 1.482 / 0.630 | all p>0.17 | flat | flat | |
      | goals / own goals | 2 / 0 | 4 / 1 | | 4 / 0 | 4 / 0 | |
      | falls a run | 0.125 | 0.417 | p=0.096, worse 8/24 | 0.125 | 0.083 | |

      The direction is the one the model predicts — the ball travels
      further up the pitch when the lines through a body are priced —
      and it shows on progress under kicks and on advance under
      push-first; the one caution is falls under kicks (3 → 10 in 24
      runs, p=0.096), which is a duck now kicking beside another one
      rather than into it.

      **Fresh block (seeds 100–123, kicks): it did not replicate.**
      Progress −0.050 (p=0.13, worse on 17 of 24), advance −0.020, spread
      −0.140 (p=0.018), depth +0.078 (p=0.08); pooled over 48 seeds:
      progress +0.009 (p=0.70), advance +0.017 (p=0.40), possession +1.24
      (p=0.082), spread −0.086 (p=0.054), depth +0.067 (p=0.026), falls
      5 → 12 (p=0.14), and the kicks it takes — more of them, 2.0 → 2.5 a
      run — carry LESS, 0.225 → 0.159 m a kick: a line that misses the
      body is a shorter, wider line, and on a 3 m pitch the body is
      usually between the ball and the goal. The discovery block's
      p=0.027 was the kind of number a second block exists to catch.
      **`kick_select_opps` ships off.** The opponent list itself
      (`Chase._opponents`) stays — the field uses it. Not built, and
      still the second half of this item: the duel — what a duck does
      when the opponent is nearer the ball than it is (B-Human's
      Zweikampf: shield, block, or contest), which needs the colour vote
      or the board to say who is who at contact range. The interception
      work (4d) found "the lever is elsewhere"; with a keeper (B.2) and an
      opponent list this is the next place to look.

#### D. Team play — after C, not before

- [x] **D.1 Passing — BUILT and MEASURED (2026-09-07): a kick cannot pass
      on this pitch, and the formations rarely offer a target.** The
      selector (A.3) gained pass lines: for every teammate the board places
      at least `pass_min_ahead` up-pitch of the ball, a candidate line
      straight at it (both feet), and every candidate — pass, shot, push —
      is valued with a `pass_bonus` for each sample that stops within
      `pass_reach` of ANY teammate (Mellmann's own next step, a teammate
      attractor in the field; B-Human scores pass targets by goal angle,
      accessibility and blocking — accessibility is what this is). The
      teammates' positions are the board's, in the shared frame C.2 gave
      them. Locked by `tests/test_kickselect.py`; the kick probe now
      records teammates and the ball's end point at every swing so a kick
      can be scored as RECEIVED.

      Measured, 24 seeds, both arms forked on one tree state, shipped
      against `kick_select_pass=1`:

      | | shipped | passing |
      |---|---|---|
      | kicks that moved the ball | 22 | 21 |
      | received, ball stops within 0.4 m of a mate | 1 (5%) | 2 (10%), p=0.52 |
      | received within 0.6 m | 3 (14%) | 5 (24%), p=0.39 |
      | possession / progress / falls / goals | flat | flat (all p > 0.3) |
      | verdicts with any received sample | | 6 of 61, max 37% |

      Two mechanisms, both in the data. **The teammate is up-pitch of the
      ball on 26% of swings** (median offset −0.22 m): in these formations
      the striker is the duck on the ball and the defender holds behind it
      on the goal line, so there is rarely anyone ahead to pass to. And
      **a kick is not a passing instrument on a 3 m pitch**: 1.4 m/s with
      35° of scatter rolls 3.3 m to the boards, so even a line laid
      straight at a mate stops within reach of it in a handful of samples.
      A pass needs a SOFT delivery — which is the push (A.4: 0.64 m of
      roll, a 30° spread), the action the selector already has and which
      is waiting on the floor. `kick_select_pass` ships off with these
      numbers.

      **The push as the passing action, measured the same night** (push-first
      selector with and without the pass bonus, 24 seeds, one tree state):
      possession +1.8 s/min (p=0.12, better on 17 of 24), progress, advance,
      falls, goals, crowd and spread all flat. A direction, not a result,
      and the formation is why: the receiver is behind the ball three
      swings in four. → **what would make passing real:** a formation that
      puts a teammate AHEAD of the ball — the striker's post is 0.8 m ahead
      of it by design (`strike_ahead`), so a defender+striker pair can pass
      only when the defender has the ball; a mid that holds ahead of the
      ball would be the receiver (D.2). Then re-run the push+pass arm.

      **Re-run in the formation that offers a receiver** (3v3 with roles, a
      striker 0.8 m ahead of the ball whenever the mid or defender has it;
      push-first with and without the bonus, 24 seeds, one tree state):
      possession −0.07 s/min (p=0.94), progress −0.005 (p=0.88), goals
      +0.08 (p=0.48) — **nothing on the ball**; spread +0.11 m (p=0.011)
      and depth −0.11 m (p=0.004), a little shape. So even with a receiver
      available, a bonus for arriving at a teammate's feet moves no ball
      measure: on this pitch the push-first selector already carries the
      ball up-pitch reliably, and "passing" as a distinct thing has no
      value left to add to it. Closed: `kick_select_pass` stays off, and
      the push IS the pass.
- [ ] **D.2 Positioning by potential field or Voronoi — the shape cost is
      now measured, and it says what the field must do.** RoboCup
      supporters stand where a potential field over the pitch (ball,
      teammates, opponents, goals) has a minimum; MSL teams tile the field
      with a weighted Voronoi tessellation. Our roles stand at fixed posts
      (`defend_depth`, `strike_ahead`, `strike_side`, `mid_side`) that do
      not see opponents. The measured win of item 3 (crowd 13% → 1.8%,
      spread +0.95 m) came from posts. What the 3v3 push-first measurement
      (A.4) adds is the specific failure a field has to fix: **a defender
      that gets the ball under push-first dribbles it up the pitch and
      leaves its post** (depth 0.57 → 0.77 m, spread 1.51 → 1.23 m, both
      p<0.001), because a post says where to stand WITHOUT the ball and
      nothing says what a defender does WITH it. The field's first job is
      not opponents; it is a defender-with-ball rule — push to the
      nearest up-pitch teammate (the receiver D.1 found nobody offered) or
      clear, then return to depth — and only then a supporter position
      that reacts to the ball's carrier.

      **The first job was built and measured the same night, and it is not
      the fix.** `ChaseParams.defender_clears`: with the push on offer, a
      defender or keeper is not offered it, kicks clear, and returns to its
      post. 3v3 with roles, push-first with and without, 24 seeds on one
      tree state:

      | | shipped (kicks) | push-first, defender carries | push-first, defender clears | clears − carries |
      |---|---|---|---|---|
      | depth from own goal | 0.571 | 0.769 | 0.723 | −0.046 (p=0.24) |
      | spread | 1.507 | 1.233 | 1.310 | +0.077 (p=0.10) |
      | crowd | 0.186 | 0.304 | 0.288 | −0.016 (p=0.54) |
      | possession / progress | 13.8 / 0.029 | 17.5 / 0.102 | 17.2 / 0.112 | flat |
      | back-kicks a run | 0.71 | 0.00 | 0.21 | +0.21 (**p=0.045**) |
      | own goals a run | 0.00 | 0.04 | 0.17 | +0.13 (p=0.17) |

      A quarter of the depth and a third of the spread come back, neither
      resolving, and the defender's clearing kicks bring back-kicks and
      own-goal risk with them (4b's geometry: a kick from our own third
      goes the wrong way some of the time). The carrying defender was a
      minor part of the shape cost. **What the numbers say the cost is:**
      a team compresses around a WALKED ball because every post — the
      striker's 0.8 m ahead, the mid's half-way, the supporter's 0.7 m
      behind — is laid out relative to the ball, and under push-first the
      ball moves with a duck attached. The lever is a supporter position
      that anticipates the carrier (holds a lane ahead and wide of it,
      rather than a distance from the ball), which is the potential-field
      positioning this item is named for. `defender_clears` ships off.

      **The field, built and measured (2026-09-07, `brain/field.py`,
      `ChaseParams.support_field`).** A striker, a midfielder or a plain
      supporter stands at the minimum of a potential field: the role's
      `ahead` along the carrier's lane (ball → goal), OUT of that lane (a
      0.2 m Gaussian repulsor strip), 0.5 m beside it on its own side (a
      cost of moving picks the near side), away from teammates — the
      carrier excepted, since the lane and the attacker's room already
      speak for it — and from opponents on or between the ball and the
      spot (duck tracks the board does not own), pulled 0.3 × the
      selector's pitch potential toward the goal, so the wide spot at the
      goal end is the far post (0.6 m off the mouth), not the corner; zone,
      boards' margin and attacker's room hard; hysteresis. ~550 spots
      costed in numpy, 36 µs a call. Defender and keeper posts untouched.
      Locked by `tests/test_field.py`. Four arms forked together on one
      package copy, 3v3 with roles, 24 seeds × 300 s:

      | | push-first | push-first + field | | shipped kicks | kicks + field | |
      |---|---|---|---|---|---|---|
      | crowd | 0.304 | **0.251** | p=0.023, better 17/24 | 0.186 | 0.226 | p=0.039, worse 16/24 |
      | spread | 1.233 | **1.374** | p=0.003, better 17/24 | 1.507 | 1.450 | p=0.17 |
      | depth | 0.769 | 0.708 | p=0.060 | 0.571 | 0.650 | p=0.003, worse |
      | possession s/min | 17.51 | 17.45 | p=0.93 | 13.83 | 15.59 | p=0.072, better 17/24 |
      | progress / advance | 0.102 / 0.227 | 0.111 / 0.215 | flat | 0.029 / 0.176 | 0.051 / 0.186 | flat |
      | goals / own goals | 3 / 1 | 3 / 0 | | 3 / 0 | 5 / 1 | |
      | falls / back-kicks | 4 / 0 | 3 / 0 | | 2 / 17 | 2 / 13 | |

      **Under push-first the field does what it was built for: it takes
      back about half of the shape cost with nothing lost on the ball** —
      crowd 0.30 → 0.25 (the kick baseline is 0.19), spread 1.23 → 1.37 (of
      1.51), depth 0.77 → 0.71 (of 0.57), possession and progress
      unchanged. **Under the shipped kicks it is the wrong shape**: crowd
      +0.04 and depth +0.08 (both p<0.04) against a possession hint of
      +1.8 s/min (p=0.07). The mechanism is the ball's speed: a walked ball
      is a carrier with a lane to stand beside; a kicked ball flies 3 m,
      and a midfielder held level with it and 0.5 m wide is simply nearer
      a flying ball than its post between the ball and the centre spot.
      So `support_field` ships OFF for the shipped brain, and is part of
      the push-first configuration
      (`kick_select_push=1,kick_select_p_whiff=0.5,support_field=1`) for
      when the rolling-resistance floor is committed and push-first is
      flipped. Same night, the shape cost itself re-measured on this tree:
      push-first crowd +0.118, spread −0.274, depth +0.198 (all p<0.001),
      possession +3.68 s/min, progress +0.074 (p=0.002), back-kicks 17 → 0
      — the fourth replication.

      **The midfielder held BEHIND the ball (`field_mid_ahead` = −0.5),
      measured the same night** (24 seeds, four arms forked together on a
      package copy; the controls reproduce the earlier arms bit for bit):

      | | shipped kicks | kicks + field, mid behind | | push-first | push-first + field, mid behind | |
      |---|---|---|---|---|---|---|
      | crowd | 0.186 | 0.240 | p=0.024, worse | 0.304 | 0.300 | flat |
      | spread | 1.507 | 1.456 | p=0.34 | 1.233 | 1.333 | p=0.046 |
      | depth | 0.571 | 0.642 | p=0.026 | 0.769 | 0.787 | flat |
      | possession s/min | 13.83 | **16.84** | **p=0.001**, better 16/24 | 17.51 | 18.95 | p=0.10 |
      | progress / advance | 0.029 / 0.176 | 0.089 / 0.231 | p=0.073 / 0.10 | 0.102 / 0.227 | 0.170 / 0.306 | p=0.038 / 0.044 |
      | goals / own goals | 3 / 0 | 4 / 0 | | 3 / 1 | 11 / 5 | p=0.075 / 0.09 |

      So the crowd under kicks is NOT the midfielder's level position —
      it is the same +0.05 with the mid behind — and what the field does
      under kicks is put the ball at our feet more: possession +3.0 s/min
      (+22%, p=0.001) and progress +0.06, at the price of five points of
      pile-up time and a deeper back line. Under push-first the two
      variants split the prize: the level mid gives the shape back (crowd
      0.30 → 0.25, spread +0.14) and moves no ball number; the mid behind
      moves the ball (progress +0.067, advance +0.079, goals 3 → 11 with
      own goals 1 → 5) and gives no shape back. `field_mid_ahead` is a
      real dial, not a fix.

      **Fresh block (seeds 100–123) for the kick-side field with the mid
      behind, and the two blocks pooled:**

      | | fresh: kicks | kicks + field | | pooled 48: kicks | kicks + field | |
      |---|---|---|---|---|---|---|
      | possession s/min | 14.02 | 15.30 | p=0.26, better 14/24 | 13.93 | **16.07** | **p=0.004**, better 30/48 |
      | ball advance | 0.206 | 0.255 | p=0.17 | 0.191 | 0.243 | p=0.032 |
      | ball progress | 0.089 | 0.093 | flat | 0.059 | 0.091 | p=0.19 |
      | crowd | 0.211 | 0.193 | p=0.46 | 0.199 | 0.217 | p=0.31 |
      | spread | 1.511 | 1.498 | flat | 1.509 | 1.477 | p=0.36 |
      | depth | 0.568 | 0.657 | p=0.005 | 0.570 | 0.649 | p<0.001 |
      | goals, both mouths / own | 5 / 1 | 10 / 2 | | 8 / 1 | 14 / 2 | |
      | falls | 2 | 3 | | 4 | 4 | |
      | turning in place (spinFrac) | 0.568 | 0.594 | p=0.002 | | | |

      The possession gain replicates in direction and shrinks in size
      (+3.0 then +1.3, pooled +2.1 s/min, +15%, p=0.004 over 48 seeds);
      the crowd cost of the discovery block does not replicate (pooled
      flat); what is consistent is a deeper back line (+0.08 m, p<0.001)
      — which, both teams running the same brain, is the OTHER side's
      defender pulled off its post by a ball that reaches its third more
      often (progress and advance up) — and a little more turning in
      place from supporters standing wide of a lane the ball crosses. So
      for the shipped brain the field with the mid behind is a possession
      gain with no shape cost that survives a fresh block.

      **The role-less supporter** (`eval-pitch`'s own roster, 2v2, 24
      seeds, field on — it stands behind the ball, wide of the lane — v
      the shipped post): possession 13.14 → 12.86 (p=0.79), advance 0.200
      → 0.158 (p=0.19), ball in view 24.1% → 22.1% (p=0.088), goals 11 →
      8, nothing significant either way. Null, leaning the wrong way. So
      the field is for the ROLES it was measured on: `field_plain` (off)
      keeps the role-less supporter on its post, and `support_field`
      concerns the striker and the midfielder.

      **2v2 with roles** (defender + striker, the lab's `pitch-2v2`, 24
      seeds, field on v off): ball progress 0.148 → 0.061 (**p=0.002**,
      worse on 18 of 24), advance 0.261 → 0.214 (p=0.15), kicks 2.4 → 3.4
      a run (p=0.024) that go nowhere, possession flat, crowd −0.04 and
      spread +0.17 (p=0.09) — the shape better, the ball worse. With no
      midfielder the striker is the only duck in the field, and a striker
      held wide of the lane on a two-duck side's smaller pitch is a
      striker the kick does not reach.

      **Verdict (2026-09-07).** The field pays on one roster and costs on
      the other two: a side with a midfielder (3v3 with roles) gains
      possession +2.1 s/min (+15%, pooled 48 seeds, p=0.004) with the
      midfielder behind the ball and no shape cost; a defender + striker
      pair loses progress (p=0.002); the role-less roster is null. So
      `support_field` ships OFF as a knob and `brain_kwargs` turns it on
      — with `field_mid_ahead=−0.5` — for a roster WITH A MIDFIELDER,
      the one it measured a win on, unless the command line names it (the
      same by-name rule as the localiser's auto-on). `pitch-3v3` in the
      lab gets it; `pitch-2v2` and `eval-pitch` do not, bit for bit.
      Under push-first (when the floor lands) the level mid is the shape
      prize and the config names `field_mid_ahead=0` itself.
#### E. Learning — where the field found it pays, and where it did not

- [ ] **E.1 The striker, with sensing in the loop (re-points item 4).**
      Three results say the same thing. "Learning Vision-Driven Reactive
      Soccer Skills for Humanoid Robots" (2025) trains search / chase /
      multidirectional-kick **with only onboard vision**, exposing the policy
      to *perceptual noise and detection failures during training*, and
      reports ball-estimate error −46% and time-to-kick −64% against a
      rule-based baseline with ~90% kick success. Dribble Master adds an
      explicit *active-sensing reward*. And DeepMind's OP3 agent — the
      closest published robot to ours, 20 joints, 40 Hz, egocentric 2-D
      observations, zero-shot sim2real — **perceived the ball and opponent
      through a motion-capture system**, not its camera, which is the one
      part of that result that does not transfer here. Our `StrikerEnv`
      already feeds the policy the *tracker's* ball (`striker.py`: "the
      TRACK's odometry position, not the truth"), through the simulated
      detector with noise presets — so `striker-v1` faced the same blind
      radius the scripted brain does and lost for the same reason. The
      recipe the field converged on: keep the honest perception, **add a
      reward for keeping the ball in view**, and train the approach as a
      closed loop on the sighting. → item 4's own bar: possession 11.8 →
      5.9 s/min was the loss; a striker that reaches the ball as often as
      `Chase` is the gate.
- [ ] **E.2 RL for the decision layer only.** WisTex United (SPL Challenge
      Shield 2024, 7 wins of 8, 39–7 on goals) kept B-Human's perception,
      localisation and motion and replaced only the high-level behaviour
      with four RL sub-policies (mid-field walk-and-kick angle, ball duel,
      near-goal precision, defensive positioning), selected by a heuristic,
      trained across a low-fidelity full-field sim and a high-fidelity one.
      Their lesson — decomposition plus heuristic selection beat one
      monolithic policy, and end-to-end was "prohibitively expensive" — is
      the split this repo already has (scripted `Chase` + learned skills),
      so the natural first RL decision here is the kick choice in A.3 or the
      supporter position in D.2, not the whole brain.
- [ ] **E.3 Learning from recordings.** SoccerDiffusion (2025) learns joint
      trajectories from RoboCup gameplay logs (vision + proprioception +
      game state) and runs on hardware after distillation, with "high-level
      tactical behaviour" still limited. We record every run (the replay
      ring, `runs/*.jsonl`). Far off; noted so it is not re-discovered.

#### F. Architecture, for when the above lands

- [ ] **F.1 A behaviour hierarchy.** B-Human writes behaviour in CABSL
      (hierarchical state machines) organised as skills and cards; NimbRo
      runs a two-layer FSM (game FSM over behaviour FSM). `Chase` is one
      flat state machine in a 3 000-line file with roles bolted on as a
      post and a zone. It has held up through this track because every
      change was measured. **Revisited 2026-09-07, after B.2, B.3, C.1–C.4
      and D.1–D.2 landed in it:** they fit — each as a knob and a hook
      (the keeper is a post plus a block; the game state is a role
      override and one more support state, "wait"; the field, the
      selector's opponents, the fused ball and the ball's sigma live in
      their own modules, `field.py`, `kickselect.py`, `team.py`,
      `tracker.py`, and the brain only calls them). What did NOT fit, and
      is the honest reason this item stays open, is legibility: `_plan`
      and `step` each carry a dozen gated branches, and the order of the
      `elif` chain in `step` (kick → look → retreat → avoid → block →
      support → yield → push → …) is a priority scheme nobody wrote down.
      The refactor that would pay is exactly that chain as a named
      priority list, not a new hierarchy — **done (2026-09-07, later):
      `Chase.PRIORITY`** names the chain in the order it runs (kick, look,
      retreat, avoid, block, support, yield, push, lineup/settle, seen,
      hunt, seek, search), each with a line on what owns the tick, and
      `tests/test_team.py` reads the `elif` heads back out of the source
      and checks the order — a branch moved by accident is a failing test.
      Still not a hierarchy; the moment one is needed is when a keeper
      needs a different top-level loop from a striker — and the measured
      keeper does not.

**What to read this list as (written 2026-09-06; revised 2026-09-07
after the survey was worked through).** A.1 was the only item that could
remove the limit item 7 hit, and it was measured: the one real camera CAN
see the kick spot, the gaze puts the ball on it, and the shipped kick
skill whiffs with the head down — so the limit is the kick skill's, and
the fix is upstream (`docs/patches/microduck_rl-kick-head-down.patch`).
A.3 (outcome-simulated kick selection) shipped on; A.4 (the push) is the
best ball-side result in the track and waits on the rolling-resistance
floor being committed; B.2 (a keeper) and B.3 (a game state) are built,
the keeper off and the game state on; C.1–C.3 (the ball's sigma, the
goal-post localiser, the fused team ball) are built and measured against
the truth; C.4's opponent list and D.2's field are built and measured in
play, each paying in one configuration and not the other. B.1 (a get-up)
came off the GPU list on 2026-09-08: `alpha_stand` already does it (36 of
36 from back, front and side) and `getup-l2` reproduces it locally in
3.5M steps, so what is left there is a controller switch in `arena.py`,
not a training run. What is left needs a GPU (A.2 a learned kick,
E.1–E.3) or a colour sense
that survives contact range (the duel, C.4's second half). E still says
the field's learned results *kept the honest camera and rewarded looking*,
which is the opposite of the shortcut that would make a striker look good
in sim. If one thing gets built next it should be A.2 on the official
stack with the head-down patch applied: the swing is the limit, and the
survey's every route around it has now been measured.

Sources: [B-Human 2024 code release](https://docs.b-human.de/coderelease2024/)
and its [WalkKickEngine](https://docs.b-human.de/coderelease2024/motion/motion-walkkickengine/);
[Mellmann et al., Simulation Based Selection of Actions for a Humanoid Soccer-Robot, RoboCup 2016](https://www.ais.uni-bonn.de/robocup.de/2016/papers/RoboCup_Symposium_2016_Mellmann.pdf);
[Haarnoja et al., Learning agile soccer skills for a bipedal robot with deep RL](https://arxiv.org/abs/2304.13653);
[RL Within the Classical Robotics Stack: A Case Study in Robot Soccer (WisTex United)](https://arxiv.org/html/2412.09417v1);
[Learning Vision-Driven Reactive Soccer Skills for Humanoid Robots](https://arxiv.org/abs/2511.03996);
[Dribble Master](https://arxiv.org/abs/2505.12679);
[NimbRo RoboCup 2023 AdultSize winner: NimbRoNet3 and waveform in-walk kicks](https://arxiv.org/abs/2401.05909);
[A Hierarchical, Model-Based System for High-Performance Humanoid Soccer](https://arxiv.org/abs/2512.09431);
[RoboCupSoccer Review: The Goalkeeper, a Distinctive Player](https://arxiv.org/pdf/2303.12635);
[A Reliability-Based Particle Filter for Humanoid Robot Self-Localization in RoboCup SPL](https://pmc.ncbi.nlm.nih.gov/articles/PMC3871090/);
[Voronoi Based Strategic Positioning for Robot Soccer](https://ceur-ws.org/Vol-1032/paper-23.pdf);
[B-Human 2022 – More Team Play with Less Communication](https://link.springer.com/chapter/10.1007/978-3-031-28469-4_24);
[SoccerDiffusion](https://arxiv.org/abs/2504.20808);
[NAO camera geometry (Aldebaran docs)](https://fileadmin.cs.lth.se/robot/nao/doc/family/robots/video_robot.html);
[RoboCup Humanoid League call for participation (KidSize 40–100 cm)](https://humanoid.robocup.org/robocup-2025/call-for-participation/);
[RoboCup SPL GameController](https://github.com/RoboCup-SPL/GameController3).

## Physics audit — 2026-09-06 (after the ball that never stopped)

The ball's zero rolling resistance (a coefficient set on a condim-3 geom,
Track 4 item 0) prompted a sweep of every other physics parameter in the
harness, measured, not read. Probe scripts were scratch; the numbers are
here. **Fixed the same day (2026-09-06, below each item) — every fix has a
test, and the measurement that moved.**

**World layer (the /sim rooms and the pitch).**

1. **In `collision="walk"` — every builtin scenario — a duck is two 13 mm
   soles.** Only the foot meshes carry contact bits against the world
   (`compose.py`, upstream's `robot_walk.xml` by design, for flat-floor
   training). Measured: a ball thrown at trunk height passes THROUGH a
   standing duck touching only its ankles; a person capsule at 0.3 m/s walks
   through the trunk (its surface 19.7 cm inside) and displaces the duck by
   7 mm; two walkers head-on overlap to 3.1 cm trunk-to-trunk (11.3 cm
   under `all`); a walker into a wall gets its beak 9 cm inside the board
   before the feet touch. The `_sense_bumps` docstring ("only the FEET carry
   collision geometry") is also wrong for duck-duck: trunk/leg slivers
   collide with each other's. `collision="all"` fixes all of it and the
   shipped walker is bit-identical on a flat floor under `walk` and `all`
   (max |Δqpos| = 0 over 10 s × 3 seeds with shoves) — but see 2 first.
   Severity: high for anything eval-tidy / eval-pitch / the person-follow
   brains measure about contact, crowding and bumps.
2. **A mocap person is an infinite-mass teleporter.** Under `collision="all"`
   a person at 0.8 m/s flings the duck at 4 m/s (233 mm penetration, 7 falls
   in 2.5 s); at 0.3 m/s it shoves it 1.1 m. Moving the mocap per substep
   does not help; a softer capsule only trims it to 2.3 m/s. `yield_m > 0`
   (the polite walker) never touches. Before adopting 1, make `yield_m`
   default nonzero or cap person speed, and say in `Person` that a mocap
   body cannot yield momentum.
3. **The ball is dead off the boards**: restitution 0.06 at a 1.4 m/s kick
   (a real hollow ball is ~0.5–0.7). Not a wrong setting — restitution is
   unmodelled by MuJoCo's soft contact at the default solref; a much stiffer
   ball contact gets to ~0.27. Matters for eval-pitch: a kicked ball sits at
   the wall.
4. **Toy sliding friction 0.8 is inert** (`compose.py`): equal priority with
   the floor takes the element-wise max, so toys slide at μ = 1.0 (measured
   0.41 cm from 0.3 m/s, the μ = 1.0 prediction). Their torsional/rolling
   entries are on condim 3 too — harmless for boxes. Give toys `priority=1`
   if 0.8 is meant, else delete the numbers.
5. `_sense_bumps` reads only the last substep's contact list (a touch under
   four substeps is invisible). Low.

**Shipped for 1–5** (`world/scenario.py`, `compose.py`, `arena.py`; tests in
`test_world.py` ×7, `test_arena.py` ×2):
- `Scenario.collision` defaults to `"all"`; the "all" variant's inertials
  are pinned to the walk file's (upstream hand-rounds three bodies
  differently per export, so the walker was NOT bit-identical as shipped —
  it is now: max |Δqpos| 0.000 over 10 s × 3 seeds). Shoe-shell hulls in
  the ankle bodies are masked off the floor so the sole stays the ground
  contact. Measured after: a trunk-height ball bounces off the jaw/trunk
  (was: through, touching nothing); walkers head-on stop at 11.2 cm with
  no fall (6.7 cm and a fall); a wall stops the beak at −0.2 cm (8.7 cm
  inside, fell).
- `Person.yield_m` defaults to 0.55 (never touches, 34 cm surface gap at
  0.3–1.5 m/s). There is NO safe speed for a yield-0 person against a duck
  with a body (0.10 m/s shoves it 0.22 m, ≥0.15 broadside topples it), so
  speed is documented, not clamped; a respawn now steps clear of a person
  capsule (the 5 m/s "fling" was a respawn inside one).
- Toys get `priority=1`, so μ = 0.8 applies (slide 0.51–0.57 cm from a
  0.3 m/s nudge, the μ = 0.8 prediction; was 0.41, the μ = 1.0 one). A
  jaw–toy contact exclude beside each grasp weld: under "all" the jaw's
  rigid hull sat on a 4 cm block and the tidy pick behind the basket fell
  from 8/8 seeds to 3/8; the exclude restores 7/8 (seed 7 picks at 78 s
  under walk and past the 90 s window under all).
- Ball restitution: negative result. Best e = 0.22 at solref (0.01, 0.1)
  and it shortens roll-outs 20%; not shipped, written on the geom.
- Bumps are sensed on every substep (+3–8% on a 0.2–0.35 ms world step).
- `World.step` now refreshes kinematics/COM/sensors after the substep loop,
  the same four calls as the walk env's fix for item 6 below, so the
  one-duck world still matches the env step for step.
- Paired benchmarks, same seeds: eval-pitch 4 seeds — goals 1 → 1, own
  goals 1 → 0, kicks 14 → 15, back-kicks 6 → 7, **falls 4 → 1**;
  eval-tidy 16 seeds — **0.82 → 0.90 tidied, 0.44 → 0.31 falls a run**
  (9 seeds better, 3 worse, 4 tied). Saved scenes that pin
  `"collision": "walk"` (`scenarios/follow-me-edit*.json`,
  `pitch-roles-2v2.json`) keep the old bodiless duck until re-saved.

Clean, measured: solver/integrator options equal `scene_walk.xml` (and
upstream's implicitfast/10-iteration choice makes zero difference to the
trajectory); every duck body mass/inertia bit-identical to upstream; box
and ball inertia right; timing (200 substeps per 50 ticks, one ctrl write a
tick); the grasp weld (0.4 mm drift, clean release); toys and boxes rest;
ToF/detector rays see what they should and never the duck's own body; the
goal line has 3.5 cm of margin and registers slow rolls.

**Training layer (walk env, BAM, vec env).** No silently-ignored
parameter: every geom is condim 3 with default torsional/rolling values;
friction and mass randomisation land in the model (300/300 episodes);
joint order, DEFAULT_POSE, obs frames, action sign, the body-velocity frame
(the `mj_objectVelocity` trap) and the 50 Hz cadence are all right; BAM is
line-exact against upstream's `bam/actuator.py` and within 8% of the
XL330-M288 datasheet. The gaps are fidelity, not errors:

6. **IMU obs are one substep (5 ms) stale** relative to the joint blocks in
   the same vector: `walk_env` reads gyro / projected gravity after the 4th
   `mj_step` without an `mj_forward`. mjlab forwards once before
   observations. Measured with `alpha_walking.onnx`: gyro median 0.06,
   max 0.76 rad/s off (obs noise band ±0.03); gravity max 0.0097 (band
   ±0.01). `infer_policy.py` has the same staleness, so the deployed policy
   sees it too. One `mj_forward` after the substep loop makes obs,
   termination and the height check consistent.
7. **Integrator**: local runs Euler / 100 Newton iterations (XML default,
   same as `infer_policy.py`); upstream trains on implicitfast / 10.
   Measured zero trajectory difference on the walker — record it as a
   deliberate choice or match it.
8. **`train-walk` trains on the XML PD servo by default** (`train.py` passes
   no `actuator`; BAM only reaches `train_behavior` forward behaviors).
   XML stall 0.96 Nm vs BAM's firmware-limited 0.64; Coulomb 0.005 vs
   0.011–0.024; 0–1 ctrl-step lag vs 3–6 substeps. Documented, not silent,
   but the default is the low-fidelity path.
9. **Domain randomisation missing vs upstream**: velocity pushes (±0.3 m/s
   every 3–6 s), trunk/head CoM offsets, armature ±10%, IMU misalignment
   ≤6°, encoder bias, 0–1 step sensor delay, joint-limit penalty; trunk
   mass DR is mass-only (upstream scales inertia too), and
   `body_subtreemass` goes stale after the write (dynamics unaffected).
10. **Open hardware questions** (not sim bugs): the 1.75 A current clamp
    BAM models may not exist on the robot — `robotd` never writes
    `operating_mode` or `current_limit`, and in plain Position mode the
    XL330's ceiling is the PWM limit (~0.98 Nm, the XML's 0.96). One
    register read settles it. And `robotd.toml` now defaults to
    `action_scale 0.9` with low-pass filters that the pinned upstream sha
    and this harness never train with.

**Shipped for 6–9** (`walk_env.py`, `train.py`, `vec_env.py`,
`behaviors/env.py`, `eval_onnx.py`; `tests/test_walk_env_physics.py`, 13
tests):
- Obs are fresh: after the substep loop the env runs
  kinematics + comPos + comVel + sensorVel (3.5 µs; `mj_forward` is 18.8
  and `mj_step1` rebuilds constraint rows the BAM friction scan reads).
  Gyro/gravity/height staleness 0.767 rad/s / 0.0098 / 6.9 mm → 0.000.
  Cost +3% a step, −1.5% ctrl steps/s at 32 envs. `infer_policy.py` on the
  robot side keeps the 5 ms lag; it is now the sim that is fresh, which is
  the direction mjlab trains in.
- Solver: implicitfast / 10 / 20 as mjlab's velocity cfg. Bit-identical to
  Euler/100/50 on the walker (the solver converges in ≤6 iterations);
  `ls_iterations` moves BAM by 1e-17, chaos-amplified to 4e-3 by step 300,
  no fall either way. Training parity wins; the deployment runtime's XML
  default is identical under the xml servo.
- `train-walk` trains on BAM by default (`--actuator xml` opts out,
  `MICRODUCK_ACTUATOR` still honoured): 56.4k → 42.4k ctrl steps/s at 32
  envs (−25%) for the honest servo.
- DR added with upstream's ranges: mass AND inertia ×U[0.95, 1.05] with
  `mj_setConst` (subtree mass consistent to 1e-16); trunk/head CoM ±3 mm
  ramping to ±15/±10 mm on upstream's step ladder; armature ×[0.9, 1.1];
  velocity pushes ±0.3 m/s every 3–6 s (measured 44 pushes in 200 s,
  intervals [3.02, 5.96] s, |Δv| ≤ 0.296). Every field lands and restores
  to the bit with DR off; the in-process shared-model path replays all 12
  fields. Pushes are OFF for every behavior env (a push in a headstand
  curriculum is an experiment, not a fix) and OFF in `eval-walk` unless
  `--push`, so eval numbers stay comparable. Skipped: IMU misalignment,
  encoder bias, 0–1 step sensor delay, the joint-limit penalty — each an
  obs/reward change to measure on its own.
- The shipped `alpha_walking.onnx` under the new env: 0/10 falls, tracking
  0.171 m/s (unchanged); with pushes on, ~1 fall per 120 pushes.
- One measured test moved with the physics: `test_mapping.py`'s
  loop-closure bound. With fresh gyro obs the walker turns 98° instead of
  83° on the same 2.2 s turn command (126° asked), so the corner is
  approached at another heading and the wall-line matcher gets 0.21 →
  0.16 m instead of 0.21 → 0.12. Replaying the identical actions with
  stale sensing gives 0.146 (one seed of three equal), so it is the path,
  not the matcher's inputs; the bound is re-measured (0.85 × raw, 0.68 of
  the map on a wall) and the docstring says why.
- **The Linux x86_64 goldens in `tests/goldens/` are invalidated** (obs
  bytes and RNG draw order moved); re-record with
  `MICRODUCK_RECORD_GOLDENS=1` on Linux before CI goes green.

## Later / parked

- **Port `find_ball` to an mjlab cfg** and retrain on GPU in upstream
  `microduck_rl`. That stack, not this one, is the sim2real recipe. Blocked on
  the items above: there is no point porting a recipe whose back-bucket
  behavior is still moving.
- **What the fake detector cannot produce.** The env fakes the detector by
  projecting a point through the MJCF camera: FOV bounds plus a range check,
  and nothing else. Three things a real one does are therefore untested, and
  they are worth doing in this order rather than as one "real detector" job:

  1. **Box size, hence RANGE.** Cheapest, and it unblocks queued work rather
     than opening new questions: the handoff gate asserts *aimed* and never
     *in range*, which is exactly why the kick whiffs in every soccer render,
     and the approach behavior (item 4's stretch) cannot start without it.
     `distance ~= focal x real diameter / box height`.
  2. **Occlusion — i.e. WALLS.** Today "not seen" ALWAYS means "not inside my
     camera cone", so the belief's dead-reckoning is never wrong about
     anything except direction. Put an occluder in the scene and that
     assumption breaks: not-seen can mean *hidden*, and the right response is
     to look around the obstacle rather than sweep past it — the memory slot
     would have to represent "hidden there", not just "it went that way".
     That is a qualitatively harder search than the one this recipe solves,
     and it is the one a robot in a real room faces. Needs geometry in the
     scene AND a ray test in `_ball_sense`; note the lab arena on the
     `robot-lab-sim-roadmap` branch may bring the geometry along anyway, in
     which case the deployed duck meets walls its training never had.
     (Worth being clear about what walls are NOT: they are not a better
     search-direction cue. Spawns are uniform in bearing, so no wall makes one
     side likelier — see section 3, where the wrong-side cost is real but the
     fix is a faster sweep, not a smarter choice.)
  3. **False positives.** A real detector reports something orange that is not
     the ball. The policy currently trusts `seen` completely, and nothing in
     training has ever lied to it.

  The fully honest version — render the head camera and run the actual
  `duck_detect` ONNX — subsumes all three and is far slower per step. Worth it
  only once the behavior is otherwise settled.
- **Other things to find.** The slot layout is not ball-specific: the same
  four head slots and scan clock would serve "find the other duck" (upstream
  wants precise bearing for gaze and following) or "find the charging dock".
  A second target is a cheap test of whether the recipe generalizes or whether
  it memorized a ball-sized blob.


## B.1 closed — the get-up is wired, and it costs nothing (2026-09-08)

`alpha_stand` already had a floor get-up (see the B.1 section above), so the
gap was a controller switch: a fallen duck was still being driven by the
WALKER, which recovers 0 of 24, and `arena.py` teleported it. `World(
getup_infer=...)` now drives a downed duck with a get-up policy, and it leaves
the down state by standing rather than by the clock; `getup_s` becomes the
timeout. `eval-pitch --getup-policy` runs a battery on it.

**It needed a dwell, and the fall count is what found that.** First cut handed
the duck back to the walker the tick it first read upright. `fallen()` is a
threshold on projected gravity and trunk height, so a duck crossing it on the
way up drops straight back over — and one real fall in seed 0 became **25
counted falls, 0.1-0.3 s apart**, which is far too close together to be
separate topples. Across 24 seeds: 3 falls with the teleport, **30** with the
get-up. `getup_hold_s` (0.3 s upright before the walker gets it back) fixes it
at the source; it is the get-up's own settle, not a metric patch.

**Measured, 24 seeds x 300 s of 3v3, ball-out on, `getup_s` 5, same seeds:**

| | teleport stand-in | real get-up |
|---|---|---|
| falls | 3 | **3** |
| got up by itself | — | **3** |
| ran the timeout out | — | **0** |
| possession s/min | 40.92 | 40.53 |
| ballAdvance | 1.076 | 1.094 |
| ballProgress | 0.444 | 0.447 |

**100% in-play recovery, and the ledger does not move.** So the teleport can
be retired wherever a get-up policy is available, and the honest price of a
fall on this robot is about a second, not the 10-20 s B.1 assumed. It stays
OFF by default (`getup_infer=None`), because every number in this roadmap was
measured with the teleport. What it cannot show on this roster is any EFFECT:
the shipped brain falls 3 times in 24 runs, so falls remain far too rare to
move a ledger, exactly as B.1 first concluded.

## Item 12f — the resting-ball memory: BUILT, MEASURED, SHIPS OFF (2026-09-08)

The reasoning was good and the result is a null. The floor has had rolling
resistance since 2026-09-06, so a ball that stops STAYS stopped, but the
tracker forgot it on the same 2.5 s clock as a ball that might have rolled
anywhere, while the kick plan is a median 3.6 s old when the swing fires. So
at the moment that decides the kick the brain was reasoning about a track that
had already expired — which is why `kick_ahead_max` could only fire on 5% of
swings on its own.

**Built** (`TrackerParams.rest_coast_s` / `rest_vel`, `Track.at_rest` and
`Tracker.disturb`, `ChaseParams.rest_predict_s` / `rest_coast_s` /
`rest_clear_m`). A ball MEASURED at rest — two hits agreeing it is slower than
`rest_vel`, never merely "we have no velocity for it", because a ball seen once
rolling also has no velocity — outlives the coast clock and stays actionable
past `predict_s`. The memory is voided by our own kick, our own push, or any
body we know the position of standing within `rest_clear_m` of it; a fresh
sighting settles it either way.

**Measured in `scripts/kick_gym.py`, which now runs arms itself** (`--arm
'label=KNOBS'`, same seeds and episodes per arm, knobs read back off the
CONSTRUCTED brain, a two-proportion z on the pooled swings). Contested gym,
ball-out on:

| | discovery (seeds 0–3, 160 ep an arm) | fresh (seeds 100–107, 320 ep an arm) |
|---|---|---|
| swing rate | 53% → 63% (p=0.070) | 59% → 58% (p=0.75) |
| effective kicks an episode | 0.406 → 0.481 (p=0.18) | 0.441 → 0.450 (p=0.81) |
| whiff | 24% → 24% (p=0.97) | 26% → 23% (p=0.47) |

The discovery block's swing-rate gain **did not replicate and is withdrawn**.
Whiff was flat in both. **Ships off**, kept with its numbers so nobody
re-derives it; `tests/test_ball_memory.py` locks the mechanism and the ways it
is voided. What would change the verdict is a configuration where the ball is
genuinely lost for longer than the gym's short episodes allow.

### …and two harness bugs it turned up, both fixed

The 12a review reported that the bench and the arena were never the same
experiment. Both halves are now closed, and the kick's verdict survives both:

- **The bench ball rolled forever.** `contract.scene_walk_ball_xml()` symlinked
  upstream's `ball.xml`, whose geom has `friction="0.5 0.005 0.0001"` and NO
  `condim` — MuJoCo defaults to 3 and silently ignores the rolling
  coefficient, the exact bug `world/compose.py` fixed for the play world on
  2026-09-06. So the kick was trained and benched on the frictionless ball the
  physics audit removed from play. Now patched to condim 6 at the play world's
  `Ball.rolling`, into the generated scene and never the pinned checkout.
  Verified identical to play: a 1.4 m/s roll travels **1.62 m in 5.6 s in both**.
  (An apparent 0.79 m gap while checking this was the measurement's own fault —
  the ball was started at the robot's feet.)
- **The bench graded on noisier observations than the arena.** `BehaviorEnv`
  defaults `obs_noise=True` independently of `domain_rand`, and the bench
  passed only `domain_rand=False`, while `WorldDuck.obs` reads straight off
  mjData with no noise at all. `bench_kick_headdown.py` now passes
  `obs_noise=False, action_delay=False`.

Re-benched on the honest ball and observations, the local right kick still
whiffs **0% from every gaze pose**; travel is 0.85–1.00 m where the
frictionless ball gave 1.0–1.3 m. The headline verdict of item 7 stands, on
numbers that now mean what they say.

## The duel (C.4 second half, bead mdl-23b): the blocker is gone, the cheap fixes are still null (2026-09-08)

The open-field duel is the largest remaining block of dead time — 263 of 508
3v3 line-ups die to `avoid` inside 0.4 s with an opponent 0.33 m ahead — and
it has been parked on one sentence: a duck cannot tell a teammate from an
opponent close in, because one frame of the classifier is a coin. Nobody had
measured the **vote**, which is what `_is_mate` actually reads.

**It survives contact range** (`scripts/probe_duck_color.py`, 4 seeds × 300 s
of 3v3, 620 000 live duck-track ticks against the truth):

| range band | ticks | voted a colour | …and RIGHT |
|---|---|---|---|
| 0.00–0.35 m | 22 862 | 100% | **95%** |
| 0.35–0.50 m | 39 620 | 100% | **95%** |
| 0.50–0.80 m | 67 154 | 100% | 92% |
| 0.80–1.50 m | 158 402 | 61% | 84% |
| ≥ 1.50 m | 269 865 | 8% | 68% |

Counterfactually (the decision is gated on `use_color`, which ships off, so
asking `_is_mate` would answer "no" every time): at contact range it would
call an opponent a teammate on **1.4%** of ticks — the dangerous error — and a
teammate a stranger on 3.6%, which is merely wasteful since unknown already
counts as an opponent. What is hopeless is the FAR field, and the duel does
not need it. **The premise the bead was parked on is refuted.**

*Caveat worth carrying:* accuracy at contact range is 98–99% for tracks with
under 20 hits and **91% at 20 or more**. More frames making a vote worse is
the signature of an identity switch keeping a stale tally, not of classifier
noise — so anything built on this should decay or reset the vote on a
re-association.

**And the sense being available does not, by itself, fix anything.** The first
arm with it switched on (`use_color=1, opp_keepout=0.5`, i.e. give an opponent
a wider berth than a teammate), 12 seeds × 300 s of 3v3 on the ball-out floor:

| paired | shipped | use_color + opp_keepout | p |
|---|---|---|---|
| kicks (events) | 63 | **48** | — |
| ballAdvance | 1.069 | 0.973 | 0.28 |
| ballProgress | 0.380 | 0.226 | 0.31 |
| possession s/min | 41.50 | 42.17 | 0.72 |
| falls (events) | 3 | 7 | 0.40 |
| back-kicks | 25% of 63 | 25% of 48 | 0.96 |

Nothing resolves, and every direction that moves is the wrong one: a quarter
fewer touches for no gain. Together with `lineup_keepout` (measured null on
the same floor), that is **two independent geometric answers to the duel, both
null**. Avoiding the opponent harder is not the fix; the duck has to CONTEST
the ball — shield it, get a body between the opponent and it — and that is a
behaviour to design, not a radius to widen. The sense it needs is now known to
be there.

## The contest — the duel's THIRD null, and what it rules out (2026-09-08)

With the colour sense measured good at contact range, the duel became
buildable, and `ChaseParams.contest_margin` is the version the earlier two
nulls pointed at. It is not a radius: a radius cannot break a symmetry, which
is why both `lineup_keepout` (shrink it on a line-up) and `opp_keepout` (widen
it for an opponent) were null. This asks **who should have the ball** — only
the duck that is nearer holds its line, the other still gives way — so exactly
one of the pair commits. It fires only against a duck the colour vote calls an
opponent, and a TOUCH still stands the duck up safely, so it declines to turn
away and never walks into anybody. `use_color` is ENFORCED, not merely
documented: with the sense off `_is_mate` answers False for everybody, so
without the gate it would contest its own teammates. A test caught that.

**Measured, 12 seeds × 300 s of 3v3 on the ball-out floor, against the shipped
brain, then a fresh block:**

| paired | discovery 0–11 | fresh 100–111 |
|---|---|---|
| kicks (events) | 63 → 68 | 56 → 60 |
| ballAdvance | +0.025 (p=0.73) | −0.096 (p=0.46) |
| ballProgress | +0.002 (p=0.99) | −0.136 (p=0.44) |
| possession s/min | +1.46 (p=0.27) | +0.12 (p=0.93) |
| **falls (events)** | **3 → 10 (p=0.043)** | **6 → 6 (p=1.00)** |
| back-kicks | 25% of 63 → 24% of 68 | 25% of 56 → 32% of 60 |

The discovery block's fall alarm — the one metric that resolved, and the one
that would have made this "keeps the touches but pays in contact" — **did not
replicate**. On the fresh block falls are identical. A separate probe of where
the falls happen was underpowered to settle the mechanism either way (2 falls
an arm over 6 seeds), and is reported as such rather than dressed up. So the
honest verdict is the plain one: **a null**, on the ball and on the falls
alike. Ships off, kept with its numbers.

**What three nulls rule out.** `lineup_keepout`, `opp_keepout` and
`contest_margin` are all answers to the same question — *whether to turn away
from the other duck* — and all three are nothing. That decision is not where
the duel lives. What is left is genuinely different in kind: a **positioning**
behaviour that gets a body between the opponent and the ball (a shield, which
changes where the duck stands rather than whether it flinches), or the WALKER,
which the repo already records as unable to do anything against another body
except stand still. The colour sense is no longer the blocker for either;
`scripts/probe_duck_color.py` says it is right 19 times in 20 where it matters.

## Track 4, item 12 — the last 30 centimetres (2026-09-08): a ball at the feet is lost, then missed — ASKS

**The complaint, from the /sim page.** A duck walks the ball to the boards, has it at its feet, loses track of it,
swings, misses, and the ball is exactly where it was. With the rolling-resistance floor (2026-09-06) the ball no longer
runs away, so it sits at somebody's feet more of the run than ever — and the funnel below says the feet are where
this stack is weakest. Nothing below is a knob sweep of what exists; each ask is a different mechanism with the number
that would settle it.

**The funnel, measured this morning** (`probe_kick_line.py --ball-out-s 5`, 60 seeds × 300 s of 2v2, 465 swings,
local kicks, exits from the sidecars; whiff = the ball travelled < 10 cm after the swing). **Corrected the same
day:** the table first written here read the probe's `moved` column (ball drift since the plan) as the kick's travel
and had the rows upside down — this is the right one.

| where the ball was at the swing (from the root, yaw frame) | swings | whiff |
|---|---|---|
| 0.00–0.08 m ahead | 46 | 24 % |
| 0.08–0.11 m ahead (`kick_ahead` = 0.08: where the plan puts it) | 103 | **15 %** |
| 0.11–0.15 m ahead | 87 | 18 % |
| 0.15–0.20 m ahead | 59 | 49 % |
| 0.20–0.30 m ahead | 98 | **81 %** |
| ≥ 0.30 m ahead | 68 | 90 % |
| inside a 0.15 × 0.12 m box (44 % of swings) | 205 | **8 %** |
| outside it | 260 | **76 %** |

Whiff overall 46 %. The line-up reaches its spot (trunk-to-spot 0.019 m against `lineup_tol` 0.03); what has moved is
the ball: the plan is a median 3.6 s old, spot-to-ball at the swing is 0.17 m against the 0.08 planned, and the swings
that miss are the ones taken at a ball that is no longer in front of the foot. **12a is done** (`probe_kick_line
--dump-state`, `bench_kick_headdown --from-swings`): 93 play swings replayed on the bench from their exact state whiff
74 %, and putting the duck back in the HOME pose or stopping the ball changes nothing (74–75 % in all four variants),
so the arrival pose, the walker's velocities and the ball's motion are not the cause; the actuator gain of the kick
window (0.8) is not either (0 % on the bench at 0.8, in a 0.5 s window). By bin the bench agrees with play (30 % whiff
at 0.08–0.11 m, 100 % at ≥ 0.20 m). The camera sees the floor from 0.12 m out at the line-up gaze, so those far balls
are *visible* — which is why an ahead gate on the fresh predicted ball (`kick_ahead_max`, 12c's first half) has the
coverage the side gate (`kick_side_max`, 4b) never had.

### The asks

**12a. A swing replay: why does the bench kick connect and the play kick not?** — DONE 2026-09-08, see above: the ball is not where the swing goes, and nothing else is.
Record, for 50 play swings, the full state at swing start (root pose, joint angles, walker phase, ball position and
velocity, head/neck joints) and replay each on the bench from that exact state. Three candidates, each falsifiable:
(i) the arrival pose — the walker's settle does not reach the standing HOME pose the kick trained from (compare joint
angles at the swing with `C.DEFAULT_POSE`; the kick recipe spawns from HOME ± nothing); (ii) the ball is moving —
pushed by the settling feet in the last 0.3 s (ball speed at swing start; the recipe trains on a resting ball);
(iii) the standing leg — with the ball at 0.08 m the support foot or the shin hits it first. Command: extend
`probe_kick_line.py` with `--dump-swings out.jsonl`, and `bench_kick_headdown.py --from-swings out.jsonl`.
Number: bench whiff from replayed play states. If it is 0 %, the sim of the swing is wrong; if it is ~90 %, one of
(i)–(iii) is the cause and the recipe has to train on it.

*Independent replication, 2026-09-08 (second agent, 386 swings over the 48 seeds the original funnel used;
`scripts/replay_kick_swings.py`, a ladder of one-change-at-a-time cells).* Same conclusion, reached with a
**positive control** the four-variant reading above cannot supply: all four of those variants leave the ball
where play had it, so they show only what does *not* matter. Move the ball back onto the recipe's sweet spot
and hold everything else — the exact recorded root pose, all 14 joint angles and velocities, the lagged
`joint_vel` and `last_action` obs blocks, the servo targets:

| replay cell (386 swings) | whiff | foot reached the ball | median travel |
|---|---|---|---|
| exact play state, ball where play had it | 56 % | 46 % | 0.00 m |
| HOME pose, still, ball where play had it | 59 % | 43 % | 0.00 m |
| **exact play state, ball on the recipe's spot** | **0 %** | **100 %** | **1.94 m** |
| …and under the arena's whole swing protocol (Kp ×0.8, condim-6 rolling ball, 0.5 s window then the walker) | **0 %** | **100 %** | **1.33 m** |
| the recipe's own reset (control) | 0 % | 100 % | 1.53 m |

So the swing as the arena runs it is not damaged in any way: from the pose a duck really arrives in, it kicks
the ball 1.3–1.9 m, *further* than from the recipe's own spawn. The three candidates are individually dead, not
merely inert — (i) every LEG joint arrives within 0.052 rad of `C.DEFAULT_POSE` (worst per swing: median
0.032, p90 0.042, never past 0.10; root speed 0.012 m/s, i.e. standing still); (ii) ball speed at the swing is
a median 0.016 m/s; (iii) in 79 of 79 swings the only thing touching the ball is the **floor** — no foot, no
shin. Nor is it the boards (ball-to-board median 0.87 m, never inside 0.10 m) or a crowding team-mate (whiff
36 % when one is within 0.25 m, against 39 % overall).

What remains is one number: how far the ball is from the spot the foot swings through. Measured radially
(against the recipe's `BALL_OFFSET`, 0.09 ahead × 0.042 to the kicking foot's side) the funnel is monotone,
which the "ahead" projection is not — a ball at the right distance on the wrong side lands in the same row as
one the foot passes straight through:

| ball's offset from the kick's sweet spot | swings | whiff |
|---|---|---|
| 0.00–0.03 m | 57 | **0 %** |
| 0.03–0.06 m | 65 | 8 % |
| 0.06–0.10 m | 60 | 20 % |
| 0.10–0.20 m | 79 | 61 % |
| ≥ 0.20 m | 125 | 86 % |

and that offset is the stale plan, almost exactly: **corr(ball drift since the plan, offset from the spot) =
0.96** (n = 386). By drift: < 0.05 m → 10 % whiff, 0.05–0.15 → 30 %, 0.15–0.30 → 68 %, ≥ 0.30 → 88 %.
Rendered frames of one swing, the same state under two ball placements (misses, then 1.3 m):
`scripts/render_kick_swing.py --which 33 --cell 4|2|10`.

Two bench/play differences found while building this, neither the cause but both meaning the two harnesses
were never the same experiment: `BehaviorEnv` defaults `obs_noise=True` even under `domain_rand=False`, so
`bench_kick_headdown` grades the kick on **noisier** observations than the arena's noise-free `WorldDuck.obs`
gives it; and the bench's ball is upstream's `ball.xml` at **condim 3**, where MuJoCo ignores the rolling
coefficient entirely — the bench's 0 % was measured on the frictionless ball the 2026-09-06 audit removed from
play. Recommendation (not applied — `behaviors/` is another agent's): if `bench_kick_headdown` is to stand in
for play, it should pass `obs_noise=False` and set the ball geom to condim 6 / rolling 0.002.

**12b. Train the kick on the ball where it actually is.** The recipe (`behaviors/kick.py`) spawns the ball at
(0.09, ±0.042) ± 0.015, standing, resting. Play puts it at a median 0.13–0.15 m ahead, 0.07 m to the side, sometimes
rolling, with the duck arriving from a walk. The local loop trains a kick in four minutes now, so this is cheap:
spawn from the *measured* joint distribution (12a's dump), over the measured ahead/side box (0.05–0.25 × 0.02–0.14),
with the ball given the measured residual velocity, and pay the same terms. Number: the funnel above re-measured with
the new kick — the target is the 0.08–0.15 m rows under 30 %. Ship rule as always: discovery block, fresh block.

**12c. Close the loop in the last metre.** — FIRST HALF SHIPPED 2026-09-08: `kick_ahead_max` 0.15 (a fresh predicted
ball further ahead than that refuses the swing and lays the line again) together with `gaze_still` (the line-up and
settle keep the head on the ball's last place, so the track is fresh when the gate reads it — the track was a median
1.8 s old at the swing before, in 100 % of the far-ball swings, and the gate alone fired on 5 % of swings). Whiff
44 → 31 % on seeds 0–23 and 51 → 41 % on the fresh block 100–123 (pooled 47 → 36 %), connected kicks a run 3.6 → 3.6
on the fresh block; the 2v2 ledger, two blocks of 24 seeds (0–23 without the sweep, 100–123 with it, 48 paired): possession
+0.4 s/min (p 0.29), progress −0.03 (p 0.42; the fresh block alone read −0.095 at p 0.048, the first +0.03 — a
sign flip, not an effect), advance −0.02 (p 0.53), goals 42 → 49 (p 0.39), back-kicks 1.9 → 1.4 (p 0.03), swings
8.2 → 6.1 a run (p < 0.001 — the blind ones), own goals and falls flat. `gaze_still` alone sees the ball on 54 % of swings and changes nothing (45 % whiff). Still open here: the
per-tick re-plan inside `approach_back`, and the head's side coverage (the side gate `kick_side_max` stays off). The plan is made once (median 3.6 s before the swing) and walked to. Re-plan
the spot every tick from the freshest ball while inside `approach_back` (0.22 m), and gate the swing on the ball being
inside the kick's box *now* (from the tracker, with the sigma the tracker already carries): no swing at a ball the
tracker has not seen for more than 0.3 s or whose sigma is over 5 cm. Number: spot-to-ball at the swing (0.166 → under
0.05 m), on-box rate (8 % → over 50 %), and the swings-per-minute that this refuses (the cost). This is the item 7
"walk-in" lever the roadmap already names, made concrete.

**12d. See the ball at the feet: a standing look-down before the swing.** The walking gaze is capped at 0.6 (0.95 rad
absolute) because deeper looks while *walking* cost falls (4c). The swing starts from a settle, standing. Let the
settle look all the way down (the joints reach 84°; 1.3 rad absolute puts the floor from 0.05 m in the frame) for the
0.3 s before the swing, and fire only on a sighting. The "settle that raises the head" (item 7, 2026-09-06) measured
off — but that was with the *shipped* kick, which whiffed 12/12 head-down; the local kicks do not. Number: sightings in
the last 0.3 s before a swing (now: the ball is below the frame, so ~0), then whiff.

**12e. The ToF as the last-20-cm ball sensor.** `tof_ball_m` exists and ships at 0: the floor-ball blob from the ToF
array. With the ball at the feet below the camera, the ToF is the sensor that is pointed at it. Turn it on for the
line-up state only, and measure the same funnel; the risk is false blobs from a teammate's foot, which `_beside`
already knows about.

**12f. Keep a resting ball.** The ball does not roll forever any more, but the tracker still forgets it: `lost_s` 2 s,
then a search that flinches (the dip) 33 times a run a duck. A ball last seen inside 0.3 m with speed under 0.05 m/s
and nothing else touching it should be *kept where it is* until something is seen to move it (the board publishes
`ball_vel`; the arena knows contacts) — a "resting ball" prior in `Tracker`, with a long memory and no search.
Number: searches a run a duck (~33 → ?), time from losing the ball to the next swing, and the own-goal ledger (a kept
ball that is not there is a swing at nothing).

**12g. When the swing is not the tool: push it.** At the boards and with the ball under the body, the push-first arms
measured +3.2 s/min of possession against the kick (A.4) and the walker touches the ball 50/50 times walking through
it. The selector already has the push as an action (`kick_select_push`, off). The ask is the *situational* rule: a ball
inside 0.12 m ahead, or against a board, is pushed out to a kickable spot first, then kicked. Number: whiff on the
0.00–0.11 rows (86–94 %) → the push's touch rate, and possession.

**12h. A learned last metre.** Item 5's learned striker could not reach the ball; the local loop can now train a
closed-loop "approach and kick" from 12a's state distribution with the ball in the observation (the striker env's
contract carries it), rewarded on ball speed along the goal line — the walk-in, the settle and the swing as one
policy instead of a planned spot plus a blind swing. Number: the funnel, against 12b+12c.

**12i. A post-kick look that looks.** — SHIPPED 2026-09-08 as `look_sweep` 0.8 rad at `look_sweep_range` 1.0 m (with 12c's knobs): connected kicks seen again within 2 s 37 → 46 % (seeds 0–23), whiff 31 → 26 % there and 41 → 38 % on the fresh block, connected kicks a run 3.8 → 4.2 and 3.6 → 3.8; the ledger check is in 12c's paragraph. (Jonathan, from the /sim page, 2026-09-08: "it kicks and then shoots off to
the opposite side".) After a swing the brain stands and looks for `look_s` 0.8 s at `look_range` 0.3 m — a range
chosen for a whiffed ball 0.17 m ahead — then hunts the PREDICTED line (aim + exit angle) for 3 s. The exit scatter is
40–70° a foot, 40–48 % of kicks leave on the other side of that line, and a kicked ball is 0.5–1.3 m out, outside the
0.3 m gaze's frame; so the look sees nothing and the hunt walks away from the ball half the time. Ask: sweep the head
yaw across the exit line's side, standing, at a gaze that covers 0.3–1.5 m, and hunt toward the sighting; the ToF-
sideways objection to `look_aim` was about a yawed head while WALKING. Number: seconds from the swing to the next fresh
sighting (probe_kick_line `reacq_s`), and the share of kicks followed by a sighting inside `hunt_s`.

**Order.** 12a first — it is a day, and it decides between 12b (recipe) and 12c/12d/12e (sensing/timing). 12f and
12g are independent of it and a morning each. 12h only after 12b has shown what a better swing is worth.


### 12j. The search freeze, re-opened against the get-up — MEASURED OFF (2026-09-08)

The dip stops a searching duck dead for `search_dip_s` 0.6 s every
`search_dip_every` 1.5 s: 21 s a duck a run frozen, 58 % of its search time,
returning 2 % of the search's sightings. It was kept in item 7 only because
deleting it drove falls 121 → 195 over 48 seeds, back when a fallen duck was
teleported away. The get-up landed today (B.1), so the fall half of that trade
had changed and it was worth re-asking.

`MICRODUCK_CHASE=search_dip_s=0`, 24 paired seeds × 300 s of 2v2, `--ball-out-s 5`,
`--getup-policy alpha_stand.onnx` on BOTH arms:

| | dip (shipped) | no dip | p |
|---|---|---|---|
| **in-place turning** | **0.612** | **0.713** | **0.000** (24/24 seeds worse) |
| spread | 0.576 | 0.522 | 0.031 |
| ballAdvance | 0.541 | 0.476 | 0.093 |
| kicks a run | 5.96 | 5.13 | 0.109 |
| goals (total) | 22 | 16 | 0.271 |
| falls | 0.333 | 0.208 | 0.443 |

**The freeze stays, for a new reason.** The get-up did change the fall half:
falls no longer rise without the dip (0.33 → 0.21, the opposite direction, not
significant). But removing it does not free the 21 s — it converts standing
into SPINNING, on 24 of 24 seeds, and the ducks lose kicks, shape and goals
with it. The pause is the only thing interrupting a search circle; without it
the duck simply keeps turning. The 21 s was never the recoverable waste it
looked like, and "a fallen duck can now get up" does not reopen it.

This also retires the reading of item 12's ledger that motivated it: the run is
not slow because the ducks stand still, it is slow because they cannot SEE
(a duck has the ball in view 24 % of ticks; nobody on a team has it 61 %).


### 12k. Why a duck cannot see the ball at its feet, and why looking further down does not fix it — MEASURED (2026-09-08)

Jonathan, from the /sim page: *"when it looks down to see where the ball is it
still doesn't scan the area next to its feet — why can't the head turn more to
that region? don't we still have room?"* There is room, the walker will use it,
and it still does not pay. All three parts measured today.

**1. The room is real, and it is in the NECK.** Joint limits: `head_pitch`
−90…+90° (home +20, the shipped gaze reaches 54°, so 36° spare), `neck_pitch`
−90…+60° (home +20, and the brain has never commanded it at all —
`Chase.step` emits `(0.0, gaze, 0.0, 0.0)`).

**2. No retrain is needed to use it.** `C.HEAD_CMD_RANGES` is ±0.05 rad, so
every gaze this brain sends is already extrapolation — worth checking, now
closed. The shipped walker driven at each pose while walking (0.3 m/s, 4 seeds):

| commanded pose | depression | forward speed | falls |
|---|---|---|---|
| level (what it trained on) | 6° | 0.136 m/s | 0/4 |
| head +0.60 (the shipped cap) | 38° | 0.120 (−11.6 %) | 0/4 |
| neck −0.30 / head +0.60 | 58° | 0.124 (−8.9 %) | 0/4 |
| neck −0.60 / head +0.60 | 69° | 0.114 (−16.3 %) | 0/4 |
| neck −1.00 / head +1.00 | 79° | 0.083 (−39.3 %) | 0/4 |

Zero falls in 28 trials at up to 20× the trained command range, and depression
rises monotonically: the walker extrapolates cleanly. The split is also
strictly cheaper than the head slot alone — 58° for −8.9 % against 38° for
−11.6 % — confirming the sweep in `gaze_neck`'s own comment.

**3. And it loses in play, because the field of view SLIDES rather than widens.**
`fov_v_deg` is 48° and fixed, so a deeper gaze trades the far half of the floor
window for the near half. Measured from the camera site, walking:

| pose | floor window, from the root |
|---|---|
| head +0.60 (shipped) | 0.19 m … 0.91 m |
| neck −0.30 (`gaze_neck` 0.5) | 0.12 m … **0.36 m** |
| neck −0.60 (`gaze_neck` 1.0) | 0.10 m … **0.27 m** |

Half the neck buys 7 cm at the feet and gives up 55 cm at the far edge — and
0.1–0.5 m is exactly where the ball is while a duck walks its line-up. In play
(24 paired seeds × 300 s of 2v2, `--ball-out-s 5`, on top of the 12c/12i
defaults):

| arm | whiff | on the sweet spot |
|---|---|---|
| no neck (shipped) | **26 %** | **11 %** |
| `gaze_neck` 0.5 | 35 % (+0.117 a seed, 15 of 24 worse) | 6 % |
| `gaze_neck` 1.0 | 28 % (+0.028, 11 of 24 worse) | 7 % |

**`gaze_neck` stays off, on a new and better reason.** It was off because the
SHIPPED kick whiffed head-down; the local kicks do not, so that reason expired
and this is the re-measurement. The ball at the feet is not a gaze problem and
no head pose solves it: it is a 48° vertical field on a camera 0.21 m up. The
two things that could — a wider or second down-pitched lens
(`DetectorSpec.bottom_pitch_deg` exists as a sensitivity test, and a camera
this robot does not have is not a fix), or **the ToF, which already points
there (item 12e, `tof_ball_m`, ships at 0)**. 12e is now the live one.


### 12e. The ToF as the last-20-cm ball sensor, gated to the line-up — MEASURED OFF (2026-09-08), third time and on much better terms

The blob had been measured off twice by pooling EVERY tick it fired. That
population was the flaw: it also fires in `search`, `avoid`, `retreat` and
`support`, where the duck is anywhere on the pitch and a ball-height thing
0.3 m away is usually somebody's foot. `scripts/probe_tof_ball.py` (6 seeds ×
180 s of 2v2, 1830 events) splits it the way the decision would use it:

| population | events | camera had it | IS THE BALL |
|---|---|---|---|
| every tick it fires | 1830 | 70 % | 85 % |
| lineup / settle | 1379 | 82 % | **97 %** |
| lineup / settle, nobody beside | 1354 | 82 % | 97 % |
| …and the camera blind (the case for it) | 241 | 0 % | **85 %** |

So the sensor is not the problem. In the population that matters it is right
85 % of the time, not the 30 % the pooled number reported, and it clears the
four-in-five bar the probe names. **`tof_ball_lineup` (new, default True)**
restricts the blob to `lineup`/`settle` with no body beside; False reproduces
the always-on behaviour the earlier measurements killed. Locked in
`tests/test_kickselect.py`.

**And it still does not pay.** `MICRODUCK_CHASE=tof_ball_m=0.5`, 2v2 ×  300 s,
`--ball-out-s 5`, get-up on, discovery (0–23) then fresh (100–123):

| | discovery | fresh | pooled (48) |
|---|---|---|---|
| ballAdvance | +0.095 (p 0.043) | +0.039 (p 0.375) | +0.067 (p **0.036**) |
| ballProgress | +0.116 (p 0.083) | +0.038 (p 0.471) | +0.077 (p 0.070) |
| goals | 22 → 33 | 24 → **24** | 46 → 57 (p 0.195) |
| falls | 0.33 → 0.50 | 0.04 → **0.29** (p 0.070) | 0.19 → 0.40 (p 0.108) |
| kicks a run | 5.96 → 5.96 | 6.29 → 6.21 | flat |

**The fresh block did not confirm it** — the headline metric went from p 0.043
to p 0.375 and goals landed dead flat — so the pooled p 0.036 is carried by the
discovery block, which is the exact pattern this repo has been wrong about
before. And falls trend worse in BOTH blocks (0.19 → 0.40 pooled), the same
direction that killed the ungated version twice; at 48 seeds that is not
resolvable (falls want ~376), which is a reason to distrust it, not to discount
it.

Also worth recording: whiff was FLAT with the blob on (26 → 29 %, 10 of 24 seeds
better, `probe_kick_line` on the same seeds) and the kick count identical, so
whatever advance gain exists is not a better swing — it is the line-up. If this
is ever re-opened, that is the mechanism to instrument.

`tof_ball_m` stays 0. The gate stays, because it makes the next attempt start
from the right population instead of the one that produced 30 %.


### 12l. The blind duck and its teammate's ball — THE OPPORTUNITY IS 1.3 %, NOT 15 % (2026-09-08)

Proposed off a coverage measurement: over 120 000 duck-ticks of 2v2 a duck has
a fresh ball on **23.9 %** of ticks and is blind while its partner is looking
straight at it on another **14.7 %**, so pointing a blind duck at the board's
ball looked worth 24 % → 39 % of coverage. `_board_ball` already collects it
and only the interceptor reads it.

**Built, then reverted, because splitting those ticks by what the duck was
DOING says the opportunity is already taken:**

| blind, but a teammate sees it | share of all duck-ticks |
|---|---|
| **support** | **7.7 %** |
| retreat | 2.1 % |
| **search** | **1.3 %** |
| avoid | 0.9 % |
| blocked | 0.8 % |
| lineup | 0.8 % |
| hunt / chase / turn / wait / settle / kick / look | 1.1 % total |

A duck whose teammate is on the ball is a SUPPORTER, and `_support` already
steers by `Team.led_ball` — the board's ball, i.e. exactly the teammate
sighting this item wanted to use. That is 7.7 of the 14.7 points, already
spent. Retreat, avoid, blocked and lineup are all deliberate. The genuinely
wasted population — a duck turning on the spot in `search` while its partner
watches the ball — is **1.3 % of ticks, about 3.9 s a duck a run**, which is
below anything this benchmark resolves and not worth a knob's surface area.

**The error worth remembering:** the coverage number was real and the
conclusion drawn from it was not. "A duck cannot see the ball" is not the same
as "a duck is not being told where the ball is", and this brain already routes
the second through the roles. Measure what the ticks are DOING before valuing
them.

(For the 61.3 % where nobody on the team sees it — support 26.7 %, search
9.9 %, retreat 8.7 % — no amount of sharing helps; that is the real ceiling
and it is item 12's blindness, not a plumbing gap.)

---

## Why everything came back null: it was the instrument, not the knobs (2026-09-09)

A run of soccer experiments — the ball memory, the ToF blob, `lineup_keepout`,
`opp_keepout`, `contest_margin` — all reported "measured off". Asked directly
why, the answer is not that the ideas were bad. **Most of those batteries could
never have resolved the effect they were denying**, and one bug meant the
p-values were the wrong distribution entirely.

`scripts/audit_power.py` (new) reads every A/B battery on disk and reports what
each one could have seen. Thirteen pairs, 108 metric readings:

| metric | median MDE (% of baseline) | median seeds for a 10% change | reading |
|---|---|---|---|
| ballProgress | 297% | 21,437 | noise — do not quote |
| goals | 48% | 551 | too blunt for a null |
| falls | 33% | 256 | too blunt for a null |
| kickCount | 28% | 192 | too blunt for a null |
| ballAdvance | 19% | 86 | too blunt for a null |
| crowd | 16% | 65 | too blunt for a null |
| possession | 10% | 23 | usable |
| spread | 9% | 20 | usable |
| depth | 5% | 6 | usable |

**A real 10% improvement in kicks is invisible at 24 seeds and comes out as a
null.** Every "measured off" verdict quoted above sat on `kicks` or
`ballAdvance` at 12–24 seeds, i.e. on an instrument with a 19–28% floor.

### The four defects, and what each one was hiding

1. **The MDE was never read — and it was on screen the whole time.** It is the
   95% half-width, which `compare_pitch.py` always printed: a difference is
   significant exactly when it exceeds it. The table now prints it as a
   percentage of baseline and gives every row a verdict — `effect`, `null`, or
   **`NO RESULT`** — where `null` is reserved for a battery tight enough to
   mean it (MDE ≤ 15% of baseline, `--tight-pct`). A footer prints the seeds
   that would settle each `NO RESULT`.

2. **Every p-value in this repo's soccer history came from the NORMAL, not
   Student's t.** `paired()` took the t critical value from a lookup table but
   computed p with `math.erfc` — the normal — in an `except ImportError` branch
   for scipy. **scipy is not a dependency of this workspace and never has
   been**, so that branch is the one that always ran. The interval and the test
   disagreed, and the normal is anti-conservative: at 24 seeds a reported p
   between **0.039 and 0.05 was significant only by that error**. Student's t
   is now written out directly (regularised incomplete beta, exact to the
   published tables to 4 dp, no new dependency). Three surviving readings flip:

   | reading | normal p | Student's t p |
   |---|---|---|
   | t4 clamp · spread | 0.047 | 0.059 |
   | t7 colour · possession | 0.043 | 0.055 |
   | t8 comp · goals | 0.045 | 0.057 |

   Only *paired* readings are affected. The two-proportion z-tests (back-kicks,
   the gym's whiff rate, the contest's falls at p = 0.043) use the normal
   correctly and stand.

3. **The paired design buys nothing here, and the docs said it bought most of
   the power.** Median between-arm correlation **r = 0.05**, variance reduction
   **1.03x**. The sim diverges within seconds of any knob that fires, so by
   300 s the arms are independent runs sharing only a layout. Pairing is kept —
   it can never be worse — but it is no longer budgeted for, and the observed
   gain is printed per metric. The one battery where it paid (`t9 hunt`,
   r = 0.6) is the one whose knob barely fired: **a high pairing gain measures
   how little your arm perturbed the run, not how good your design is.**

4. **`ballProgress` is noise and has been quoted.** MDE has never once been
   under 100% of baseline. It is now labelled `unquotable` in the table.

### What to do instead

Measured cost per kick event: the 3v3 pitch is **~15 CPU-seconds** (86 CPU-s a
seed, 5.6 kicks a run), `scripts/kick_gym.py` is **~0.8** (129 CPU-s for 202
swings) — about **19x cheaper**, because a gym episode is one placement and one
swing while a pitch run is six bodies and a mostly dead ball. So:

- Anything about the kick itself goes in the gym, where the whiff rate is a
  proportion over hundreds of events instead of a mean over 24 runs.
- Screen on `possession` / `spread` / `depth` (10% / 9% / 5% MDE), then confirm
  the ball metrics — do not open on `kicks`.
- Measure the **opportunity rate** before spending a battery at all: a rule that
  fires on 1.3% of the run cannot move a whole-match average whatever it does
  when it fires. That measurement is what closed the ball-memory arm, and it
  costs minutes.

### What this does and does not overturn

It does **not** resurrect the five knobs. Their point estimates were flat or
negative, not merely unresolved; the honest restatement is "**not shown to
help, on an instrument that could not have shown a 10% gain**" rather than
"measured off". The distinction matters for what gets retried: `contest_margin`
and the ball memory are worth one more look **in the gym**, where the same
compute buys ~19x the events.

Locked by `tests/test_compare_power.py` (57 tests): the MDE identity, the
three-way verdict, the seed budget's quadratic scaling, Student's t against the
published tables, the anti-conservatism of the old normal, and the measured
pairing gain on the real batteries — with the historical finding pinned to the
files it was measured on, so a future pitch that makes it false fails the test
rather than silently passing.


### 12m. "It walks at the wall and gets stuck looking at it" — the ball is RIGHT THERE and it cannot see it (2026-09-09)

From the /sim page: *"the robot is just walking putting its head down and
walking towards the wall and the ball is obviously not there and now its stuck
looking at the wall"*. Counted per tick — a duck within 0.35 m of a board,
pointing at it within 35°, with no sighting of the ball fresher than 0.3 s
(4 seeds × 180 s of 2v2):

| | share of that stuck time |
|---|---|
| **the ball is within 0.5 m of the duck** | **79.7 %** |
| the ball is also at the boards | 48.5 % |
| state `blocked` | 34.5 % |
| state `support` | 21.5 % |
| state `retreat` | 20.4 % |
| state `search` | 12.6 % |
| …and the head is down while it happens | 12.5 % |

**The premise is inverted: the ball is not absent, it is invisible.** Four
times out of five the duck is standing within half a metre of the ball it is
looking for. This is the near-field blind radius (§3b of
`docs/camera-hardware.md`: 0.50 m head-level, 0.19 m at the gaze, on the
frustum the sim assumes) meeting a ball that spends most of a run at the
boards, where a duck cannot get an angle on it.

Total cost: **9 s a duck in a 180 s run**, and `gaze_still` — shipped the same
day (12c) — **doubled it from 4 s**, which is part of the flat ledger that
knob already carries.

**Two fixes built and reverted, because neither moved the total:**

1. `wall_back` — reverse out of a board instead of turning on the spot.
   `gait.back_up`'s own measurement favours it (1.6 s to clear 0.30 m against
   2.7 s turn-90-and-walk, 3.2 s turn-180; and backwards is the fastest this
   walker moves, 0.23 m/s against 0.185 forward). Measured: it halves the
   `blocked` share (34.5 → 18.6 %) and the total is **unchanged at 9 s** — the
   time moves into `support`. It escapes the wall and the duck goes back.
2. `post_margin` — clamp every hold target inside the boards, on the theory
   that posts laid out from a ball at the wall land inside one. **A no-op:**
   the striker's post for a ball jammed at (1.5, 1.35) is (1.25, 0.49), well
   inside 1.70 × 1.425. The posts were never the problem.

**So it is not a planning bug and not a walker bug**, and neither better logic
nor more training addresses it: the duck goes to the ball, gets within half a
metre, and the ball drops inside its own blind radius. That is the same
finding as 12k from the other side, and it is what the camera work below
measures directly.


### 12n. The soccer ledger on the cameras that actually exist — MEASURED (2026-09-09)

`docs/camera-hardware.md` measured all three frustums on *tidy* and on the
soccer *line-up*, but never the soccer ledger. Jonathan asked whether the
device's field of view is not larger than the sim's, which it is — in both
directions, because **62 × 48 is the IMX219's FULL-ARRAY figure and the robot
pins the sensor to a 1080p CROP** (39 × 22.5). Ball coverage first, 120 000
duck-ticks of 2v2 an arm, a sighting counted fresh within 0.3 s:

| camera | a duck sees the ball | nobody on the team does |
|---|---|---|
| **the robot today** — IMX219 1080p crop, 39 × 22.5 | **8.2 %** | **85.2 %** |
| what the sim assumes — 62 × 48, 320 px | 23.9 % | 61.3 % |
| the replacement as it ships — 116 × 60, 320 px, uncalibrated | 27.9 % | 54.5 % |
| **the replacement calibrated** — 116 × 60, 640 px | **35.5 %** | **42.7 %** |

So the blindness that item 12 calls its ceiling is substantially a property of
a camera **neither candidate is**: the real robot today is three times blinder
than the sim, and the replacement cuts team blindness by a third.

And the ledger, 24 paired seeds × 300 s of 2v2, `--ball-out-s 5`, get-up on.
**Read through `scripts/compare_pitch.py` as corrected in 7fff78c** — the p
values first published here came from `math.erfc`, the normal, inside a
`except ImportError: scipy` branch that has always run, and the normal is
anti-conservative. Nothing below flipped, but two rows that were quoted are
NO RESULT and one sits exactly on the line:

| against the sim's camera | the crop (today) | the replacement @640 |
|---|---|---|
| possession (both teams) | **−3.73 s/min (p 0.007)** | **+3.11 s/min (p 0.011)** |
| ballAdvance | **−0.377 m/min (p 0.000)** | −0.070, MDE 18% — NO RESULT |
| crowd | −0.013 (null) | +0.040 (p 0.039) |
| goals (22 baseline) | 11 — **NO RESULT**, MDE 54% | 16 — NO RESULT, MDE 65% |
| falls (8 events baseline) | 9 — NO RESULT, MDE 109% | 1 — **p 0.050, MDE 87%** |
| kicks (events) | 143 → 123 | **143 → 84** |
| ballProgress | unquotable | unquotable |

`goals` needs 708 seeds and `falls` 2840 to resolve 10% of baseline here, so
neither the crop's goal collapse nor its fall count is evidence of anything.
**The falls row on the replacement is the one claim not to lean on**: 8 events
against 1 is a large drop and it is on the significance line with an 87% MDE.
It is suggestive, and the mechanism is plausible — a duck that can see its
near field does not walk into things — but it is one battery.

**Every row above is one of nine read at once, and not one of them was named
before the run.** Three arms x nine metrics is 27 reads; if the nine were
independent, the chance that *some* row clears p < 0.05 on noise alone is 37 %.
They are not independent — possession/possessionWide, ballProgress/ballAdvance
and spread/depth/crowd are three correlated families, so the effective count is
nearer five and the honest threshold nearer 0.01 than Bonferroni's 0.0056.
Against that threshold **possession on the crop (0.007) sits on the line and
possession on the replacement (0.011) is past it**; crowd (0.039) and falls
(0.050) are well past it. What keeps the possession pair alive is not either p
value on its own but that the two arms move possession in OPPOSITE directions,
in the order the visibility column predicts — a dose-response, not a coin flip.
**RESOLVED BY 12u (read it before quoting anything here).** That confirmatory
battery has since been run — fresh seeds 100-123, possession named one-sided in
the predicted direction before launch — and **both contrasts replicated**:
crop −4.843 s/min (one-sided p < 0.0001) and replacement +3.082 (p 0.0015),
against −3.73 and +3.11 here. The possession rows below are therefore
**established**, not suggestive. The crowd and falls rows are not: they were
re-run too and did not replicate, exactly as this caveat predicted. Two things are untouched by any of this: the
crop's ballAdvance row (p 0.000) survives every threshold above, and the
visibility fractions — 85.2 / 61.3 / 42.7 — are descriptive counts over 120 000
duck-ticks rather than tests, so no correction applies to them at all.

**The crop is a straight loss** and it is what the robot runs today: every
soccer number in this repo is optimistic about the current hardware, the same
caveat §3c already recorded for tidy.

**The replacement is not a straight win, and that is the interesting part.**
It buys possession, and falls drop 8 events to 1 (suggestive, on the line, see
above) — a duck that can see the near field does not walk into things. But it
takes **41% fewer kicks** (143 events → 84), which is `kick_ahead_max`
(12c) doing its job: better sight means the gate can see that the ball is out
of reach and refuse the swing. Goals do not move. So the wide lens converts
blind swinging into possession and safety, not yet into goals, and what to do
with the extra possession is a brain question this branch has not answered.

Not chased here: 640 px is a bet on the NPU sustaining it (§5.1 is still the
open question), and the uncalibrated 320 px arm — the module as it would ship
without lens calibration — is the one to fear, since §2's 9.7° bearing error
is past the chase brain's aim tolerance. `MICRODUCK_CAMERA` can now set
`projection`, so that arm is finally runnable from a command line; it was
numeric-only until today.

### The fixed instrument, used: two unearned nulls become earned ones (2026-09-09)

Re-run in `scripts/kick_gym.py` on fresh seeds 600-615, 25 episodes a seed,
with the MDE now printed:

| arm | swings | whiff | vs base | ±MDE | p | verdict |
|---|---|---|---|---|---|---|
| shipped (clean gym) | 333 | 24% | — | — | — | — |
| ball memory `rest_predict_s=6,rest_coast_s=20` | 341 | 28% | +4 pts | 7 pts | 0.254 | **null** |
| shipped (`use_color=1`, 1 opponent) | 233 | 22% | — | — | — | — |
| `contest_margin=0.15` (+`use_color=1`) | 225 | 25% | +3 pts | 8 pts | 0.385 | **null** |
| `contest_margin=0.15` alone | 233 | 22% | +0 pts | 8 pts | 1.000 | **BROKEN** |

**Both are now earned nulls** — 7-8 percentage points of resolution, for 32 s
and 176 s of wall clock, against the 28%-of-baseline the pitch battery that
originally judged them could manage. Both point estimates are also slightly
*worse*, so the conclusion does not change; what changes is that it is now
supported.

**The third row is the point of the exercise.** `contest_margin=0.15` on its
own reproduced the baseline **episode for episode** — 233 swings against 233,
whiff 22% against 22%, p = 1.000. That is not a null, it is playbook rule 0: a
knob that changes nothing is BROKEN. The rule is gated on `use_color`, which
defaults to `False`, so the arm ran the shipped path and returned a clean
result about nothing — and the new MDE machinery called it `null`, which is
precisely the failure this work is about. `is_identical` now catches it and
names the gate. (Note what the same row proves in passing: `use_color=1` alone
is also a no-op here, since the base sets it and the two runs match.)

**Cost of the two batteries: under four minutes of wall clock**, for a
resolution the pitch could not reach at any size this project would ever run.

### Caveat on every null above: they were measured through a camera that does not exist (2026-09-09)

microduck-62's item 12n changes what the nulls in this section mean. The sim's
62x48 field of view is neither of the real candidates: 62x48 is the IMX219's
**full-array** figure, and the robot pins 1080p, which on that sensor is a
**crop of 39x22.5**. Ball coverage, over 120k duck-ticks — the share of time
where **nobody on the team can see the ball**:

| camera | nobody sees it |
|---|---|
| the robot **today** (1080p crop, 39x22.5) | **85.2%** |
| what this sim assumes (62x48) | 61.3% |
| the replacement, calibrated (116x60 @640) | 42.7% |

So the duel's three nulls — `lineup_keepout`, `opp_keepout`, `contest_margin` —
were all measured at 61.3% blindness, and the hardware in the room runs at
85.2%. A rule about **which duck turns away from which** cannot plausibly
matter when nobody can see the ball for six-sevenths of the run. The honest
restatement is not "the duel is a null" but **"the duel was never the binding
constraint, on either camera"** — and the constraint that binds is coverage.

That also reframes this section's own conclusion. The instrument had two
defects: it could not resolve a 10% effect (fixed above), and it was pointed at
a machine that does not exist (not fixable here — it is a hardware fact). The
first made nulls unearned; the second makes even an earned null a statement
about the wrong robot. **Before spending more compute on chase knobs, the
coverage question outranks all of them.**

One thread worth pulling, from 12n's own ledger: the replacement camera buys
possession (+1.55 s/min, p=0.006) but takes a third fewer kicks (5.96 -> 3.50,
p=0.001), which 12n attributes to `kick_ahead_max` refusing swings it can now
see are out of reach. That is a knob, not a law, and it is a kick question —
so it belongs in `scripts/kick_gym.py` at ~0.8 CPU-seconds an event rather than
on the pitch at ~15.

### The duel's three nulls were guaranteed: the rule fires on 0.07% of ticks (2026-09-09)

microduck-62's re-frame of the camera result (12n) was that the duel nulls are
"underpowered by population" rather than evidence the rule does nothing. That
is a number, and nobody had taken it — including me, whose own playbook rule 5
says measure the opportunity rate BEFORE spending a battery.

`scripts/probe_contest.py` (new) runs the contest arm with its gate set and
counts how far down the chain of preconditions a duck actually gets. 135,000
duck-ticks, 3 seeds x 150 s of 3v3, on the sim's camera:

| the chain of preconditions | ticks | share |
|---|---|---|
| the duck can see the ball | 51,346 | 38.03% |
| sees the ball **and** a live opponent | 41,247 | 30.55% |
| …and that opponent is within `duck_touch` | 560 | **0.41%** |
| **the rule fires** (`Chase.contesting`) | 97 | **0.07%** |

**One tick in fourteen hundred.** No whole-match average can move on a
population that size whatever the rule does when it fires, and the battery
that judged it could only resolve a 19% change in ballAdvance to begin with.
The null was not a finding; it was arithmetic.

That settles the reading of all three duel results. `lineup_keepout`,
`opp_keepout` and `contest_margin` are **not** "the duel does not matter" —
they are "a rule that acts on 0.07% of ticks cannot be measured by a
whole-match metric". The two are completely different claims and this repo
published the first one three times.

**What it does not license.** It is not evidence the duel WOULD pay; the
population is the ceiling on the effect, not a promise about it. The way to
find out is to build the population instead of waiting for it — a contested
placement in `scripts/kick_gym.py`, where the event is created rather than
hoped for, at ~0.8 CPU-seconds each. And per 12n, the population itself is a
function of the camera: at 42.7% team-blindness (the calibrated replacement)
contests become common, where at the crop's 85.2% they essentially never
happen. **The duel is a question about the camera, not about geometry.**


### 12o. The corner post: a fix built on a case that does not occur — RETRACTED (2026-09-09)

**Claimed and then withdrawn the same hour.** From /sim: a duck walking into
the corner and holding there, its own camera preview showing board and **0
det**. The proposed mechanism was that the DEFENDER's post is laid from its own
goal along the line to the ball, so with the ball in its own corner the post
lands inside the wall — hand-computed as (−1.66, 0.49) against bounds of
1.70 × 1.425, four centimetres out. `post_margin` clamped every post inside the
boards and shipped at 0.25.

**It does not happen in play.** Swept directly on `eval-striker` with a
defender+striker roster on both sides, 2 seeds × 60 s, everything else fixed:

| `post_margin` | cream possession, seeds 0 and 1 |
|---|---|
| 0.0 (off) | 9.32, 21.46 |
| **0.25 (as shipped)** | **9.32, 21.46 — bit-identical** |
| 0.30 | 9.32, 21.46 — bit-identical |
| 0.40 | 32.42, 11.76 — binds |

A clamp that changes nothing at 0.30 is a clamp no post ever came within 0.30 m
of triggering. So posts do not reach the boards in real play, the hand-built
corner case is not one the brain produces, and **the knob is reverted** rather
than shipped inert. At 0.40 it does bind, but that is no longer a correctness
fix for an impossible post — it is a positional knob that moves supporters
generally, and it would need its own discovery and fresh block.

**Four ways I got a confident zero from a question I had not asked**, all on
this one item, all in one hour, recorded because the pattern is the lesson:

1. **Wrong subject** — tested the clamp on the STRIKER, whose post is never
   near a wall, and called it a no-op.
2. **Wrong harness** — ran the A/B through `eval-pitch`, which is the
   deliberately role-free control, so a role-only knob could not fire. Nine
   metrics identical to three decimals; the skill file says this in bold.
3. **Measured through the fix** — a population probe that called
   `_hold_target` while `post_margin` was already the default, so it counted
   CLAMPED posts and reported 0.000%.
4. **Instrumentation that lied** — a wrapper that called `_hold_target` twice
   per tick to compare clamped against unclamped, reporting a 7.56% firing
   rate that the direct sweep contradicts. Re-entering a method with side
   effects is not a measurement of it.

Only the fourth needed the direct sweep to catch; the other three would have
fallen to one positive control — show the measurement produce a non-zero on
something already believed, before quoting a zero (playbook rule 6, 077a0a0).

**What this leaves of the /sim observation:** the corner-walking is NOT
explained by post geometry. 12m's measurement stands — the duck is at a board
with the ball within 0.5 m on 79.7% of that time — and 12p gives the mechanism:
a ball at a board is a ball almost nothing can be done with. Corners are also
not traps (3.79 s a visit against 5.17 s at a flat wall).

### 12p. Why the boards eat the game: the swing rate against distance to a board — MEASURED (2026-09-09)

The dead-ball budget said the boards are 72% of dead-ball time with **0 kicks
taken there**, which is a statement about a whole match and does not say
whether that is the ball's fault or the duck's. `kick_gym --at-boards`
(microduck-4a, eadbe47) asks it directly: one duck, one ball, one placement,
one swing, the real `chase` brain doing the walk-in and the settle, with the
ball drawn near a board instead of in open play and the duck spawned the SAME
0.45–1.40 m walk-in either way — so the two modes differ in where the ball is
and not in how far the duck walks, which matters because approach length
already drives the whiff through plan staleness.

**The swing rate falls monotonically as the ball is drawn nearer a board, from
85% in open play to 4% when it is drawn within 0.15 m** (200 episodes an arm,
8 seeds 800–807, shipped brain throughout, the open-play control from the same
seeds and episode count):

| ball drawn within | mean ball-to-board distance sampled | episodes producing a swing |
|---|---|---|
| 0.15 m | 0.10 m | **4%** |
| 0.30 m | 0.18 m | 9% |
| 0.60 m | 0.33 m | 26% |
| 1.00 m | 0.53 m | 35% |
| open play | (much larger) | **85%** |

**READ THE ROWS AS CUMULATIVE, NOT AS A RESPONSE CURVE.** `--at-boards M`
draws uniformly in [radius + 0.01, M], so the 0.60 m arm contains balls at
0.06 m as well as at 0.59 m; each row is the average over everything inside
its cap, which is why the mean-distance column is there. The endpoints are
clean and the monotonicity is real; the intermediate points are not a
per-distance response and must not be quoted as one. A proper response curve
means binning by each episode's ACTUAL placement, which needs the ball's
position kept in the row (`kick_gym` currently drops it — `swing.pop("ball0")`,
and a no-swing row keeps nothing). Four minutes of compute once someone adds
the field; not done.

**This is the strongest statement of item 12's problem yet**, and it is a
positive control as well as a result: the rate recovers toward open play as
the ball comes off the board, so the harness is working rather than failing to
find swings. It also reframes what the boards cost. It is not that the ducks
get stuck there — 12m measured them leaving a corner in 3.8 s, faster than a
flat wall — it is that a ball at a board is a ball almost nothing can be done
with: the kick spot lies 8 cm behind the ball along the kick line, which for a
ball at a wall is inside the wall, and `kick_clear` refuses a swing at
anything with a board right in front of it.

**And the knob for it has been off the whole time.** `board_margin` gates the
along-the-boards kick spot AND is the clearance it demands, and it ships at
**0.0** — so the boards-line spot is disabled today, and the pitch null that
retired it (11b: kicks 8.7 → 8.9, p 0.85) was measured on a knob whose shipped
value turns the feature off. Independently, that null failed on RESOLUTION and
not population: at 24 seeds the MDE on kicks was 28% of baseline, about 2.4
kicks, against an observed difference of 0.2. Anything up to a 28% improvement
was invisible. Unlike the duel's nulls (0.07% firing rate) this one
is worth re-running, and the gym is the instrument — at 4–9% swing rates the
metric is **swings per episode**, never whiff rate, or a knob that takes fewer
but better swings reads as a win.

#### Reading a null against how often the rule fires

A rule that only changes outcomes on the ticks it fires moves a whole-match
metric by roughly (firing rate) × (per-firing effect), so the smallest
per-firing effect a battery could have seen is

    required per-firing effect  =  MDE (as a fraction of baseline) ÷ firing rate

which turns a null into a POSITIVE claim — *this rule does not produce more
than an X% improvement on the ticks it fires* — instead of "no significant
difference". (microduck-4a, 2026-09-09.)

**State it one-sided.** The division above assumes a firing's effect is
consumed within its own tick. Most things here are not: a firing that lands in
STATE — the ball somewhere else, the duck somewhere else — is inherited by
every tick after it, so the real footprint is F ≈ min(1, k·f) for a persistence
of k ticks, and the requirement is MDE/F. Since F ≥ f, **the formula
OVERSTATES what you would need and therefore UNDERSTATES what the null rules
out.** It cannot claim more than it should, which is the safe direction and the
reason it is publishable. So: *"rules out per-firing effects above X%, and
possibly smaller ones."* Never "rules out exactly X".

| rule | fires on | MDE | bound at k=1 | with k=10 |
|---|---|---|---|---|
| contest_margin (the duel) | 0.07% | 19% | 27,143% | 271% at k=100 |
| kick_ahead_max alone | 5% | 19% | 380% | 38% |
| **board_margin (the 11b null)** | 15% | 28% | 187% | **28%** |

Persistence is the axis, not action-versus-positional — `board_margin` is an
action knob whose effect is maximally persistent, because a kick puts the ball
somewhere else and the rest of the run inherits it. So the 11b null is more
informative than the k=1 bound suggests, and "never measured in any meaningful
sense" (an earlier draft of this section) was too strong. The duel's verdict is
the one that needs no assumption: 27,143% survives a 100× persistence
multiplier at 271%, two orders of magnitude clear.

Rule of thumb where the confined case does hold: **a null is quotable when the
firing rate exceeds about twice the MDE.** Measuring k directly means comparing
paired trajectories after a single firing and counting ticks to reconvergence;
with a between-arm correlation of r = 0.05 at 300 s the prior is that k is
large for nearly everything. Not done.

**The consequence is about the instrument, not this knob.** At this repo's
MDEs almost nothing fires often enough for a whole-match null to be quotable on
the confined reading, which retroactively weakens much of Track 4's
shelved-knob list. That does NOT license re-opening them: it licenses the
smaller claim that they were not shown to be ineffective. Anyone re-opening one
should compute MDE ÷ firing rate first, with a persistence estimate if the
effect lands in state, and proceed only if the answer is a number worth having.


### 12q. The along-the-boards kick line WORKS, and ships disabled — MEASURED (2026-09-09)

`board_margin` gates the kick spot laid ALONG a board (rather than 8 cm behind
the ball, which for a ball at a wall is inside the wall) and supplies the
clearance it demands. It ships at **0.0**, i.e. off, and its only prior
measurement — 11b, kicks 8.7 → 8.9 on a 24-seed pitch battery, p 0.85 — was
taken on the disabled knob's own scenario at an MDE that could not have seen a
187% per-firing effect (see 12p).

Measured properly in `kick_gym --at-boards`, **swings per episode** as the
metric (at these rates whiff is secondary: a knob that takes fewer but better
swings would read as a win). 2240 episodes an arm, 8 seeds × 280, shipped
brain otherwise:

| ball drawn within | arm | swings/episode | |
|---|---|---|---|
| **0.30 m of a board** | shipped (`board_margin` 0.0) | **7.05%** | |
| | **`board_margin` 0.25** | **11.25%** | **1.59×, z +4.87, p < 0.0001** |
| | `board_margin` 0.40 | 7.10% | inert (p 0.95) |
| 0.15 m | shipped | 3.66% | |
| | `board_margin` 0.25 | 3.48% | nothing (p 0.75) |
| | `board_margin` 0.40 | 3.66% | **flagged BROKEN — baseline episode for episode** |

Whiff among the swings that did happen went 26% → 22% at 0.30 m (p 0.34, not
resolvable) — so this is MORE swings, not fewer-and-better ones.

#### The headline is diluted about 2.5× — the rule cannot act on most episodes

A boards spot exists exactly when **`board_margin ≤ ball_gap + 0.05`**, where
`ball_gap` is the ball's distance from the board. Measured by calling
`_along_the_boards` at 1 cm steps for eight gaps on the side board (gym
bounds), not reconstructed:

| ball gap | 0.05 | 0.10 | 0.15 | 0.20 | 0.25 | 0.30 | 0.40 | 0.60 |
|---|---|---|---|---|---|---|---|---|
| largest feasible margin | 0.11 | 0.16 | 0.20 | 0.26 | 0.31 | 0.36 | 0.46 | 0.65 |

The feasible FRACTION of a draw has to be measured, not derived from that law:
the draw is 2-D (a position along the board as well as a gap), a placement near
a corner is bound by both boards, and `_along_the_boards` picks which board to
line along — none of which a 1-D estimate expresses. Measured by drawing as
`_place_at_boards` does and calling the rule on each actual (x, y), 4000 draws:

| | 1-D estimate | measured, wrong pitch | **measured, `gym_scenario()`** |
|---|---|---|---|
| `--at-boards 0.30`, margin 0.25 | 39.2% | 36.5% | **35.8%** |
| `--at-boards 0.15`, margin 0.10 | 95.2% | 96.0% | **95.2%** |
| `--at-boards 0.15`, margin 0.15 | 47.6% | 53.5% | **52.8%** |
| `--at-boards 0.15`, margin 0.25 | 0% | — | **0.00%** (4000 draws) |

The estimate errs in BOTH directions — worse at 0.30 where corners bind, better
at 0.15/0.15 where the rule can line along the board it is further from — so it
could not have been patched with a correction factor. (Independently measured
by microduck-4a at 35.8 / 95.2 / 52.8 on its own draw; agreement to a point.)

**Name the pitch a draw came from, not just "a real call".** The feasibility
LAW is pitch-invariant, so a draw built on the wrong pitch
(`make_pitch(per_side=2)`, bounds 1.7 × 1.425, instead of the gym's own
1.5 × 1.25) still gives the right law — but the FRACTIONS are not invariant,
because `_place_at_boards` picks a board in proportion to its length and
corners bind differently, and the corner SHARE is a ratio of areas and moves
further still (16.3% wrong-pitch against 19.2% right). The wrong pitch was
caught on the corner share, where the error was large, and only then found to
have shifted the fractions too — the same mistake hiding in four numbers and
visible in one. Every measurement in this section is on `gym_scenario()`
unless it says otherwise. That is the whole shape of 12q:

- the 0.15 m "null" is not a result — the knob was never once able to act;
- 0.40 is inert because it disables itself, as above;
- and the 0.30 m headline averages 39.2% acting episodes with 60.8% that ran
  the baseline path.

~~Undiluted on the measured 35.8%, 7.05% → 18.8% per acting episode, 2.66×.~~
**WITHDRAWN — see 12r.** The division assumes the per-acting effect is constant
in the margin, and the `--at-boards 0.15` arm shows it is not (a margin feasible
on 52.8% of the draw out-performed one feasible on 95.2%). The headline
1.46–1.71× stands; no per-acting figure does.

**And 19.2% of the `--at-boards 0.30` placements are corner-ish** — both board
distances under 0.30 m, drawn on the gym's own `gym_scenario()` floor
(3.5 × 3.0, bounds 1.5 × 1.25) — where margin 0.25 provably never fires — so the corner limit
below is not about some other scenario, it is about a sixth of the episodes in
this very result.

#### Pitch-size invariant, and structurally dead in corners

Both by direct call (1 cm sweep, bounds read from `make_pitch` and the
`floor/2 − 0.25` the brain uses — microduck-4a, verified independently here):

| largest feasible margin | gap 0.05 | 0.10 | 0.15 | 0.20 | 0.30 |
|---|---|---|---|---|---|
| mid-board, 1v1 / 2v2 / 3v3 (identical) | 0.11 | 0.16 | 0.20 | 0.26 | 0.36 |
| **corner**, 1v1 / 2v2 / 3v3 (identical) | **none** | 0.02 | 0.06 | 0.11 | 0.21 |

**Pitch size does not matter.** The law depends on the ball's gap from the NEAR
board and the spot's offset from the ball, not on where the other boards are,
so a margin tuned in the gym cannot go inert on 3v3 bounds. That ship criterion
is satisfied rather than deferred.

**Corners are a different law** — roughly `margin ≤ gap − 0.09` against
`gap + 0.05` mid-board — and at 0.05 m into a corner **no margin is feasible at
all**. The candidate 0.25 would need the ball 0.34 m off BOTH boards, which is
not a corner. So this rule helps along a board and **has never once fired in a
corner**, which matters because 12m measured corners as a real population.
Anyone reading "the boards line works" should not expect it there.

This argues for BUILDING the knob split independently of any battery: a corner
needs a large trigger (the ball is deep inside any sensible radius) and a tiny
clearance (or nothing is feasible), which is exactly the pair one number cannot
be. **It does not argue that acting in a corner helps** — that is the
"the rule can now act" versus "the rule improves anything" gap this whole
section is about, and the existing arms cannot close it, since 0.25 never fires
in the corner population. Three separate fates: the split is NECESSARY (settled
by the table), the split is worth BUILDING (follows), and acting in corners
PAYS (open, wants an `--at-corners` arm).

**This caveat is stronger than the MDE/firing-rate bound's, and should not be
discounted the same way.** There, confinement is an assumption: an effect may
persist past the ticks it fires on. Here it is not assumed — on a non-acting
episode `_along_the_boards` returns None and the code runs the *identical*
baseline path, and the BROKEN flag at `board_margin` 0.40 (baseline episode for
episode) is the proof rather than the argument.

**It has a working range, and the reason is a FLAW IN THE KNOB rather than a
fact about walls** (microduck-4a's diagnosis, confirmed against the code
2026-09-09). `board_margin` does two opposing jobs: it is the TRIGGER (the
boards line is used when the normal spot is `not _clear_of_boards`, so a bigger
margin fires more often — controllers.py:2416) and it is also the FEASIBILITY
test the replacement spot must itself pass (`_along_the_boards` returns None
when neither foot's spot is ≥ `board_margin` from every bound). Forcing them
equal means asking to switch early *and* demanding more room, which are
opposite wants. Asked of the code directly, for a ball at a given gap from a
side board:

| margin | ball 0.30 m off a board | ball 0.15 m off |
|---|---|---|
| 0.15 | never triggers | triggers, **spot found** |
| 0.25 | triggers, spot found | triggers, **no spot** |
| 0.40 | triggers, **no spot** | triggers, no spot |

So **0.40 is inert because it disables itself at the top of its range**, and
the 0.15 m null is the feasibility half refusing a spot that a SMALLER margin
finds — not the geometry running out. (An earlier draft of this section said
"even the along-the-board spot is inside the wall that close". That was wrong
and is corrected here rather than silently edited.) The fix is to split the
knob — `board_trigger` large, `board_clear` small — and the prediction that
small margins work at 0.15 m is under test before that code is written.

And **0.40 is inert at both distances**:
the gym's broken-versus-null flag (microduck-4a, c4c8138 lineage) caught it
reproducing the baseline episode for episode and refused to call it a null.
Sweeping only the larger value would have published a clean null on a knob that
never reached its code — the same failure class as the 11b null it was
re-testing.

**Ship criterion added:** the feasible window depends on `kick_ahead` AND on
the pitch bounds, so a margin tuned in the 3.0 × 2.5 gym can ship INERT on
another pitch — this same failure, silently, on a knob believed to be on.
Anything that ships must be checked on the 2v2 and 3v3 bounds.

Status: **discovery block only.** Fresh seeds (100–107), the `--opponents 1`
match-realistic arm and the small-margin prediction test are running; 12p's curve is one duck, and a real ball at a
board usually has a body beside it, which can only suppress swings further. Not
shipped until both land.


### 12r. The boards line REPLICATES and survives an opponent — but the registered prediction FAILED and the 2.66× is withdrawn (2026-09-09)

All on `gym_scenario()`, 2240 episodes an arm, swings per episode:

| | baseline | `board_margin` | | |
|---|---|---|---|---|
| **discovery**, seeds 0–7, `--at-boards 0.30` | 7.05% | 11.25% (0.25) | 1.59× | p < 0.0001 |
| **fresh block**, seeds 100–107, same | 7.41% | 10.80% (0.25) | **1.46×** | **p = 0.0001** |
| **with one opponent**, seeds 0–7, same | 4.38% | 7.50% (0.25) | **1.71×** | **p < 0.0001** |

**It replicates on fresh seeds and it survives a contested ball.** The opponent
arm is the one that matters for play: a second duck suppresses swings overall
(baseline 7.05% → 4.38%, as 12p predicted it would), and the rule's effect is
undiminished — if anything larger — inside that harder population.

**And the pre-registered prediction failed.** At `--at-boards 0.15`, where
`board_margin` 0.25 is infeasible on 100% of the draw, both small margins act,
which confirms the mechanism's SIGN:

| margin | feasible on | swings/episode | |
|---|---|---|---|
| 0.10 | 95.2% | 3.66% → 6.16% | 1.68×, p = 0.0001 |
| 0.15 | 52.8% | 3.66% → 6.70% | 1.83×, p < 0.0001 |

But the registered ratio was **1.80:1** (from the feasible fractions) and the
observed ratio is **0.82:1** — the margin feasible on half the draw produced
slightly MORE total effect than the one feasible on nearly all of it.

**Per the falsifier as registered, the confinement assumption is wrong and the
2.66× per-acting figure is WITHDRAWN.** What fails is not the code-path
identity (that is established by grep: `board_margin` has three sites, the
fall-through returns the identical shipped spot). It is the assumption that the
per-acting effect is CONSTANT in the margin. It is not: a larger clearance
yields a spot further off the wall, which is evidently a better spot, so total
effect = fraction × per-acting-effect with the second term rising as the first
falls. Dividing a headline by a feasible fraction is therefore invalid here,
and the same objection applies to the MDE ÷ firing-rate bound whenever a knob's
value changes the quality of what it does and not only how often it does it.

**What stands:** the measured 1.46–1.71× across three independent populations.
**What goes:** the 2.66× per-acting figure, and any per-acting number derived
by dividing by a feasibility fraction.

The prediction was registered before the run with an explicit falsifier, and it
lost. That is the falsifier working, not a setback — an unfalsifiable version
of it would have let the 2.66× stand.


### 12s. The referee moved the ball and told nobody — FIXED, partially (2026-09-09)

Jonathan, watching /sim: *"does this teleporting also really mess up the
positioning?"* It does. `World._check_ball_out` moves the ball up to
`ball_out_in` 0.45 m and zeroes its qvel, but `ball_outs` was a counter **no
harness watched** — `grep ball_outs` across `brain/` returned nothing. Every
harness watches `goal_seq` and calls `kickoff_brains`; the throw-in was built
as a second teleport path with no notification. A duck that saw the ball on
both sides simply differences the two positions, and `Team.vel_max` 4.0 does
not reject it: 0.45 m over a 0.15–1.0 s baseline is 0.45–3.0 m/s.

Measured against a matched control window (random seconds with no throw-in),
4 seeds × 300 s of 2v2:

| in the second after a throw-in | before | control | **after the fix** |
|---|---|---|---|
| prediction > 0.30 m from the true ball | 49.1% | 4.6% | 47.3% |
| **prediction > 0.60 m off** | 4.9% | 3.0% | **0.0%** |
| tracker reads the parked ball as moving > 0.3 m/s | 33.3% | 19.9% | **27.4%** |
| **board publishes an invented ball velocity** | 15.2% | 9.7% | **7.5%** |

**Shipped:** `World.ball_out_seq` beside `goal_seq`, and
`brain/team.throw_in_brains` — `Tracker.disturb(cls)` unconditionally (at a
throw-in every belief is stale; note the selective form needs BOTH `xy` and a
non-zero `radius`, or it silently marks everything), the ball tracks'
`vel/vel_hits/vel_sig` zeroed, and `Team.throw_in` clearing the board's
published velocity and its fix history. Wired into `eval_pitch`,
`eval_striker` and `world_server` at the three sites that already watch
`goal_seq`. Deliberately NOT `kickoff_brains`: a throw-in is not a restart and
must not reset roles, plans or counters — locked by a test.

The velocity zero lives in the handler, not in `disturb`, because `disturb`
has three live callers on the shipped path (a duck near the ball, a push, a
kick) and those are NUDGES where the remembered velocity is still about right.
A teleport invalidates both position and velocity, and only the throw-in knows
that. `Track.predict` never consults `rest_block`, so a test on the at-rest
flag alone would pass while the coasting continued; the test asserts
`predict()` no longer moves the ball.

**What is NOT fixed, and why the headline barely moved.** Zeroing the velocity
stops the invented motion, which is why the > 0.60 m errors vanish. It does
nothing about the remembered POSITION, which after a 0.45 m teleport is simply
wrong — so "> 0.30 m off" stays at 47%. The honest fix is to expire the
position too, so a duck knows it does not know.

**A first attempt at measuring that hit the selection trap (12r's sixth
shape).** Expiring `xy` reported 63% — *worse* — but the metric skips ticks
where the track has no position, so the change shrinks its own denominator
(1005 duck-ticks → 451) and the survivors are a biased subset. Not evidence of
anything. Measuring it needs a denominator fixed across arms (all duck-ticks in
the window, with "no prediction" as its own outcome). Left open rather than
guessed at.

### 12t. `board_margin` at match level — the metric registered BEFORE the read (2026-09-09)

The gym says the along-the-boards line raises swings/episode by 1.46-1.71x on
three separate blocks (12q, 12r). The remaining ship criterion is whether that
survives a whole match, where a swing has to compete with everything else a
duck does. Three arms, `eval-pitch --seeds 24 --seconds 300 --per-side 2
--ball-out-s 5` with the get-up on: `board_margin` 0.0 (control), 0.25, 0.10.
Both teams carry the knob.

**Registered at 13:16:38Z, while the arms stood at 8 of 24 rows and were still
running** (`scratchpad/prereg-board-margin.txt`, sha256 550f5d29...). Written
out here because a prediction that lives only in a scratchpad is a prediction
nobody can check:

> PRIMARY METRIC: possession (s/min, SUMMED over the pair by compare_pitch).
> Chosen because (a) the power audit says only possession (~10% MDE), spread
> (9%) and depth (5%) can resolve at 24 seeds, and (b) possession is the only
> one of those three that board_margin has a mechanism for: the knob is meant
> to turn dead ball at the boards into contested ball.
>
> DECISION RULE: two contrasts (m25 vs off, m10 vs off), paired by seed.
> Bonferroni over the two: an arm SHIPS on possession only at p < 0.025.
> A non-significant possession row is a null only if its MDE is tight; else
> NO RESULT. Direction predicted: possession UP.
>
> EVERYTHING ELSE IS EXPLORATORY. compare_pitch prints 9 metrics x 3 arms =
> 27 reads; P(some p<0.05 by chance) = 37%. Non-primary rows are quoted with
> that count attached and never called an effect.
>
> PRIOR: I expect the match effect to be much SMALLER than the gym's
> 1.46-1.71x, and NO RESULT on kicks, falls, goals and ballAdvance because 24
> seeds cannot resolve them. If possession does not move at p < 0.025,
> `board_margin` does NOT ship, and the gym result stands as a gym result only.

One check had to come before naming the metric, because `eval-pitch` gives
**both** sides the same knob: a metric that is a SHARE between the teams cannot
move under a symmetric change, however well the change works. `compare_pitch`
takes possession as `("possession", "sum", "s/min")` — total time the ball is
in someone's control, not who has it — so it is free to rise for both teams at
once. Had it been a share, spread would have had to be the primary instead.

**RESULT: the primary is a NULL, and a tight one. `board_margin` does not ship.**

> **Read 12v first if you are arriving at this item cold.** The mechanism found
> after these arms landed reframes what they were testing: the gate is whether
> the plan put the kick spot where the duck's body can stand, `board_margin =
> 0.10` is the duck's footprint rather than a tuning value, and the shipped 0.0
> is an unphysical default rather than a neutral one. The null below is still
> the null — the knob does not ship — but the reason is sharper than "it does
> not pay".

Read through the registered lens first and alone (`scratchpad/read_primary.sh`
prints the possession row and withholds the other eight, so the primary was
seen before anything that could have tempted a different story):

| possession, s/min | off | arm | delta | ±MDE | MDE% | p | verdict |
|---|---|---|---|---|---|---|---|
| 0.25 (m25) | 36.268 | 37.118 | **+0.850** | 1.677 | **5%** | 0.305 | null |
| 0.10 (m10) | 36.268 | 36.203 | **−0.065** | 1.911 | **5%** | 0.944 | null |

Neither contrast comes near the registered p < 0.025, and neither is a NO
RESULT dodge: the MDE is **5% of baseline**, twice as tight as the ~10% the
power audit predicted at 24 seeds, so the instrument was good enough to have
seen an effect half the size of the smallest one worth having. This is a real
null, not an underpowered one. By the rule registered before the read, the knob
does NOT ship, and 12q/12r's 1.46-1.71x stands **as a gym result only**.

**A first draft of this item said the match proves "a lever that moves,
attached to nothing". That was written before the kick counts were read and it
is wrong** — the arithmetic below says the match cannot support that claim, and
it is retracted here rather than left standing.

The exploratory table is 18 rows with no p under 0.16, but the events block
carries the one number that bears on the mechanism: **kicks are flat**, 168 →
161 (m25) and 168 → 160 (m10), where the gym says 1.46-1.71x. Per seed that is
7.00 kicks baseline and −0.29 / −0.33 observed, against a kick MDE of ±1.24
(18%) and ±1.00 (14%). Two hypotheses, and this battery separates them:

| what the gym effect would have to be | predicted | vs MDE |
|---|---|---|
| 1.46x on **all** kick chances | +3.22 kicks/seed | far above ±1.24 — **EXCLUDED** |
| 1.46x confined to the boards | depends on how often the knob fires | see below |

So the **broad** reading is refuted: `board_margin` does not raise kicking
across the board, and if 12q/12r had implied it did, this rules that out. The
**scoped** reading is not refuted, because it is arithmetically invisible here.
Solving for the firing rate this battery could have seen:

- at the gym's 1.46x, the knob must fire on **more than 39%** (m25) or **31%**
  (m10) of kick chances to clear the MDE;
- at the most favourable 1.71x, **more than 25%** (m25) or **20%** (m10).

The corner census puts the boards-adjacent share well under those thresholds,
so a real, gym-sized, boards-confined effect would look exactly like this
table. **The match is not a test of the scoped claim and never could have
been** — which is a limitation of the harness, not a rescue of the knob.

What that leaves: the knob ships OFF, by the rule registered before the read
(possession null at 5% MDE, p 0.305 / 0.944, nowhere near p < 0.025) and
because no match-level benefit has been demonstrated. But the open question is
now sharp and cheap, and it is a **firing-rate census**, not another battery:
count what fraction of match kick plans occur within `margin` of a board. If
that fraction is under ~25%, no `eval-pitch` battery of any affordable size can
settle this, and the question belongs in the gym with the census beside it.
That is the same instrument 12r needed and the same one `--at-corners` is
building.

**AND THEN THE CENSUS TURNED OUT TO BE UNNECESSARY, because the question is
geometry and not statistics.** `scripts/probe_board_geometry.py` calls
`Chase._along_the_boards` and `_clear_of_boards` directly — no simulation, a
few thousand trig calls — and answers "could the knob ever act here?" which no
battery can separate from "did it pay?".

First, the feasibility law of 12q is exact and its constant was never
empirical. A legal along-the-boards spot exists iff

    gap >= board_margin - kick_side          (kick_side = 0.06; kick_ahead does not enter)

a clean step at every margin swept (0.10/0.15/0.20/0.25/0.30). What 12q fitted
as `margin <= gap + 0.05` is `kick_side` and should be written that way.

Second, and this is what explains the arms — the branch in `_hold_target` runs
only when the DEFAULT spot is inside the margin, and helps only if a legal
alternative exists. **Both tests use the same `board_margin`, so raising it
makes the knob fire more and fail more.** Sweeping aim direction uniformly
(an assumption, and the only one here):

| gap | m=0.10 act / wasted | m=0.25 act / wasted |
|---|---|---|
| 0.04 | **69.4%** / 0.0% | 0.0% / **100.0%** |
| 0.08 | 55.6% / 0.0% | 0.0% / **100.0%** |
| 0.15 | 33.3% / 0.0% | 0.0% / **100.0%** |
| 0.20 | 0.0% / 0.0% | **66.7%** / 0.0% |
| 0.30 | 0.0% / 0.0% | 33.3% / 0.0% |
| 0.40 | 0.0% / 0.0% | 0.0% / 0.0% |

**`board_margin = 0.25` cannot act anywhere below gap 0.20 — it triggers and
falls through 100% of the time.** For a ball within 19 cm of a board it is a
no-op that still runs the branch: the same self-disabling behaviour 12q found
at 0.40, reached at a value we were treating as live. So **the m25 arm above
was testing a knob that structurally cannot reach the region the whole idea is
about, and its null is expected rather than informative.** The arm that carried
the hypothesis was m10, which acts across gap 0.04-0.16.

**Third — and CORRECTED, because the first version of this paragraph reached
for the weaker of two instruments.** It originally said the legal-spot
explanation for 12p's cliff was "not the cause", arguing from the m10 *match*
null. The other session pointed out that the direct test was a gym arm already
on disk: `--at-boards 0.15` draws gap in [0.045, 0.15], which IS the collapse
region, and there m10 gave **3.66% → 6.16%, 1.68x, p = 0.0001**. In the heart
of the cliff, supplying a legal spot moves the swing rate by an amount that is
not in doubt. So the legal spot is not unsupported. It is **supported and
small**:

| | swing rate |
|---|---|
| collapse region (gap ≤ 0.15), shipped | 3.66% |
| same, `board_margin = 0.10` | **6.16%** (p = 0.0001) |
| far field (gap 0.65-1.01) | 79.0% |
| **the cliff to explain** | **75.34 points** |
| **the knob recovers** | **2.50 points = 3.3% of it** |

**A legal spot accounts for about 3% of the collapse; the other ~97% (72.8
points) is downstream** — reaching the spot, seeing the ball at that range, or
the swing itself. The operational conclusion is unchanged (do not spend a
battery on the legal-spot hypothesis) but the honest sentence is "a real but
~3% effect", NOT "not the cause" — the latter invites someone to stop measuring
spot availability altogether, and it is a live 3%.

Why the first version went wrong is worth more than the correction: **the match
arm was blind here (18% MDE on kicks, firing rate uncounted) while the gym arm
was decisive (p = 0.0001) — same knob, same question, two instruments — and the
weaker one was reached for because it was the more recent.** That is the
per-metric MDE lesson one level up, applied to choosing an instrument rather
than to reading one.

**Fourth, the corners are the case that differs, and it changes what a census
must count.** Sweeping each board type (probe section 3): the side board, their
end and our end all behave identically — act 69/50/33% at gap 0.04/0.10/0.15
with **zero** waste at m=0.10. The corners do not: there the branch **wastes as
often as it acts** (47/47, 38/38, 29/29), and at m=0.25 every corner ball inside
gap 0.20 is 100% waste. So a census that counts *plans within margin of a board*
counts fall-throughs as reach and overstates the knob badly — at m=0.25 below
gap 0.20 it would report the knob firing constantly when not one of those
firings changes anything. **The census must count the ACT set, not the trigger
set, and must report corners separately from the flat boards.**

The lesson worth keeping is the cheapness. Three 20-minute match arms, a gym
block, and a planned census were all pointed at a question that a pure-geometry
probe answers in under a second, because the knob's reachable set is a property
of the code and not of the football. **Ask what a knob CAN do before paying to
find out whether it does.**

### 12u. The camera ledger, confirmatory block — the metric registered BEFORE launch (2026-09-09)

12n read nine metrics across three camera arms with **none named in advance**,
and its own multiplicity note says possession on the crop (p 0.007) and on the
replacement (0.011) do not survive the family-wise correction. They are recorded
there as *suggestive*. This block exists for one purpose: to promote that claim
or kill it. It is the confirmatory run 12n says is owed.

**Registered at 13:39:43Z, BEFORE the arms were launched**
(`scratchpad/prereg-camera-confirm.txt`, sha256 76528d0e...):

> PRIMARY: possession (s/min, summed over the pair). Fresh seeds **100-123** —
> the discovery block was 0-23, and a confirmatory run on the discovery seeds
> would prove nothing.
>
> DIRECTIONAL PREDICTIONS from the discovery block: crop vs sim possession
> **DOWN** (discovery −3.73), replacement vs sim possession **UP** (+3.11).
>
> DECISION RULE: two contrasts, paired by seed, **ONE-SIDED in the predicted
> direction**, Bonferroni over the two → alpha = 0.025 each. A contrast
> REPLICATES only at one-sided p < 0.025 in the predicted direction. The
> two-sided p is reported too, and a significant effect in the OPPOSITE
> direction is a **FAILED REPLICATION**, never a null.
>
> POWER: possession ran at 5% MDE on 24 seeds in 12t; the discovery effects are
> ~10% and ~9% of baseline. If they are real this block should see them
> comfortably, so a null here is informative rather than a shrug.
>
> PRIOR: I expect the crop contrast to replicate — its ballAdvance row survived
> every threshold in 12n, so something real is there — and I am genuinely
> unsure about the replacement. If neither replicates, 12n's possession rows
> come out of the roadmap as a finding and stay only as a caveat.

The arms: `cam-sim` (shipped 62 × 48, 320 px), `cam-crop`
(`fov_h_deg=39,fov_v_deg=22.5`), `cam-repl`
(`fov_h_deg=116,fov_v_deg=60,px_h=640`), 24 seeds × 300 s of 2v2,
`--ball-out-s 5`, get-up on.

**Positive control run before trusting a single row**, on the lesson that a
harness which cannot see the effect reports a clean zero: `MICRODUCK_CAMERA` is
read on the world path (`world/arena.py:476`, `spec=DetectorSpec.from_env()`),
a mistyped field raises rather than being ignored, and the three frustums admit
**9.3% / 31.0% / 79.3%** of a fixed grid of sample directions. The knob acts.

**RESULT: BOTH CONTRASTS REPLICATE. 12n's possession claim is promoted from
suggestive to established.**

Read through the registered lens alone first
(`scratchpad/read_camera_primary.sh` prints the possession row, converts the
two-sided p to the registered one-sided one, and withholds the other eight):

| contrast | predicted | discovery (seeds 0-23) | **confirmatory (seeds 100-123)** | one-sided p | verdict |
|---|---|---|---|---|---|
| crop vs sim | DOWN | −3.73 s/min | **−4.843** (37.109 → 32.265) | **< 0.0001** | **REPLICATES** |
| replacement vs sim | UP | +3.11 s/min | **+3.082** (37.109 → 40.191) | **0.0015** | **REPLICATES** |

Both clear the registered alpha = 0.025 with room, on **fresh seeds**, in the
**direction named before the run**, at a 5% MDE. The replacement's +3.082
against the discovery block's +3.11 is agreement to 0.03 s/min on independent
seeds — closer than the measurement deserves, and the crop's effect came back
*larger* rather than shrinking, which is the opposite of what regression to the
mean does to a claim that was only ever noise.

This is the confirmatory run 12n said it was owed, and it passes. The camera's
effect on possession is now a finding rather than a caveat: **the crop the robot
runs today costs ~4.8 s/min of possession against the sim's assumed camera, and
the calibrated replacement buys ~3.1 s/min.** Note the crop contrast would
survive even the harshest correction applied anywhere in this document
(0.05/9 = 0.0056) without the pre-registration; the replacement needed the
registration to be quotable, and now has it.

**THE EXPLORATORY EIGHT DID SOMETHING BETTER THAN AGREE — THEY TESTED THE
MULTIPLICITY CORRECTION ITSELF.** 12n's caveat sorted its rows into strong
(possession, `ballAdvance`) and marginal-and-not-to-be-leaned-on (crowd 0.039,
falls 0.050). The confirmatory block reproduces **exactly that split**:

| 12n row | discovery | confirmatory | |
|---|---|---|---|
| crop `ballAdvance` | −0.377 (p 0.000) | **−0.356 (p 0.000)** | **replicates** |
| replacement kicks | 143 → 84 (−41%) | 163 → 94 (**−42%**) | **replicates** |
| replacement crowd | +0.040 (p 0.039) | +0.017 (p 0.341) | **does not** |
| replacement falls | 8 → 1 events (p 0.050) | 4 → 3, **NO RESULT** (MDE 139%) | **does not** |

The two rows the correction said to trust came back at −0.356 against −0.377 and
−42% against −41%. **The two it said not to lean on evaporated.** 12n called the
falls row "the one claim not to lean on" on the strength of an 87% MDE, and it
was right — 4 → 3 here, needing 4661 seeds to resolve. That is the family-wise
caveat earning its keep on real data rather than in principle, and it is the
best argument in this document for registering a primary: the correction did not
merely make us cautious, **it correctly predicted which findings would survive.**

One new exploratory row worth naming with its read count attached (1 of 27): the
crop's `goals` fell 26 → 12, p 0.013 at a 41% MDE. Suggestive and unquotable as
an effect — `goals` needs ~581 seeds here — but it points the same way as
everything else the crop touches.

**What the 12n/12u pair now supports, and its limit.** The camera drives
possession and ball advance, monotonically in team blindness (85.2% crop /
61.3% sim / 42.7% replacement). It does **not** support a falls claim. And the
replacement still takes ~42% fewer kicks — replicated now, and still
unexplained, with `kick_ahead_max` the leading suspect. A wider camera is not a
free win, and 12n's original framing of that trade survives intact.

**THE ~42% KICK DROP IS PROBABLY THE GATE WORKING, NOT A COST — a mechanism
read out of the code FIRST, then a prediction, then the numbers.** Flagged
exploratory throughout: these are derived ratios, not the registered primary.

`_too_far` (controllers.py:3154) aborts a swing when the predicted ball is more
than `kick_ahead_max` = 0.15 m ahead. Its docstring carries the key clause:
*"False when the knob is off **or nothing fresh has been seen**."* **The gate is
disabled whenever the duck has no live track.** So a blinder camera loses the
ball, `predicted` goes None, the gate silently switches itself off, and the duck
swings anyway — at a ball its plan has lost. A camera that keeps the ball in
view keeps the gate armed, and the gate declines those swings.

That predicts, before looking: the better camera should take **fewer** kicks and
each kick should be **worth more**. Metres of ball advance per kick taken:

| camera | team blindness | m of advance per kick |
|---|---|---|
| crop (39 × 22.5) | 85.2% | **0.747** |
| sim (62 × 48) | 61.3% | 1.055 |
| replacement (116 × 60 @640) | 42.7% | **1.785** |

**Monotone in camera quality, and the replacement is +0.730 m/kick against the
sim (z +2.41).** It takes 42% fewer kicks worth 69% more each, which is why its
total `ballAdvance` came back flat (+0.002, p 0.980) while possession rose. The
same construction in possession terms: the replacement spends 66.0 s of
possession per kick against the sim's 36.0 (z +2.59) — it holds the ball and
declines to swing — while the crop is unchanged at 33.3 (z −0.36).

Why this is worth more than its p values (z 2.41 and 2.59 would not survive a
strict family-wise threshold, and these are two of several ratios I could have
formed): the direction was **derived from the code before it was computed**, and
it is a three-arm dose-response in the predicted order, which no single p value
captures. It is still hypothesis-generating, not established.

**The falsifier, and it is cheap.** If this is right, running the replacement
camera with `kick_ahead_max = 0` (gate off) should push kicks back up toward the
sim's count AND push advance-per-kick back down. Two `kick_gym` arms at 0.8
CPU-s a swing settles it. If kicks do *not* recover with the gate off, the
explanation is wrong and the kick drop is something else — detection latency or
approach length are the next suspects. **Registering that prediction here, in
advance, because that is the only thing that makes the check worth running.**

**THE FALSIFIER PASSED, DECISIVELY, IN THE REGISTERED DIRECTION (2026-09-09).**
Run by the other session, 2240 episodes an arm, `--at-boards 1.00`, `gate_on=`
against `gate_off=kick_ahead_max=0`:

| | swings | whiff | **connected kicks** |
|---|---|---|---|
| gate ON (shipped 0.15) | 846 | **35%** | **550** |
| gate OFF (disabled) | 1044 | **60%** | **418** |

Disabling the gate: swings **+23%**, whiff **+25 points** (p < 0.0001, MDE 5%).
So the gate **buys +32% more connected kicks from 19% FEWER swings** — exactly
the registered prediction, fewer kicks each worth more. **And it is not a ratio
artefact:** the absolute count of connected kicks is higher with the gate armed,
550 against 418. A duck swinging blind is not trading quantity for quality, it
is losing on both.

**So the replacement's ~42% kick drop moves from the cost column to the benefit
column, and 12n's "a wider camera is not a free win" was too pessimistic.**
`_too_far` disables itself when `predicted` is None; a blind camera swings with
the safety off; the replacement keeps the ball in view, keeps the gate armed,
and the gate declines exactly the swings that would have missed.

The boundary the other session put on it, kept: this is the **gym** at
`--at-boards 1.00`, not a match, and the gym's blindness profile is not the
crop's. The *mechanism* is confirmed and its *direction* matches the three-arm
dose-response above, but the "69% more advance per kick" figure is still the
match arms' and is not reproduced here. **Two independent routes agreeing on
sign and mechanism, not one number confirmed twice.**


### 12v. The gate is the SPOT, not the ball — and `board_margin` was never a tuning knob (2026-09-09)

The whole boards line was framed as "should we bias the kick line near a
board". The measurement says the question was wrong. **The gate is whether the
planner put the kick spot somewhere the duck's body can physically be.**

**First, what was already known, because this is a confirmation and not a
discovery.** `controllers.py:544-556` (roadmap 11b) already documents it: *"A
kick spot laid closer than this to the boards — inside them, or inside the
walker's own `tof_stop` of them — is never reached: the line-up stands against
the wall for `lineup_s` and times out (measured: the ball is at the boards 72%
of a 3v3 run and 0 of 36 kicks were taken there, every boards line-up a
timeout)."* The gate is `lineup_tol = 0.03` — the trunk must reach within 3 cm
of the planned spot to swing — and `lineup_s = 4.0` ends the attempt. What
follows is that behaviour measured at 2240 episodes with a denominator, which
11b's 36 kicks could not support.

**The other session's mechanism re-bin** (2239 of 2240 episodes carrying a
latched plan, the spot recorded on no-swing rows — the fix for a selection
error one level down from the `place_board` one):

| where the PLAN put the spot | episodes | swings | rate |
|---|---|---|---|
| inside the board (< 0) | 165 | **0** | **0%** |
| 0.000 - 0.050 m | 187 | 0 | 0% |
| 0.050 - 0.095 m | 197 | 2 | 1% |
| 0.095 - 0.150 m | 269 | 10 | 4% |
| 0.150 - 0.250 m | 299 | 41 | 14% |
| 0.250 - 0.400 m | 328 | 134 | 41% |
| 0.400 m and beyond | 794 | 672 | **85%** |

Split at **0.095 m**, the body figure the code derives itself: **549 episodes
where the body cannot stand, 0.4% swing; 1690 where it can, 50.7% — a factor of
127.**

**The discriminating test, and it is the reason this is causal and not a
correlation.** Fix the spot's distance and the *ball's* distance stops
mattering in the collapse region:

| spot→board | ball 0.04-0.20 | ball 0.20-0.45 | ball 0.45+ |
|---|---|---|---|
| 0.000-0.095 | **1%** (206) | **0%** (140) | **0%** (38) |
| 0.095-0.250 | 6% (249) | 13% (242) | 6% (77) |
| 0.250-0.450 | — | 45% (220) | 57% (178) |
| 0.450+ | 58% (26) | 63% (84) | 90% (599) |

In the top row the ball's position adds nothing — 1%, 0%, 0%. **The ball's
distance predicts the swing only THROUGH the spot**, exactly where the 12p
cliff lives. Higher up the table both still matter, so the spot is not the whole
story above the body radius (approach length is the obvious remaining
candidate); that is stated rather than papered over.

**Why corners are ~50x worse than a flat board at the same distance
(`scripts/probe_board_geometry.py`).** The shipped brain runs
`board_margin = 0.0`, so `_clear_of_boards` accepts any spot inside the pitch
line and ignores the body entirely. Measuring the duck's footprint from the
compiled MJCF — 70 collision geoms, **max horizontal extent 0.116 m** from the
trunk centre, 90th percentile **0.091 m** — and asking how often the planned
spot is unreachable:

| gap | flat board (r .116 / .091) | CORNER (r .116 / .091) |
|---|---|---|
| 0.04 | 77.8% / 66.7% | **100.0%** / 91.7% |
| 0.10 | 55.6% / 47.2% | 80.6% / 72.2% |
| 0.15 | 38.9% / 30.6% | 63.9% / 55.6% |
| 0.20 | 18.1% / 0.0% | 36.1% / 0.0% |
| 0.25 | 0.0% | 0.0% |

**In a corner at gap 0.04 it is 100% — every aim direction, both feet.** The
constraint is *conjunctive*: the spot must clear two walls at once, so the
unreachable fraction SATURATES where on a flat board it merely rises. Same
gate, two populations: one it sometimes lets through, one it never does. That
is the shape of the other session's corner arm — **2 swings in 2240 episodes,
0.09%**, against 79% in the far field.

**Two numbers that agree by different routes.** The code derives the body
clearance as *"a ball against the wall (radius 0.035) with the `kick_side`
offset (0.06) puts the trunk 0.095 m from it"*; the compiled model gives 0.091
at the 90th percentile. **Agreement to 4 mm, from arithmetic and from geometry
independently.**

**WHICH REFRAMES THE KNOB.** `board_margin = 0.10` sits on the duck's body. It
is not a tuning value, and **the shipped 0.0 is not a neutral default — it is
an unphysical one**: it permits the planner to choose spots inside a wall, and
165 episodes above did exactly that and swung zero times. The question was never
"should we bias the kick line"; it is **"should the planner be allowed to plan
somewhere the robot cannot stand".**

**What this does NOT license.** Turning `board_margin` up is still not the fix,
and 12t's null stands: the knob conflates two jobs — *rejecting* an unreachable
spot (the trigger) and *escaping* along the wall (the remedy) — behind one
number, so raising it makes both fire more and fail more. In a corner the escape
has nowhere to go: at 0.25 every corner ball inside gap 0.20 is a guaranteed
fall-through, and at 0.10 the corner branch wastes as often as it acts. **A
corner needs a different move, because in a corner there is no other side to
stand on.**

**The next item is a design question, not a sweep.** Reachability belongs in
`kickselect` as a CONSTRAINT on candidate spots — the selector already ranks by
p_goal then value, and it can rank only reachable spots — rather than as a
post-hoc rescue in `_hold_target` after a spot has been chosen. That separates
rejection from escape, gives the corner case somewhere to fall back to (a push,
a different aim, an approach from the open side), and would be measured on
`kick_gym` where a swing costs 0.8 CPU-s. Not attempted here; written down so
the next person starts from the mechanism rather than the knob.

**THE CENSUS SETTLES THE BOARDS QUESTION, AND A HARD GEOMETRIC FLOOR FALLS OUT
OF IT (2026-09-09, added after 12t/12u).** The other session's firing-rate
census — 4 seeds × 180 s, **144 000 duck-ticks, 14 184 kick plans**, act /
wasted / clear per margin, corners split from flat boards, every kick plan in
the denominator:

| | FLAT boards (13 704 plans, 97%) | | | CORNERS (480 plans, 3%) | | |
|---|---|---|---|---|---|---|
| margin | act | wasted | clear | act | wasted | clear |
| 0.10 | 5% | 1% | 94% | **83%** | 3% | 14% |
| 0.15 | 3% | 5% | 92% | 66% | 20% | 13% |
| 0.20 | 5% | 9% | 87% | 13% | 74% | 12% |
| 0.25 | 5% | 13% | 82% | **1%** | **86%** | 12% |

**ACT over all kick plans: 7.9% at m=0.10 and 4.6% at m=0.25**, against the 31%
and 39% that 12t's MDE-run-backwards said were needed to see the gym effect in a
match. Short by 3.9x and 8.5x, and seeds scale as the square: **370 seeds (8.8
CPU-hours an arm) at 0.10, and 1 725 seeds (41 CPU-hours) at 0.25.** No
affordable `eval-pitch` could ever have settled the scoped claim. 12t's null was
never the test it looked like, and "the boards line belongs in the gym with the
census beside it" is a conclusion rather than an excuse.

Two things neither of us predicted. The structural no-op is confirmed exactly
where the geometry put it — **at m=0.25 corners are 1% act and 86% wasted**, the
branch running and falling through almost every time. And the **inversion**:
corners are 3% of plans but the knob reaches **83%** of them at m=0.10, against
5% of flat-board plans. **The knob has its best access to precisely the
population where the geometry saturates and every aim direction is
unreachable** — its reach is concentrated where it can do least.

**And the floor.** A legal along-the-boards spot needs `gap ≥ margin −
kick_side`; a *usable* spot needs `margin ≥ the body's own extent`. Substituting:

    gap  >=  body - kick_side  =  0.129 - 0.060  =  0.069 m

**A ball closer than ~6.8 cm to a board — about 2 ball radii — has no legal
kick spot at ANY margin.**

> **Corrected from 5.6 cm, and my caveat pointed the wrong way.** I first used
> the nominal-pose extent (0.116 m) and warned the floor might be a *range* as
> low as 3.1 cm, because a 90th percentile over the duck's geoms is 0.091. That
> percentile is meaningless here: it is taken **over geoms**, and a body's
> footprint is set by its **outermost** geom, so the max is the physical
> quantity and a percentile over parts is not a smaller body. The variation
> worth checking was across **poses**, and measuring that (the other session
> first, reproduced here independently with the walk policy driving) gives
> **0.1292 m max over 499 walking ticks with 0.4 mm of spread** — theirs 0.1275
> with 0.5 mm. A walking duck is **wider** than a standing one by ~13 mm, so the
> floor goes UP, and it is a constant rather than a range. One caveat kept from
> them: `geom_rbound` is a bounding-*sphere* radius, so 0.129 is an upper bound
> on the true horizontal extent; the pose-to-pose *difference* is the
> trustworthy part. My own first attempt at this measured 0.215 m and was
> invalid — I built the World with `infer_for=None`, so no policy was driving
> the ducks and I measured a limp one sagging; the 98 mm spread was the tell. The probe confirms it: at m=0.116 the act region
covers gap 0.06-0.20 and gap 0.04 is 0% act, 78% wasted. This is not a tuning
problem and no value of `board_margin` reaches it. It also shows why raising the
margin to the true body extent is a wash rather than a fix — it buys reach at
gap 0.175-0.20 and loses gap 0.04-0.06, because the trigger and the acceptance
move together.

**AND THERE IS A ONE-LINE STATEMENT OF THE WHOLE GEOMETRY, which is the other
session's and is better than the derivation above.** Sweeping every margin from
0.01 to 0.60 for each ball gap and keeping the best clearance achieved:

| ball gap | 0.056 | 0.060 | 0.080 | 0.100 | 0.150 | 0.200 | 0.300 |
|---|---|---|---|---|---|---|---|
| best spot clearance | 0.116 | 0.120 | 0.140 | 0.160 | 0.210 | 0.260 | 0.360 |

**Clearance = `gap + kick_side`, exactly, at every gap — and the best margin is
the smallest one tried, in every row.** Verified here independently against
`_along_the_boards` (7/7 gaps, agreement to 1e-6).

So **the along-the-boards spot always stands exactly `kick_side` further from
the board than the ball does. The margin never moves the spot; it is purely an
acceptance test on a placement it does not control.** That makes the floor
immediate rather than derived — you need `gap + kick_side ≥ body`, and no
acceptance threshold can change a placement — and it explains why raising the
margin to the body radius had to be a wash: the spot is already where it is
going to be, so the margin only shifts *which* gaps are accepted, gaining at one
end and losing at the other.

**Which sharpens the design conclusion one final time. `board_margin` is the
wrong KIND of parameter for this problem: an acceptance threshold cannot fix a
placement rule.** Whatever goes into `kickselect` has to be a constraint that
**rejects the kick and selects a push**, not a margin that accepts or refuses a
spot the geometry has already fixed. Every arm in 12p-12v was sweeping the
acceptance test of a placement nobody was changing.

Near a board, then, the duck **does not need a better place to stand, it needs a
different action** — a push,
or a nudge that puts the ball more than ~6.8 cm out before any kick is planned.
`kickselect` already ranks pushes; giving it the reachability constraint would
let it choose one exactly when no kick spot exists, which is the design change
above and is now bounded by a number rather than an intuition.

### 12w. Does the whiff gate work on the camera the ROBOT has? — MEASURED (2026-09-09)

`_too_far` returns `False` when `predicted is None`: **the gate that took whiffs
47% → 36% disables itself whenever the duck has no live track.** The robot today
runs a 1080p crop at 85.2% team blindness (12n). So the question that decides
whether any of tonight's whiff work reaches hardware: does the gate still pay on
that camera?

**Registered at 14:05:46Z before launch** (`scratchpad/prereg-gate-on-crop.txt`,
sha 9e073124...): primary is the gate's benefit in **connected kicks**
(swings × (1 − whiff)), compared between cameras; prediction **smaller under the
crop**; and a stop rule — with the gate held fixed the crop must differ from the
sim on some outcome, else **NO RESULT**, because `kick_gym` is one duck and one
ball at close range while 85.2% is a *match* statistic. The stop rule was written
into the reader itself rather than kept in mind, so it executes before the
primary is visible.

**Positive control: PASSES, decisively.** With the gate armed, whiff is 34.5%
(sim) against 63.6% (crop), z = 11.84. The harness reproduces close-range
blindness. The other session measured the mechanism variable directly and
independently: `predicted is None` on **33.6%** of sim line-up ticks against
**56.2%** of crop ones.

**PRIMARY, 2240 episodes an arm, `--at-boards 1.00`:**

| camera | gate | swings | whiff | connected | **gate's benefit** |
|---|---|---|---|---|---|
| sim | ON | 892 | 34.5% | 584 | **+96** |
| sim | OFF | 1109 | 56.0% | 488 | |
| **crop** | ON | 662 | **63.6%** | 241 | **+38** |
| **crop** | OFF | 738 | 72.5% | 203 | |

**crop/sim benefit ratio = 0.396**, bootstrap 95% CI **[0.04, 0.95]**,
P(ratio < 1) = **0.98**. **The registered prediction holds: the gate is worth
substantially less on the camera the robot actually has.**

**What this does NOT establish.** The other session's prior bracketed the ratio
at 0.71 (availability × fire-rate) to 0.99 (also weighted by how far past the
threshold the caught swing was). The point estimate is well below that floor,
but P(ratio < 0.71) is only **0.90** — suggestive, not decisive, and **their
availability model is not refuted by this**. A ratio of two differences of counts
is a noisy statistic and the CI says so honestly.

**The robust number, which needs none of that precision, is the one that
matters: with everything we shipped tonight armed, whiff on the robot's camera
is 63.6% against the sim's 34.5%.** Nearly double, at z = 11.84. Whatever the
gate is worth there, the duck it protects is missing most of its swings anyway.

**So the whiff work does not transfer, and this is the third time the same thing
has been found** (12n for the soccer ledger, §3c for tidy): **every number in
this repo is measured on a camera the robot does not have, and is optimistic.**
The 47% → 36% in item 12a is a sim result. On the crop, the same brain is at
roughly 64% in the gym's hard case, and the gate recovers ~40% of what it
recovers in the sim. Two candidate reasons the gate is worth less, neither
tested here: it is armed less often (measured, 43.8% vs 66.4% of line-up ticks),
and — not in anyone's model so far — **when it does decline a swing under the
crop the replacement swing is nearly as bad**, since even gate-armed crop swings
miss 63.6% of the time. Declining is only worth what the next attempt is worth. **That second one is NOT testable in
`kick_gym`** — checked rather than assumed: every episode contains exactly one
swing (2001 swings across 2001 episodes, none with two), so there is no
"subsequent swing" to compare and no gym arm can answer it as the harness
stands. It needs repeated attempts on the same ball, which means
`probe_kick_line.py` reading swings out of a contested match with the camera as
the arm, or a gym that does not end the episode at the first swing. Recorded so
nobody spends a battery discovering the harness cannot answer the question.

> **This limit is narrower than the sentence above may suggest, and the
> distinction matters.** What one swing per episode rules out is comparing a
> *replacement* swing to the one it replaced — that needs two swings in one
> episode. It does **not** rule out asking whether the gate changes *which*
> swing ends up being the episode's first one: the gate suppresses an attempt,
> the episode continues, and the swing that eventually happens is still that
> episode's first. That question is answered by comparing the swing
> **populations** between arms, on exactly these rows. Same data, different
> comparison — and it is the test the deferral mechanism below actually needs.

**THREE RESULTS FROM THE OTHER SESSION'S VALUE-MODEL ARM (2240 episodes, gate
off, `pred_ahead` added to every swing row — the quantity `_too_far` actually
gates on, the brain's own belief rather than the true ball).**

**(a) The `predicted is None` swings were a selection effect, and it was called
before the analysis ran.** 1037 gate-off swings:

| | n | median true \|ahead\| | whiff |
|---|---|---|---|
| `predicted = None` | 625 | **0.108 m** | **37%** |
| has a prediction | 412 | **0.290 m** | **91%** |

The blind swings are at balls **2.7x closer**. `predicted` goes None precisely
when the ball is too close to see (12k, ~0.3 m) and a ball at the duck's feet is
unmissable — so an 8-swing smoke test showing "blind swings do not whiff" was
measuring *close balls are easy*. **The population was defined by the thing that
makes the outcome easy**, which is the same shape as 12o's corner case and the
throw-in expiry probe: the third time tonight.

**(b) THE LINEAR-VALUE ASSUMPTION IS FALSE, and the session that made it tested
it rather than defending it.** Whiff by how far past the gate the belief sat:

| excess beyond 0.15 m | n | whiff | median travel |
|---|---|---|---|
| 0.15-0.25 | 170 | 88% | 0.019 m |
| 0.25-0.35 | 177 | **97%** | 0.027 m |
| 0.35-0.50 | 50 | 96% | 0.033 m |

**It saturates.** Declining a swing 0.40 m out is worth no more than declining
one 0.20 m out — both whiff. So the severity weighting that produced the 0.99
ceiling is unearned, and **that bracket collapses to its floor, 0.71.** Recorded as
its author asked it be recorded: not "tested their own assumption well" but **a
number that should not have been offered as a number** — 0.99 was stated as a
bound when it was a conditional (*this is what it would be IF value scales with
excess*), and on the strength of that invented ceiling this session reasoned
against its own registered prediction. Stating an assumption as a bound is the
error; withdrawing it is only the repair. This item's 0.396 still
does not beat 0.71 (P = 0.90) and is not claimed to — but the comparison is now
one number against one number, and the one that moved was measured away rather
than argued away.

**(c) The gate declines almost pure losers, which is a better argument for
`kick_ahead_max` than the A/B that shipped it.** Of the 412 swings where the
brain had a prediction, **409 were beyond the gate**, whiffing **92%** with a
median travel of 0.026 m. The 3 inside it whiff **0%** and travel 0.812 m. When
the duck can see the ball at swing time, the swing is nearly always one the gate
should refuse.

**AN UNCONFIRMED MECHANISM, REGISTERED BEFORE ITS TEST.** Gate-off gives 1037
swings of which the gate would refuse 409, so removal predicts 628 with the gate
on — but the gate-on arm has **846**, leaving **+218 unexplained**. So declining
does not *remove* a swing, it **defers** one, and the deferred swing may arrive
later, closer and blind: the 37%-whiff population. If so the gate's value comes
from making the duck wait until the ball is at its feet, which is a stranger
mechanism than either session's model. **The registered prediction, before the
confirming arm runs: `predicted is None` on ~99% of gate-ON swings, against
60.3% measured on gate-off. Under ~80% and the mechanism is unsupported, not
merely "higher".**

> **THAT FALSIFIER WAS UNDERSPECIFIED, and this refinement is registered before
> the arm lands.** A single None fraction collapses two *opposite* worlds into
> one number — the other session spotted it and this one did not. The
> discriminator is where the surviving *predicted* swings sit; gate-off had only
> **3 of 412 inside the gate (0.7%)**:
>
> | outcome | reading |
> |---|---|
> | ~99% None | deferral to close-and-blind: the registered mechanism, supported |
> | ~60% None, predicted swings mostly **beyond** 0.15 | the gate is failing to refuse swings it should — unsupported, and alarming |
> | ~60% None, predicted swings mostly **inside** 0.15 | the duck waited until its belief agreed with where it stands: **deferral as registered is WRONG, and the knob is better than either session has described it** — the value is alignment, not blindness |
>
> The third case refutes the registered mechanism *while improving the knob*,
> and the original rule would have filed it as "unsupported" and stopped there.
> **A falsifier that cannot tell a refutation from a better finding is a badly
> built falsifier.**

**RESULT: the registered prediction held to a percentage point, and the third
case is refuted on its own evidence.**

    REGISTERED   ~99% None on gate-ON; under 80% unsupported
    OBSERVED      97.9%  (829 of 847)      gate-OFF was 60.3%

The third case is **filed as "exists, does not account for the result" rather
than "refuted"**, at its author's request and correctly: of the gate-ON swings
that did have a prediction, 94.4% were inside the gate against gate-off's 0.7%,
so the alignment mechanism is **real and correctly signed** — there are simply
only **18 of them, 2% of the arm**. "Refuted" would discard a true thing. The other 829 are blind.

**And the `\|ahead\|` distribution separates "deferred until close" from
"deferred until merely lost", which the None fraction alone could not:**

| | n | median | q25 | q75 |
|---|---|---|---|---|
| gate OFF, `predicted = None` | 625 | 0.108 | 0.086 | 0.150 |
| gate OFF, had a prediction | 412 | 0.290 | 0.229 | 0.349 |
| **gate ON, all swings** | 847 | **0.111** | 0.090 | 0.145 |

Gate-ON *is* the close mode, quartiles on top of each other, and the far mode
has disappeared. **So `kick_ahead_max` does not improve swings — it replaces
them.** The gate suppresses the far, stale, *seen* swing (92% whiff, 26 mm of
travel) and the duck swings later at a ball 0.11 m away it can no longer see.
Whiff 59% → 34%. **The knob's entire value is converting a bad seen swing into a
good blind one**, which is neither session's original model of it.

**WHICH EXPLAINS THE CROP, AND MAKES THE CAMERA CASE STRONGER RATHER THAN
WEAKER.** The obvious objection to 12w's ratio was that a camera going blind
*earlier* should reach the valuable close-blind state *sooner* and lose less —
yet the crop's benefit measured 0.396. The resolution is in this item's own
arms: gate-ON swings are ~98% blind, and

    gate ON, sim camera   whiff 34.5%
    gate ON, crop camera  whiff 63.6%

**the same swing type, nearly double the whiff.** A blind swing is not
camera-independent: *blind* means flying on the last thing seen, so a camera
that loses the ball earlier hands the swing a staler starting estimate. The
camera therefore matters even for swings taken with the camera contributing
nothing at the moment of the swing — which is the strongest form of the ceiling
argument in 12n and 12u.

**Stated as an inference, not a measurement:** it assumes the crop's gate-ON
swings are also ~98% blind. That is very likely — the crop is blinder
everywhere — but it is **unmeasured**, and `pred_ahead` on a crop gate-ON arm
would settle it. Recorded as the next question if this line continues.

**THE PRECISE CLAIM THE ARMS SUPPORT, since two nearby ones are wrong.** The
other session's framing — that the camera's help is "upstream of the swing
entirely" — is right about the *sighting* and too strong about the *estimate*:

- **Wrong:** "a wider camera lets the duck see the ball as it kicks." These arms
  actively contradict it. Gate-ON swings are 97.9% blind in *both* arms; the
  duck never sees the ball at the moment of a good swing.
- **Also wrong:** "the camera only helps upstream, so the swing is
  camera-independent." Blind is not information-free.
- **Supported:** *the camera never contributes at the instant of the swing — the
  duck is blind by then, by design. It determines how good the last sighting
  was, and every swing, seen or blind, flies on that.*

**STILL OPEN.** The inference above narrows the fourth question but does not
close it: it explains why the crop's close-blind swing is worse, not why the
crop's gate benefit lands *below* the collapsed 0.71 rather than at it. Left
open in these terms rather than implying the chain is understood end to end.

**AND THE INFERENCE IS BEING CONVERTED TO A MEASUREMENT — expectation registered
before it lands.** The other session is running the crop at gate-ON with
`pred_ahead` recorded (2240 episodes, `--at-boards 1.00`, seed0 800). Their
registered expectation is ~98%, matching the sim arm. **Mine is ≥98%** — the
crop is blinder everywhere, so if anything it should be *more* blind, not less.

**The outcome that would refute me is more plausible than it looks, and naming
it is the point of registering.** The crop takes far fewer swings (662 against
892 on the same design), so its swing population is *selected*. If the duck's
blind approach fails more often under the crop, the episodes that still produce
a swing may be disproportionately the ones where it kept sight of the ball —
which would push the None fraction **down**, not up. That would mean the crop
does not reach the close-blind state less *effectively* but less *often*, and
"the blind state it reaches is worse" would become "the crop reaches a different
state" — a different claim, and one that weakens rather than strengthens the
brief. Registered so that outcome cannot be read afterwards as a variation on
the same story.

**AND THE INFERENCE IS ROBUST TO THE ANSWER, which is worth deriving before it
arrives rather than after.** Decompose the crop's gate-ON whiff as a mix of the
two swing types, using the rates already measured (seen swings 92%, sim's blind
swings 37%):

    0.636 = f x w_blind + (1 - f) x 0.92          f = the blind fraction being measured

| f | implied crop blind whiff |
|---|---|
| 1.00 | 0.636 |
| 0.90 | 0.604 |
| 0.70 | 0.514 |
| 0.60 | 0.447 |
| **0.516** | **0.370 — the sim's blind rate** |

**The crossover is f ≈ 0.52.** For any blind fraction above that, the crop's
blind swing whiffs more than the sim's, and "the blind state the crop reaches is
worse" holds *whatever the arm returns* — 0.52 is far below the sim arm's 97.9%
and below even the gate-OFF 60.3%. So the measurement can still surprise on the
selection question (how the crop gets to a swing at all), but the claim the
brief rests on does not hinge on its exact value.

**The action this implies is not a knob.** No tuning of `kick_ahead_max` fixes a
gate that is off because the camera cannot see; the fix is the camera, which is
what 12u now supports on possession and ball advance. If the replacement ships,
the whiff work starts paying what the sim says it should.
