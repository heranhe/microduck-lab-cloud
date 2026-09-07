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
   against +2.4° with tracking off. Paired per seed every kick metric is
   flat: kicks p=0.68, whiffs p=0.64, |error| p=0.73, ball-ahead p=0.21.

10. **A shared frame for the blackboard** (4.3). At `datasheet` drift two
   teammates' frames wander 0.456 m apart over a run, so "the ball is at
   (x, y)" stops being a place the teammate can act on. Everything soccer
   here runs at `ideal`, where the frames agree exactly — so nothing measured
   is affected, and nothing measured is evidence about a robot either.

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
   whole whiff on this floor, and it is not a sensing problem: either plan
   the spot closer (`kick_ahead` 0.05?) or decline on AHEAD rather than
   side (`kick_side_max` gates the wrong axis here). Judge on whiff and
   on-spot with `probe_kick_line.py` — the shipped floor baseline is
   `runs/raise/k_ship2.jsonl` — and only after the parallel session's
   floor change is committed, since every number in this paragraph is on
   its uncommitted physics.
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
- [ ] **A.3 Choose the kick by simulating its outcomes — and stop kicking
      out.** Mellmann, Schlotter & Blum (Berlin United, RoboCup 2016):
      before every kick, each candidate action (long, short, sidekick left,
      sidekick right, turn) is forward-simulated **30 times**, sampling the
      kick's velocity and direction from Gaussians fitted to real kicks and
      the ball from the tracker's uncertainty; each sample rolls out under a
      rolling-resistance model (d_max = v₀²/2c_R·g) until it stops or hits
      the goal box or an obstacle; each is labelled INFIELD / OUT / GOALOPP /
      GOALOWN / COLLISION; actions with p(INFIELD ∪ GOALOPP) < 0.85 or any
      own-goal sample are discarded, and the rest are scored by a potential
      field (linear slope to the opponent goal, Gaussian attractor at it,
      Gaussian repulsor at own goal). On labelled video of real games it cut
      kicks out at the opponent goal line **5× (6.1% → 1.2%)** and raised
      strategically-good kicks from 67% to 78%. **We already own every number
      this needs**: v₀ = 1.4 m/s and decel 0.04 m/s² (`predict_s` note), the
      per-foot exit angles +23.6°/−28.7° AND their sd 33–49° (4b), the goal
      geometry, and now +1.90°/cm of side offset. Our `aim_mode="clamp"` is a
      one-line deterministic version of the potential field with no notion
      of risk; the ledger's `kicksBack` (34% after the clamp) and out-of-play
      counts are exactly what this would move. → **what settles it:**
      `kicksBack` as a proportion over kick events (24 seeds), plus a new
      `kicksOut` in `PitchMetrics`. Cheap: it is a function over numbers the
      brain has, called once per settle.
- [ ] **A.4 Dribbling — carry the ball rather than stop and strike it.**
      B-Human ships a `Dribble` behaviour beside kicks; Dribble Master (2025)
      learns dribbling with RL using **a virtual camera in the simulator that
      models the field of view**, plus rewards for *active sensing* — keeping
      the ball in view — and transfers to hardware. The point for us:
      dribbling keeps the ball in continuous contact inside the walk, so
      there is never a 3 s blind approach to a stale spot. Our `push` mode
      (`push_beyond`, a "push spot squarely behind the ball") is a crude
      dribble that nothing has ever measured against the kick. → **what
      settles it:** signed `ballProgress` and `possession`, push-only vs
      kick-only vs shipped, 24 seeds. If push moves the ball as far forward
      with fewer falls, the kick is not the right primitive for this robot.

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
- [ ] **B.2 A goalkeeper.** The review "RoboCupSoccer Review: The Goalkeeper,
      a Distinctive Player" (2023) lists the role: hold the ball–goal line,
      track the ball continuously, block or dive on a shot, clear when in
      possession, decide when to leave the goal, return. We have three roles
      and none is this. Most of the pieces exist: `block` state, the threat
      geometry in `brain/intercept.py` (4d, 18 tests), the ledger's
      `goalsAgainst`. → **what settles it:** `ownGoals` and `goalsAgainst`
      need 347/136 seeds, so do NOT judge it on those. Judge it on the
      deflection-agent's measures: shots on target reaching the line, and
      the keeper's time-on-line.
- [ ] **B.3 Game state and set plays.** Every league runs a GameController
      with `initial / ready / set / playing / penalized`: in *ready* the
      robots walk to legal kickoff positions, in *set* they stand still, and
      teams script set plays off it. We have `kickoff_brains` (a reset) and
      nothing else; there is no notion of a duck being penalised, a
      kick-in, or a formation to return to. Small, and it makes 3v3 look
      like a game: a `GameState` on the World that the roles read. → **what
      settles it:** it is a correctness feature, so tests, plus `depth` and
      `spread` at t=0 after each goal.

#### C. The world model — what "tracking is working" leaves out

- [ ] **C.1 A ball model with uncertainty.** Berlin United's selector (A.3)
      runs on a *multi-hypothesis extended Kalman filter* for the ball;
      B-Human's ball model carries covariance. Our `Track` is an
      exponentially-smoothed point with a velocity from consecutive hits and
      **no covariance**. That is why nothing in this track could reason
      about risk, and why the shot gate (item 7) had a fresh estimate on 34%
      of swings and no way to say how much to trust it on the rest. → the
      prerequisite for A.3 and C.3; settle it with the estimate error
      `probe_shot_gate.py` already prints (median 2.3 cm, 90th 8.3 cm) and
      whether the covariance predicts it.
- [ ] **C.2 Self-localisation from the pitch.** SPL teams localise with
      particle filters over field lines, goals and corners; the survey
      literature calls the limited unique landmarks the hard part. **Our
      detector has no landmark class at all** — `DETECT_CLASSES` is duck,
      person, ball, marker, toy, basket; there is no goal, post or line — so
      the duck is pure dead reckoning and the goal is "where it was at
      spawn" (4.4.3). Item 10 measured the cost: at `datasheet` drift,
      teammates' frames wander **0.456 m** apart. Two steps: a `goal`
      detection class (the pitch has two distinct goal mouths; the sim
      detector is pinhole + noise so this is an afternoon), then a particle
      filter on odometry + goal sightings. → **what settles it:**
      `probe_odom_goal.py` at `datasheet` and `hostile`: miss-at-goal-line
      and mates-disagree, both of which it already prints.
- [ ] **C.3 A shared world model, not a shared point.** SPL teams fuse
      teammates' ball estimates weighted by their covariances into a team
      ball; B-Human 2022 ("More Team Play with Less Communication") rebuilt
      the behaviour to play pass-oriented soccer while *sending fewer
      messages*, because the league capped team traffic. Our blackboard
      sends one point estimate a second with no confidence and no frame
      correction. After C.1 and C.2 it can carry covariance and a frame.
- [ ] **C.4 An opponent model and a duel.** B-Human has a `Zweikampf`
      (one-on-one) behaviour; every stack tracks opponents as first-class
      objects. Ours has duck tracks, a colour vote that failed confirmation
      (4.2), and `avoid`/`blocked`/`yield`. The interception work (4d) found
      "the lever is elsewhere". Low priority until a keeper exists.

#### D. Team play — after C, not before

- [ ] **D.1 Passing.** B-Human scores candidate pass targets by goal angle,
      teammate accessibility and opponent blocking, picks the best, and
      executes it with an in-walk kick. We have zero passing, and cannot
      have any until a teammate's "I am here" means the same place to both
      ducks (C.2/C.3). Then it is A.3's simulator with a teammate attractor
      in the potential field — which is precisely what Mellmann's paper
      names as its own future work.
- [ ] **D.2 Positioning by potential field or Voronoi.** RoboCup supporters
      stand where a potential field over the pitch (ball, teammates,
      opponents, goals) has a minimum; MSL teams tile the field with a
      weighted Voronoi tessellation and assign robots to cells. Our roles
      stand at fixed posts (`defend_depth`, `strike_ahead`, `strike_side`)
      that do not see opponents at all. The measured win of item 3 (crowd
      13% → 1.8%, spread +0.95 m) came from posts; a field would let the
      posts move. → `crowd`, `spread`, `depth` resolve at 11 seeds.

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
      flat 13-state machine in a 2 300-line file with roles bolted on as a
      post and a zone. It has held up through this track because every
      change was measured, but B.2 + B.3 + D.1 will not fit in it. Not
      urgent; the moment it becomes urgent is when a keeper needs a
      different top-level loop from a striker.

**What to read this list as.** A.1 is the only item that removes the
limit item 7 hit; A.2–A.4 route around it; B and C are the parts of a
soccer stack that were never started; D depends on C; E says the field's
learned results *kept the honest camera and rewarded looking*, which is the
opposite of the shortcut that would make a striker look good in sim. If
one thing gets built next it should be A.1 in the simulator — a day of
work — because its number ("swings it can see") decides whether A.2–A.4
and items 7's dead knobs are worth reopening at all.

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
## Physics audit — 2026-09-06 (after the ball that never stopped)

The ball's zero rolling resistance (a coefficient set on a condim-3 geom,
Track 4 item 0) prompted a sweep of every other physics parameter in the
harness, measured, not read. Probe scripts were scratch; the numbers are
here. Nothing below was changed — each item is a decision to make.

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

  `duck_detect` ONNX — subsumes all three and is far slower per step. Worth it
  only once the behavior is otherwise settled.
- **Other things to find.** The slot layout is not ball-specific: the same
  four head slots and scan clock would serve "find the other duck" (upstream
  wants precise bearing for gaze and following) or "find the charging dock".
  A second target is a cheap test of whether the recipe generalizes or whether
  it memorized a ball-sized blob.
