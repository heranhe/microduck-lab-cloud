# MOSS stow: the seat hands back dead starts — the aim-chain bisect, 2026-09-26

**Every varied-shape stow number before this is partly a pick number.** The
stow env's reset seats the object on the floor, closes the jaw and raises it
to the rung's start pose. For six shapes the seat FAILS on ~7 of 8 draws —
ball, card, most off-size cans — and the reset fell through silently, handing
back the object LYING ON THE FLOOR under an open jaw at the grasp pose, at
every rung. `_carrying()` even reported True for it (inside the radius, within
the debounce of `_last_contact = 0`).

* rung 0, six shapes: held at reset **8/60** (ball 0/15, card 0/11, tall 0/3,
  can-family 1/13, block 6/11); the reference can alone: 60/60.
* rung 2 likewise ~5/40 lifted. The shipped stow's ~58% at rung 2 is partly
  the policy RE-PICKING the object from the failed seat, which rung 2's open
  jaws allow; at rung 0 that is the whole job, so the aim chain's stage 1
  (from scratch, rung 0) collapsed to 3/40 in 10-step episodes.

The bisect (rung 0, from scratch, one knob per arm, 34-83k steps): the can-only
control scored +62.6 per episode in 13-step deliveries; every six-shape arm
(-28 to -40) failed whatever else it added — so clutter, the drop point and the
release bonus were never tested by the aim chain at all.

Fix: `valid_start` (`MICRODUCK_MOSS_VALID_START`) — a failed seat-and-carry
redraws the shape (up to 6) then falls back to the reference can: held AND
lifted 59-60/60 at rungs 0 and 2, with and without clutter; ~a quarter to a
third fall back; cards never seat (the separate grasp problem). Default OFF so
old runs replay identically; `test_a_valid_start_stow_episode_begins_...`
fails on the planted regression. Round 2 of the bisect runs on it.

## Round 2 — valid starts (rung 0, from scratch, 250k steps, 60 unseen seeds)

| arm | delivered (spot real / zeroed) | landing err | perched |
|---|---|---|---|
| V0 six shapes | 59/60 | 38 mm | 0 |
| V1 + drop point | 60 / 59 | 40 / 38 mm | 0 / 0 |
| V2 + release bonus 0.6 | 59 / 59 | 36 / 37 mm | 0 / 0 |
| V3 + 0-6 clutter | 52 / 53 | 43 / 46 mm | 26 / 28 |

* The seat WAS the collapse: with valid starts every arm delivers (3/40 -> 52-60/60).
* No arm USES the spot: zeroing slots 28-29 changes nothing. Only V3 tests it (an
  empty bin's spot is always the centre); its +0.3 slope survives zeroing — physics
  deflecting objects toward open space, not aim. The policy lets go in 10-12 steps.
* Aiming would matter: V3, 180 deliveries, perched by release distance from the
  spot — 0-2 cm 5/12, 2-4 16/50, 4-6 11/41, 6-20 cm 56/77 (73%).

## Round 3 — scripted aim, and the FILL test (the metric that matters)

Scripted aim: IK the jaws over the chosen spot (`scratchpad` prototype), 0.03 rad/tick,
open. Findings on the way:
* **Reach**: the jaws only get over the +y ~58% of the bin (pan stops at -1.92 rad; same
  at every height). `data/moss_bin_reach.npz`; `choose_drop_point` now masks by it (1-cell
  margin). Unmasked, a spot past it was 45 mm out of reach and landed ON THE RIM.
* **"Perched" is fill, not aim**: single drops, high/centre, perched by items already in
  the bin: 0:0/28 1:0/22 2:7/21 3:15/29 4:19/29 5:26/30 6:21/23. Resting on other litter in a
  full bin is how a bin fills — a bad success metric.
* Single drops, 200 seeds: low release (2 cm under the rim) wedges on pads/clutter (75%
  delivered); high release (as the learned policy: 11 cm above) 87-91%, aim makes no
  paired difference (10 v 10 perches).

FILL TEST (items delivered one after another into ONE bin, each left where it landed; 10
attempts per bin; same shape sequence for every controller):

| release | aim at spot | aim at centre | sequences |
|---|---|---|---|
| high, items in bin | 6.06 | 6.08 | 36 |
| high, items before 1st miss | 3.72 | 5.00 | 36 |
| mid (+5 cm, lift away) | 5.92 | 5.58 | 12 |
| low (-2 cm, lift away) | 4.58 | 3.00 | 12 (10-0 paired, p~0.002) |

**Verdict: in this sim, aiming does not fill the bin fuller.** It helps only when forced to
release low, and releasing low fills worse. At the best height the centre is as good and
misses LATER (spot aims go toward the edges; the centre leaves bounce room). The bin holds
~6 mixed items. Caveat: placed items are welded where they land (static clutter), so a new
drop cannot shove them — a real bin settles; that could favour either side.

## The "cheap release fix" — a null, and what it says about where the yard loses objects

`STOW_RELEASE_HIGH` (tool point over the bin centre, 11 cm above the rim, carry wrist)
in place of the descent to `STOW_INSIDE`: moss-yard, 6 seeds x 300 s, paired —
**14/36 objects in the bin against 13/36** (per seed +0 +1 +1 +1 -2 +0). In the world
~93% of RELEASES already land in the bin (13 of 14). An env replica of the brain's
carry lost 32/72 at release — the replica was wrong, which is why it was not trained
on. The yard's losses are upstream: block 0/6 and ball 0/6 are almost never gripped,
and fingertip grips (>45 mm from the tool point) never survive the carry (0/7).
`release_high` kept, default OFF. Also found: `_held()`'s 2.0 mm servo-stall test
reads a physically carried can as dropped at the brain's 27 mm carry squeeze.

## Pick retrain, step 1 — the pick policies scored IN THE YARD (6 seeds x 300 s, 6 objects)

| pick | in bin | attempts | lifts | kept past lift | reached release |
|---|---|---|---|---|---|
| teach-moss_pick-5e9df7 (brain's) | 13 | 77 | 39 | 21 | 14 |
| moss-pick-v1 | 12 | 78 | 34 | 19 | 15 |
| teach-moss_pick-d879c3 | 11 | 83 | 44 | 22 | 16 |
| teach-moss_pick-ad9876 (record.json pick) | 4 | 90 | 32 | 18 | 14 |

* The three can-only picks are indistinguishable at 6 seeds (+-1): the yard cannot rank them
  at this sample, so no env criterion can be validated against a ranking among them.
* Funnel (pooled): attempt->lift ~49%, lift->kept ~54%, kept->release ~71%, release->bin ~80%.
  Ball: 0 lifts in any run. Block: 1-3 lifts, 0 kept. Squat: about half.
* ad9876 — the only six-shape pick (88% in its env) — trained with publish_attitude,
  publish_size, publish_proximity AND base_lock. tidy_moss publishes none of them and does
  not lock the base; the world has no wrist camera at all (head-camera detections only).
  Its 4/36 is a deployment mismatch, not its skill: 14 releases, 4 in the bin at the end.

## ad9876 given what it trained with — the wrist camera in the yard (robots/moss_wrist.py)

The world now produces MOSS's wrist/front-camera readings of the object nearest the jaws
by calling the pick env's OWN `_sense_arm` / `_front_attitude` / `_true_attitude`
(`Senses.target_obs`), and tidy_moss fills exactly the slots each leg trained with and
zeroes the base twist for a leg trained base-locked (read from its run.json). Slot diff
yard vs env: all alive; the yard's wrist range is 0.27 m vs 0.06 m in the env (the yard
hands over FARTHER), and yard objects stand (uprightness 0.96 vs 0.38).

ad9876 in the yard, 6 seeds x 300 s (in bin / lifts / kept / releases):
no inputs, base free 4/32/18/14; inputs, base free 5/23/16/14; no inputs, base locked
1/25/12/7; inputs + lock 0/28/7/1 — against 5e9df7's 13/39/21/14. The inputs were not
the gap (+1); the base lock is fatal because the yard hands over beyond the arm's reach;
and it never lifts block/ball/squat in the yard. Its env skill does not transfer: the
handover situations differ. Also open: only 5 of its 14 releases end in the bin.
Default brain unchanged (5e9df7 has no extra inputs).

## The pick handover, diffed (and the release wedge, measured directly)

* First pick observation, yard vs env (ad9876): the LAST-ACTION slots carried the previous
  leg's action (up to 680 sd off the env's zeros), a stale pick command carried between
  attempts, the pick ran at 50 Hz (trained 25), and the arm was off GRASP_POSE (pan / wrist
  sd 0.21-0.29) because 16-30% of deploys TIME OUT. `pick_clean_start` (on) zeroes the last
  action and re-seeds the command at `creep`; `pick_at_control_hz` (off: ad9876 8 -> 1,
  5e9df7 11 -> 10 at 25 Hz). Clean start moved the yard count inside the noise.
* Handover bank (`data/moss_pick_handovers_yard.npy`, 310): 81% nearer than the rung-2
  box, 42% lying. From these starts IN THE ENV ad9876 makes deep (<=40 mm) picks 64% with
  the base locked (ball 16/20, block 11/16) — in the yard, ~none. `handover_bank` and
  `deep_grip_m` added to the pick env, off by default.
* RELEASE WEDGE: at the end of each release, the STOW_INSIDE descent left the can in the open
  jaws 3/12 (the fold then carries it off); `release_high` 0/13. End count 11 -> 13 (noise
  alone), folds home 66/78 v 60/72, over-speed 6 v 2. `release_high` now ON. (The first
  end-count-only A/B, 14 v 13, could not see the wedge.)
* A bin check without an upper height counted cans CARRIED over the footprint as "in the
  bin" and invented a leak of delivered cans; with the rim bound nothing leaves the bin.

## The too-close stop was the RANGE, and the lift opened the jaws on small objects

* tidy_moss ranged every detection with a CAN's radius (`can_radius_m`, a workaround from
  before the arena passed each prop's own size to the detector). In moss-yard every other
  shape read ~0.2 m farther than it stood (block 0.52 v 0.26, ball 0.53 v 0.30, squat 0.51 v
  0.31; cans +-0.01): the robot drove 20 cm too close (deploys timed out) and the pick reached
  20 cm past the object. `range_from_detector` (on) takes the detector's per-object
  `range_est`: all shapes within +-4 cm, handovers at 0.35-0.49 m — inside the pick's box.
* The lift set the jaw to the can's carry squeeze (27 mm) — WIDER than a block or ball, so it
  opened on them: blocks lifted 21 times, balls 16, none kept. `carry_jaw_relative` (on):
  carry at the achieved jaw minus 6 mm (the interference the 27 mm gave a can).

moss-yard, 12 seeds x 300 s, 6 objects: 5e9df7 37/72 (can 22/36 block 1 squat 12 ball 2 of 12),
ad9876 (base locked, wrist inputs) 38/72 (can 20/36 block 6 squat 7 ball 5); paired +0.08 +-
0.42 per seed. Both ~3.1 objects per run against ~2.2 before these fixes; only ad9876 delivers
blocks and balls. The brain still ships 5e9df7 — switching is a decision, not yet made.

## Teaching the pick to grasp CLOSE IN failed three ways — fine-tuning ad9876 degrades it

ad9876 deep picks (<=40 mm) by object distance, base locked, its env: 65% at 0.22-0.28 m,
75% at 0.28-0.34, 93% at 0.36-0.42 — it learned the stretched lunge, not the top-down grasp.
Warm-started from it (every recorded setting matched):
| run | change | 0.22-0.28 | 0.28-0.34 | 0.36-0.42 |
|---|---|---|---|---|
| teach-moss_pick-b57bbb | box 0.22-0.34 AND deep grip | 43% | 31% | 62% |
| pickA (battery) | box 0.22-0.34 only | 31% | 47% | 68% |
| pickB (battery) | deep grip only | 7% | 31% | 65% |
Every fine-tune is worse everywhere, whatever it changed; pickB's training reward ROSE
(-15 -> +75) while its deterministic picks fell. Not exploration noise (ad9876 ended at the
same capped std 0.607). Suspected, NOT confirmed: ad9876 trained (lab chain from
teach-moss_pick-combo) under reward knobs that are import-time env constants run.json does
not record, so a fine-tune optimises a different reward. Next: recover its launch env, or
diff the per-term reward budget of ad9876 vs a fine-tune on the same episodes.

## 2026-09-27 — the unrecorded reward knob, 478dad, and the carry

* ad9876 trained with `MICRODUCK_MOSS_GAP_TCP=1` (pay the gripper's approach), which nothing
  recorded: its replay earns +86/ep with it, -3 without (training reported +88). The three
  failed fine-tunes all trained without it and learned to DRAG objects toward the chassis
  (+14..+17 cm). Now a recorded kwarg (`gap_from_tcp`), with a test.
* 478dad = ad9876 + spawn 0.22-0.34 m, knob on: deep picks 81/78/90% at 0.22-0.28/0.28-0.34/
  0.36-0.42 m (ad9876 65/75/93), card close in 8/11 (2/11), no dragging. Shipped.
* Handing over at 0.30 m: grips hold (38/44 kept past the lift) but carries were lost on the
  FIRST swing (to STOW_HIGH) — close-in grips are taken with the tool axis near horizontal
  (|cos| 0.23 v 0.84 at 0.41 m) and the swing turns them against gravity. Carry squeeze
  10 mm (was 6) halves those losses (9 -> 4) and doubles 0.30-m delivery (6 -> 13/32); at the
  default 0.41 m it gives 33/64 v 28/64 (8 seeds, paired +0.62 +- 0.68). 0.30 m + 10 mm =
  24/64: the close-in handover does NOT yet beat 0.41 m — losses during the LIFT (8-13 per 4
  seeds) are what remains. Handover stays 0.41 m; squeeze 10 mm is the default.

## Small things, straight down — a size-aware handover

The jaws can point straight down at floor height from 0.10 to 0.38 m (9 deg at 0.40, 41 at
0.46; joint sweep). 478dad USES that close in — at 0.24-0.30 m it shuts on a card at 4 deg
from vertical (7/7), a block at 5 (12/12); at 0.38-0.42 m everything is 20-27 deg off, card
5/7. tidy_moss now sizes each detection (range x angular width) and hands an object smaller
than 8 cm over at 0.28 m, bigger ones at 0.41 m (a close handover lost cans on the lift).
moss-yard 8 seeds: 37/64 against 32/64 (paired +0.62 +- 0.56; cans 9 -> 14). The card stays
0/8: the head camera sees a 4 mm card 15 times in 300 s — it is rarely targeted at all.

## The litter set, and why the cigarette butt fails (2026-09-27)

Litter (`litter=True`): paper (sphere), butt (lying 32x8x8 mm box), cap. 478dad close in picks
butts 11-16 of 30-40. Three things that DON'T move it (30-40 episodes each, 478dad close in):
* softer contact (`physics_contact`, soft 1.0): 11 -> 3/30. MuJoCo's soft contact pushes back
  less, so grips less — the opposite of a real filter. Plumbing kept, off.
* grippier friction (1.2): 12/30. Kept (free).
* a gentler close — cap the jaw command at 4/8/12 mm past the achieved jaw (a current limit):
  10, 8, 13/30 v 12 unclamped. Force is not what loses it.
* the "correct" axis: `_true_attitude` reports body z, which for the lying butt is a SHORT
  axis. Reporting the long one: 16/40 -> 0/40 (cards 36 -> 31/40). Why: all 16 picks gripped
  END TO END (jaw opening along the long axis, sin^2 < .3 in 16/16); all 7 grips across the
  8 mm width failed. The short-axis reading steers to the grip that works. Kept as is,
  documented on `_true_attitude`.
The failures that remain: 8/24 never got both pads on it, 7/24 closed across the width,
7/24 were end-to-end and still slipped. Next lever: jaw alignment (the existing `jaw_align`
term pays turning towards the end-to-end grip under this axis) in a litter fine-tune.
* NOT the jaw-align fine-tune: 478dad and ad9876 already train with `jaw_align` 6 /
  `align_hold` 0.3, so "478dad + litter + align" IS d49520 (butts 6/19, not shipped).
* the wrist's STARTING angle: pre-rotated for the end-to-end grip 12/40, for across 13/40,
  as spawned 16/40 (same seeds). The policy re-aims during the approach; the start is not
  the lever. End-to-end in 16/16 picks is what survives, not what the policy chooses.

## Lift losses are shallow grips (2026-09-27)

Shipped brain (478dad, size-aware handover), moss-yard 12 seeds x 300 s, 69 carries held at
the lift: tool-point-to-object distance at the lift start decides it. Under 35 mm: 50
carries, 3 lost in the lift, 16 in the stow, 31 kept. 35 mm and over: 19 carries, 12 lost
in the lift, 3 in the stow, 3 kept. Most handovers were at 0.40-0.54 m (not close in).
* NOT the squeeze: the lift resets the jaw command from the pick's 0-14 mm to achieved-10 mm
  and the can slides out 0.2 s later, which looks causal — but keeping the pick's tighter
  command (8 seeds, same seeds) gave 24 v 23 in the bin, early lift losses 4 v 4. Removed.
* NOT sensed: the wrist camera's range at the lift does not separate deep from shallow
  (shallow 47-101 mm, deep 62-106), so the brain cannot gate on it.
* The ball is its own problem: 11/17 lost in the STOW, all deep grips (a sphere slips on
  the swing).
* The lever tried: 478dad + `deep_grip_m` 0.035 (only "picked" when held deep), one knob,
  every other 478dad setting (b6cef3). b57bbb (0.04 from ad9876) is not evidence: it trained
  without `gap_from_tcp`.
* RESULT, b6cef3 (478dad + deep_grip_m 0.035): env 35/40 deep, but moss-yard 24 seeds
  76 v 71 in bin (paired +0.21 +- 0.29) and shallow grips at the lift 55 v 48. NOT shipped.
  The env does not produce the yard's shallow grips — shallow share of picks: spawn box
  0.22-0.34 m 2-5%, 0.40-0.54 m 12-14%, from the 310 real yard handovers 9-13%, yard ~33% at
  the lift — and the brain's lift trigger replayed in the env almost never fires on a shallow
  grip (2/31, 1/12). Yard can sizes are inside the env's range. Unfound: what makes the yard's
  grips shallow. Next instrument: snapshot the yard's full state (arm qpos, object pose and
  size) at `creep` for shallow cases and replay it in the env — if the env grips deep from
  the same state, the difference is the world's physics or the brain, not the start.

## The replay (2026-09-27): the start is not the cause; two instrument errors on the way

* Replay: snapshot the yard at every `creep` entry (arm joints, the brain's TARGET object —
  from its own `_fix`, pose and size), pose it in the pick env, run 478dad. 12 seeds, 162
  starts, 72 yard lifts (69 on the target), 23 shallow. From the 23 states that ended in a
  shallow yard lift the env picks 17; from the 90 that never lifted in the yard, 34. Same
  policy, same start — the yard runs it worse. `snap_yard4.py` / `replay_env.py` (scratch).
* TWO WRONG TURNS, both from taking "the object nearest the jaws" as the target: it matches
  the brain's `_fix` only 42/80 times. It produced "shallow lifts are never-steady grips"
  (31/34) and "the jaws grab a neighbour" (16/39); on the brain's own target, 38/39 lifts
  hold it and steadiness does not predict depth (18/59 v 5/13). A `lift_needs_steady_grip`
  built on the first was measured anyway: 64 v 71, paired -0.29 +- 0.19. Removed.
* A REAL train/deploy difference: the pick env's object inherits MOSS's MJCF default
  (solref 0.008 / solimp 0.95 0.99); yard props get MuJoCo's softer default (0.02 / 0.9
  0.95). Stiffened in the yard, 24 seeds: 84 v 71 in bin (paired +0.54 +- 0.34, not
  resolved); lift losses 40 v 39, shallow 47 v 48 — it is not the lift cause; stow losses
  28 v 37. Not landed.
* Untested suspect for the lift: in the env the POLICY lifts (success = held 8 cm up); in
  the yard the brain takes over at the grip and runs a scripted joint-space ramp to
  LIFT_POSE. The replay's env success is the policy's own lift.
* LETTING THE POLICY LIFT (`policy_lift_m` 0.08, max 2 s after the grip; patch in scratch):
  the wrist depth camera's z does not update while the object is in the jaws, so it ran on
  the 2 s clock, and in 2 s the policy lifts to 30-36 cm by itself. 24 seeds: 68 v 71 in
  bin (paired -0.12 +- 0.19). Carries reaching the scripted lift fell 141 -> 98 and kept
  better (lift losses 39 -> 21): the shallow grips now fall out during the POLICY's lift.
  Who lifts does not matter; the losses moved upstream. Removed.
  Open: the replay's env "picked 17/23" is the env's success over a whole 8 s episode, which
  allows a drop and a re-grasp; the yard aborts on the first drop. That, not the lift, may be
  the difference the replay saw.
  Checked: NOT re-grasping — of the 17 env picks from shallow-yard states, 15 came on the
  first contact, none after a lift-and-drop. NOT the contact softness — the same 23 states
  with the yard's soft contact: 16/23 picked. What is left is the pick's EXECUTION: from the
  same state the env's policy lifts the object and the yard's (with `policy_lift_m`) drops
  it, so the observation or the stepping during the pick differs. Next instrument: step the
  yard and the replayed env side by side from one snapshot and diff the 32 slots per tick.

## FOUND: the yard ran the pick at 50 Hz (2026-09-27)

Side by side from one snapshot (seed 4, t 213.36, can1; `side_by_side.py` in scratch), the
yard's brain stepped the pick every 0.02 s — `pick_at_control_hz` was off, so each action's
0.03 rad / 4 mm nudge was applied 50 times a second against the 25 it trained under: arm and
jaw at double speed. It had been switched off on an ad9876 measurement (8 -> 4 of 18 cans)
that could not resolve it and was never re-measured. 478dad at 25 Hz, moss-yard 24 seeds x
300 s: **99 v 71 in the bin, paired +1.17 +- 0.36 per seed (+1.17 in each half)**; lift
losses 39 -> 24, kept 65 -> 92, cans kept 20 -> 45. Grips are as often shallow (53 v 48) —
they hold. Now ON by default, with a test planted against the 50 Hz regression.
Also in that diff, not yet chased: the axis slots (28-30) disagree yard v env (mean 0.67 over
40 steps; partly the env replay starting with a cleared tracker), and the brain does not
clip arm commands to the joint range where the env does.

## The axis-slot diff, chased (2026-09-27)

In the yard (6 seeds, 10761 pick steps at 25 Hz), slots 28-30 against the true axis of the
object the BRAIN is going for (its `_fix`):
* For LYING targets with the sensor on them the axis is right: corr 0.90 / 0.88, doubled-
  angle offset median -0.3 deg (IQR +-6.7). The big mean error in the side-by-side is
  upright objects, whose axis is noise in the env too — not a bug.
* The wrist sensor (`moss_wrist`) targets the object NEAREST THE JAWS, which is not the
  brain's target on 29% of pick steps. Pointing it at the brain's fix (a hint from
  world_server; 876/879 on target after) measured 89 v 99 in the bin over 24 seeds (paired
  -0.42 +- 0.44): null, leaning worse — fewer carries reached the lift (125 v 145), those
  that did kept slightly better. Reverted (patch `sensor_hint.patch` in scratch). The
  mismatch is real but it is not costing objects.

## The ball in the carry (2026-09-27): squeezed out, and none of three levers fixes it

Mechanism (12 yard seeds, 12 ball carries, 7 lost; `ball_stow.py` in scratch): at a
constant jaw command the achieved jaw creeps shut (20 -> 13 mm over ~1.5 s) while the
ball's centre moves 24 -> 37 mm from the tool point, then it pops out — the pads meet the
sphere AHEAD of its centre, so the squeeze has an outward part; the jaws point down
(tool axis z -0.87) through the lift, so gravity helps. As often in the lift as the stow.
Against the 25 Hz baseline (99 in bin, 24 seeds):
* lighter squeeze for all (12 seeds): 6 mm ball kept 9/12 v 5/12 but bin 51 v 50 (other
  objects lost in the stow); 4 mm 46 v 50.
* ease off on slip (`carry_slip_m` 2 mm -> re-squeeze 4 mm below the jaw): 99 v 99; lift
  losses 24 -> 13, stow 29 -> 37 — the same objects fall later. Reverted (patch in scratch).
* jaws LEVEL through the lift (LIFT_POSE wrist_flex 1.43 -> 0.4): 106 v 99, paired
  +0.29 +- 0.33 (+0.33 / +0.25 per half) — not resolved; lift losses 24 -> 14, ball kept
  12 -> 14 but lost in the stow 10 times. Jaws UP (0.0): 43 v 50 on 12 seeds. Not landed.
The ball's stow losses are the stow's orientation (tool axis back to -0.4..-0.85 on the way
to the bin) or the grasp point (centre 20-25 mm from the tool point at the lift) — a pick
that takes a sphere through its centre, or a stow path that keeps the jaws level.
* JAWS LEVEL THROUGH THE STOW TOO (lift wrist 0.415, STOW_HIGH/TURNED 1.25, solved by FK
  for tool axis z = 0; the stow waypoints were already near level, -0.18): 98 v 99, paired
  -0.04 +- 0.38 over 24 seeds; ball still lost in the stow 10 times. Traced with the jaws
  at z -0.02, the ball extrudes exactly as before (jaw 15 -> 13 mm, centre 33 -> 36 mm from
  the tool point, then out). GRAVITY IS NOT THE DRIVER — the pinch ahead of the centre is.
  Carry orientation cannot fix it; the grasp point can. The ball's centre sits 20-25 mm
  from the tool point at the lift, inside `deep_grip_m` 0.035, so b6cef3 never asked for
  it (its yard balls: 6 kept of 24). A sphere needs its centre BETWEEN the pads.
* The criterion, measured first: 478dad's env ball picks (120 episodes) run through the
  yard's lift ramp and swing to STOW_HIGH at the 10 mm carry squeeze — centre under 16 mm
  from the PAD MIDPOINT 33/33 kept, 16-20 mm 9/12, over 20 mm 2/9 (median 14 mm). [CORRECTED
  2026-09-28: the pad midpoint IS the tool point at every jaw opening (0.0 mm measured with
  both fingers moving); "10-15 mm off" came from moving one finger alone.] `sphere_centre_m` (recorded kwarg, `MICRODUCK_MOSS_SPHERE_CENTRE`):
  a sphere is only "picked" with its centre within that of the pad midpoint. Test planted
  against a tool-point reference. Fine-tune 0281e5 = 478dad + sphere_centre_m 0.016,
  one knob (run.json diff checked).
* RESULT 0281e5: did NOT learn to centre. Env ball picks 58/120 (478dad 54/120), centre
  from the pad midpoint median 15.7 mm (14.1), held through lift + swing 44/58 (44/54).
  Its training "picked" 30-33/40 is across all six shapes. Balls are 1/6 of episodes and
  the criterion only withholds the success bonus — too little pressure, or the centred
  grasp is never sampled (see "reward cannot fix exploration"). Not shipped; no yard run.
  Next if pursued: a ball-only fine-tune (all episodes spheres), then the yard.

## The wrist camera was mounted in the wrong frame (2026-09-27)

`gripper_frame_link`'s +z is the APPROACH (tool point at (-0.008, 0, -0.014) in it at
GRASP, LIFT, DROP, TUCK_POSE_V04 and home; housing z -0.098..-0.042, ~12 x 12 cm; jaws
slide on the 41/221 deg diagonal). `ARM_CAMERA_POS` (-0.060, 0, +0.065) read it as "up":
the lens sat 6.5 cm PAST the jaw tips, under the floor in 32% of frames within 10 cm of a
grasp, looking up. The old sweep searched only that half-space. Every wrist reading every
pick leg trained on (slots 26, 28-30 via `_sense_arm` / `moss_wrist`) came from there.
* Re-swept with occlusion (478dad frames, 5 rays per object against the robot's visual
  meshes and the floor): side face at the housing's back edge, (0.021, -0.080, -0.100),
  aimed at (-0.008, 0, 0.15): seen 89% inside 10 cm (old 60%), 94% inside 5 cm; 2.2 cm
  clear of the housing, 2.1 cm of the forearm over the whole wrist range. A plateau
  (88-90% over 290-340 deg, r 8.5-9.5 cm). The jaw-travel diagonal sees 27-46%.
* What it does to the slots (478dad, trained settings): wrist attitude fresh 57 -> 92% at
  10-20 cm, 7 -> 70% at 20-40 cm. Old slot 26 read a near-CONSTANT 7-9 cm from 40 cm to
  contact (the lens rode past the jaws); new reads 15-20 cm. z +1.6 on 478dad's
  normalizer: inside its support, so a warm start is not the frozen-normalizer trap.
* 478dad under the new mount, 248 paired seeds: eval env (cans) 87.1 -> 86.3% (null,
  sign p 0.86); its trained six-shape settings 89.1 -> 84.7% (-4.4 +- 1.8, 16 v 5
  discordant, p 0.03). moss-yard 24 x 300 s: 100 v 93 in the bin, paired -0.29 +- 0.46
  per seed (CI -1.24..+0.65, both halves) — null, unresolved at this size.
* `moss_bin.BIN_LOOK_POSE` sees 0% of the bin from the new mount (drop_target only; off by
  default). Re-searched: (-1.776, -1.195, 0.559, 1.507, 2.338), 94% of the floor; the whole
  floor needs wrist_flex on its stop.
* LANDED as 723a1a3 (every pick baseline above it is on the old mount). Retrain 0a9726
  (478dad's recipe, 1.5M steps, new mount), 248 paired seeds under its trained settings:
  87.1% v 478dad-on-new 84.7% (+2.4 +- 1.9) and v 478dad-on-old 89.1% (-2.0 +- 1.8);
  base-free cans env 79.8% v 86.3%; moss-yard 93 v 93. Nothing resolved: 478dad stays.

## The wrist camera moved (723a1a3): no retrain needed (2026-09-27)

Only the pick reads the wrist camera (478dad: attitude, proximity, size). 478dad under old v
new mount, env 120 episodes: 105 v 105 picked — the proximity slot is live 88% v 61% and
reads ~4 cm longer (median 0.141 v 0.099 m, +0.6 sd on its normaliser), and the policy does
not care. moss-yard 24 seeds: 94 v 99 in bin (paired -0.21 +- 0.43). A fine-tune under the
new camera (0a9726, 478dad's settings, 1.5M steps): env 101/120, yard 91 v 94 (paired
-0.12 +- 0.45). Not shipped; 478dad stays. The mount is a code constant, so run.json cannot
say which camera a run trained under — written on 0a9726's record by hand.
* Since b515d93 it can: every MOSS run records `arm_camera_mount` in its env_kwargs (a
  run.json without it reads as mount 1, the retired one), and MossPickEnv / tidy_moss warn
  when a leg that saw the wrist camera runs behind another mount. 0a9726 backfilled to 2.

## The 3-D wrist fix and a grab check (2026-09-27)

* The wrist sensor passed on ONE number, the lens-to-object range, and it cannot see grip
  depth: 122 yard lifts with the new camera, range median 120-124 mm whether the object was
  deep or at the fingertips (corr -0.1). A depth camera's fix is a 3-D point, so
  `moss_wrist` now also publishes `grip` — the object in the tool frame, same 8 mm noise per
  axis, only when 3 of 5 rays from the lens reach it past the robot (mj_ray, visual meshes
  included). In the yard: available on 96% of pick ticks within 10 cm (optimistic against
  the mount sweep's 88%), corr 0.93 with true grip depth, mean error 7.8 mm; at lift entry
  deep grips read ~14 mm, fingertip ones ~53 mm. No policy or brain reads it yet.
* Using it, 24 seeds v 478dad new camera (94 in bin) — both WORSE, resolved, both halves:
  - grab check (re-open and let the pick continue if > 30 mm, 2 re-seats): 72, paired
    -0.92 +- 0.38. Shallow grips at the lift 41 -> 3, lift losses 12 -> 4 — the filter
    works — but carries 125 -> 91: re-opening rarely becomes a deeper grip, and many
    fingertip grips would have been carried anyway.
  - local retry (restart the pick in place once when its 8 s window runs out): 61, paired
    -1.38 +- 0.42. A pick restarted from a failed state rarely succeeds; it costs 8 s.
  Reverted (patch `grab_check_retry.patch` in scratch). Rejecting a grip does not make a
  better one: the pick has to LEARN to grip deep, with the 3-D fix as an input (new obs
  contract, pick retrain).

## CORRECTION: the yard's fingertip grips come from its handover STATES (2026-09-27)

The replay above ("the start is not the cause") started the env with its target tracker
EMPTY, so the pick's first steps had no target — a replay bug. Seeded with a fresh fix (as
the yard's is), 12 yard seeds at 25 Hz with the new mount, 150 pick starts, 61 lifts, 23
shallow: from the 23 states that ended in a fingertip lift the env's settled grips are
shallow 5 of 6 (picked 16/23); from the 38 deep ones, 4 of 18. From the env's OWN spawns,
3 of 43. So the pick behaves the same in both; the yard hands it over in states that lead
to fingertip grips, and the env scores "picked" at 8 cm of lift, which a fingertip grip
survives and the carry does not. That is why b6cef3 (deep_grip_m) and 0281e5
(sphere_centre_m) were nulls: their spawns rarely produced the grip they penalised.
Retrain design this implies: start episodes from FULL yard handover states (arm joints +
object pose + shape; `handover_bank` holds only object x, y and uprightness, with the arm
at GRASP_POSE), score a deep grip, and give the pick the 3-D grip fix as an input.
