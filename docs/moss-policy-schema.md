# MOSS policy I/O schema — `moss-arm-32-v1`

What an ONNX exported by this lab expects and emits, in order, with units and
scaling, so a code skill on the Jetson can drive it at 25 Hz.

Written from `microduck_local/src/microduck_local/robots/moss.py`; if this
file and that module disagree, the module is right and this is stale.
Regenerate the numbers with `uv run python scripts/moss_schema.py`.

**Untested on hardware.** Nothing in this repo has driven a MOSS. Every
sensing constant below is a datasheet figure or a stated assumption, marked
as such — the ones to replace first are the camera's.

## The loop

    25 Hz. Every tick: build 32 floats, run the graph, apply 8 floats.

**It fits, with room to spare.** MEASURED single-threaded on a laptop CPU:
the graph costs **~8 µs** a tick and the whole sense-decide-command loop
**~26 µs** (p95 33 µs) against a 40,000 µs budget — 0.06%. Whether MOSS holds
25 Hz is therefore a question about the detector, not about any of this. A
reference implementation of the arithmetic below, runnable and checked, is
`moss_runner.py` beside this file.

    The observation normalizer is BAKED INTO the graph — do not scale the
    inputs yourself and do not keep a separate scaler in sync.

## Observation — 32 floats, float32, shape [1, 32]

| idx | name | n | units | how to build it |
|---|---|---|---|---|
| 0–6 | `joint_pos` | 7 | rad ×5, m ×2 | measured joint position **minus** `DEFAULT_POSE`. Order: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, finger_left, finger_right`. `DEFAULT_POSE` = `1.34, −0.65, 0.40, 1.25, −0.05, 0.037, 0.037`. Both fingers are *reported* even though only one is *commanded*. |
| 7–13 | `joint_vel` | 7 | rad/s ×5, m/s ×2 | same order, raw. |
| 14–21 | `last_action` | 8 | −1…+1 | the previous tick's action vector, exactly as emitted. Zeros on the first tick. |
| 22–23 | `base_twist` | 2 | m/s, rad/s | **measured** body-frame `(vx, wz)` — from the encoders through `tracks_to_twist`, not the commanded value. |
| 24–26 | `target_base` | 3 | m | the can in the **base frame**, x forward / y left / z up. The tracker's *belief*, not ground truth — see below. |
| 27 | `target_seen` | 1 | 0 or 1 | 1.0 while the fix is fresher than 0.6 s, else 0.0. |
| 28–31 | `reserved` | 4 | — | zeros. Keep them zero. |

### Building `target_base` (24–26)

The policy was trained on a *tracked belief*, so feed it one:

1. Detect the can in RGB. Range is inverted from apparent width:
   `range = R / tan(width/2)`, `R` the radius your detector models the class
   as. **Not a stereo-depth reading** — valid depth on a D455f starts near
   0.52 m and the grasp standoff is 0.26 m, so there is no usable depth
   where this matters.
2. `x = 0.156 + range·cos(bearing)`, `y = range·sin(bearing)` — the 0.156 m
   is the camera's front-face offset ahead of the rover origin, your CAD
   figure. `z` is the can's half-height, 0.0575 m.
3. Between frames, hold the last fix and re-express it with odometry as the
   rover moves. Do **not** extrapolate the can's own motion: measured here, a
   velocity differenced from two 10 Hz fixes carries ~0.2 m/s of noise, and
   predicting made the belief worse (22.2 mm against 16.5 mm).
4. After 0.6 s with nothing reported, zero slots 24–26 **and** slot 27.

## Action — 8 floats, float32, shape [1, 8]

> **CLIP EVERY ELEMENT TO −1…+1 BEFORE APPLYING IT.** The graph is the
> policy's *mean* action and its raw output is not bounded: measured on the
> three shipped files, the largest magnitude from a zero observation is 1.73
> (approach), **3.13** (pick) and 1.74 (stow). Every one of these policies was
> trained inside an env that clips first (`np.clip(action, -1, 1)`), so an
> unclipped 3.13 moves a joint three times further in one tick than anything
> it ever experienced. One line, and it is the difference between the arm you
> trained and an arm that overshoots every increment.

    a = np.clip(onnx_out[0], -1.0, 1.0)   # then apply the scalings below


| idx | name | applies to | scaling |
|---|---|---|---|
| 0–4 | `arm_delta` | the five arm joints, order as above | **an increment, not a target**: `cmd[j] += a[i] × 0.03 rad`, clamped to the joint range. A nudge has a zero and zero means *hold*; an absolute target has no way to say "stay". |
| 5 | `gripper_delta` | the single gripper servo | `cmd += a[5] × 0.004 m`, clamped 0…0.041 m of finger travel. **One command for both jaws.** Pad-centre separation is `16 + 2·cmd` mm; a 66 mm can grips at `cmd ≈ 0.027`. Re-measured after the jaws were coupled: **every closure that touches the can holds it** (3/5 lifted at `cmd` = 0.000 through 0.026) and only a gap fails (0/5 at 0.030, a 2 mm gap). The earlier "2–6 mm holds, 10 mm ejects" was an artefact of the soft two-servo coupling and does not apply to the one-servo model. 0.027 is used because it is the gentlest setting that still holds — 6 mm of interference — which is what a thin aluminium can wants even though this simulator no longer punishes a crush. |
| 6 | `vx` | forward speed | `vx = a[6] × 0.20 m/s`, then through your own `twist_to_tracks` scaling with track spacing **0.266 m** and the 0.6 m/s per-track limit, so an unreachable twist slows both tracks instead of flattening the turn. |
| 7 | `wz` | yaw rate | `wz = a[7] × 1.0 rad/s`, same scaling step. |

The policy holds its own integrated arm command. Start it at the deploy pose
(`GRASP_POSE`, jaws open at 0.041) and keep applying increments to that — do
not re-read the measured position into the command, or backlash integrates.

## Three policies, one contract

The mission loop is **three separate ONNX files**, all under this same
32-in/8-out contract, so the runtime loads whichever the current state needs
and the wiring never changes. `brain/tidy_moss.py` sequences them; only
`search` — turn on the spot until the detector reports a can — is still
scripted, because there is nothing in it to learn.

| leg | file | hand it control when | take control back when |
|---|---|---|---|
| `approach` | `moss-approach-v1` | a can is seen and further than ~0.55 m | the can is within ~0.55 m ahead |
| `pick` | `moss-pick-v1` | the arm is deployed and the can is within ~0.55 m | it reports a grip |
| `stow` | `moss-stow-v1` | a grip is held | the can is at rest inside the bin |

Each expects the arm to BE somewhere when it takes over — the pose its
training reset used — and this is not a detail. Handing the pickup policy
control in the wrong pose scored 0/3 in the room against 4/12 in its own
env, which is why the handover waits for the arm to arrive rather than for a
timer:

| leg | arm pose at handover |
|---|---|
| `approach` | folded (see the tuck note below) |
| `pick` | `GRASP_POSE` — jaws open at can height, ~0.26 m ahead |
| `stow` | wherever the grip was made; the policy takes it from there |

**Which slots are live depends on the leg.** `target_base` / `target_seen`
carry the tracker's belief, and it is MEASURED fresh on 98% of steps during
`pick` and on only **8%** during `stow`. That is not a training shortcut to
be corrected — a can held in the gripper is inside its own jaws, occluded
and far nearer than the 0.52 m depth minimum, so the stow policy runs on
PROPRIOCEPTION (joint positions, joint velocities, the grip) and that will
be just as true on the robot. Two consequences for the hardware loop: the
stow leg does not need a detector fix to be running, and it must not be
handed a stale belief dressed up as a fresh one — keep `target_seen` honest
and let it read 0.

**The scripted release touches the bin, and you should know that before it
runs on metal.** The delivery is not the learned stow by default — it is a
three-waypoint ramp (up, round the shoulder, down) ending with the jaws
opening over the bin mouth. Measured 2026-09-24: the pose that ramp was
originally commanded to is **not reachable while carrying**. The arm stalls
0.44 rad short with 33 N of forearm and wrist against the bin's own front
wall, on every release of 51 — five servos held in a stall for the whole
release. It still delivered, because that stall is what parks the gripper
over the mouth (can at x = −0.128 m in the base frame, mouth −0.164…−0.010,
z = 0.318 over a 0.261 m rim).

It now commands the pose the arm actually **settles** at, which reaches the
same place with the arm resting on the wall rather than driving into it:
same delivery rate, 33.6 N → 7.3 N. Two things follow for the robot. First,
searching for a genuinely contact-free release pose made it worse, not
better — eleven candidates reached the mouth with zero wall force and the
best delivered 12/12 in the isolated stow env and **5 of 16** in the full
room, letting go 20 cm too high and past the mouth. So the contact is doing
real work and removing it needs a different route, not a different pose.
Second, a real release that leans on the bin depends on the bin being where
the model says: if the V0.4 bin geometry differs from the 244/266 mm
collision set we have not yet reconciled, this is one of the places it will
show up first.

**How wrong may the drive be?** Measured on the approach policy, by scaling
what the robot actually achieves against what it is told:

| drive error | arrived | disturbed the can |
|---|---|---|
| turn rate 0.20x to 3.00x | 12/12 at every setting | 0/12 |
| speed 1.00x | 12/12 | 0/12 |
| speed 0.50x | 9/12 | 0/12 |
| speed 0.30x | 4/12 | 0/12 |

TURN-RATE ERROR DOES NOT MATTER to this policy. It closes the loop on the
can's bearing, so it corrects whatever the yaw actually does — which means
the effective skid-steer width does not have to be calibrated for the
approach leg to work. Speed matters only below about half, and those
failures are the clock rather than control (at 0.2x the rover covers 0.34 m
in an 8.4 s episode against spawns to 1.5 m). Nothing disturbed the can at
any setting, so the property the pickup depends on survives drive error.

Worth knowing why that question needed asking: in this simulator the tracks
never touch the floor — the collision proxies sit 4 mm proud — so the base
follows its twist almost exactly and the sim cannot tell you anything about
slip. The table above is the answer to "does that matter", and for this leg
it does not.

**The arm does not reach its tuck.** Commanded to `TUCK_POSE` it jams on the
hull and the track (`hull/palm`, `track_1/palm`) 0.74 rad short and stays
there, riding ~75 mm proud of the chassis; routed via the drop pose it jams
25 mm out instead. No commanded path tried reaches the pose. The approach
policy is therefore paid to fold the arm in using the arm actions it already
holds, rather than being handed a pose the robot has never held. Worth
reconciling against the real arm — it may be a model artefact, and if it is
real it is the wall-catching you described.

**The documented drop pose is not reachable while carrying.** `DROP_POSE`
stalls `shoulder_lift` 0.379 rad short against `bin_x1`, the bin's own front
wall; the jammed servo stores energy and opening the jaws flings the can
clear of the bin. The stow policy has to find its own way over the rim, and
the reachable set does contain one — of 60,000 random arm poses, 179 put the
gripper contact-free just above the rim and 93 inside the bin. **This is a
geometry question for the real robot: can the SO-101 on MOSS actually place
an object in its own bin, or does the arm foul the bin's front wall there
too?**

## Numbers that are assumptions, not measurements

| quantity | value here | status |
|---|---|---|
| camera FOV | 87° × 62° | D455f **product page**; the family datasheet says 90° × 65°. Replace with your stream intrinsics. |
| detector rate | 10 Hz | assumption |
| detector latency | 50 ms | assumption |
| bearing noise | 0.02 rad, 1σ | assumption |
| range noise | 15 mm, 1σ | assumption |
| dropout | 8% of frames | assumption |
| RGB near limit | 0.12 m | geometric: 75 mm lens height against a 62° vertical field |
| depth near limit | 0.52 m | published; **nothing here uses depth** |
| track spacing, drive | 0.266 m | V0.4 CAD belts. Collision proxies are still 0.244 m — unreconciled, and turning must be validated on the real robot. |
| base mass | 3.18 + 0.14 + 0.18 kg | weighed by you; the arm's 0.82 kg is the model's, against your 0.682 kg revised estimate |
| COM | composed, ≈ (−0.006, 0, 0.083) m | from your components; the 75 mm height is your estimate |
| loaded-bin COM | −20 mm per kg in the bin | **simulated, not weighed** — see below |
| loaded-bin inertia | not modelled | a can in the bin contributes its mass by contact; no revised tensor exists |


## Loaded bin: what the model says, and what still needs your scale

You flagged that we had not measured the loaded-bin COM or inertia and that
weight in the rear bin shifts the combined centre rearward. That is right, and
here is the size of it — composed over every body about the composite centre,
three cans in the bin:

| per can | total | COM x | shift | Ixx | Iyy | Izz |
|---|---|---|---|---|---|---|
| 15 g (empty can) | 4.365 kg | +19.7 mm | −1.1 mm | 0.0448 | 0.1039 | 0.1114 |
| 200 g | 4.920 kg | +7.6 mm | −13.1 mm | 0.0516 | 0.1073 | 0.1177 |
| 350 g (full can) | 5.370 kg | −0.3 mm | −21.1 mm | 0.0561 | 0.1092 | 0.1227 |

About **−20 mm of COM travel per kilogram** in the bin, and +7.5 mm up.

Two readings. For the litter the robot is actually for — empty cans at ~15 g —
the payload is negligible: three of them move the centre about a millimetre,
so a pickup with the arm extended is not a loaded-stability problem. At the
other end you are exactly right: three full cans put the composite centre at
x = −0.3 mm, across the balance point you measured, and that is the regime
where a measured tensor would change decisions.

**This is the model, not a measurement.** The rover carries your weighed
components at their own centroids and the cans are free bodies resting on the
bin floor, so their weight and moment arrive through contact rather than
through an assumed inertia — but nothing here has been on a scale with a load
in the bin. If you do weigh it, the row that matters is the tensor at a known
payload; the COM shift above is the prediction it would confirm or correct.
