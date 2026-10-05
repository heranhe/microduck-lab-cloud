# Reply to Laurent — 2026-09-24

All five corrections are in, and one of them changed a result rather than a
constant. Taking them in your order:

**Fingers coupled, eight actions.** You were right that nine would let a
policy learn something the hardware can't do. The two finger joints are now
tied by an equality constraint *in the model* rather than by convention in
the brain, so the simulator enforces it and the follower's servo is gone —
six actuators, five arm plus one gripper. The contract is `moss-arm-32-v1`:
32 observations, 8 actions, 25 Hz. A brain that says "open both jaws" is
still accepted and folded onto the one servo, since that's your MJCF's old
vocabulary, but it can't give them different values.

That coupling invalidated a table we'd measured the day before. With the soft
two-servo default, contact impulses on the unactuated follower could throw
the can out, so we had "2–6 mm of interference holds, 10 mm ejects". With one
servo and a stiff coupling there is no narrow window at all: every closure
that touches a 66 mm can holds it, and only a gap fails. We kept 0.027 m
(6 mm interference) as the gentlest setting that still holds. The old numbers
are recorded as *wrong* rather than deleted, because they're exactly the kind
that gets quoted for a year.

**Mass and balance.** 3.18 + 0.14 + 0.18 are now carried as three separate
components at their own centroids, not lumped at the axle — the cover 25 mm
behind and 124 mm up, the bin 87 mm behind and 185 mm up, read off your V0.4
manifest's mesh bounds. So your balance point stays attached to the 3.18 kg
configuration it was measured in. The loaded-bin COM and inertia are flagged
explicitly as unmeasured; a can in the bin shifts the combined COM rearward
by itself here because it's a free body resting on the bin floor, so its
weight and moment arrive through contact rather than an assumed tensor — but
nothing claims a revised inertia. On the arm: your supplied model contains
0.820 kg of links against your 0.682 kg revised estimate, so our total is
4.32 kg today and will be ~4.18 kg whenever that revision lands.

**D455f.** 87° × 62° is what we declare, with the family's 90° × 65° recorded
beside it, and we'll take stream intrinsics whenever you have them. The
depth minimum is 0.52 m. We've kept RGB visibility and valid depth as
separate questions as you asked: the 0.12 m figure is now labelled as what it
actually is — a geometric near limit from the 75 mm lens height against the
vertical field — and nothing in the loop uses depth. Latency, dropout and the
noise sigmas are all marked assumptions rather than measurements.

**Track spacing.** Taking your 25 September call: the collision geometry
matches the V0.4 CAD. We run `prepare_candidate.py` as published — it does
all three things we'd each got wrong ourselves, including the bin walls
(we'd raised them 35 mm off a mis-transformed mesh reading, exactly the
coordinate-transform error you predicted) and the arm hulls (we'd widened
your 13 mm capsules to a uniform 35 mm, and with self-contacts opened our
fattened version admits *zero* legal tucks out of 500,000 sampled poses;
yours admits some).

One compatibility note: your patcher needs an explicit rover inertia, so it
has to run *after* our inertial edit rather than on your raw file. It
preserves masses, joints, actuators, equalities, keyframes and the camera
through that ordering.

**And the thing you most need to know, which your geometry uncovered.**
Adopting it puts MOSS on the ground for the first time. Your original track
boxes sit with their underside 4 mm proud of the floor, so the robot has
never made a single floor contact in our simulator — **every drive number we
have ever sent you is from a hovering robot.** The −1.5% yaw tracking, and
our finding that 266 vs 244 mm "doesn't affect turning", are both artefacts
of tracks touching nothing to turn against.

With your belt hulls on the floor, commanded 0.2 m/s for 2 s the rover
travels 3 mm instead of 400. That isn't a gain to tune. Our base applies a
force to the chassis and treats belt-to-ground friction as pure drag, so it
pays the resistance twice and collects the propulsion never; a real tracked
vehicle is *propelled* by that same friction. Sweeping it: 0.9 → 3 mm,
0.2 → 33 mm, 0.08 → 140 mm, 0.03 → 314 mm. Reaching the commanded speed
needs about 0.02, which is not a rubber track on a floor.

So the geometry is right and our driver is what needs the work — either the
belts get modelled as driven surfaces, or the drive is replaced with
something traction-limited. **Please don't calibrate skid-steer width against
hardware until that's settled**, because there is currently no friction model
on our side to calibrate against. The candidate ships available and
unadopted behind a flag for that reason alone: switching it on fails 7 of
116 tests, including the drive ones, and every trained policy would need
retraining against it.

**One more that touches your bin.** The scripted release reaches the bin
mouth by *leaning on it*. The pose it was commanded to isn't reachable while
carrying — the arm stalled 0.44 rad short with 33 N of forearm and wrist
against the bin's front wall on every release, five servos held in a stall —
and it delivered anyway, because that stall is what parks the gripper over
the mouth. It now commands the pose the arm actually settles at, which
reaches the same place resting rather than driving: same delivery rate,
33.6 N → 7.3 N. Searching for a genuinely contact-free release pose made it
*worse* (12/12 in the isolated stow env, 5 of 16 in the full room, letting go
20 cm too high). So the contact is doing real work, and a release that leans
on the bin depends on the bin being where the model says — another place the
unreconciled collision set would show up first.

**The schema.** Ordered observation and action tables, with units, scaling,
and how to build each slot: `docs/moss-policy-schema.md`, sent alongside.
The parts that will bite if they're missed:

- Actions are **increments, not targets** — `cmd[j] += a[i] × 0.03 rad` for
  the arm, `× 0.004 m` for the gripper. A nudge can say "hold" by being
  zero; an absolute target can't.
- **Clip the ONNX output to ±1 before scaling.** Raw output reaches 3.13.
- `base_twist` must be the **measured** twist from the encoders, not the
  commanded value.
- The normalizer is baked into the exported graph — don't normalise again.
- `target_seen` must be honest. During the stow it is legitimately 0 on 92%
  of steps (the can is inside the jaws, occluded, and far nearer than the
  0.52 m depth minimum), and that leg runs on proprioception. Handing it a
  stale belief dressed as a fresh one is the one mismatch that reliably
  breaks it.
- 25 Hz: inference is 8 µs and the whole tick 26 µs, so 0.06% of the budget.

Happy to take you up on weighing things separately — the loaded-bin COM and
an inertia tensor for the bin with a can or two in it are the two numbers
that would most improve fidelity, in that order.


---

## Later the same day: your collision candidate, measured

Three things, and the third is a question only you can answer.

**Your geometry is right and our drive was the blocker — that is fixed.**
Adopting the candidate used to stop the rover dead: commanded 0.2 m/s for 2 s
it travelled 3 mm instead of 400. Not a gain. Our base is a planar joint, so
the rover's 42.4 N is carried by the JOINT and the tracks bear 0.00 N at rest
— an unloaded track cannot make traction, so friction there was pure drag. We
now declare the belts a SUPPORT surface rather than a traction one, and the
yaw loop's standing error turned out to be a gain (a constant 64% of every
commanded rate, 0.2 through 1.0 rad/s, which is a gain's signature and not
saturation). On your hulls MOSS now holds station to 0.0 mm, travels 0.385 m
of a commanded 0.400 and turns 1.152 rad of a commanded 1.200 — 96% on both
axes. What that does NOT buy is a friction model: mu = 0.01 is not a rubber
track, and skid-steer width is still uncalibratable on our side, so please
don't measure against it yet.

**Your geometry is also better for the task, which we did not expect.** All
three trained policies transfer to it with no retraining — approach 12/12
either way, pickup 22/24 on the old geometry and 23/24 on yours, stow 11/12
on both — and the whole litter mission goes from 32% to 36% of cans delivered
over 72 attempts. We had assumed adoption would cost a full retrain. It does
not.

**Correction to the paragraph that was here: your arm hulls are fine.**

We first measured zero legal tucks out of 400,000 poses on your candidate and
were about to ask you whether the hulls were meant as a conservative envelope.
That measurement was broken, and by our own change an hour earlier: putting
the belts on the floor means every pose now carries two track-floor contacts,
and the tuck probe's test was "no contacts at all" — written back when the
legacy boxes sat 4 mm proud and the robot touched nothing. The test could not
pass for any pose. The contact census gave it away, `floor|track_left` and
`floor|track_right` in 8,977 of 8,977 poses.

Asking what the tuck actually asks — does the ARM touch anything — your hulls
admit a tuck at `(-0.5014, -1.7374, 1.5055, 1.3084, 1.9457)`: reachable to
0.010 rad, 0.00 mm outside the rover's shell, top 0.2583 m against the
0.261 m rim, zero arm contacts. It is a different pose from the one we use on
the old proxies, which is expected — that one stands 13.31 mm proud on your
hulls, and the new one stands 4.28 mm proud on the old ones. Each collision
model needs its own folded pose.

So nothing in your candidate is worse than what it replaces. Drive fixed on
our side, all three policies transfer with no retraining, the mission improves
32% -> 36%, and the tuck exists. What remains before it becomes our default is
bookkeeping on our side: four tests that encode the old geometry, two of them
tolerances describing a robot that used to hover.
