"""The floor get-up: lying on the ground -> a stand it can hold (bead
mdl-0ad, roadmap B.1, 2026-09-08).

`arena.py` teleports a fallen duck upright and counts a `falls`; `World.
getup_s` makes it lie there first for as long as a real recovery would
cost. Neither is a get-up, and bead mdl-0ad assumed the real one needed
the upstream GPU stack (`Mjlab-StandUp-Flat-MicroDuck`).

READ THIS BEFORE TRAINING ANYTHING. Building the bench for this recipe
(`scripts/bench_getup.py`) turned up something the roadmap said was not
there: **the shipped `alpha_stand.onnx` already does the get-up.** From a
settled lie on the floor, honest BAM servos, 12 seeds per pose with
observation noise, domain randomization and the action delay on, it
recovers 36 of 36 — back, front and side — reaching a full stand (jaw at
0.234 m against the STAND keyframe's 0.233) in 0.22-0.55 s median and
holding it to the buzzer. Rendered and read, not inferred from a sum.
Nothing else shipped comes close: `alpha_walking` 0 of 24, `alpha_sitstand`
0 of 24, `alpha_ground_pick` 75% from the front only.

So the gap in the soccer world is not a missing POLICY, it is a missing
CONTROLLER SWITCH: a duck that falls on the pitch is still being driven by
the walker, which cannot get up, and `arena.py` teleports it. Handing a
fallen duck to `alpha_stand` for a second and a half would replace the
teleport with a real recovery today, with nothing trained at all.

This recipe is still worth having — it is the bench's definition of
standing, it is the thing to retrain when the walker's own recovery is
wanted, and it is how the skill gets measured — but it should not be sold
as the only way to have a get-up here.

WHAT THE LADDER MEASURED (2026-09-08, seed 3, 12 envs).
  * The control, `runs/getup-flat-scratch` — this recipe's LAST rung trained
    from scratch, 2M steps — learns the front push-up (83%) and nothing
    else: 0% from the back, 0% from the side. That is the exploration gap,
    and it is not a reward problem: scored under this very recipe,
    `alpha_stand` earns 13.2/step against that run's 3.3.
  * The ladder closes it by rung TWO. `runs/getup-l2` (3.5M steps, still on
    the XML servos) recovers 36 of 36 on honest BAM with noise, DR and the
    action delay on — 0.22 / 0.52 / 1.26 s to stand from front / back /
    side — and holds the stand unbroken for 20 s.
  * Rungs 3-5 keep the recovery (100%) and LOSE THE HEAD. The bench's
    head-up column falls 100% -> 3% -> 0% -> 0% while every gate below
    still passes: the trunk is at full height and perfectly upright, and
    the jaw hangs at 0.15 m instead of 0.23. Nothing in this recipe prices
    head HEIGHT — `head_up` scores joint ANGLES and is worth 0.8 — so once
    the starts get hard the head becomes a free counterweight. That is the
    term this recipe is missing, and the next revision should put head
    height inside the salary gate rather than beside it.

WHY IT IS A CURRICULUM AND NOT A REWARD. A duck flat on its back never
rolls itself upright by accident, so no term, weight or ramp can teach the
last mile of a get-up from there: the value of a state no rollout visits is
never learned (AGENTS.md, "Reward design cannot fix an exploration gap").
So the PHYSICS ladders and the pay does not. One spawn function poses the
duck at a TILT away from upright, and the stages walk that tilt down from
"tipped over and falling" (45-75 deg, still catchable by a fresh brain) to
"flat on the floor" (80-115 deg), stepping the servos from the XML's
phantom-strong ones to the honest BAM XL330s on the way. The reward terms
are identical in every stage - only the world gets harder.

The pay has one salary (upright AND tall AND on both feet, ramped so
DWELLING beats flickering) plus two potential-based shepherds (uprightness
and trunk height, symmetric deltas per the headstand's `_hs_update`, so
climbing pays, sliding back charges, and parking nets exactly zero). Both
potentials are things the policy can SEE: uprightness is the projected
gravity that rides in obs[3:6], and trunk height is a function of the leg
angles in obs[6:20] once the feet are down.

Everything about a fall is observable, so the three poses share one policy:
a duck on its back reads gravity (-1, 0, 0) in the trunk frame, on its
front (+1, 0, 0), on its side (0, -/+1, 0). Nothing is hidden, so nothing
has to be remembered.
"""

import numpy as np

from .. import contract as C
from .core import (
    Behavior,
    CurriculumStage,
    RewardTerm,
    _both_feet_down,
    _flat_feet,
    _head_up_blend,
    _limit_parking_pen,
    _register,
    _spawn_knob,
    _upright,
)
from .poses import _pose_home, _stand_tall

# The stand the salary is willing to pay for. -1 is perfectly upright;
# -0.85 is ~32 deg of lean, comfortably inside "standing" and well clear of
# the walk env's own fall line (-0.342, ~70 deg).
STAND_GZ = -0.85
# ... and how far the trunk may sit below the STAND keyframe height and
# still count. 4.5 cm is deeper than `crouch` (3.5) and shallower than
# `deep_squat` (5.5): a duck that has got its feet under it but is still
# folded is not yet standing.
STAND_DROP = 0.045
# Default tilt window (degrees off upright) for an UNSTAGED run — the
# ladder below overrides it per stage. 80-115 is flat on the floor.
TILT_LO, TILT_HI = 80.0, 115.0


# --------------------------------------------------------------- the spawns

def _getup_place(env) -> None:
    """Drop whatever pose is in qpos until its lowest geom rests on the
    floor, then prime the controller and the BAM state at that pose.

    Lifted from `poses._stand_spawn_ground`, plus the BAM reset it omits:
    the spawn families run AFTER `MicroduckWalkEnv.reset`, which primed the
    servo delay buffer at the STAND keyframe. A duck spawned on its back
    with a standing pose in the buffer gets one control step of a target it
    never asked for (the kick's `reset_fn` resets it for the same reason).
    """
    import mujoco

    d, m = env.data, env.model
    d.qpos[2] = 0.6
    d.qvel[:] = 0.0
    mujoco.mj_forward(m, d)
    lows = [float(d.geom_xpos[g][2]) - float(m.geom_rbound[g])
            for g in range(m.ngeom) if g != env.floor_geom]
    d.qpos[2] = 0.6 - min(lows) + 0.003
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    if getattr(env, "bam", None) is not None:
        env.bam.reset(d.qpos[env.joint_qpos_adr])

    # SETTLE, and this is not cosmetic. Placed-and-forwarded, the duck is
    # 3 mm clear of the floor with the pose it was posed in and zero
    # velocity — it is ARRIVING, not lying. Measured on the first version
    # of this spawn, a back-lying duck released limp went on falling for
    # ~1 s, trunk 0.10 -> 0.047, as the shells rolled it flat and the legs
    # flopped. Starting the episode at frame 0 of that hands the policy
    # several centimetres of unspent potential energy and a leg geometry
    # that has not collapsed yet, and `alpha_stand` "recovered" from it in
    # 0.4 s — which measures the arrival, not a get-up. So the physics runs
    # first, with the servos holding the pose they fell in (a real robot's
    # controller is still running when it lands), and the episode starts
    # from what is left.
    settle = float(_spawn_knob(env, "MICRODUCK_GETUP_SETTLE_S", "1.0"))
    if settle > 0.0:
        target = d.qpos[env.joint_qpos_adr].copy()
        if env.bam is None:
            for _ in range(int(round(settle / C.PHYSICS_DT))):
                mujoco.mj_step(m, d)
        else:
            env.bam.set_target(target)
            for _ in range(int(round(settle / C.PHYSICS_DT))):
                env.bam.before_step()
                mujoco.mj_step(m, d)
                env.bam.after_step()
        d.qvel[:] = 0.0            # at rest, not mid-bounce
        d.ctrl[:] = d.qpos[env.joint_qpos_adr]
        mujoco.mj_forward(m, d)
        if env.bam is not None:
            env.bam.reset(d.qpos[env.joint_qpos_adr])
    env.prev_joint_vel = env._joint_vel().copy()


def _getup_spawn(env, kind: str):
    """Pose the duck `kind`-down, tilted by the stage's tilt window.

    Tilt is the angle away from upright, so it IS the curriculum: at 45 deg
    the duck is merely toppling and a fresh brain can still catch it; at
    100 deg it is flat, which is the skill we actually want and the state
    nothing samples by accident.
    """
    r = env._rng
    d = env.data
    lo = float(_spawn_knob(env, "MICRODUCK_GETUP_TILT_LO", str(TILT_LO)))
    hi = float(_spawn_knob(env, "MICRODUCK_GETUP_TILT_HI", str(TILT_HI)))
    tilt = np.deg2rad(r.uniform(lo, hi))
    if kind == "back":
        pitch, roll = -tilt, r.uniform(-0.2, 0.2)
    elif kind == "front":
        pitch, roll = tilt, r.uniform(-0.2, 0.2)
    else:  # side — rolled onto the left or the right shoulder
        pitch, roll = r.uniform(-0.2, 0.2), tilt * float(r.choice((-1.0, 1.0)))
    d.qpos[:] = 0.0
    # Same composition as `_spawn_inverted`: pitch about y, roll about x.
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    d.qpos[3:7] = [cp * cr, cp * sr, sp * cr, sp * sr]
    # Legs somewhere between straight out and folded under — a fall does not
    # arrange them, and a get-up that only works from one leg configuration
    # is not one. Signs mirror `_stand_spawn_ground` (left hip/knee positive,
    # right negative).
    fold = r.uniform(0.0, 1.0)
    q = d.qpos
    for adr in env.joint_qpos_adr:
        q[adr] = r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[2]] = 1.1 * fold + r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[11]] = -1.1 * fold + r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[3]] = 1.2 * fold + r.uniform(-0.15, 0.15)
    q[env.joint_qpos_adr[12]] = -1.2 * fold + r.uniform(-0.15, 0.15)
    # Head flopped somewhere in its range, as it would be after a fall.
    q[env.joint_qpos_adr[5]] += r.uniform(-0.3, 0.3)
    q[env.joint_qpos_adr[6]] += r.uniform(-0.3, 0.3)
    _getup_place(env)
    return env._get_obs()


def _getup_spawn_family(kind: str):
    def fn(env):
        return _getup_spawn(env, kind)
    # BehaviorEnv.reset labels the duck by everything after "_spawn_" — this
    # is what puts "back" / "front" / "side" on the viewer's duck and in the
    # contact-sheet caption, so a rendered sheet says which fall it recovered
    # from.
    fn.__name__ = f"_getup_spawn_{kind}"
    return fn


# ---------------------------------------------------------------- the terms

def _getup_upright(env) -> float:
    """Two-layer uprightness straight off the projected gravity the policy
    observes: 1.0 standing, ~0.25 lying flat, ~0 upside down.

    Wide layer first, for the same reason every other recipe here has one:
    a tight-only Gaussian pays nothing where a fallen duck actually IS, and
    a term with no gradient at the current behaviour teaches nothing (the
    `head_up` / `flat_stance_foot` postmortems)."""
    gz = float(env._projected_gravity()[2])
    d2 = (gz + 1.0) ** 2          # 0 upright, 1 flat on the floor, 4 inverted
    return (0.5 * float(np.exp(-d2 / 1.2 ** 2))
            + 0.5 * float(np.exp(-d2 / 0.35 ** 2)))


def _getup_height(env) -> float:
    """Trunk height as a 0..1 potential (1 = the STAND keyframe height)."""
    return float(min(1.0, max(0.0, env._trunk_xpos[2] / env.stand_z)))


def _getup_props_down(env) -> int:
    """Floor contacts from anything that is NOT a foot — the trunk, the hips,
    the shins, the beak. On a duck that has finished getting up this is zero;
    on every way of half-getting-up it is not.

    Measured, not guessed: the first getup run (getup-probe-s3, 400k steps)
    converged to a TRIPOD — both feet planted, body folded forward, jaw on
    the floor for 96% of frames, parked there for the whole 8 s at tilt 40
    deg. Every term in the recipe was happy: trunk_z 0.105 of 0.120, both
    feet down, legs near home. Nothing charged for the third leg, so the
    cheapest way to be tall was to lean on its face. This is the price.
    """
    cache = env._step_cache if env._cache_active else None
    if cache is not None:
        v = cache.get("getup_props")
        if v is not None:
            return v
    feet = set(env.foot_geoms.values())
    v = 0
    n = int(env.data.ncon)
    if n:
        con = env.data.contact
        g1 = con.geom1.tolist()
        g2 = con.geom2.tolist()
        floor = env.floor_geom
        for i in range(n):
            a, b = g1[i], g2[i]
            if a == floor:
                other = b
            elif b == floor:
                other = a
            else:
                continue
            if other not in feet:
                v += 1
    if cache is not None:
        cache["getup_props"] = v
    return v


def _getup_prop_pen(env) -> float:
    """Rent (<= 0) for every part still on the ground that isn't a foot.
    Bounded — it saturates at four contacts, so a duck lying flat pays a
    fixed rent rather than an unbounded one that would swamp the salary and
    make the attempt itself the expensive part."""
    return -min(0.25 * _getup_props_down(env), 1.0)


def _getup_hold_raw(env) -> float:
    """The one salary, gated four ways: both feet on the floor, NOTHING else
    on the floor, the trunk upright, and the trunk TALL. All four, because
    each of the first three alone is a pose the reward batteries here have
    already mistaken for standing — a face-down crouch passes the height
    test, a duck sitting on its tail passes the upright one, a heap with two
    feet touching passes the contact one, and the tripod this recipe's own
    first run found passes three of the four."""
    c = env.foot_contact_state
    if not (c["left"] and c["right"]):
        return 0.0
    if _getup_props_down(env) > 0:
        return 0.0
    if float(env._projected_gravity()[2]) > STAND_GZ:
        return 0.0
    if float(env._trunk_xpos[2]) < env.stand_z - STAND_DROP:
        return 0.0
    return _upright(env)          # tight polish, 0..1, inside the gate


def _getup_hold(env) -> float:
    """The salary scaled by a persistence ramp — the headstand's fix for the
    flicker equilibrium. A policy that pops upright and falls back collects
    30% of the pay; one that holds for a full second collects all of it, so
    DWELLING is strictly richer than cycling, and nothing farmable is added
    (the ramp only scales pay that already cleared every gate)."""
    raw = _getup_hold_raw(env)
    streak = getattr(env, "_gu_streak", 0)
    return raw * (0.3 + 0.7 * min(streak / 50.0, 1.0))


# Potentials the two shepherd terms read back. Keep this tuple and the
# registered `*_gain` terms in one-to-one correspondence (the headstand grew
# a term that summed a potential nobody paid).
_GU_POTENTIALS = (("rise", _getup_upright), ("lift", _getup_height))


def _getup_update(env) -> None:
    """Per-step bookkeeping: the hold streak, and symmetric potential-based
    shaping deltas (Ng et al.) for the two shepherds.

    Symmetric, not best-so-far: climbing pays +, sliding back charges -, so
    an episode's shaping income telescopes to phi(end) - phi(start). Parking
    nets zero, cycling nets zero, and every re-approach still has dense
    gradient — the headstand learned the hard way that best-so-far gating
    kills parking AND kills the come-back-this-way pull that a scratch brain
    needs. Baselines anchor to the first post-spawn state, so a spawn that
    hands out half the climb banks none of it."""
    env._gu_streak = ((getattr(env, "_gu_streak", 0) + 1)
                      if _getup_hold_raw(env) > 0.05 else 0)
    prev = getattr(env, "_gu_prev", None)
    cur = {k: fn(env) for k, fn in _GU_POTENTIALS}
    env._gu_gain = ({k: 0.0 for k in cur} if prev is None
                    else {k: cur[k] - prev[k] for k in cur})
    env._gu_prev = cur


def _getup_gain_term(key: str):
    def term(env) -> float:
        gains = getattr(env, "_gu_gain", None)
        return 0.0 if gains is None else gains[key]
    term.__name__ = f"_getup_gain_{key}"
    return term


def _getup_report(env) -> list[str]:
    """Extra episode-summary lines for render-rollout: did it stand, and
    when. This is the measurement the bench script and the contact sheet
    both want, computed by the env that owns the definition of "standing"
    rather than by two callers that would drift apart."""
    return [f"spawn: {getattr(env, 'last_spawn', '?')}",
            f"upright: gz {float(env._projected_gravity()[2]):+.2f} "
            f"(stand <= {STAND_GZ:+.2f})",
            f"trunk z: {float(env._trunk_xpos[2]):.3f} "
            f"(stand {env.stand_z:.3f})",
            f"non-foot floor contacts: {_getup_props_down(env)}",
            f"standing now: {'yes' if _getup_hold_raw(env) > 0.0 else 'no'}"]


def _getup_caption(env) -> str:
    c = env.foot_contact_state
    return (f"gz {float(env._projected_gravity()[2]):+.2f} "
            f"z {float(env._trunk_xpos[2]):.3f} "
            f"feet {int(c['left'])}{int(c['right'])} "
            f"props {_getup_props_down(env)} "
            f"{'STAND' if _getup_hold_raw(env) > 0.0 else '-'}")


_register(Behavior(
    id="getup",
    emoji="🛌",
    title="Get up off the floor",
    description=(
        "Start lying on the ground — on its back, its front or its side — "
        "roll over, get its feet under it and stand up, then hold the stand."
    ),
    how_it_learns=(
        "It wakes up on the floor and is paid for one thing: being upright "
        "AND tall AND on both feet AND with nothing else touching the "
        "ground, worth more the longer it stays there, so popping up and "
        "falling back over is worth a third of holding still. Every part "
        "still resting on the floor charges rent — the first version of "
        "this recipe had no such rent and the duck learned to prop itself "
        "on its beak. "
        "Two shepherds get it there — every step it gets more upright, or "
        "its body gets higher, it earns; every step it slides back down, it "
        "pays. Lying flat is a state random flailing never escapes, so the "
        "practice starts half-toppled on strong servos and works down to "
        "flat on the floor on the real ones."
    ),
    keywords=("get up", "getup", "get back up", "stand up", "stand back up",
              "off the floor", "off the ground", "recover", "fell over",
              "fallen", "pick itself up", "roll over"),
    terms=(
        RewardTerm("getup_hold",
                   "The salary: upright, at full height, on both feet — worth "
                   "more the longer it holds",
                   8.0, _getup_hold),
        RewardTerm("rise_gain",
                   "Points every step it gets more upright than it was (and a "
                   "charge every step it slides back)",
                   30.0, _getup_gain_term("rise")),
        RewardTerm("lift_gain",
                   "Points every step its body gets higher off the floor (and "
                   "a charge every step it sinks)",
                   30.0, _getup_gain_term("lift")),
        RewardTerm("stay_upright", "Points for being upright at all", 2.0,
                   _getup_upright),
        RewardTerm("stand_tall", "Points for standing at full height", 2.0,
                   _stand_tall),
        RewardTerm("both_feet", "Points for both feet on the ground", 1.0,
                   _both_feet_down),
        RewardTerm("flat_feet", "Points for getting the feet flat under it",
                   0.8, _flat_feet),
        RewardTerm("normal_pose",
                   "Points for finishing in the normal ready pose, so it can "
                   "hand straight back to the walker",
                   1.0, _pose_home),
        RewardTerm("head_up", "Points for bringing the head up", 0.8,
                   _head_up_blend),
        RewardTerm("still_on_the_floor",
                   "Rent for every part still resting on the ground that "
                   "isn't a foot",
                   1.5, _getup_prop_pen, is_penalty=True),
        RewardTerm("no_limit_parking",
                   "Penalty for levering off joints jammed against their end "
                   "stops",
                   0.8, _limit_parking_pen, is_penalty=True),
        # smooth_moves / gentle_joints / save_energy are deliberately OUT of
        # the recipe: on the headstand's scratch runs they charged -1.9/step
        # against a +1.3 salary, so correcting a wobble cost more than the
        # catch paid and training stalled. A get-up is a VIOLENT move — it
        # has to throw the legs — and taxing that during discovery is the
        # attempt tax AGENTS.md warns about. They are catalog terms; a
        # finished get-up can be polished with them through --weights-json.
    ),
    default_steps=2_000_000,
    success_metric="fraction of lying starts that reach a held stand, and the seconds it takes",
    episode_s=8.0,      # ~2 s to rise, ~6 s to prove the stand is a stand
    scene="all",        # the trunk and head must be able to REST on the floor
    terminate_on_fall=False,    # lying on the floor IS the walk env's "fallen"
    height_termination=False,   # ... and the trunk starts under the z-kill
    state_fn=_getup_update,
    caption_fn=_getup_caption,
    report_fn=_getup_report,
    # Mix for an UNSTAGED run (a fine-tune, or a preview env built without
    # stage knobs); the ladder overrides it per stage. 5% of episodes start
    # standing, so the policy is also asked to leave a stand alone.
    spawn_families=(
        (0.50, _getup_spawn_family("back")),
        (0.30, _getup_spawn_family("front")),
        (0.15, _getup_spawn_family("side")),
    ),
    # THE LADDER. Identical terms in every stage (AGENTS.md: a stage may
    # ladder physics, spawns and strictness, never the pay). What moves is
    # the TILT window — how far from upright the duck wakes up — the actuator
    # model, and which falls are in the mix.
    curriculum=(
        CurriculumStage("finding its feet", 1_500_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_GETUP_TILT_LO": "20",
                         "MICRODUCK_GETUP_TILT_HI": "50",
                         "MICRODUCK_GETUP_SETTLE_S": "0.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.80,0.20,0.0"},
                        detail=(
                            "It wakes up leaning, not lying — far enough over "
                            "that it has to push itself up, close enough that "
                            "a brand-new brain manages it in the first minute "
                            "and finds out what standing is worth. Nothing "
                            "later on the ladder is learnable until this "
                            "is.")),
        CurriculumStage("catching a topple", 2_000_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_GETUP_TILT_LO": "45",
                         "MICRODUCK_GETUP_TILT_HI": "75",
                         "MICRODUCK_GETUP_SETTLE_S": "0.15",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.75,0.20,0.0"},
                        detail=(
                            "It wakes up already toppling backwards, on "
                            "phantom-strong servos — far enough over that "
                            "standing up is the only way back, close enough "
                            "that a brand-new brain can actually manage it "
                            "and learn what upright is worth.")),
        CurriculumStage("from further down", 2_000_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_GETUP_TILT_LO": "65",
                         "MICRODUCK_GETUP_TILT_HI": "100",
                         "MICRODUCK_GETUP_SETTLE_S": "0.4",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.60,0.35,0.0"},
                        detail=(
                            "The same catch from most of the way down, "
                            "including properly flat on its back — still on "
                            "the strong servos, so the new part is the pose "
                            "and not the strength.")),
        CurriculumStage("on real servos", 2_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_BAM_CURRENT_SCALE": "1.3",
                         "MICRODUCK_GETUP_TILT_LO": "75",
                         "MICRODUCK_GETUP_TILT_HI": "110",
                         "MICRODUCK_GETUP_SETTLE_S": "0.7",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.50,0.35,0.15"},
                        detail=(
                            "Flat on the floor, and the servos step down "
                            "toward the real XL330s. Falls onto the side "
                            "join the mix.")),
        CurriculumStage("off the floor for real", 2_500_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_GETUP_TILT_LO": "80",
                         "MICRODUCK_GETUP_TILT_HI": "115",
                         "MICRODUCK_GETUP_SETTLE_S": "1.0",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.45,0.30,0.20"},
                        detail=(
                            "Honest XL330s, flat on the floor, all three "
                            "ways of falling, and 5% of episodes already "
                            "standing so it also learns to leave a stand "
                            "alone.")),
    ),
))
