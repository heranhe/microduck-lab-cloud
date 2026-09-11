"""The last metre, learned: a kick that SEES the ball (roadmap 12h / E.1,
2026-09-10).

12b put the ball anywhere in the box play produces and paid the swing that
connects across it - blind. That recipe took the grid's box coverage from
the vendored strike's 69-82 % to 82-99 %, and the gym whiff from 18 % to
4 %, and there it stops: a blind sweep cannot refuse a ball it cannot reach
and cannot step to one. 12aj measured what is left - the ball is a median
10 cm ahead at the swing, but 11 % of swings are at a ball more than 15 cm
ahead and those whiff 44 %, and *nothing that reads the belief can refuse
them, because the belief is what is wrong*. 12ak then bought the fresh
sighting (58 % of swings, track age 1.54 s -> 0.14 s) and every way of
ACTING on it - declining, re-laying, gating - whiffed more, and closed with
the ask this recipe answers: "the lever that would USE a fresh sighting is
a kick that adapts to where the ball is".

So: the wide kick's world, the wide kick's pay, and the ball in the
observation.

**The slots.** `find_ball`'s rule (AGENTS.md: "a task the robot must SENSE
puts its sensing in the command slots, in the robot's own terms"), with the
same four HEAD slots, obs[51:55], and the 61-dim layout untouched:

  [51] bearing  the ball's bearing in the DUCK's own yaw frame, psi/(pi/2),
                + = to the LEFT, clipped to +-1. 0 while nothing is known.
  [52] range    ground distance from the trunk to the ball, / LM_RANGE_SCALE,
                clipped to 0..1. 0 while nothing is known.
  [53] seen     1.0 while the detector's last report had the ball in it.
  [54] conf     freshness of the estimate: 1.0 on a fresh report, fading
                exp(-t/LM_MEM_TAU) while the report says nothing, 0.0 before
                the first sighting of the episode.

Body frame, not camera frame, and that is deliberate: the detector reports a
bearing across the FRAME, and the daemon's tracker adds the camera's own yaw
to it before anyone consumes it (`Tracker._associate`; AGENTS.md verification
rule 8 is about exactly this frame). A camera-frame bearing confounds the
head's pose into the signal - the same number means a different ball
depending on where the neck is - and this policy is asked to put a FOOT on
the ball, which is a body-frame errand. What rides the slots is therefore
what the daemon hands a consumer, and every step of producing it is
something the robot can do:

  * the ball is projected through the MJCF `head_camera` exactly as
    `find_ball` does it (`_ball_camera`), with find_ball's own FOV knobs -
    one source for the camera, so a lens change cannot mean two things here
    (AGENTS.md: "a characterisation that silently inherits a default");
  * `seen` is the detector's, not the truth's: in frame in BOTH axes and
    inside MAX_RANGE, sampled at the detector's cadence (25 Hz against the
    50 Hz control loop) and jittered in normalized bearing units when the
    env is randomizing;
  * the jittered bearing pair is turned back into a ray and intersected with
    the floor plane - the ground-plane placement a daemon does with head
    encoders and the IMU, no range sensor needed;
  * that world point is HELD while the report says nothing and re-expressed
    in the current body frame every step, which is odometry. Over a 2 s
    episode that is what the robot's own odometry gives it.

The reward reads the truth (the ball's speed along the kick line), as every
reward here does; the obs never does.

**The pay is the wide kick's, to the term** (`kick._kick_terms` plus 12ab's
`face_line` anchor) - so this recipe and `kick_{side}_wide` differ in the
OBSERVATION and in nothing else that is paid. AGENTS.md: a stage may ladder
the world, never the pay.

**The ladder is spawns only.** Stage 1 puts the ball in the 6 x 6 cm box
round the sweet spot AND pins the gaze to the down half of its range, where
the geometry says the ball is actually in frame (at a level head the camera
sits 0.21 m above a ball on the floor, so a ball 0.10 m ahead is 65 deg below
the optical axis and outside the 58 deg half-VFOV - a level duck cannot see
its own feet, which is 12k's finding). Stage 2 opens both: the full box play
produces, and the whole gaze range the wide kick trains on, so the finished
policy still has to swing blind at the poses where there is nothing to see.
"""

import math

import mujoco

from .. import contract as C
from .ball import _ball_camera, _ball_knob
from .core import Behavior, CurriculumStage, RewardTerm, _face_home_pen, _register, _trunk_yaw
from .kick import (
    BALL_NOISE,
    BALL_OFFSET,
    BALL_Z,
    HEAD_DOWN,
    KICK_BOX_AHEAD,
    KICK_BOX_SIDE,
    KICK_BOX_STAGE1,
    NECK_DOWN,
    _box_knob,
    _kick_ball_ids,
    _kick_terms,
    _yaw,
)

LM_RANGE_SCALE = 0.25        # m of ground range at slot [52] = 1.0 (the box tops out ~0.21)
LM_DETECT_EVERY = 2          # control steps between detector reports (25 Hz, find_ball's cadence)
LM_JITTER = 0.02             # normalized bearing units, find_ball's MICRODUCK_BALL_JITTER
LM_MEM_TAU = 1.0             # s: the confidence slot's fade while the report says nothing
LM_MAX_RANGE = 3.0           # m — the detector has no box beyond this (find_ball's)
# Stage 1's gaze: the down half of the range, where a ball in the box is in
# frame. Full range (NECK_DOWN / HEAD_DOWN) is stage 2 and the default.
LM_GAZE_STAGE1 = ("-0.30,-0.15", "0.45,0.60", "0.30,0.50")  # neck, head, |head yaw| "lo,hi"
# Rung 1's box: the POINT-STRIKE spot, +-BALL_NOISE — `kick.BALL_OFFSET` to
# the millimetre, derived from it so the two cannot drift apart. 12b measured
# why this rung exists: the box FROM SCRATCH loses the strike ("a random swing
# at a ball spread over 12 x 12 cm is paid a little everywhere and the
# optimiser settles on the nudge", box coverage 46-63 %), and it was rescued
# there by warm-starting from the vendored strike. That warm start is not
# available to this recipe: the strike's `VecNormalize` was fitted with these
# four slots carrying keep-alive noise (std 0.009-0.029 over a 2M count), so a
# bearing of 1.0 would enter the network at ~34 and the running statistics
# would take another 2M steps to notice. So the strike is found HERE instead,
# on the spot, and the box opens under it.
LM_BOX_STAGE0 = (f"{BALL_OFFSET[0] - BALL_NOISE:.3f},{BALL_OFFSET[0] + BALL_NOISE:.3f}",
                 f"{BALL_OFFSET[1] - BALL_NOISE:.3f},{BALL_OFFSET[1] + BALL_NOISE:.3f}")
# Head yaw at spawn is HOME in the finished world, as it is for the wide kick
# and as the bench and the arena hand it over.
LM_GAZE_YAW = (0.0, 0.0)


def _lm_gaze(env) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """(neck, head, |head yaw|) spawn windows for this stage — offsets off HOME.

    The yaw window is a MAGNITUDE, turned toward the kicking foot by the
    reset, because which way "at the ball" is depends on the foot.
    """
    return (_box_knob(env, "MICRODUCK_LM_GAZE_NECK", NECK_DOWN),
            _box_knob(env, "MICRODUCK_LM_GAZE_HEAD", HEAD_DOWN),
            _box_knob(env, "MICRODUCK_LM_GAZE_YAW", LM_GAZE_YAW))


def _lm_ground_point(cam, fwd, right, up, ax: float, ay: float):
    """Where a detection at camera angles (ax, ay) meets the floor plane the
    ball's centre rides on, in world xy — the daemon's own placement, from
    head encoders and the IMU. None if the ray never gets there."""
    tx, ty = math.tan(ax), math.tan(ay)
    dx = fwd[0] + tx * right[0] + ty * up[0]
    dy = fwd[1] + tx * right[1] + ty * up[1]
    dz = fwd[2] + tx * right[2] + ty * up[2]
    if dz > -1e-6:
        return None                       # pointing at or above the horizon
    t = (BALL_Z - cam[2]) / dz
    if t <= 0.0:
        return None
    return cam[0] + t * dx, cam[1] + t * dy


def _lm_reset_for(side: str):
    sgn = -1.0 if side == "right" else 1.0

    def _lm_reset(env) -> None:
        r = env._rng
        _, qadr, dadr = _kick_ball_ids(env)
        yaw = _yaw(env)
        ox = r.uniform(*_box_knob(env, "MICRODUCK_KICK_BOX_AHEAD", KICK_BOX_AHEAD))
        oy = sgn * r.uniform(*_box_knob(env, "MICRODUCK_KICK_BOX_SIDE", KICK_BOX_SIDE))
        x = float(env.data.qpos[0]) + math.cos(yaw) * ox - math.sin(yaw) * oy
        y = float(env.data.qpos[1]) + math.sin(yaw) * ox + math.cos(yaw) * oy
        env.data.qpos[qadr:qadr + 7] = [x, y, BALL_Z, 1.0, 0.0, 0.0, 0.0]
        env.data.qvel[dadr:dadr + 6] = 0.0
        env._kick_dir = (math.cos(yaw), math.sin(yaw))
        neck_w, head_w, yaw_w = _lm_gaze(env)
        env.data.qpos[env.joint_qpos_adr[5]] = C.DEFAULT_POSE[5] + r.uniform(*neck_w)
        env.data.qpos[env.joint_qpos_adr[6]] = C.DEFAULT_POSE[6] + r.uniform(*head_w)
        # An OFFSET on the spawn pose, not an assignment: at the finished
        # world's (0, 0) this is a no-op and the walk env's own head-yaw
        # spawn noise survives, exactly as it does for the wide kick.
        env.data.qpos[env.joint_qpos_adr[7]] += sgn * r.uniform(*yaw_w)
        env.data.ctrl[:] = env.data.qpos[env.joint_qpos_adr]
        if getattr(env, "bam", None) is not None:
            env.bam.reset(env.data.qpos[env.joint_qpos_adr])
        mujoco.mj_forward(env.model, env.data)
        env.last_spawn = f"ball {ox:.2f}m ahead {abs(oy):.2f}m {side}"
        # Sensing state, fresh per episode (a leaked estimate is one free
        # sighting on the first step of the next one).
        env._lm_episode = env.episode_id
        env._lm_step_done = -1
        env._lm_det_step = -10 ** 9
        env._lm_det_seen = False
        env._lm_world = None
        env._lm_conf = 0.0
        env._lm_seen_steps = 0
        env._lm_half_h = math.radians(_ball_knob(env, "MICRODUCK_BALL_HFOV_DEG")) / 2
        env._lm_half_v = math.radians(_ball_knob(env, "MICRODUCK_BALL_VFOV_DEG")) / 2
        _lm_sense(env, force=True)

    _lm_reset.__name__ = f"_lm_reset_{side}"
    return _lm_reset


def _lm_sense(env, force: bool = False) -> None:
    """Project the ball through the head camera, run the detector at its own
    cadence, and write the four head slots. Scalar math: this runs every
    control step."""
    cam, fwd, right, up = _ball_camera(env)
    _, qadr, _ = _kick_ball_ids(env)
    q = env.data.qpos
    vx, vy, vz = float(q[qadr]) - cam[0], float(q[qadr + 1]) - cam[1], float(q[qadr + 2]) - cam[2]
    dist = math.sqrt(vx * vx + vy * vy + vz * vz)
    f = vx * fwd[0] + vy * fwd[1] + vz * fwd[2]
    half_h, half_v = env._lm_half_h, env._lm_half_v
    if f > 1e-6:
        bx = math.atan2(vx * right[0] + vy * right[1] + vz * right[2], f) / half_h
        by = math.atan2(vx * up[0] + vy * up[1] + vz * up[2], f) / half_v
        seen = -1.0 < bx < 1.0 and -1.0 < by < 1.0 and dist < LM_MAX_RANGE
    else:
        bx = by = 0.0
        seen = False
    if seen:
        env._lm_seen_steps += 1
    if force or env.step_count - env._lm_det_step >= LM_DETECT_EVERY:
        env._lm_det_step = env.step_count
        env._lm_det_seen = seen
        if seen:
            jit = LM_JITTER if env.obs_noise else 0.0
            r = env._rng
            ax = (bx + (float(r.uniform(-jit, jit)) if jit else 0.0)) * half_h
            ay = (by + (float(r.uniform(-jit, jit)) if jit else 0.0)) * half_v
            p = _lm_ground_point(cam, fwd, right, up, ax, ay)
            if p is not None:
                env._lm_world = p
                env._lm_conf = 1.0
    if not force and not env._lm_det_seen:
        env._lm_conf *= math.exp(-C.CTRL_DT / LM_MEM_TAU)
    hc = env.head_cmd
    if env._lm_world is None:
        hc[0] = hc[1] = 0.0
    else:
        # The held world estimate, re-expressed in the body frame the duck is
        # in NOW — odometry, which is what carries it while the report is
        # stale and while the duck steps toward the ball.
        t, yaw = env._trunk_xpos, _trunk_yaw(env)
        dx, dy = env._lm_world[0] - float(t[0]), env._lm_world[1] - float(t[1])
        c, s = math.cos(yaw), math.sin(yaw)
        ahead, beside = c * dx + s * dy, -s * dx + c * dy      # + beside = to the LEFT
        hc[0] = max(-1.0, min(1.0, math.atan2(beside, ahead) / (math.pi / 2)))
        hc[1] = max(0.0, min(1.0, math.hypot(ahead, beside) / LM_RANGE_SCALE))
    hc[2] = 1.0 if env._lm_det_seen else 0.0
    hc[3] = max(0.0, min(1.0, env._lm_conf))


def _lm_obs(env) -> None:
    """Behavior.obs_fn: sense once per control step (the obs can be rebuilt
    more than once per step; the detector runs once)."""
    if getattr(env, "_lm_episode", None) != env.episode_id:
        return                        # reset in progress; reset_fn seeds first
    if env._lm_step_done == env.step_count:
        return
    env._lm_step_done = env.step_count
    _lm_sense(env)


def _lm_body_xy(env) -> tuple[float, float]:
    """The TRUE ball, ahead/beside the trunk (+ beside = left) — read-side only."""
    _, qadr, _ = _kick_ball_ids(env)
    t, yaw = env._trunk_xpos, _trunk_yaw(env)
    dx, dy = float(env.data.qpos[qadr]) - float(t[0]), float(env.data.qpos[qadr + 1]) - float(t[1])
    c, s = math.cos(yaw), math.sin(yaw)
    return c * dx + s * dy, -s * dx + c * dy


def _lm_caption(env) -> str:
    hc = env.head_cmd
    ahead, beside = _lm_body_xy(env)
    return (f"slots psi{hc[0]:+.2f} r{hc[1]:.2f} seen{hc[2]:.0f} conf{hc[3]:.2f} | "
            f"true ahead {ahead:+.3f} side {beside:+.3f}")


def _lm_report(env) -> list[str]:
    ahead, beside = _lm_body_xy(env)
    steps = max(int(env.step_count), 1)
    return [f"ball in frame {env._lm_seen_steps / steps:.0%} of steps; "
            f"ball now {ahead:+.3f} m ahead, {beside:+.3f} m left of the trunk"]


for _side in ("right", "left"):
    _register(Behavior(
        id=f"kick_{_side}_sensed",
        emoji="👁",
        title=f"Kick the ball it can see ({_side} foot)",
        description=(
            f"Kick a ball off the {_side} foot down the line it is facing, from standing, "
            "with the ball's position in its observation - so it can step to a ball "
            "that is not where the swing wants it."
        ),
        how_it_learns=(
            "The ball starts anywhere in the box a duck really arrives at in play, "
            "and the head anywhere between level and looking at its feet - but this "
            "time a detector on the robot projects the ball through the head camera "
            "and writes where it is, in the duck's own frame, into the four head "
            "command slots: a bearing, a range, whether it is in frame, and how "
            "fresh the estimate is. The pay is the wide kick's, to the term: the "
            "ball rolling away along the line, the other foot planted, legs and head "
            "near home, and no turning off the line. Nothing pays for looking or for "
            "stepping - the two seconds do that, because a ball it has to walk to is "
            "a ball it is not yet being paid for."
        ),
        # No bare "kick <side>": the plain strike owns that phrase and the
        # matcher scores by how much of the message a keyword explains, so a
        # generic substring of a specific request would win it (core's own
        # "jump backflip" note). The side-named "last metre" is the handle.
        keywords=(f"kick {_side} sensed", f"kick_{_side}_sensed", f"sensed kick {_side}",
                  f"seeing kick {_side}", f"{_side} foot sensed kick",
                  f"last metre {_side}", f"last meter {_side}"),
        terms=_kick_terms(_side) + (
            RewardTerm("face_line", "Docked for the body turning off the kick line through the swing",
                       4.0, _face_home_pen, is_penalty=True),
        ),
        default_steps=4_000_000,
        success_metric=("ball speed along the kick line at 0.5 s from every spot in the box and every "
                        "gaze pose; coverage of the play box above the blind pair's 69-82 %; body turn under 20 deg"),
        symmetric=False,
        episode_s=2.0,
        scene="ball",
        terminate_on_fall=True,
        reset_fn=_lm_reset_for(_side),
        obs_fn=_lm_obs,
        caption_fn=_lm_caption,
        report_fn=_lm_report,
        curriculum=(
            CurriculumStage("finding the strike, with the ball in view", 1_000_000,
                            {"MICRODUCK_KICK_BOX_AHEAD": LM_BOX_STAGE0[0],
                             "MICRODUCK_KICK_BOX_SIDE": LM_BOX_STAGE0[1],
                             "MICRODUCK_LM_GAZE_NECK": LM_GAZE_STAGE1[0],
                             "MICRODUCK_LM_GAZE_HEAD": LM_GAZE_STAGE1[1],
                             "MICRODUCK_LM_GAZE_YAW": LM_GAZE_STAGE1[2]},
                            detail=("The ball on the kicking foot's sweet spot, a centimetre and a "
                                    "half either way — the point-strike recipe's own world — with "
                                    "the gaze already on it. Nothing but a real strike pays here, "
                                    "which is what the box cannot teach from scratch (12b).")),
            CurriculumStage("widening to a box, with the ball in view", 1_000_000,
                            {"MICRODUCK_KICK_BOX_AHEAD": KICK_BOX_STAGE1[0],
                             "MICRODUCK_KICK_BOX_SIDE": KICK_BOX_STAGE1[1],
                             "MICRODUCK_LM_GAZE_NECK": LM_GAZE_STAGE1[0],
                             "MICRODUCK_LM_GAZE_HEAD": LM_GAZE_STAGE1[1],
                             "MICRODUCK_LM_GAZE_YAW": LM_GAZE_STAGE1[2]},
                            detail=("The ball in a 6 x 6 cm box round the sweet spot and the gaze "
                                    "already ON it — pitched into the down half of its range and "
                                    "yawed toward the kicking foot, which is the only pose the box "
                                    "is really in frame from (measured: 96 % of the box, against "
                                    "47 % looking straight down and 0 % level). A swing AND a "
                                    "sighting are both in the rollouts from the first minute.")),
            CurriculumStage("the box play produces, at any gaze", 2_000_000,
                            {"MICRODUCK_KICK_BOX_AHEAD": f"{KICK_BOX_AHEAD[0]},{KICK_BOX_AHEAD[1]}",
                             "MICRODUCK_KICK_BOX_SIDE": f"{KICK_BOX_SIDE[0]},{KICK_BOX_SIDE[1]}",
                             "MICRODUCK_LM_GAZE_NECK": f"{NECK_DOWN[0]},{NECK_DOWN[1]}",
                             "MICRODUCK_LM_GAZE_HEAD": f"{HEAD_DOWN[0]},{HEAD_DOWN[1]}",
                             "MICRODUCK_LM_GAZE_YAW": f"{LM_GAZE_YAW[0]},{LM_GAZE_YAW[1]}"},
                            detail=("The full box — 4-16 cm ahead, 1-13 cm to the kicking foot's "
                                    "side — the whole gaze range, and the head yaw back at HOME, "
                                    "which is how the bench and the arena hand a kick over: the "
                                    "ball is in frame on about a quarter of spawns and the policy "
                                    "has to turn its head to the rest or swing blind.")),
        ),
    ))
