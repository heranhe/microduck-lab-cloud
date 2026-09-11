"""Locks for roadmap 12as's wiring: a kick ONNX whose sidecar says
`"sensed": true` gets the BALL in its four head command slots while it swings,
and everything without that field is the blind world, to the bit.

12as trained `kick_{left,right}_sensed` (`behaviors/lastmetre.py`) with the
ball's bearing / range / seen / freshness riding `obs[51:55]`, measured
94-99 % box coverage against the vendored pair's 67-82 %, and closed with the
reason it could not be played: `World._skill_cmd` zeroes `head_cmd` for the
whole kick window, so a sensed kick dropped into the arena runs BLINDFOLDED —
the row in that item's own table which is worse than the blind pair.

So the two halves both need a lock: the units the slots carry (they are the
recipe's, imported from it, and a bearing that came out mirrored would be a
policy kicking at the ball's reflection), and the silence of the default path
(no sidecar field, no change — the shipped sidecars predate this and must
behave as they always did).

The byte-identical half of that second claim is checked OUT of band, against a
PYTHONPATH copy of the pre-edit package (AGENTS.md / memory note
shared-checkout-ab-on-copies.md): a 2 s 2v2 rollout digest, ball and every
duck's qpos each tick, same on both. This file locks the mechanism that makes
it so — the slots stay zero through a whole kick window on the vendored pair.
"""

from __future__ import annotations

import json
import math
import os
import shutil

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local.behaviors.lastmetre import LM_MEM_TAU, LM_RANGE_SCALE
from microduck_local.brain.brain_env import POLICIES_DIR
from microduck_local.brain.tracker import Track, Tracker
from microduck_local.world import World
from microduck_local.world.arena import KICK_S, SENSED_BALL_CLS
from microduck_local.world.scenario import Ball, Duck, Scenario, Wall

CTRL_DT = 0.02
WALKER = POLICIES_DIR / "alpha_walking.onnx"
needs_walker = pytest.mark.skipif(not WALKER.exists(), reason="upstream policies not checked out")


def _standing_world(sc: Scenario) -> World:
    """A duck on the shipped walker under a zero command. `policy=None` is a
    LIMP duck (memory note open-loop-holds-topple: zero/limp fall in ~1 s), and
    a duck on the floor has no kick window to read."""
    from microduck_local.brain.brain_env import onnx_infer  # noqa: PLC0415
    return World(sc, infer_for={d.id: onnx_infer(WALKER) for d in sc.ducks}, seed=0)


def _pin(monkeypatch, tmp_path, sidecar_for) -> None:
    """Pin both kicks to a scratch copy of the VENDORED onnx with a sidecar of
    our own — the swing itself is not what these tests read, the command block
    is, so the blind pair's network is the right one to run under both."""
    for foot in ("left", "right"):
        src = World.skill_path(f"kick_{foot}")
        dst = tmp_path / f"kick_{foot}.onnx"
        shutil.copyfile(src, dst)
        side = sidecar_for(foot)
        if side is not None:
            (tmp_path / f"kick_{foot}.json").write_text(json.dumps(side))
        monkeypatch.setenv(f"MICRODUCK_SKILL_KICK_{foot.upper()}", str(dst))


def _one_duck(ball_xy=(0.5, 0.0)) -> Scenario:
    """The kick gym's world, minus the goal-mouth arithmetic: one duck facing
    +x at the origin, one ball, boards."""
    hx, hy = 1.5, 1.25
    corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
    walls = [Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)]
    return Scenario(name="sensed-kick-test", floor=(2 * hx + 0.5, 2 * hy + 0.5), walls=walls,
                    balls=[Ball(ball_xy)],
                    ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "datasheet", "datasheet", "chase",
                                team="cream")],
                    goal_width=0.7)


def _place_ball(w: World, x: float, y: float) -> None:
    j = w._ball_joint
    q = int(w.model.jnt_qposadr[j])
    w.data.qpos[q:q + 2] = [x, y]
    w.data.qvel[int(w.model.jnt_dofadr[j]):int(w.model.jnt_dofadr[j]) + 6] = 0.0
    mujoco.mj_forward(w.model, w.data)


# -- the sidecar ------------------------------------------------------------

def test_the_shipped_sidecars_are_not_sensed_and_an_old_three_field_one_still_loads(monkeypatch, tmp_path):
    """`kick_exits` has read `exit_rad` out of these files since item 7; the
    `sensed` flag is a new key in the same file and its ABSENCE is the whole
    compatibility claim."""
    assert World.skill_sensed("kick_left") is False
    assert World.skill_sensed("kick_right") is False
    assert "exit_rad" in World.skill_sidecar("kick_left")

    # The oldest shape a sidecar ever had: run, behavior, exit_rad.
    _pin(monkeypatch, tmp_path,
         lambda foot: {"run": f"old-{foot}", "behavior": f"kick_{foot}", "exit_rad": 0.25})
    assert World.kick_exits() == (0.25, 0.25)
    assert World.skill_sensed("kick_left") is False
    # …and no sidecar at all is neither an exit nor a sensed kick.
    (tmp_path / "kick_left.json").unlink()
    assert World.kick_exits() is None
    assert World.skill_sensed("kick_left") is False
    assert World.skill_sidecar("kick_left") == {}


def test_a_sensed_sidecar_is_read_onto_the_constructed_world(monkeypatch, tmp_path):
    """Playbook rule 0: the flag a battery thinks it is measuring has to be
    readable off the World it actually built, not off the file it wrote."""
    _pin(monkeypatch, tmp_path,
         lambda foot: {"run": f"lastmetre-{foot}", "sensed": True, "exit_rad": -0.1})
    w = World(_one_duck(), seed=0)
    assert w._skill_sensed["kick_left"] is True and w._skill_sensed["kick_right"] is True
    assert w._sensed_kick is True
    assert World.kick_exits() == (-0.1, -0.1)          # the exit still comes out of the same file


# -- the encoding -----------------------------------------------------------

def _fake_track(bearing: float, rng: float, last_t: float) -> Track:
    return Track(id=1, cls=SENSED_BALL_CLS, bearing=bearing, elevation=0.0, width=0.1,
                 range=rng, conf=1.0, born_t=0.0, last_t=last_t, hits=3)


def test_the_four_slots_carry_the_recipes_units(monkeypatch, tmp_path):
    """`behaviors/lastmetre.py`'s docstring, term by term: psi/(pi/2) with
    POSITIVE to the LEFT, ground range over LM_RANGE_SCALE clipped to 0..1,
    seen, and exp(-age / LM_MEM_TAU)."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    w = World(_one_duck(), seed=0)
    d = w.ducks["d0"]
    trk = w._ball_tracker(d)

    # A track with no POSITION: bearing and range are the measurement itself.
    w.t = 1.0
    trk.tracks = [_fake_track(math.pi / 4, 0.125, last_t=0.5)]
    w._sensed_head(d)
    assert w._skill_sensed["kick_left"]
    assert d.head_cmd[0] == pytest.approx(0.5, abs=1e-6)          # +45 deg = half of pi/2, to the LEFT
    assert d.head_cmd[1] == pytest.approx(0.125 / LM_RANGE_SCALE, abs=1e-6)
    assert d.head_cmd[2] == 0.0                                    # no frame of ours hit it
    assert d.head_cmd[3] == pytest.approx(math.exp(-0.5 / LM_MEM_TAU), abs=1e-6)

    # The mirror image is the mirror value, and nothing else moves.
    trk.tracks = [_fake_track(-math.pi / 4, 0.125, last_t=0.5)]
    w._sensed_head(d)
    assert d.head_cmd[0] == pytest.approx(-0.5, abs=1e-6)

    # Both ends are CLIPPED, the way the recipe's own slots are.
    trk.tracks = [_fake_track(2.0, 4.0, last_t=1.0)]
    w._sensed_head(d)
    assert d.head_cmd[0] == pytest.approx(1.0) and d.head_cmd[1] == pytest.approx(1.0)

    # Nothing known at all is the all-zero block the recipe spawns with.
    trk.tracks = []
    w._sensed_head(d)
    assert np.array_equal(d.head_cmd, np.zeros(4, np.float32))


def test_a_coasting_track_is_read_off_its_position_not_off_the_stale_bearing(monkeypatch, tmp_path):
    """12ar / 12af: `Track.bearing` is the pose at the last HIT turned by yaw
    alone, so a duck that WALKED while coasting carries a bearing that stopped
    meaning bearing. A kick window is entered walking."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    w = World(_one_duck(), seed=0)
    d = w.ducks["d0"]
    trk = w._ball_tracker(d)
    w.t = 1.0
    tr = _fake_track(0.0, 1.0, last_t=1.0)
    tr.xy = (0.10, 0.10)                    # the remembered ball, in the odometry frame
    trk.tracks = [tr]
    d.odom_est[:] = (0.0, 0.0, 0.0)         # …and the duck is at the origin facing +x
    w._sensed_head(d)
    assert d.head_cmd[0] == pytest.approx(math.atan2(0.10, 0.10) / (math.pi / 2), abs=1e-6)
    assert d.head_cmd[1] == pytest.approx(math.hypot(0.10, 0.10) / LM_RANGE_SCALE, abs=1e-6)
    assert d.head_cmd[0] > 0.0 and d.head_cmd[1] < 1.0          # to the LEFT, and inside the box


# -- in the world ------------------------------------------------------------

@needs_walker
def test_a_sensed_kick_sees_the_ball_through_a_whole_swing(monkeypatch, tmp_path):
    """End to end, with the duck's own camera and no hand-made track: a ball
    to the LEFT reads positive in the slots for every tick of the window."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    w = _standing_world(_one_duck())
    d = w.ducks["d0"]
    _place_ball(w, 0.45, 0.22)                     # ahead and to the duck's left, in frame at a level head
    for _ in range(15):                            # the detector runs at its own rate; give it frames
        w.step()
    assert w._ball_tracker(d).best(SENSED_BALL_CLS, w.t, min_hits=1) is not None, "the camera never saw it"
    assert w.start_skill(d, "kick_left")
    assert d.skill_sensed is True
    rows = []
    for _ in range(int(round(KICK_S / CTRL_DT))):
        w.step()
        if d.skill is None:
            break                    # the window ended — its 0.5 s, or a fall
        rows.append(d.head_cmd.copy())
    # The vendored BLIND kick swinging at a ball 0.45 m away topples this cold
    # standing duck about 0.12 s in (measured: 6 ticks, every seed tried), and
    # a fall ends the window early. What is locked here is the command block,
    # not the swing: every tick the window DID run carried the ball.
    assert len(rows) >= 5
    assert all(r[0] > 0.0 for r in rows), [float(r[0]) for r in rows]   # + is LEFT, all the way through
    assert all(0.0 < r[1] <= 1.0 for r in rows)
    assert any(r[2] > 0.0 for r in rows) and all(r[3] > 0.0 for r in rows)
    assert d.skill is None and d.skill_sensed is False                  # …and the window let go


@needs_walker
def test_without_the_field_the_window_is_all_zeros(monkeypatch, tmp_path):
    """The compatibility claim, as a mechanism: same world, same ball, same
    swing, sidecar without `sensed` — the command block the vendored pair was
    trained on, every tick. (No sidecar field means no tracker either: nothing
    is built and nothing is stepped.)"""
    _pin(monkeypatch, tmp_path, lambda foot: {"run": "old", "exit_rad": 0.0})
    w = World(_one_duck(), seed=0)
    d = w.ducks["d0"]
    assert w._sensed_kick is False
    _place_ball(w, 0.45, 0.22)
    for _ in range(15):
        w.step()
    assert w._ball_trackers == {}
    assert w.start_skill(d, "kick_left") and d.skill_sensed is False
    for _ in range(int(round(KICK_S / CTRL_DT))):
        w.step()
        assert np.array_equal(d.head_cmd, np.zeros(4, np.float32))


@needs_walker
def test_the_ball_memory_goes_when_the_duck_is_respawned(monkeypatch, tmp_path):
    """A track carried across a respawn is one free sighting of a ball that is
    no longer where it was — the same leak `lastmetre`'s reset closes per
    episode."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    w = _standing_world(_one_duck())
    d = w.ducks["d0"]
    _place_ball(w, 0.45, 0.22)
    for _ in range(15):
        w.step()
    assert w._ball_tracker(d).tracks
    w.reset_duck("d0")
    assert w._ball_tracker(d).tracks == []


def test_the_tracker_is_the_brains_own_over_the_ducks_own_frames(monkeypatch, tmp_path):
    """Not a second model of the ball: `brain/tracker.py`'s `Tracker`, with the
    uncertainty model of THIS duck's detector preset (`for_detector`), fed the
    frames the duck's own camera produced."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    w = World(_one_duck(), seed=0)
    d = w.ducks["d0"]
    trk = w._ball_tracker(d)
    assert isinstance(trk, Tracker)
    from microduck_local.sensors.detector import DetectorNoise
    nz = DetectorNoise.preset("datasheet")
    assert trk.p.meas_bearing_sigma == pytest.approx(float(nz.bearing_sigma_rad))
    assert trk.p.meas_range_frac == pytest.approx(float(nz.width_sigma_frac))


# -- the truth ablation (roadmap 12as follow-up H) ---------------------------
#
# `MICRODUCK_SENSED_TRUTH=1` makes `_sensed_head` write the four slots from the
# recipe's own projection of the TRUE ball instead of off the track, so a gym
# arm can price what the tracker's ~5.5 cm placement error costs the kick. It
# is a SIM-ONLY ablation — the robot has no `qpos` for the ball — so the two
# things worth locking are that it is inert when off and that when on it really
# is the recipe's projection of the truth and not a second, tidier tracker.

def test_the_truth_knob_is_off_unless_asked_and_keeps_no_state(monkeypatch, tmp_path):
    """Off is the shipping value and costs one attribute: no memory, no RNG
    stream, and the slots still come out of the track."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    monkeypatch.delenv("MICRODUCK_SENSED_TRUTH", raising=False)
    w = World(_one_duck(), seed=0)
    assert w.sensed_truth is False
    d = w.ducks["d0"]
    trk = w._ball_tracker(d)
    w.t = 1.0
    trk.tracks = [_fake_track(math.pi / 4, 0.125, last_t=1.0)]
    w._sensed_head(d)
    assert d.head_cmd[0] == pytest.approx(0.5, abs=1e-6)      # the TRACK's bearing, not the ball's
    assert w._truth_sense == {} and w._truth_rng is None      # …and nothing was computed beside it

    assert w.sensed_truth_mode == ""

    for value in ("1", "true", "yes"):
        monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", value)
        built = World(_one_duck(), seed=0)
        assert built.sensed_truth is True and built.sensed_truth_mode == "all"
    for value in ("", "0", "false"):
        monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", value)
        built = World(_one_duck(), seed=0)
        assert built.sensed_truth is False and built.sensed_truth_mode == ""


def test_the_split_modes_are_read_onto_the_constructed_world_and_a_typo_raises(monkeypatch, tmp_path):
    """The two SPLITS (roadmap 12as follow-up I) are modes of the same knob, so
    the same rule 0 applies to them: the arm a battery is running has to be
    readable off the World it built. A value that is not a mode RAISES — the
    failure this prevents is `MICRODUCK_SENSED_TRUTH=fesh` running the truth
    arm (or, worse, the play arm) and being quoted as the split."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    for value in ("xy", "fresh", "all"):
        monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", value)
        built = World(_one_duck(), seed=0)
        assert built.sensed_truth is True and built.sensed_truth_mode == value
    for value in ("fesh", "XY", "2", "truth"):
        monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", value)
        with pytest.raises(ValueError, match="is not a mode"):
            World(_one_duck(), seed=0)


@needs_walker
def test_the_truth_knob_off_is_byte_identical_to_the_package_before_it(monkeypatch, tmp_path):
    """The out-of-band half, in band when a snapshot is available:
    MICRODUCK_PRE_PKG=<dir holding a pre-edit microduck_local> makes this run
    the same seeded rollout under both packages and compare the digest of every
    qpos of every tick (memory note shared-checkout-ab-on-copies).

    It is env-gated on purpose rather than built from `git show HEAD:`: once
    this change is committed, HEAD carries the knob and a self-built snapshot
    would be comparing the file with itself — a test that passes by being
    vacuous. Run it with a real snapshot:

        cp -R src/microduck_local <snap>/microduck_local     # at the pre-edit sha
        ln -s $PWD/policies <snap>/policies                  # or it silently runs the Hub kicks
        MICRODUCK_PRE_PKG=<snap> uv run --with pytest pytest tests/test_sensed_kick_world.py -k byte
    """
    import subprocess
    import sys
    import textwrap
    from pathlib import Path

    pre = os.environ.get("MICRODUCK_PRE_PKG")
    if not pre:
        pytest.skip("set MICRODUCK_PRE_PKG to a pre-edit package copy to run the digest A/B")
    src = textwrap.dedent("""
        import hashlib, sys
        import numpy as np
        import microduck_local
        from microduck_local.brain.brain_env import POLICIES_DIR, onnx_infer
        from microduck_local.world import World
        from microduck_local.world.scenario import Ball, Duck, Scenario, Wall
        hx, hy = 1.5, 1.25
        corners = [(-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)]
        sc = Scenario(name="t", floor=(2 * hx + 0.5, 2 * hy + 0.5),
                      walls=[Wall(corners[i], corners[(i + 1) % 4], 0.3, 0.02) for i in range(4)],
                      balls=[Ball((0.45, 0.22))],
                      ducks=[Duck("d0", (0.0, 0.0, 0.0), None, "datasheet", "datasheet", "chase",
                                  team="cream")], goal_width=0.7)
        walker = onnx_infer(POLICIES_DIR / "alpha_walking.onnx")
        w = World(sc, infer_for={"d0": walker}, seed=3)
        h = hashlib.sha256()
        for k in range(100):
            if k == 20:
                w.start_skill(w.ducks["d0"], "kick_left")
            w.step()
            h.update(np.ascontiguousarray(w.data.qpos, dtype=np.float64).tobytes())
        print(microduck_local.__file__, file=sys.stderr)
        print(h.hexdigest())
    """)
    script = tmp_path / "digest.py"
    script.write_text(src)
    _pin(monkeypatch, tmp_path, lambda foot: {"run": "old", "exit_rad": 0.0})   # the VENDORED shape
    import microduck_local
    here_pkg = str(Path(microduck_local.__file__).resolve().parents[1])
    env = dict(os.environ)
    env.pop("MICRODUCK_SENSED_TRUTH", None)
    # The snapshot trap (memory note shared-checkout-ab-on-copies): a package
    # copy resolves POLICIES_DIR relative to ITSELF and would silently run the
    # Hub's walker, or none. Pin the upstream checkout for both runs; the kicks
    # are already pinned by `_pin`.
    env["MICRODUCK_RL_DIR"] = str(C.MICRODUCK_RL_DIR)

    def digest(pkg: str) -> str:
        out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                             env=dict(env, PYTHONPATH=pkg))
        assert out.returncode == 0, out.stderr[-2000:]
        assert pkg in out.stderr, f"ran the wrong package: {out.stderr[-400:]}"
        return out.stdout.strip()

    here, there = digest(here_pkg), digest(pre)
    assert here == there, f"the knob-off rollout moved: {here} != {there}"


@needs_walker
def test_the_truth_knob_on_writes_the_recipes_projection_of_the_true_ball(monkeypatch, tmp_path):
    """On, the slots are the TRUE ball seen through the recipe's own camera
    maths — within the projection's own jitter, and regardless of what the
    tracker believes. The poisoned track is the discriminator: a second, better
    tracker would still be reading the track."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", "1")
    w = _standing_world(_one_duck())
    assert w.sensed_truth is True
    d = w.ducks["d0"]
    _place_ball(w, 0.45, 0.22)
    for _ in range(15):
        w.step()
    assert w._truth_sense["d0"]["world"] is not None, "the recipe's camera never saw the ball"

    # The held estimate is the ball, to about a centimetre.
    est = w._truth_sense["d0"]["world"]
    assert math.dist(est, (0.45, 0.22)) < 0.03, est

    # …and the four slots are that estimate in the recipe's units.
    w._sensed_head(d)
    x, y, yaw = w.odom(d)
    ahead = math.cos(yaw) * (0.45 - x) + math.sin(yaw) * (0.22 - y)
    beside = -math.sin(yaw) * (0.45 - x) + math.cos(yaw) * (0.22 - y)
    assert d.head_cmd[0] == pytest.approx(math.atan2(beside, ahead) / (math.pi / 2), abs=0.05)
    assert d.head_cmd[1] == pytest.approx(min(1.0, math.hypot(ahead, beside) / LM_RANGE_SCALE), abs=0.06)
    assert d.head_cmd[0] > 0.0                      # the ball is to the LEFT and reads left

    # A poisoned track cannot move it: the ablation does not read the tracker.
    poisoned = d.head_cmd.copy()
    trk = w._ball_tracker(d)
    bogus = _fake_track(-math.pi / 3, 2.0, last_t=w.t)
    bogus.xy = (-1.0, -1.0)
    trk.tracks = [bogus]
    w._sensed_head(d)
    assert np.allclose(d.head_cmd, poisoned)

    # …and the memory goes with the duck, like the track's.
    w.reset_duck("d0")
    assert "d0" not in w._truth_sense


# -- the two SPLITS of the ablation (roadmap 12as follow-up I) ---------------
#
# `=1` swaps THREE things at once — placement, `seen`, and the held estimate's
# age — so it prices the tracker as a whole and cannot say which term the kick
# pays for. `xy` and `fresh` cut it in half: `xy` is the truth's PLACEMENT with
# the track's own freshness, `fresh` is the track's placement with the truth's
# freshness. What has to be locked is that each really writes the mix it claims
# and nothing else, which needs a track that is WRONG in one half and RIGHT in
# the other — a tidy track would let a mode read the wrong producer and pass.

def _slots(w: World, d, mode: str) -> np.ndarray:
    """The four slots this World would write in `mode`, on the state it is in
    right now. The mode is an attribute read per call, so all four arms can be
    taken off ONE world state — which is the only way the mix is checkable."""
    w.sensed_truth_mode, w.sensed_truth = mode, mode != ""
    w._sensed_head(d)
    return np.asarray(d.head_cmd, np.float64).copy()


@needs_walker
def test_the_splits_write_exactly_the_mix_they_claim(monkeypatch, tmp_path):
    """Both poisonings, on one warm world. First a track whose PLACEMENT is
    nonsense but whose freshness is real, then a track placed on the true ball
    but a second and a half STALE — in each the mode that is supposed to take
    the poisoned half must carry it, and the mode that is supposed to replace
    it must not."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})
    monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", "1")     # …so the truth memory runs and is warm
    w = _standing_world(_one_duck())
    d = w.ducks["d0"]
    _place_ball(w, 0.45, 0.22)
    for _ in range(15):
        w.step()
    assert w._truth_sense["d0"]["world"] is not None, "the recipe's camera never saw the ball"
    assert w._ball_tracker(d).best(SENSED_BALL_CLS, w.t, min_hits=1) is not None, "the camera never saw it"
    trk = w._ball_tracker(d)
    last = d.detector.last
    assert last is not None

    # (A) placement POISONED, freshness REAL: the ball is behind and to the
    # right of where it is, and the newest detector frame is the track's hit.
    bogus = _fake_track(-math.pi / 3, 2.0, last_t=last.t)
    bogus.xy = (-1.0, -1.0)
    trk.tracks = [bogus]
    track, allm = _slots(w, d, ""), _slots(w, d, "all")
    xy, fresh = _slots(w, d, "xy"), _slots(w, d, "fresh")

    # the composition itself: each half is one producer's own output, exactly.
    assert np.array_equal(xy[0:2], allm[0:2]) and np.array_equal(xy[2:4], track[2:4])
    assert np.array_equal(fresh[0:2], track[0:2]) and np.array_equal(fresh[2:4], allm[2:4])
    # …and the poison is visible, so the check above is not vacuous.
    assert track[0] < -0.3 and allm[0] > 0.0          # the track says RIGHT, the truth says LEFT
    assert abs(xy[0] - track[0]) > 0.3               # `xy` did not inherit the poisoned placement
    assert abs(fresh[0] - track[0]) < 1e-9           # …and `fresh` did
    assert track[2] == 1.0 and track[3] == 1.0       # the track's freshness is REAL here
    assert xy[2] == 1.0 and xy[3] == 1.0             # …and `xy` keeps it

    # (B) placement REAL, freshness POISONED: the track sits on the true ball
    # but its last hit is 1.5 s old, so `seen` is off and confidence has faded.
    stale = _fake_track(0.0, 1.0, last_t=w.t - 1.5)
    stale.xy = (0.45, 0.22)          # ON the true ball, in the odometry frame `_track_head` reads from
    trk.tracks = [stale]
    # …and the truth memory pinned to a sighting on THIS tick, so the contrast
    # between the two freshness halves is the poisoning and not the cadence
    # (`truth_update` only refreshes every LM_DETECT_EVERY ticks, and whether
    # the last one landed is a property of where the head happened to be).
    w._truth_sense["d0"]["seen"], w._truth_sense["d0"]["conf"] = True, 1.0
    track, allm = _slots(w, d, ""), _slots(w, d, "all")
    xy, fresh = _slots(w, d, "xy"), _slots(w, d, "fresh")

    assert np.array_equal(xy[0:2], allm[0:2]) and np.array_equal(xy[2:4], track[2:4])
    assert np.array_equal(fresh[0:2], track[0:2]) and np.array_equal(fresh[2:4], allm[2:4])
    assert track[2] == 0.0 and track[3] == pytest.approx(math.exp(-1.5 / LM_MEM_TAU), abs=1e-6)
    assert allm[2] == 1.0 and allm[3] > 0.9          # the truth path saw it on this very frame
    assert fresh[2] == 1.0 and fresh[3] > 0.9        # …and `fresh` takes that, not the stale fade
    assert xy[2] == 0.0                              # …while `xy` keeps the track's staleness
    # the placement halves agree with each other AND with the ball, both ways
    assert track[0] == pytest.approx(allm[0], abs=0.08) and fresh[0] == pytest.approx(track[0])


@needs_walker
def test_a_split_mode_runs_a_whole_kick_window_and_off_is_unchanged_beside_it(monkeypatch, tmp_path):
    """End to end, through `_skill_cmd`, with the duck's own camera: the split
    arms are arms a gym can actually run. The ball is to the LEFT, so every
    tick of every mode reads left — and the OFF world beside them, built from
    the same seed, is bit-for-bit the world it always was."""
    _pin(monkeypatch, tmp_path, lambda foot: {"sensed": True, "exit_rad": 0.0})

    def window(mode: str) -> list[np.ndarray]:
        if mode:
            monkeypatch.setenv("MICRODUCK_SENSED_TRUTH", mode)
        else:
            monkeypatch.delenv("MICRODUCK_SENSED_TRUTH", raising=False)
        w = _standing_world(_one_duck())
        assert w.sensed_truth_mode == mode
        d = w.ducks["d0"]
        _place_ball(w, 0.45, 0.22)
        for _ in range(15):
            w.step()
        assert w.start_skill(d, "kick_left")
        rows = []
        for _ in range(int(round(KICK_S / CTRL_DT))):
            w.step()
            if d.skill is None:
                break
            rows.append(d.head_cmd.copy())
        assert len(rows) >= 5
        return rows

    # (The vendored blind network topples this cold standing duck partway
    # through the window, and under the truth halves the bearing follows the
    # ball LIVE as the body rotates — so what is locked is the window's OPENING
    # ticks, before the fall turns the duck past the ball.)
    for mode in ("xy", "fresh", ""):
        rows = window(mode)
        assert rows[0][0] > 0.0 and rows[1][0] > 0.0, (mode, [float(r[0]) for r in rows])
        assert all(0.0 < r[1] <= 1.0 for r in rows), mode
        assert all(r[3] > 0.0 for r in rows), mode
