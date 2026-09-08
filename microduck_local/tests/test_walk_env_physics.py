"""Locks for the training-layer fixes of the 2026-09-06 physics audit
(docs/roadmap.md, "Physics audit", items 6-9):

6. the IMU blocks of the obs (gyro, projected gravity) and the trunk height
   the fall check reads describe the INTEGRATED state after the substep
   loop, not the state one substep before it;
7. the solver runs the upstream velocity cfg's integrator / iteration budget;
8. `train-walk` trains on the BAM servo unless told otherwise;
9. every domain-randomized quantity LANDS in the model, inside upstream's
   range, and restores to the compile-time model rather than accumulating;
   velocity pushes fire at upstream's cadence and magnitude.

Each test measures the model/data, never the code's own bookkeeping alone.
"""

import mujoco
import numpy as np
import pytest

from microduck_local import contract as C
from microduck_local import walk_env as W
from microduck_local.walk_env import MicroduckWalkEnv, shared_model_scope

QUIET = dict(obs_noise=False, action_delay=False, random_yaw=False)


@pytest.fixture(autouse=True)
def _no_actuator_env_override(monkeypatch):
    monkeypatch.delenv("MICRODUCK_ACTUATOR", raising=False)


# ------------------------------------------------------- 6. obs freshness


@pytest.mark.parametrize("actuator", ["xml", "bam"])
def test_imu_obs_and_height_equal_a_fresh_forward(actuator):
    """After step(), the gyro and projected-gravity blocks and the trunk
    height must equal what a full mj_forward on the integrated state gives —
    to the bit, since they are the same float64 numbers cast the same way.
    Before the refresh they trailed qpos/qvel by one substep (measured on
    the shipped walker: gyro up to 0.77 rad/s off, gravity 0.0098)."""
    env = MicroduckWalkEnv(actuator=actuator, domain_rand=False, seed=0, **QUIET)
    obs, _ = env.reset(seed=0)
    m, d = env.model, env.data
    rng = np.random.default_rng(0)
    trunk = env.trunk_body_id
    checked = 0
    for _ in range(80):
        obs, _, term, trunc, _ = env.step(rng.uniform(-0.4, 0.4, C.NUM_JOINTS).astype(np.float32))
        gyro, gravity = obs[0:3].copy(), obs[3:6].copy()
        z = float(d.xpos[trunk][2])
        if actuator == "bam":
            # The refresh must not clobber the BAM torque readout.
            np.testing.assert_array_equal(d.actuator_force[env.bam._act_ix],
                                          env.bam.applied_torque)
        mujoco.mj_forward(m, d)
        np.testing.assert_array_equal(gyro, d.sensordata[env.gyro_adr].astype(np.float32))
        np.testing.assert_array_equal(
            gravity, C.quat_rotate_inverse(d.xquat[trunk], W._NEG_Z).astype(np.float32))
        assert abs(z - float(d.xpos[trunk][2])) <= 1e-12
        checked += 1
        if term or trunc:
            obs, _ = env.reset()
    assert checked == 80


def test_the_refresh_is_what_makes_them_fresh():
    """Blind the refresh and the staleness comes back — so the test above is
    testing the fix, not an accident of the rollout."""
    env = MicroduckWalkEnv(actuator="xml", domain_rand=False, seed=0, **QUIET)
    env.reset(seed=0)
    env._refresh_derived = lambda: None
    rng = np.random.default_rng(0)
    worst = 0.0
    for _ in range(40):
        obs, *_ = env.step(rng.uniform(-0.4, 0.4, C.NUM_JOINTS).astype(np.float32))
        gyro = obs[0:3].astype(np.float64)
        mujoco.mj_forward(env.model, env.data)
        worst = max(worst, float(np.abs(gyro - env.data.sensordata[env.gyro_adr]).max()))
    assert worst > 1e-3, f"stale-by-a-substep gyro should be visible, got {worst:.2e}"


# --------------------------------------------------- 7. solver parity


@pytest.mark.parametrize("actuator", ["xml", "bam"])
def test_solver_settings_match_the_upstream_velocity_cfg(actuator):
    """mjlab/tasks/velocity/velocity_env_cfg.py: implicitfast, iterations=10,
    ls_iterations=20 (the microduck velocity cfg inherits them on flat
    ground). infer_policy.py runs the MJCF default — a documented
    difference, see walk_env.UPSTREAM_*."""
    env = MicroduckWalkEnv(actuator=actuator, seed=0)
    opt = env.model.opt
    assert opt.integrator == int(mujoco.mjtIntegrator.mjINT_IMPLICITFAST)
    assert opt.iterations == 10
    assert opt.ls_iterations == 20
    assert opt.timestep == pytest.approx(C.PHYSICS_DT)


# ---------------------------------------------- 8. train-walk trains on BAM


def test_train_walk_defaults_to_bam_and_the_flag_opts_out(monkeypatch):
    from microduck_local.train import DEFAULT_ACTUATOR, env_kwargs_from_args, parse_args

    assert DEFAULT_ACTUATOR == "bam"
    kw = env_kwargs_from_args(parse_args([]))
    assert kw["actuator"] == "bam" and "actuator_force" not in kw
    env = MicroduckWalkEnv(seed=0, **kw)
    assert env.actuator_model == "bam" and env.bam is not None
    assert env.domain_rand and env.obs_noise and env.push_robot

    # The explicit flag is the opt-out and beats the process env...
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "bam")
    kw = env_kwargs_from_args(parse_args(["--actuator", "xml"]))
    assert MicroduckWalkEnv(seed=0, **kw).actuator_model == "xml"
    # ...while MICRODUCK_ACTUATOR still overrides the bare default, as before.
    monkeypatch.setenv("MICRODUCK_ACTUATOR", "xml")
    kw = env_kwargs_from_args(parse_args([]))
    assert MicroduckWalkEnv(seed=0, **kw).actuator_model == "xml"
    assert env_kwargs_from_args(parse_args(["--no-domain-rand"]))["domain_rand"] is False


# ------------------------------------------------- 9. domain randomization


def _pristine_model():
    m = mujoco.MjModel.from_xml_path(str(C.SCENE_WALK_XML))
    for gid in ("left_foot_collision", "right_foot_collision"):
        m.geom_priority[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, gid)] = 1
    return m


@pytest.mark.parametrize("actuator", ["xml", "bam"])
def test_every_randomized_field_lands_in_range_and_restores(actuator):
    env = MicroduckWalkEnv(actuator=actuator, domain_rand=True, seed=5, **QUIET)
    m = env.model
    trunk = env.trunk_body_id
    heads = env._head_com_body_ids
    joints = env.joint_qvel_adr
    base = {k: v.copy() for k, v in env._defaults.items()}
    pristine = _pristine_model()
    # The baseline IS the compiled model (BAM's armature retune aside).
    for name in ("body_mass", "body_inertia", "body_ipos", "geom_friction"):
        np.testing.assert_array_equal(base[name], getattr(pristine, name))
    if actuator == "bam":
        np.testing.assert_allclose(base["dof_armature"][joints], env.bam.p["armature"])
    else:
        np.testing.assert_array_equal(base["dof_armature"], pristine.dof_armature)
    assert len(heads) == len(W.HEAD_COM_BODIES), "a head body went missing by name"

    mass, inertia, com_t, com_h, arm = [], [], [], [], []
    for i in range(50):
        env.reset(seed=100 + i)
        mass.append(m.body_mass[trunk] / base["body_mass"][trunk])
        inertia.append(m.body_inertia[trunk] / base["body_inertia"][trunk])
        com_t.append(m.body_ipos[trunk] - base["body_ipos"][trunk])
        com_h.append(m.body_ipos[heads] - base["body_ipos"][heads])
        arm.append(m.dof_armature[joints] / base["dof_armature"][joints])
        # Derived constants follow the draw (the audit found them stale).
        assert m.body_subtreemass[trunk] == pytest.approx(m.body_mass[1:].sum(), abs=1e-12)
        # Nothing outside the randomized bodies/joints moves.
        others = np.setdiff1d(np.arange(m.nbody), np.r_[trunk, heads])
        np.testing.assert_array_equal(m.body_ipos[others], base["body_ipos"][others])
        np.testing.assert_array_equal(np.delete(m.body_mass, trunk), np.delete(base["body_mass"], trunk))
    mass, inertia, com_t, com_h, arm = map(np.array, (mass, inertia, com_t, com_h, arm))

    lo, hi = W.MASS_SCALE_RANGE
    assert lo - 1e-9 <= mass.min() and mass.max() <= hi + 1e-9
    assert mass.min() < 0.97 and mass.max() > 1.03, "50 draws never left the middle"
    np.testing.assert_allclose(inertia, np.broadcast_to(mass[:, None], inertia.shape),
                               rtol=1e-12)  # ONE factor, mass and inertia
    r = W.TRUNK_COM_STAGES[0][1]
    assert np.abs(com_t).max() <= r + 1e-12 and np.abs(com_t).max() > 0.5 * r
    assert np.abs(com_h).max() <= r + 1e-12 and np.abs(com_h).max() > 0.5 * r
    assert (np.abs(com_h).reshape(50, -1).max(0) > 0).all(), "a head body never moved"
    lo, hi = W.ARMATURE_SCALE_RANGE
    assert lo - 1e-9 <= arm.min() and arm.max() <= hi + 1e-9
    assert arm.min() < 0.93 and arm.max() > 1.07
    assert len({round(float(x), 6) for x in arm[0]}) > 1, "armature must draw per joint"

    # Non-accumulating: after 50 draws, DR off restores EVERY field — the
    # randomized ones and the mj_setConst outputs alike — to the bit.
    env.domain_rand = False
    env.reset(seed=999)
    for name, value in base.items():
        np.testing.assert_array_equal(getattr(m, name), value, err_msg=name)


def test_com_offset_follows_the_upstream_curriculum():
    env = MicroduckWalkEnv(actuator="xml", domain_rand=True, seed=1, **QUIET)
    m, trunk, heads = env.model, env.trunk_body_id, env._head_com_body_ids
    base = env._defaults["body_ipos"]

    def widest(lifetime, n=40):
        env._lifetime_steps = lifetime
        wt = wh = 0.0
        for i in range(n):
            env.reset(seed=i)
            wt = max(wt, float(np.abs(m.body_ipos[trunk] - base[trunk]).max()))
            wh = max(wh, float(np.abs(m.body_ipos[heads] - base[heads]).max()))
        return wt, wh

    wt0, wh0 = widest(0)
    assert wt0 <= 0.003 + 1e-12 and wh0 <= 0.003 + 1e-12
    wt3, wh3 = widest(36_000)
    assert 0.010 < wt3 <= 0.015 + 1e-12, "trunk range should have ramped to ±15 mm"
    assert 0.005 < wh3 <= 0.010 + 1e-12, "head range should have ramped to ±10 mm"
    # A pinned knob overrides the ladder.
    env2 = MicroduckWalkEnv(actuator="xml", domain_rand=True, seed=1,
                            trunk_com_offset_m=0.0, head_com_offset_m=0.0, **QUIET)
    env2._lifetime_steps = 36_000
    env2.reset(seed=3)
    np.testing.assert_array_equal(env2.model.body_ipos, env2._defaults["body_ipos"])


def test_velocity_pushes_fire_at_upstream_cadence_and_magnitude():
    env = MicroduckWalkEnv(actuator="xml", domain_rand=True, seed=2, max_episode_s=200.0,
                           terminate_on_fall=False, height_termination=False, **QUIET)
    assert env.push_robot
    d = env.data
    env.reset(seed=2)
    deltas, at = [], []
    orig = env._push

    def spy():
        before = d.qvel[0:2].copy()
        orig()
        deltas.append(d.qvel[0:2] - before)
        at.append(env.step_count)
    env._push = spy

    zero = np.zeros(C.NUM_JOINTS, np.float32)
    for _ in range(4000):                      # 80 s of episode time
        env.step(zero)
    lo, hi = W.PUSH_INTERVAL_S
    at = np.array(at) + 1                      # step_count increments after the push
    assert 80 / hi - 1 <= len(at) <= 80 / lo + 1, f"{len(at)} pushes in 80 s"
    assert at[0] * C.CTRL_DT <= hi + C.CTRL_DT, "first push is drawn at reset"
    gaps = np.diff(at) * C.CTRL_DT
    assert gaps.min() >= lo - C.CTRL_DT and gaps.max() <= hi + C.CTRL_DT, gaps
    deltas = np.array(deltas)
    vlo, vhi = W.PUSH_VEL_RANGE
    assert deltas.min() >= vlo and deltas.max() <= vhi
    assert deltas.min() < 0.5 * vlo and deltas.max() > 0.5 * vhi, "never used the range"
    assert np.abs(deltas[:, 0]).max() > 0 and np.abs(deltas[:, 1]).max() > 0
    np.testing.assert_array_equal(deltas[-1], env.last_push)
    assert env.push_count == len(at)


def test_pushes_follow_domain_rand_and_stay_off_for_behaviors():
    from microduck_local.behaviors import BehaviorEnv

    off = MicroduckWalkEnv(actuator="xml", domain_rand=False, seed=0, **QUIET)
    off.reset(seed=0)
    assert off.push_robot is False and off._push_countdown is None
    explicit = MicroduckWalkEnv(actuator="xml", domain_rand=True, push_robot=False, seed=0, **QUIET)
    explicit.reset(seed=0)
    assert explicit._push_countdown is None
    on = MicroduckWalkEnv(actuator="xml", domain_rand=True, seed=0, **QUIET)
    on.reset(seed=0)
    assert on._push_countdown is not None
    # Behaviors: a push in a trick's curriculum would be a new experiment.
    # The run recipe's velocity-DR subset stays as it was tuned: no pushes.
    run = BehaviorEnv("run", domain_rand=True, random_yaw=True, actuator="xml", seed=0)
    run.reset(seed=0)
    assert run.push_robot is False and run._push_countdown is None
    assert BehaviorEnv("run", domain_rand=True, push_robot=True, actuator="xml", seed=0).push_robot


def test_shared_model_is_repointed_at_every_randomized_field():
    """In-process sharing replays THIS env's draw — now every DR field and
    the mj_setConst outputs, not just mass and friction — before its physics."""
    with shared_model_scope(exclusive=False):
        a = MicroduckWalkEnv(seed=11, domain_rand=True, **QUIET)
        b = MicroduckWalkEnv(seed=22, domain_rand=True, **QUIET)
    assert a.model is b.model
    a.reset(seed=1)
    b.reset(seed=2)
    act = np.zeros(C.NUM_JOINTS, np.float32)
    for name in W.DR_MODEL_FIELDS:
        assert not np.array_equal(a._dr_fields[name], b._dr_fields[name]), name
    for _ in range(3):
        a.step(act)
        for name, value in a._dr_fields.items():
            np.testing.assert_array_equal(getattr(a.model, name), value, err_msg=name)
        b.step(act)
        for name, value in b._dr_fields.items():
            np.testing.assert_array_equal(getattr(b.model, name), value, err_msg=name)


def test_eval_walk_does_not_push_unless_asked():
    """`eval-walk` is the deployment-contract eval; its numbers are compared
    across months. The training env's pushes (2026-09-06) are opt-in there
    (`--push`), and the behavior env leaves them off for every trick."""
    from microduck_local.walk_env import MicroduckWalkEnv
    from microduck_local.behaviors.env import BehaviorEnv
    assert MicroduckWalkEnv(seed=0, push_robot=False).push_robot is False
    assert MicroduckWalkEnv(seed=0).push_robot is True            # training default
    assert BehaviorEnv("stand", seed=0).push_robot is False
    import microduck_local.eval_onnx as eo
    src = open(eo.__file__).read()
    assert "push_robot=args.push" in src
