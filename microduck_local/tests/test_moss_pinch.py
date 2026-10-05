"""The scripted top-down pinch for small objects (`robots/moss_pinch`,
`TidyMoss.pinch_small`): the arm's own kinematics, the locked target, and
which objects it is used for."""
import math

import mujoco
import numpy as np
import pytest

from microduck_local.robots import moss
from microduck_local.robots import moss_env as ME
from microduck_local.robots.moss_pinch import MossKinematics

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


def _base(env, p):
    bx, by, byaw = env.driver.pose(env.data)
    c, s = math.cos(-byaw), math.sin(-byaw)
    return np.array([(p[0] - bx) * c - (p[1] - by) * s,
                     (p[0] - bx) * s + (p[1] - by) * c, p[2]])


def test_the_depth_fix_lands_on_the_object_in_the_base_frame():
    """The wrist camera reports the object in the TOOL frame; through the
    arm's own kinematics that must be where the object really is."""
    K = MossKinematics()
    env = ME.MossPickEnv(seed=4, pick_rung=2)
    env.reset(seed=4)
    m, d = env.model, env.data
    arm = {j: float(d.qpos[m.joint(j).qposadr[0]]) for j in moss.ARM_JOINTS}
    jaw = float(d.qpos[m.joint(moss.GRIPPER_JOINT).qposadr[0]])
    R = d.site_xmat[env.tcp_site].reshape(3, 3)
    grip = R.T @ (d.xpos[env.can_body] - d.site_xpos[env.tcp_site])
    est = K.object_in_base(arm, jaw, grip)
    assert np.linalg.norm(est - _base(env, d.xpos[env.can_body])) < 1e-4


def test_a_solved_pose_puts_the_pads_over_the_target_jaws_down():
    """Solved in the robot-only model, checked in the pick env's model: the
    midpoint between the pads at the target, the tool pointing down. (With
    both fingers moving, the tied jaw keeps that midpoint ON the tool point
    at every opening — measured 0.0 mm — so aiming either is the same.)"""
    K = MossKinematics()
    env = ME.MossPickEnv(seed=1, pick_rung=2, pick_box="0.22,0.30,0.06")
    env.reset(seed=1)
    m, d = env.model, env.data
    arm = {j: float(d.qpos[m.joint(j).qposadr[0]]) for j in moss.ARM_JOINTS}
    target = _base(env, d.xpos[env.can_body]) + np.array([0.0, 0.0, 0.07])
    pose, res = K.solve(target, K.roll_for(arm, 0.4), arm, jaw=0.0)
    assert res < 1e-3
    for j in moss.ARM_JOINTS:
        d.qpos[m.joint(j).qposadr[0]] = pose[j]
    for j in moss.FINGER_JOINTS:
        d.qpos[m.joint(j).qposadr[0]] = 0.0
    mujoco.mj_kinematics(m, d)
    mid = (d.geom_xpos[m.geom("pad_left").id] + d.geom_xpos[m.geom("pad_right").id]) / 2
    assert np.linalg.norm(_base(env, mid) - target) < 0.002
    assert d.site_xmat[env.tcp_site].reshape(3, 3)[2, 2] < -0.99


def test_the_pinch_stays_on_its_object_when_the_head_fix_jumps():
    """As a small thing leaves the head camera's view its fix jumps to the
    next toy (9 of 12 butt pinches drove toward one). A fix further than
    `pinch_lock_m` from the locked spot must not move the target."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss

    b = TidyMoss()
    b._to("pinch", 0.0)
    arm = dict(zip(moss.ARM_JOINTS, moss.GRASP_POSE))
    arm[moss.GRIPPER_JOINT] = 0.041
    b._odom = (0.0, 0.0, 0.0)
    b._fix = (0.40, 0.0)
    b._pinch_step(Senses(t=0.02, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm), 0.02, 0.02)
    locked = b.pinch_target_world
    assert locked is not None and abs(locked[0] - 0.40) < 1e-6
    b._fix = (0.95, 0.5)                                 # another toy
    b._pinch_step(Senses(t=0.04, odom=(0.0, 0.0, 0.0), speed=0.0, arm=arm), 0.04, 0.04)
    assert b.pinch_target_world == locked


def test_something_the_pinch_picked_up_swings_to_the_bin_more_slowly():
    """Most stow losses happen on the swing round to the bin, and a pinched
    butt was lost there 58 times in 96 yard runs. After a pinch the swing
    takes `pinch_turn_s` (878 v 838 objects in the bin over 96 seeds); an
    object the learned pick took keeps the usual swing."""
    from microduck_local.brain.runtime import Senses
    from microduck_local.brain.tidy_moss import TidyMoss
    from microduck_local.robots import moss_env as ME

    def arm_at(pinched: bool) -> dict:
        b = TidyMoss()
        b._to("stow", 0.0)
        b._pinched = pinched
        b._carry_from = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
        arm = dict(zip(moss.ARM_JOINTS, moss.LIFT_POSE))
        arm[moss.GRIPPER_JOINT] = 0.01
        t = b.p.stow_high_s + b.p.stow_turn_s - 0.02     # the usual swing: done
        for k in range(int(t / 0.02) + 1):
            it = b.step(Senses(t=0.02 * k, odom=(0.0, 0.0, 0.0), speed=0.0,
                               arm=arm, holding=True))
        assert b.state == "stow"
        return it.arm

    turned = dict(zip(moss.ARM_JOINTS, ME.STOW_TURNED))
    usual, slow = arm_at(False), arm_at(True)
    assert abs(usual["shoulder_pan"] - turned["shoulder_pan"]) < 0.05
    assert abs(slow["shoulder_pan"] - turned["shoulder_pan"]) > 0.2
