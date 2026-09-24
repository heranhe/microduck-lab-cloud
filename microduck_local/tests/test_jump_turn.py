"""Synthetic trajectories lock success semantics; they are NOT learned skill."""
from types import SimpleNamespace
import numpy as np
from microduck_local import contract as C, goal_training as G
from microduck_local.behaviors import BehaviorEnv, BEHAVIORS, match_behavior
from microduck_local.behaviors.jump_turn import _jt_advance, _jt_reset, _jt_turn


def test_jump_turn_rejects_ground_spins_drops_rolls_and_unstable_landings():
    both = {'left': True, 'right': True}
    air = {'left': False, 'right': False}

    def trial(ground=0., angle=np.pi, clearance=.025, clean=True, stable=True):
        e = SimpleNamespace()
        _jt_reset(e)
        s = e._jt
        for step in range(31):
            _jt_advance(s, step, ground*step/30, both, 0., True, True)
        assert s['stage'] == 1
        _jt_advance(s, 31, ground, air, clearance, clean, False)
        for i in range(1, 21):
            _jt_advance(s, 31+i, ground+angle*i/20, air, clearance, clean, False)
        for i in range(50):
            _jt_advance(s, 52+i, ground+angle, both, 0., clean, stable)
        return s

    good = trial()
    assert good['best'] == 50 and good['flight_ok']
    assert trial(ground=np.pi, angle=0.)['best'] == 0  # floor pirouette
    assert trial(ground=np.pi/2, angle=np.pi/2)['best'] == 0
    assert trial(angle=3*np.pi)['best'] == 0  # no wrapping 540 into 180
    assert trial(angle=np.pi/2)['best'] == 0
    assert trial(clearance=.004)['best'] == 0  # contact flicker
    assert trial(clean=False)['best'] == 0  # rolling/body-floor contact
    assert trial(stable=False)['best'] == 0
    _jt_advance(good, 102, np.pi, both, 0., True, False)
    assert good['hold'] == 0 and good['best'] == 50
    e = SimpleNamespace()
    _jt_reset(e)
    for step in range(30):
        _jt_advance(e._jt, step, step*np.pi/29, air, .02, True, True)
    for step in range(30, 100):
        _jt_advance(e._jt, step, np.pi, both, 0., True, True)
    assert e._jt['best'] == 0  # initial spawn falling down isn't a jump
    goal = G.new_goal('jump_turn_180')
    assert goal['hold_seconds'] == 1
    assert G.record_evaluation(goal, {'holds':[1.]*8+[0.]*2}, 1)['status']=='training'
    assert G.record_evaluation(goal, {'holds':[1.]*9+[0.]}, 2)['status']=='passed'


def test_jump_turn_real_bam_contract_and_observable_commands():
    assert match_behavior('起跳转身180度').id == 'jump_turn_180'
    assert not BEHAVIORS['jump_turn_180'].symmetric
    e = BehaviorEnv('jump_turn_180', obs_noise=False, domain_rand=False,
                    random_yaw=False, actuator_force='xml', bam_current_scale=2.,
                    action_delay=False, standing_spawns=True)
    try:
        obs, _ = e.reset(seed=23)
        assert e.bam is not None and e.bam.max_current == 1.75 and e.action_delay
        assert obs.shape == (61,) and e.action_space.shape == (14,)
        assert e.behavior.spotter_fn is None and not e.spotter
        for _ in range(20):
            previous = e.prev_joint_vel.copy()
            obs, _, terminated, _, info = e.step(np.zeros(14, dtype=np.float32))
            np.testing.assert_allclose(obs[20:34], previous, atol=1e-6)
            assert np.isfinite(obs).all() and info['best_hold_s'] == 0
            assert obs[59] == e._jt['stage']/4
            assert e._jt['step'] == e.step_count
            for term in e._terms:
                value = term.fn(e)
                assert np.isfinite(value)
                if term.is_penalty:
                    assert value <= 0
            if terminated:
                break
        # Same visible yaw-rate target/gyro/phase => same reward, whatever
        # absolute yaw a simulation happens to assign.
        e._jt['stage'] = 2
        e.twist_cmd[2] = 5.
        a = _jt_turn(e)
        e._jt['total'] = 100.
        assert _jt_turn(e) == a
        e.reset(seed=24)
        assert e._jt['best'] == 0 and e._jt['stage'] == 0 and e._jt['clean']
    finally:
        e.close()


def test_jump_turn_air_rehearsal_never_certifies_supplied_history():
    from microduck_local.behaviors.jump_turn import _jt_update
    e = BehaviorEnv('jump_turn_180', obs_noise=False, domain_rand=False,
                    random_yaw=False, spawn_overrides={'MICRODUCK_SPAWN_FAMILY_PROBS':'1'})
    try:
        obs, _ = e.reset(seed=55)
        assert e.last_spawn == 'air-rehearsal' and e._jt['rehearsal']
        assert e._jt['stage'] == 2 and obs[59] == .5
        assert not any(e._foot_contacts().values())
        assert e.bam.max_current == 1.75 and e.action_delay
        assert not np.any(e.data.qfrc_applied) and not np.any(e.data.xfrc_applied)
        e._jt.update(best=50, step=-1)
        _jt_update(e)
        assert e._one_leg_best_steps == 0  # injected history cannot certify
        previous = e.prev_joint_vel.copy()
        obs, _, _, _, info = e.step(np.zeros(14, dtype=np.float32))
        np.testing.assert_allclose(obs[20:34], previous, atol=1e-6)
        assert info['best_hold_s'] == 0 and not info['is_success']
        # The exact same env and force-spawn knob cannot bypass certification's
        # standing_spawns flag. Reset removes all supplied history.
        e.standing_spawns = True
        obs, _ = e.reset(seed=55)
        assert e.last_spawn == 'standing' and not e._jt['rehearsal']
        assert e._jt['stage'] == 0 and e._jt['air_s'] == e._jt['air_yaw'] == 0
        assert e.step_count == 0 and obs[59] == 0
    finally:
        e.close()
