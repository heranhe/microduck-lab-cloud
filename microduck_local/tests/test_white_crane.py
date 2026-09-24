"""White crane: forward lean, slow entry, counterbalance and six-second gate."""
import numpy as np
import mujoco
from microduck_local.behaviors import (
    BehaviorEnv, match_behavior, WHITE_CRANE_POSE, _wc_place, _wc_update,
    _wc_balance, _wc_rear_leg, _wc_hold, _wc_slow_entry, _wc_reset_entry,
    _wc_slow_pen, _wc_lean, WHITE_CRANE_PREP_S, WHITE_CRANE_RAISE_S,
)
from microduck_local import goal_training as G


def test_white_crane_pose_balance_and_consecutive_timer():
    assert match_behavior('白鹤亮翅').id == 'white_crane'
    env = BehaviorEnv('white_crane', actuator_force='xml', bam_current_scale=2.,
                      action_delay=False, obs_noise=False, domain_rand=False,
                      random_yaw=False, standing_spawns=True, seed=0)
    try:
        end_step=round((WHITE_CRANE_PREP_S+WHITE_CRANE_RAISE_S)/.02)
        assert env.actuator_model == 'bam' and env.action_delay
        assert env.observation_space.shape == (61,) and env.action_space.shape == (14,)
        env.reset(seed=0)
        _wc_place(env)
        env.foot_contact_state={'left': True, 'right': False}
        _wc_update(env)
        s=env._wc_state
        assert s['flat'] < np.deg2rad(15)
        assert s['clearance'] > .05 and s['foot'][0] < -.10
        assert np.deg2rad(15) <= s['lean'] <= np.deg2rad(30)
        assert s['margin'] >= .005  # leave >5 mm, not the old 0.5 mm edge spawn
        assert _wc_lean(env) > .99  # calibrated reference is the orientation target
        assert env._one_leg_hold_steps == 0  # rehearsal spawn is not success
        _wc_reset_entry(env)
        for step in range(1,101):
            env.step_count = step
            moving = {**s, 'foot': np.array([-.12*step/100,-.06,.1*step/100]),
                      'clearance': .02, 'lean': s['lean']*step/100,
                      'foot_speed': .07}
            # Stay just outside the complete pose until the 100th step.
            if step < 100: moving['foot'][0] = max(moving['foot'][0],-.099)
            assert _wc_slow_entry(env,moving,False) == (step == 100)
        env.step_count = end_step
        _wc_update(env)
        good = _wc_balance(env)
        assert _wc_rear_leg(env) > .9 and _wc_hold(env) > .5
        for _ in range(298): _wc_update(env)
        assert env._one_leg_hold_steps == 299
        _wc_update(env)
        assert env._one_leg_hold_steps == 300
        env.foot_contact_state['right']=True
        _wc_update(env)
        assert env._one_leg_hold_steps == 0 and _wc_hold(env) == 0
        # Head mass is in the balance objective, not just decorative joint matching.
        env.data.qpos[env.joint_qpos_adr[5]] += .5
        mujoco.mj_forward(env.model,env.data)
        _wc_update(env)
        assert abs(_wc_balance(env)-good) > .01
        env.reset(seed=1)
        assert env._one_leg_best_steps == 0
        _wc_place(env, noise=.012)
        q = env.data.qpos[env.joint_qpos_adr]
        limits = env.model.jnt_range[env.bam.joint_ids]
        assert np.all(q >= limits[:, 0]) and np.all(q <= limits[:, 1])
        # Waiting on the floor before a fast kick cannot satisfy slow entry.
        _wc_reset_entry(env)
        env.step_count = 400
        assert not _wc_slow_entry(env,{**s,'foot_speed':.5},False)
        env.step_count = 800
        assert not _wc_slow_entry(env,{**s,'foot_speed':0},False)
        env._wc_state = {**s,'foot_speed':.5}
        assert _wc_slow_pen(env) < -.9
        # Same pose at t=0 is too early: the policy observes the clip phase.
        env.reset(seed=3)
        obs0 = env._get_obs()
        env._wc_state = s
        early = _wc_rear_leg(env)
        env.step_count = end_step
        assert _wc_rear_leg(env) > early
        assert not np.allclose(obs0[59:61],env._get_obs()[59:61])
        assert np.allclose(env.clip.at(end_step)[0],WHITE_CRANE_POSE)
        assert env.clip.duration >= WHITE_CRANE_PREP_S+WHITE_CRANE_RAISE_S+6
        # The intended slow path itself must pass: flight starts only above
        # the clearance gate, and the arrival threshold is before full reach.
        _wc_reset_entry(env)
        for step in range(end_step+1):
            env.step_count=step
            p=np.clip((step*.02-WHITE_CRANE_PREP_S)/WHITE_CRANE_RAISE_S,0,1)
            moving={**s,'foot':np.array([-.12*p,-.03,.10*p]),
                    'clearance':max(.1*p-.006,0),'lean':np.deg2rad(20)*p,
                    'foot_speed':.05}
            _wc_slow_entry(env,moving,moving['clearance']<=0)
        assert env._wc_entry_ok
    finally:
        env.close()


def test_white_crane_uses_six_seconds_not_old_three():
    from microduck_local import viz_server as V
    assert V.trainee_env_kwargs(match_behavior('白鹤亮翅'))['standing_spawns']
    goal=G.new_goal('white_crane')
    assert goal['hold_seconds'] == 6
    assert G.record_evaluation(goal,{'holds':[3.0]*10},100)['status'] == 'training'
    assert G.record_evaluation(goal,{'holds':[5.98]*10},100)['status'] == 'training'
    assert G.record_evaluation(goal,{'holds':[6.0]*9+[0]},100)['status'] == 'passed'
    assert G.record_evaluation(G.new_goal(),{'holds':[3.0]*10},100)['status'] == 'training'


def test_forward_lean_cannot_hide_excess_side_tilt():
    from types import SimpleNamespace
    from microduck_local.behaviors import _wc_lean, WHITE_CRANE_PITCH
    env=SimpleNamespace(step_count=200, _wc_state={'lean':WHITE_CRANE_PITCH})
    scores=[]
    for tilt in (28,35,40):
        # Keep the SAME forward lean, changing only lateral inclination.
        x=np.sin(WHITE_CRANE_PITCH)
        z=-np.cos(np.deg2rad(tilt))
        gravity=np.array([x,np.sqrt(1-x*x-z*z),z])
        env._projected_gravity=lambda: gravity
        scores.append(_wc_lean(env))
    assert .99 < scores[0] <= 1
    assert 0 <= scores[2] < .5 < scores[1] < scores[0]
    # Holding excess roll fixed must NOT move the optimal forward pitch:
    # the old total-tilt penalty paid for standing straighter instead.
    for phase in (.5,1.):
        env.step_count=round((WHITE_CRANE_PREP_S+phase*WHITE_CRANE_RAISE_S)/.02)
        scores=[]
        for offset in (-.01,0,.01):
            pitch=WHITE_CRANE_PITCH*phase+offset
            roll=np.deg2rad(-35)
            gravity=np.array([np.sin(pitch),-np.cos(pitch)*np.sin(roll),
                              -np.cos(pitch)*np.cos(roll)])
            env._projected_gravity=lambda: gravity
            env._wc_state={'lean':pitch}
            scores.append(_wc_lean(env))
        assert scores[1] > max(scores[0],scores[2])
