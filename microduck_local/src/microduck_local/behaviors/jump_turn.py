"""A single airborne half-turn, followed by an independently certified landing.

The commander exposes its phase/time in obs[59:61] and yaw-rate request in
obs[50]. Relative yaw is commander/evaluator state, never a hidden reward.
This commander is part of the local prototype, not a hardware deployment.
"""
from .core import *  # noqa: F401,F403
from .airflip import _af_airborne, _jaw_bid, _stand_tall
from .white_crane import _wc_sole_z


def _jt_reset(env):
    env._jt = dict(step=-1, stage=0, yaw=None, total=0., ground=0., air_yaw=0.,
                   air_s=0., clearance=0., planted=0, clean=True,
                   flight_ok=False, hold=0, best=0, rehearsal=False)


def _jt_spawn_air_rehearsal(env):
    """Reverse curriculum ONLY: supplied airborne state, not a learned launch.

    Like backflip's mid-roll family, the preceding rotation is credited for
    reward propagation. Certification bypasses all spawn families, and even
    training telemetry must never count this supplied history as success.
    No force/torque is applied after reset and BAM/current/delay stay intact.
    """
    d, m, r = env.data, env.model, env._rng
    angle = r.uniform(np.deg2rad(90), np.deg2rad(175))
    rate = float(np.clip(6*(np.pi-angle), .5, 8.))
    mujoco.mj_resetDataKeyframe(m, d, env.key_stand)
    d.qpos[3:7] = [np.cos(angle/2), 0., 0., np.sin(angle/2)]
    mujoco.mj_forward(m, d)
    clearance = r.uniform(.025, .06)
    d.qpos[2] += clearance-min(_wc_sole_z(env, s) for s in ('left','right'))
    d.qvel[:] = 0.
    d.qvel[2], d.qvel[5] = r.uniform(-.15, 0.), rate
    d.ctrl[:] = d.qpos[env.joint_qpos_adr]
    mujoco.mj_forward(m, d)
    env.bam.reset(d.ctrl)
    env.prev_joint_vel = env._joint_vel().copy()
    env.step_count = 25
    _jt_reset(env)
    env._jt.update(stage=2, yaw=_trunk_yaw(env), total=angle, air_yaw=angle,
                   air_s=angle/rate, clearance=clearance, rehearsal=True)
    return env._get_obs()


def _jt_advance(s, step, yaw, contacts, clearance, clean, stable):
    """One attempt only: initial spawn-drop and ground turns never qualify."""
    if s['step'] == step:
        return
    s['step'] = step
    delta = 0. if s['yaw'] is None else float(np.arctan2(
        np.sin(yaw-s['yaw']), np.cos(yaw-s['yaw'])))
    s['yaw'] = yaw
    s['total'] += delta
    s['clean'] &= clean
    both, air = all(contacts.values()), not any(contacts.values())
    s['planted'] = s['planted']+1 if both else 0
    if s['stage'] == 0 and step*C.CTRL_DT >= .5 and s['planted'] >= 5:
        s['stage'] = 1
    if s['stage'] <= 1:
        s['ground'] += abs(delta)
        if s['stage'] == 1 and air and clearance > .003:
            s['stage'] = 2
    elif s['stage'] == 2:
        if air:
            s['air_s'] += C.CTRL_DT
            s['air_yaw'] += delta
            s['clearance'] = max(s['clearance'], clearance)
        else:
            s['flight_ok'] = (s['air_s'] >= .08 and s['clearance'] >= .01
                              and s['air_yaw'] >= np.deg2rad(150)
                              and s['ground'] <= np.deg2rad(30) and s['clean']
                              and abs(s['total']-np.pi) <= np.deg2rad(15))
            s['stage'] = 4 if s['flight_ok'] else 3
    error = abs(s['total']-np.pi)  # no wrapping: 540 degrees is not 180
    valid = (s['stage'] == 4 and s['clean'] and both and stable
             and error <= np.deg2rad(15))
    s['hold'] = s['hold']+1 if valid else 0
    s['best'] = max(s['best'], s['hold'])


def _jt_update(env):
    if not hasattr(env, '_jt'):
        _jt_reset(env)
    s = env._jt
    if s['step'] == env.step_count:
        return
    contacts = env._foot_contacts()
    feet = set(env.foot_geoms.values())
    clean = True
    for c in env.data.contact:
        if c.geom1 == env.floor_geom and c.geom2 not in feet:
            clean = False
        if c.geom2 == env.floor_geom and c.geom1 not in feet:
            clean = False
    g = env._projected_gravity()
    clean &= bool(g[2] < -np.cos(np.deg2rad(45)))
    flat = True
    for side, gid in env.foot_geoms.items():
        gravity = -env.data.geom_xmat[gid][6:9]
        flat &= bool(gravity @ env.foot_flat_ref[side] > np.cos(np.deg2rad(15)))
    stable = (g[2] < -np.cos(np.deg2rad(20)) and flat
              and env._trunk_xpos[2] >= .9*env.stand_z
              and env.data.xpos[_jaw_bid(env), 2] >= .18
              and np.linalg.norm(env.data.qvel[:2]) <= .15
              and np.linalg.norm(env._gyro) <= 1.)
    _jt_advance(s, env.step_count, _trunk_yaw(env), contacts,
                min(_wc_sole_z(env, side) for side in ('left', 'right')),
                clean, stable)
    env._one_leg_hold_steps, env._one_leg_best_steps = (
        (0, 0) if s['rehearsal'] else (s['hold'], s['best']))


def _jt_commands(env):
    _jt_update(env)
    s = env._jt
    # Explicit relative-heading commander; not an unobservable yaw reward.
    # No ground correction after touchdown: the turn must happen in flight.
    env.twist_cmd[:] = 0.
    if s['stage'] in (1, 2):
        env.twist_cmd[2] = np.clip(6*(np.pi-s['total']), 0., 14.)
    env.body_cmd[4] = s['stage']/4.
    env.body_cmd[5] = min(env.step_count*C.CTRL_DT/6., 1.)


def _jt_turn(env):
    stage = env._jt['stage']
    if stage not in (1, 2):
        return 0.
    target = float(env.twist_cmd[2])
    if target < .1:
        return 0.
    baseline = np.exp(-target**2/25.)
    raw = np.exp(-(float(env._gyro[2])-target)**2/25.)
    return max(0., float(raw-baseline))/(1-baseline) * (1. if stage == 2 else .15)


def _jt_air(env):
    return _af_airborne(env)*_upright(env) if env._jt['stage'] == 2 else 0.


def _jt_ready(env):
    stage = env._jt['stage']
    # Same symmetric leg fold as the viewer's squat rig; a soft guide only.
    fold = .45*min(env.step_count*C.CTRL_DT/.5, 1.) if stage == 0 else (
        -.15 if stage == 1 else .25 if stage == 2 else 0.)
    target = C.DEFAULT_POSE.copy()
    target[[2,3,4,11,12,13]] += np.array([-1,-2,-1,1,2,1])*fold
    return float(np.exp(-np.mean(((env._joint_qpos()-target)/.6)**2)))


def _jt_land(env):
    # Phase 4 is exposed in obs[59]; invalid landings (phase 3) earn nothing.
    if env._jt['stage'] != 4 or not env._jt['clean']:
        return 0.
    return _both_feet_down(env)*_stand_tall(env)*_upright(env)*float(
        np.exp(-np.sum(env._gyro**2)))


_register(Behavior(
    id='jump_turn_180', emoji='🪽', title='起跳转身180°',
    description='双脚起跳，在空中转身180度，双脚落地后站稳1秒。',
    how_it_learns=('一半从站立探索起跳，一半从预置空中状态练习转身落地。'
                   '预置腾空不计成功，最终仅验收从站立自主完成。'
                   '真实BAM舵机、不加外力；地面转圈、翻滚和出生下落都不计成功。'),
    keywords=('起跳转身180度', '起跳转身180°', '起跳转身180', '跳转180',
              'jump turn 180', 'jump_turn_180'),
    terms=(RewardTerm('get_air', '双脚真正离地、保持身体直立', 5., _jt_air),
           RewardTerm('air_turn', '跟随可观测的转向指令，主要在空中转身', 4., _jt_turn),
           RewardTerm('jump_shape', '屈腿准备、伸腿起跳、收腿与落地姿态', .5, _jt_ready),
           RewardTerm('stay_upright', '抑制翻滚，保持竖直轴转身', .5, _upright),
           RewardTerm('land_stable', '有效空中转身后双脚站稳', 5., _jt_land),
           RewardTerm('smooth_moves', '减少不必要的动作抖动', .15,
                      _action_rate_pen, is_penalty=True),
           RewardTerm('save_energy', '限制舵机负担', .1, _torque_pen, is_penalty=True),
           RewardTerm('soft_landings', '减少落地冲击', .25,
                      _soft_landing_pen, is_penalty=True)),
    default_steps=8_000_000, episode_s=6., scene='all', symmetric=False,
    success_metric='正常起步，空中转身180°±15°，双脚稳定落地1秒；独立10次至少9次',
    state_fn=_jt_update, spawn_families=((.5, _jt_spawn_air_rehearsal),),
))

__all__ = [n for n in dir() if not n.startswith('__')]
