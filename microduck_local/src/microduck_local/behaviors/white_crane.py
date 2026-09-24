"""白鹤亮翅: rear-leg lift with head/neck counterbalance, honest BAM only.

Geometry rewards are relative to the stance foot/heading, never world yaw or
home position. Joint positions (obs 6:20) + gravity (3:6) determine these
kinematics, including the head's contribution to the whole-robot CoM.
"""
from .core import *  # noqa: F401,F403

WHITE_CRANE_POSE = np.array([
    .5235888, .3839624, -1.3237144, -1.0093294, .1834719,
    -.6144055, -.7824481, .0386336, -.2372928,
    .4363223, .3839624, -.7500418, .0268340, -.6177363,
])
WHITE_CRANE_ROLL = -.356013448
WHITE_CRANE_PITCH = .346264735
WHITE_CRANE_FOOT_TARGET = np.array([-.120, -.030, .100])
WHITE_CRANE_PREP_S = .6
# Takeoff is measured after 8 mm clearance, and arrival before 100% extension.
# 2.4 s total leaves <2 s between those gates even on the exact reference.
WHITE_CRANE_RAISE_S = 3.2


def _wc_phase(env):
    # The existing clip phase in obs[59:61] makes this clock observable.
    return float(np.clip((env.step_count*C.CTRL_DT-WHITE_CRANE_PREP_S)
                         / WHITE_CRANE_RAISE_S, 0, 1))


def _wc_reset_entry(env):
    env._wc_lift_start = None
    env._wc_lift_finished = False
    env._wc_lift_peak = 0.0
    env._wc_entry_ok = False


def _wc_place(env, fraction=1.0, noise=0.0):
    """Reset-state curriculum only, never called while stepping/evaluating."""
    d, m = env.data, env.model
    mujoco.mj_resetDataKeyframe(m, d, env.key_stand)
    q = C.DEFAULT_POSE + fraction * (WHITE_CRANE_POSE - C.DEFAULT_POSE)
    q = q + env._rng.uniform(-noise, noise, C.NUM_JOINTS)
    limits = m.jnt_range[env.bam.joint_ids]
    q = np.clip(q, limits[:, 0], limits[:, 1])
    d.qpos[env.joint_qpos_adr] = q
    a, b = fraction * np.array([WHITE_CRANE_ROLL, WHITE_CRANE_PITCH]) / 2
    d.qpos[3:7] = [np.cos(a)*np.cos(b), np.sin(a)*np.cos(b),
                  np.cos(a)*np.sin(b), -np.sin(a)*np.sin(b)]
    mujoco.mj_forward(m, d)
    d.qpos[2] -= min(_wc_sole_z(env, side) for side in ('left', 'right'))
    d.qvel[:] = 0
    d.ctrl[:] = q
    mujoco.mj_forward(m, d)
    env.bam.reset(q)
    env.prev_joint_vel = env._joint_vel().copy()
    env.step_count = round((WHITE_CRANE_PREP_S + fraction*WHITE_CRANE_RAISE_S)/C.CTRL_DT)
    return env._get_obs()


def _wc_spawn(env):
    # Shift rehearsal distribution, not rewards or servo strength. Independent
    # certification ALWAYS bypasses this and starts on both feet.
    progress = min(env._lifetime_steps / 250_000, 1.0)
    if env._rng.random() < .50 + .35 * progress:
        return env._get_obs()
    # Linear joint interpolation lifted the foot BEFORE shifting the CoM:
    # 25/50/75% spawns started 21/14/7 mm outside the support rectangle.
    # Rehearse the balanced endpoint; actual entries still start on both feet.
    return _wc_place(env, noise=.006)


def _wc_sole_z(env, side):
    g = env.foot_geoms[side]
    bb = env.model.geom_aabb[g]
    row = env.data.geom_xmat[g][6:9]
    # Conservative collision-AABB bottom, not the geometric center height.
    return float(env.data.geom_xpos[g, 2] + row @ bb[:3] - np.abs(row) @ bb[3:])


def _wc_init(env):
    env._wc_head = env.model.body('jaw_soft').id
    env._wc_robot_root = int(env.model.jnt_bodyid[0])
    ref = env.foot_flat_ref['left']
    env._wc_vertical = int(np.argmax(np.abs(ref)))
    env._wc_plane = [i for i in range(3) if i != env._wc_vertical]
    # Read reference head attitude from FK without changing the env's state.
    d = mujoco.MjData(env.model)
    mujoco.mj_resetDataKeyframe(env.model, d, env.key_stand)
    mujoco.mj_forward(env.model, d)
    env._wc_stand_head_rotation = d.xmat[env._wc_head].reshape(3, 3).copy()
    env._wc_stand_foot = (d.geom_xpos[env.foot_geoms['right']]
                          - d.geom_xpos[env.foot_geoms['left']]).copy()
    env._wc_velocity = np.empty((2,6))
    d.qpos[env.joint_qpos_adr] = WHITE_CRANE_POSE
    a,b = WHITE_CRANE_ROLL/2, WHITE_CRANE_PITCH/2
    d.qpos[3:7] = [np.cos(a)*np.cos(b),np.sin(a)*np.cos(b),
                  np.cos(a)*np.sin(b),-np.sin(a)*np.sin(b)]
    mujoco.mj_forward(env.model, d)
    env._wc_head_gravity = -d.xmat[env._wc_head][6:9].copy()


def _wc_measure(env):
    d, m = env.data, env.model
    l, r = (d.geom_xpos[env.foot_geoms[s]] for s in ('left', 'right'))
    forward = env._trunk_xmat.reshape(3,3)[:,0].copy()
    forward[2] = 0
    forward /= max(np.linalg.norm(forward), 1e-8)
    lateral = np.array([-forward[1],forward[0],0.])
    delta = r-l
    foot = np.array([delta@forward, delta@lateral, delta[2]])
    com = d.subtree_com[env._wc_robot_root].copy()
    com[2] = l[2]  # project along gravity into the horizontal support plane
    g = env.foot_geoms['left']
    R = d.geom_xmat[g].reshape(3,3)
    local = R.T @ (com-l) - m.geom_aabb[g,:3]
    plane = env._wc_plane
    # ponytail: conservative inner rectangle of the shoe AABB, not a force-
    # based support polygon; replace with contact-hull margins for rough terrain.
    half = m.geom_aabb[g,3:][plane] * .75
    error = local[plane]
    margin = float(np.min(half-np.abs(error)))
    flat = float(np.arccos(np.clip(np.dot(-R[2],env.foot_flat_ref['left']),-1,1)))
    headg = -d.xmat[env._wc_head][6:9]
    head_error = float(np.arccos(np.clip(headg@env._wc_head_gravity,-1,1)))
    speed = float(np.hypot(*_base_vel(env)[:2]))
    gyro = float(np.linalg.norm(env._gyro))
    for i, side in enumerate(('left','right')):
        mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_GEOM,
                                 env.foot_geoms[side],env._wc_velocity[i],0)
    # Relative velocity cancels global translation; observable from joints/IMU.
    foot_speed = float(np.linalg.norm(env._wc_velocity[1,3:]-env._wc_velocity[0,3:]))
    lean = float(np.arcsin(np.clip(env._projected_gravity()[0],-1,1)))
    return dict(foot=foot, com_error=error, margin=margin, flat=flat,
                clearance=_wc_sole_z(env,'right'), speed=speed, gyro=gyro,
                head_error=head_error, head_z=float(d.xpos[env._wc_head,2]),
                lean=lean, foot_speed=foot_speed)


def _wc_slow_entry(env, s, right_contact):
    """Judge actual takeoff-to-pose motion, not waiting before a fast kick."""
    if right_contact or s['clearance'] < .008:
        _wc_reset_entry(env)
    else:
        if env._wc_lift_start is None:
            env._wc_lift_start = env.step_count
        if not env._wc_lift_finished:
            env._wc_lift_peak = max(env._wc_lift_peak,s['foot_speed'])
            if (s['foot'][0] <= -.10 and s['foot'][2] >= .08
                    and s['lean'] >= np.deg2rad(15)):
                env._wc_lift_finished = True
                duration = (env.step_count-env._wc_lift_start+1)*C.CTRL_DT
                env._wc_entry_ok = duration >= 2.0 and env._wc_lift_peak <= .18
    return env._wc_entry_ok


def _wc_update(env):
    s = env._wc_state = _wc_measure(env)
    c = env.foot_contact_state
    entry_ok = _wc_slow_entry(env,s,c['right'])
    valid = (entry_ok and c['left'] and not c['right'] and s['clearance'] >= .05
             and s['foot'][0] <= -.10 and s['foot'][2] >= .08
             and np.deg2rad(15) <= s['lean'] <= np.deg2rad(30)
             and s['margin'] >= 0 and s['flat'] <= np.deg2rad(15)
             and env._projected_gravity()[2] <= -np.cos(np.deg2rad(35))
             and env._trunk_xpos[2] >= .9*env.stand_z
             and s['head_z'] >= .20 and s['head_error'] <= np.deg2rad(30)
             and s['speed'] <= .08 and s['gyro'] <= .8)
    env._one_leg_hold_steps = env._one_leg_hold_steps+1 if valid else 0
    env._one_leg_best_steps = max(env._one_leg_best_steps,env._one_leg_hold_steps)


def _wc_balance(env):
    return float(np.exp(-np.sum((env._wc_state['com_error']/.025)**2)))


def _wc_rear_leg(env):
    p = _wc_phase(env)
    target = (1-p)*env._wc_stand_foot + p*WHITE_CRANE_FOOT_TARGET
    return float(np.exp(-np.sum(((env._wc_state['foot']-target)
                                 / np.array([.05,.04,.035]))**2)))


def _wc_lean(env):
    p = _wc_phase(env)
    # Score pitch and roll separately: a total-tilt penalty also paid for
    # reducing the desired forward lean (380723). Gravity is in obs[3:6].
    g = env._projected_gravity()
    roll = np.arctan2(-g[1],-g[2])
    return float(np.exp(-((env._wc_state['lean']-WHITE_CRANE_PITCH*p)/np.deg2rad(10))**2
                        -((roll-WHITE_CRANE_ROLL*p)/np.deg2rad(12))**2))


def _wc_slow_pen(env):
    excess = max(env._wc_state['foot_speed']-.08,0)
    return -float(1-np.exp(-(excess/.10)**2))


def _wc_head(env):
    s = env._wc_state
    return float(np.exp(-(s['head_error']/.6)**2)
                 * np.clip((s['head_z']-.10)/.12,0,1))


def _wc_hold(env):
    s=env._wc_state
    # No contact-only jackpot: the whole shape, head balance, clearance and
    # stillness must improve together. Timer is metadata, not hidden reward state.
    return (float(env.foot_contact_state['left'] and not env.foot_contact_state['right'])
            * _wc_rear_leg(env) * _wc_balance(env) * _wc_head(env) * _wc_lean(env)
            * float(np.clip(s['clearance']/.05,0,1))
            * float(np.exp(-(s['speed']/.15)**2-(s['gyro']/1.5)**2)))


_register(Behavior(
    id='white_crane', emoji='🪽', title='白鹤亮翅 · 慢抬舒展 · 6 秒',
    description='身体前倾约20度，右脚向后舒展约12厘米，缓慢抬腿后完整姿态连续保持6秒。',
    how_it_learns='用真实BAM舵机练习。先从接近目标的姿势练平衡，逐渐增加双脚起步；'
                  '时间轴指导约3.2秒慢抬腿；头颈参与整机配重，不锁死角度。'
                  '独立验收从双脚起步，拒绝快踢后等待；姿态到位后才计6秒。',
    keywords=('白鹤亮翅','white crane','baihe'), symmetric=False,
    terms=(
        RewardTerm('crane_hold','完整后抬腿姿态、头颈配重及稳定支撑共同得分',10.,_wc_hold),
        RewardTerm('crane_rear_leg','右脚向后抬高，接近髋部高度；不是脚尖短暂离地',3.,_wc_rear_leg),
        RewardTerm('crane_balance','头颈、躯干和腿协同，将整机重心移到支撑脚内',2.,_wc_balance),
        RewardTerm('crane_head','保持抬头造型，允许头颈调节配重',.5,_wc_head),
        RewardTerm('flat_stance_foot','支撑脚掌平放',.5,_stance_flat('left')),
        RewardTerm('crane_lean','身体逐渐前倾约20度，同时避免过度侧倾',3.,_wc_lean),
        RewardTerm('slow_raise','惩罚后脚快速甩动，按时间轴缓慢抬起',3.,_wc_slow_pen,True),
        RewardTerm('smooth_moves','减少突变，保留学习抬腿所需的动作空间',.25,_action_rate_pen,True),
        RewardTerm('gentle_joints','减少关节甩动',.25,_joint_vel_pen,True),
        RewardTerm('save_energy','控制舵机负荷',.15,_torque_pen,True),
    ), state_fn=_wc_update, spawn_families=((1.0,_wc_spawn),),
    episode_s=20., default_steps=8_000_000, clip_name='white_crane_slow',
    success_metric='双脚起步，实际抬腿不少于2秒且不甩腿，前倾15–30度、后伸至少10厘米，完整姿态连续300控制步（6秒）',
))

__all__ = [n for n in dir() if not n.startswith('__')]
