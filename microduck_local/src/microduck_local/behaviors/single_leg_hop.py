"""Ground-launched left-foot hops; no supplied flight or stronger actuators.

Joint/gravity shaping observes obs[3:34]; the crouch/extend clock is supplied
in obs[59:61]. Contact-history certification is metadata, never reward.
"""
from .core import *  # noqa: F401,F403
from .white_crane import (
    WHITE_CRANE_FOOT_TARGET, WHITE_CRANE_POSE, _wc_balance, _wc_head,
    _wc_init, _wc_lean, _wc_measure, _wc_rear_leg, _wc_sole_z,
)

HOP_START_S = 4.4
# Allow the required .5 s landing recovery plus loading/thrust/flight.
HOP_GAP_STEPS = round(1.2 / C.CTRL_DT)
# FK-calibrated whole-body poses: forward gaze, rear foot >6 cm clear,
# CoM >9 mm inside the support shoe. These are soft targets, not driven qpos.
HOP_READY_POSE = np.array([
    .513599,.373972,-.998883,-1.000170,.067676,
    -.553316,-.701249,-.075600,-.426332,
    .426332,.373972,-.900478,-.355287,-.370094,
])
# Calibrated at the SAME root attitude as READY, not at unrelated FK poses.
# Old load put CoM 38 mm outside the shoe and the free foot only 38 mm behind,
# contradicting the rear-leg gate. These targets are geometrically feasible,
# NOT evidence that a dynamic jump has been learned.
HOP_LOAD_POSE = HOP_READY_POSE.copy()
HOP_LOAD_POSE[[2, 3, 4]] = [-.72, -.25, .30]
HOP_THRUST_POSE = HOP_READY_POSE.copy()
HOP_THRUST_POSE[[2, 3, 4]] = [-.35, -.65, .05]
HOP_FLIGHT_POSE = HOP_READY_POSE.copy()
HOP_FLIGHT_POSE[[2, 3, 4]] = [-.20, -.75, -.05]
HOP_CROUCH = HOP_LOAD_POSE[[2, 3, 4]]
HOP_EXTEND = HOP_THRUST_POSE[[2, 3, 4]]
HOP_LANDING_STEPS = round(.5 / C.CTRL_DT)


def _slh_reset(env):
    env._slh = dict(step=-1, ready=0, locked=False, failed=False,
                    loaded=False, knee_peak=-10., shortest=1., released=-10_000,
                    airborne=False, air_start=0, takeoff_xy=None, forward=None,
                    peak=0., hops=0, last_hop=-10_000, chain_start=None,
                    landed=False, hold=0, best=0, rehearsal=False,
                    stage=0, stage_start=0, command_step=-1, prepared=0,
                    recovery=False, stable=0, recovery_best=0, bad_pose=0,
                    forward_m=0.0, best_forward_m=0.0)


def _slh_spawn_ground(env):
    """Grounded load rehearsal only; supplied posture never certifies success."""
    # Mixed with ordinary entries, not a stronger motor or a supplied jump.
    recovery = _spawn_knob(env, 'MICRODUCK_HOP_RECOVERY', '0') == '1'
    # 覆盖从单脚直立落地(0.0)到深蹲蓄势(0.80)全谱系，教会小鸭子自主深屈膝蓄势
    fraction = env._rng.uniform(0.0, 0.80)
    q = HOP_READY_POSE + fraction*(HOP_LOAD_POSE-HOP_READY_POSE)
    roll, pitch = -.477302, .178626
    if recovery:
        # Subtle tilt perturbations (about ±0.46°) to train active COM recovery
        roll += float(env._rng.uniform(-.008, .008))
        pitch += float(env._rng.uniform(-.008, .008))
    a, b = roll/2, pitch/2
    d, m = env.data, env.model
    d.qpos[env.joint_qpos_adr] = q
    d.qpos[3:7] = [np.cos(a)*np.cos(b), np.sin(a)*np.cos(b),
                  np.cos(a)*np.sin(b), -np.sin(a)*np.sin(b)]
    mujoco.mj_forward(m, d)
    # Tiny contact penetration seats the actual shoe, no height or velocity kick.
    d.qpos[2] -= _wc_sole_z(env, 'left') + .0002
    d.qvel[:] = 0.
    d.ctrl[:] = q
    mujoco.mj_forward(m, d)
    # The rotated collision AABB can sit BELOW the actual mesh. Seat against
    # real contacts, so a supposed ground drill never starts with a drop.
    for _ in range(150):
        if env._foot_contacts()['left']:
            break
        d.qpos[2] -= .0001
        mujoco.mj_forward(m, d)
    if env._foot_contacts() != {'left': True, 'right': False} or not _slh_clean(env):
        raise RuntimeError('Single-leg load rehearsal must start on the left shoe only')
    env.bam.reset(q)
    env.prev_joint_vel[:] = 0.
    env.last_action[:] = q-C.DEFAULT_POSE
    env.prev_action[:] = env.last_action
    env.step_count = round(HOP_START_S/C.CTRL_DT)
    _slh_reset(env)
    env._slh.update(rehearsal=True, recovery=recovery, locked=True, ready=5,
                    loaded=True, knee_peak=float(q[3]),
                    stage=4 if recovery else 1, stage_start=env.step_count)
    return env._get_obs()


def _slh_command_advance(s, step, *, left, right, knee, head, clearance, valid_landing,
                         knee_speed: float = 0.0):
    """Contact/pose-driven commander, explicitly exposed in obs[59:61]."""
    if s['command_step'] == step:
        return
    s['command_step'] = step
    age = (step-s['stage_start'])*C.CTRL_DT
    stage = s['stage']
    next_stage = stage
    if stage == 0:
        if step*C.CTRL_DT >= HOP_START_S:
            next_stage = 1
    elif stage == 1:
        # 支持落地微屈蓄能后迅速起跳
        prepared = left and not right and knee >= -.65 and head > .20 and clearance >= .020
        s['prepared'] = s['prepared']+1 if prepared else 0
        spring_thrust = knee_speed < -0.7
        if spring_thrust or (s['prepared'] >= 2 and age >= .08) or age >= .30:
            next_stage = 2
        elif age >= 1.5:
            s['failed'] = True
    elif stage == 2:
        if not left and not right:
            next_stage = 3
        elif age >= .8:
            s['failed'] = True
    elif stage == 3:
        if left:
            next_stage = 4
        elif age >= .6:
            s['failed'] = True
    elif stage == 4:
        if not left and not right:
            next_stage = 3
        # 0.08~0.12秒落地吸震后顺势弹起下一跳
        landing_target = HOP_LANDING_STEPS if s['recovery'] else round(.08 / C.CTRL_DT)
        if (s['stable'] >= landing_target or age >= .12) and not s['recovery']:
            next_stage = 1
        elif age >= .6 and not s['recovery']:
            s['failed'] = True
    if next_stage != stage:
        s.update(stage=next_stage, stage_start=step, prepared=0)


def _slh_commands(env):
    _slh_update(env)
    s = env._slh
    c = env.foot_contact_state
    _slh_command_advance(s, env.step_count, left=c['left'], right=c['right'],
                         knee=float(env._joint_qpos()[3]),
                         knee_speed=float(env._joint_vel()[3]),
                         head=_slh_head(env),
                         clearance=env._wc_state['clearance'], valid_landing=s['landed'])
    if s['stage']:
        # Radius distinguishes commands from the original entry clock;
        # each stage has a distinct angle, with small within-stage age sweep.
        age = min((env.step_count-s['stage_start'])*C.CTRL_DT/2., 1.)
        angle = (s['stage']-1)*np.pi/2 + age*.3
        env.body_cmd[4:6] = .5*np.array([np.sin(angle), np.cos(angle)])


def _slh_head(env):
    """World-level gaze along body heading; neck remains free to balance."""
    R = env.data.xmat[env._wc_head].reshape(3, 3) @ env._wc_stand_head_rotation.T
    forward = env._trunk_xmat.reshape(3, 3)[:, 0].copy()
    forward[2] = 0.
    forward /= max(np.linalg.norm(forward), 1e-8)
    error = np.sum((R[:, 0]-forward)**2) + .5*np.sum((R[:, 2]-[0, 0, 1])**2)
    return float(np.exp(-error/.6**2))


def _slh_clean(env):
    feet = set(env.foot_geoms.values())
    return all((c.geom1 != env.floor_geom or c.geom2 in feet)
               and (c.geom2 != env.floor_geom or c.geom1 in feet)
               for c in env.data.contact)


def _slh_advance(s, step, *, left, right, lifted, good, knee, knee_speed, leg_height, vz,
                 clearance, xy, forward):
    """Certification only: measured compression → extension → flight → landing."""
    if s['step'] == step:
        return
    s['step'], s['landed'] = step, False
    s['ready'] = s['ready']+1 if left and not right and lifted else 0
    s['locked'] |= s['ready'] >= 5
    # Initial two-foot entry is allowed. After lifting, a right-foot contact
    s['failed'] |= s['locked'] and right
    if s['failed'] or not good:
        s.update(loaded=False, airborne=False,
                 chain_start=None, released=-10_000, knee_peak=-10., shortest=1.)
        return
    if step-s['last_hop'] > HOP_GAP_STEPS:
        s.update(chain_start=None)

    if left and not right:
        if s['airborne']:
            delta = xy-s['takeoff_xy']
            fwd = float(delta @ s['forward'])
            side = abs(float(delta @ np.array([-s['forward'][1], s['forward'][0]])))
            air_steps = step - s['air_start']
            valid = (air_steps >= 1 and side <= 0.15)
            # 用户铁律：单步跳跃距离至少要 1cm (0.010m) 以上才能认证和奖励！
            is_hop_over_1cm = (fwd >= 0.010)
            if valid and is_hop_over_1cm:
                s['forward_m'] = s.get('forward_m', 0.0) + fwd
                s['best_forward_m'] = max(s.get('best_forward_m', 0.0), s['forward_m'])
                s.update(hops=s['hops']+1, last_hop=step, landed=True, last_step_fwd=fwd)
                if s['chain_start'] is None:
                    s['chain_start'] = s['air_start']
                s['hold'] = step-s['chain_start']+1
                s['best'] = max(s['best'], s['hold'])
            elif valid:
                # 安全单脚着陆，但若不足1cm不计入有效跳跃与里程，允许安全存活继续起跳
                s.update(landed=True, last_step_fwd=fwd)
            s.update(airborne=False, loaded=False, knee_peak=-10., shortest=1.,
                     released=-10_000)
        if s['locked']:
            s.update(loaded=True, knee_peak=max(s['knee_peak'], knee),
                     shortest=min(s['shortest'], leg_height))

    if not left and not right:
        if not s['airborne'] and s['locked']:
            s.update(airborne=True, air_start=step, takeoff_xy=xy.copy(),
                     forward=forward.copy(), peak=0., loaded=False)
        if s['airborne']:
            s['peak'] = max(s['peak'], clearance)


def _slh_update(env):
    # The live lab reloads recipe functions before rebuilding old preview
    # ducks. Discard an old-format attempt rather than killing the frame loop.
    if 'recovery_best' not in env._slh or 'command_step' not in env._slh:
        _slh_reset(env)
    s = env._slh
    if s['step'] == env.step_count:
        return
    env.foot_contact_state = env._foot_contacts()
    env._wc_state = crane = _wc_measure(env)
    c = env.foot_contact_state
    good = (_slh_clean(env) and env._projected_gravity()[2] <= -np.cos(np.deg2rad(65))
            and _slh_head(env) > .15 and crane['clearance'] >= .015
            and crane['foot'][0] <= -.03)
    stable = (good and c['left'] and not c['right'] and crane['margin'] >= -.008
              and crane['flat'] <= np.deg2rad(15) and crane['speed'] <= .15
              and crane['gyro'] <= 2.2 and env._trunk_xpos[2] >= .9*env.stand_z)
    s['stable'] = (s['stable']+1 if stable and s['stage'] == 4
                   and env.step_count > s['stage_start'] else 0)
    if s['recovery']:
        s['recovery_best'] = max(s['recovery_best'], s['stable'])
    # Actual rear touchdown/body collision is terminal; short pose excursions
    # get shaping, but cannot turn into an alternate head-back locomotion mode.
    if s['stage']:
        bad_pose = (_slh_head(env) < .15 or crane['foot'][0] > -.03
                    or crane['clearance'] < .015)
        s['bad_pose'] = s['bad_pose']+1 if bad_pose else 0
        s['failed'] |= not _slh_clean(env) or s['bad_pose'] >= 5
    forward = env._trunk_xmat.reshape(3, 3)[:2, 0].copy()
    forward /= max(np.linalg.norm(forward), 1e-8)
    # A retracting leg can move the floating trunk down while the robot's
    # total mass still rises. Certification follows whole-robot momentum,
    # not one body's internal motion. This is telemetry, not a reward input.
    mujoco.mj_subtreeVel(env.model, env.data)
    _slh_advance(s, env.step_count, left=c['left'], right=c['right'],
                 lifted=crane['clearance'] >= .03, good=good,
                 knee=float(env._joint_qpos()[3]),
                 knee_speed=float(env._joint_vel()[3]),
                 leg_height=float(env._trunk_xpos[2])-_wc_sole_z(env, 'left'),
                 vz=float(env.data.subtree_linvel[env._wc_robot_root, 2]),
                 clearance=_wc_sole_z(env, 'left'),
                 xy=env.data.subtree_com[env._wc_robot_root, :2], forward=forward)
    env._one_leg_hold_steps, env._one_leg_best_steps = (
        (0, 0) if s['rehearsal'] else (s['hold'], s['best']))


def _slh_body(env):
    # Forward lean (~15 deg pitch) + upright roll for active forward hopping balance
    g = env._projected_gravity()
    pitch_lean = float(np.arcsin(np.clip(g[0], -1, 1)))
    pitch_error = (pitch_lean - np.deg2rad(15)) / np.deg2rad(10)
    roll = float(np.arctan2(-g[1], -g[2]))
    # 单腿支撑时维持左侧微倾（约-26度，与地面单腿静平衡一致），消除向右翻倒的错误引导
    roll_error = (roll - (-0.45)) / np.deg2rad(12)
    return float(np.exp(-(pitch_error**2 + roll_error**2)))


def _slh_gate(env):
    # Relative foot geometry + gravity + head joints: obs[3:20]. No world
    # height/contact reward. Physical contact is checked by certification.
    s = env._wc_state
    return (_slh_body(env) * _slh_rear_gate(env)
            * float(np.clip((s['foot'][2]-.02)/.04, 0, 1))
            * float(np.clip((_slh_head(env)-.15)/.35, 0, 1)))


def _slh_rear_gate(env):
    """Full hop credit only while the free foot stays clearly behind."""
    return float(np.clip((-float(env._wc_state['foot'][0])-.06)/.04, 0, 1))


def _slh_raised_leg(env):
    """The free (right) leg must stay held back and up as an active counterbalance."""
    foot = env._wc_state['foot']
    target = np.array([-.120, -.030, .095])
    d2 = float(np.sum(((foot - target) / [.045, .040, .035])**2))
    score = float(np.exp(-d2))
    s = env._slh
    if s.get('stage') == 1:
        age = (env.step_count - s['stage_start']) * C.CTRL_DT
        if age > 0.25:
            score *= float(np.exp(-(age - 0.25) / 0.15))
    # 刚性高度门控：右腿离地低于5cm直接线性衰减，守护第二跳及后续连跳右腿不擦地
    clearance = float(env._wc_state['clearance'])
    height_gate = float(np.clip(clearance / 0.05, 0.0, 1.0))
    return score * height_gate


def _slh_shape(env):
    if env._slh['stage'] == 0:
        return _wc_lean(env)*_wc_rear_leg(env)
    target = ({1: HOP_LOAD_POSE, 2: HOP_THRUST_POSE, 3: HOP_FLIGHT_POSE}
              .get(env._slh['stage'], HOP_READY_POSE))
    error = (env._joint_qpos()[[2, 3, 4]]-target[[2, 3, 4]])/.45
    # Broad and narrow kernels keep useful gradient outside the target pose.
    d2 = float(np.mean(error**2))
    return _slh_gate(env)*float(.5*np.exp(-d2)+.5*np.exp(-d2/4))


def _slh_task(env):
    """No independent salary for standing forever once the drill begins."""
    s = env._slh
    if s['stage'] == 0:
        return _slh_shape(env)
    right = env._wc_state['foot']
    d2 = float(np.sum(((right-WHITE_CRANE_FOOT_TARGET)/[.08,.07,.06])**2))
    rear = float(.4*np.exp(-d2/4)+.6*np.exp(-d2))
    base = rear*(.25+.75*_slh_head(env))
    return _slh_shape(env)*base


def _slh_thrust(env):
    if env._slh['stage'] != 2:
        return 0.
    # 纯相对运动学：用雅可比计算支撑腿撑长速度（完全可观测），避免世界坐标 vx 不可观测陷阱
    if not hasattr(env, '_slh_jac'):
        env._slh_jac = (np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv)))
    trunk, foot = env._slh_jac
    mujoco.mj_jacBody(env.model, env.data, trunk, None, env.trunk_body_id)
    mujoco.mj_jacGeom(env.model, env.data, foot, None, env.foot_geoms['left'])
    # extension_speed：躯干-支撑脚竖向相对速度，蹬伸爆发时为正
    extension_speed = float((trunk[2]-foot[2]) @ env.data.qvel)
    # 前向分量用躯干朝向投影，仍在局部坐标系内，策略可观测
    fwd_dir = env._trunk_xmat.reshape(3, 3)[:, 0]
    trunk_vel = trunk @ env.data.qvel
    fwd_speed = float(fwd_dir @ trunk_vel)
    up_score  = float(np.clip(extension_speed / .25, 0., 1.))
    fwd_score = float(np.clip(fwd_speed / .18, 0., 1.))
    thrust_power = .55 * up_score + .45 * fwd_score
    return _slh_gate(env) * _slh_rear_gate(env) * thrust_power


def _slh_air(env):
    # Observable stage/age (obs[59:61]), bounded by the flight-stage timeout.
    # Reward reaching with the support leg, not tucking it higher in a fall.
    return _slh_shape(env) if env._slh['stage'] == 3 else 0.


def _slh_landing(env):
    if env._slh['stage'] != 4:
        return 0.
    s = env._wc_state
    calm = float(np.exp(-np.sum(env._gyro**2)/.6**2
                        -np.mean(env._joint_vel()**2)/1.5**2))
    flat = float(np.exp(-(s['flat']/np.deg2rad(12))**2))
    # Steep continuous margin gradient around boundary (-8mm to +8mm)
    margin_norm = float(np.clip((s['margin'] + .010)/.018, 0., 1.))
    # Gate out aesthetic pose points when COM leaves the support footprint
    margin_gate = float(np.clip((s['margin'] + .004)/.012, 0., 1.))
    balance = .50*_wc_balance(env) + .50*margin_norm
    style = (.60*calm + .40*flat) * margin_gate
    # 用户要求：单步跳跃必须达到 1cm (0.010m) 以上才给予单步达成奖励
    step_fwd = env._slh.get('last_step_fwd', 0.0)
    step_1cm_bonus = float(np.clip((step_fwd - 0.010) / 0.020, 0.0, 1.0)) if step_fwd >= 0.010 else 0.0
    progress_bonus = float(np.clip(env._slh.get('forward_m', 0.0) / 10.0, 0.0, 1.0))
    return _slh_gate(env) * (.30*balance + .20*style + .30*step_1cm_bonus + .20*progress_bonus)


def _slh_forward(env):
    # 空中阶段奖励前向速度：使用躯干朝向投影（可观测），无硬截断避免梯度消失
    if env._slh['stage'] != 3:
        return 0.
    if not hasattr(env, '_slh_jac'):
        env._slh_jac = (np.zeros((3, env.model.nv)), np.zeros((3, env.model.nv)))
    trunk, _ = env._slh_jac
    mujoco.mj_jacBody(env.model, env.data, trunk, None, env.trunk_body_id)
    fwd_dir  = env._trunk_xmat.reshape(3, 3)[:, 0]
    fwd_speed = float(fwd_dir @ (trunk @ env.data.qvel))
    # 连续渐进：从 0 开始线性给分，0.25 m/s 时满分，无硬截断
    score = float(np.clip(fwd_speed / 0.25, 0., 1.))
    return _slh_shape(env) * score


def _slh_takeoff(env):
    """Stage 1 蓄势→起跳过渡奖励：奖励快速蹬伸膝关节速度，打破站立局部最优。"""
    if env._slh['stage'] != 1:
        return 0.
    # 膝关节速度：蹬伸（向伸展方向，负速度）时为正奖励
    knee_speed = float(env._joint_vel()[3])
    # 向蹬伸方向（负方向）运动时给奖励；越快越高分
    extend_score = float(np.clip(-knee_speed / 1.5, 0., 1.))
    return _slh_gate(env) * extend_score


def _slh_gaze(env):
    p = float(np.clip((env.step_count*C.CTRL_DT-3.8)/.6, 0, 1))
    return (1-p)*_wc_head(env)+p*_slh_head(env)


def _slh_straight_pen(env):
    return -float(min((float(env._gyro[2])/.8)**2, 1.))


_register(Behavior(
    id='single_leg_hop', emoji='🪶', title='单脚向前连续跳 · 累计10米',
    description='右脚全程悬空，左腿缓慢屈曲、快速蹬伸，小步连续向前跳跃，累计前进10米。',
    how_it_learns=('先练稳定单脚姿态与前倾平衡，再练深蹲蓄力爆发起跳与小步向前跃进；'
                   '每次落地恢复后立即接下一跳，不断单脚小步跳跃，直到累计向前达到10米。'),
    keywords=('单脚跳跃前进', '单脚向前跳', '单脚连续跳', 'single leg hop'),
    symmetric=False, episode_s=90., default_steps=10_000_000, scene='all',
    success_metric='右脚不落地；单脚连续向前跳跃，累计前进达到10米',
    state_fn=_slh_update, spawn_families=((.7, _slh_spawn_ground),),
    terms=(
        RewardTerm('hop_shape',   '屈腿、蹬伸、空中伸脚；后腿悬空且头向前', 1., _slh_task),
        # hop_thrust 权重提高：8.0（原6.0），强化起跳蹬伸探索
        RewardTerm('hop_thrust',  '蹬地阶段髋膝踝协同撑长支撑腿爆发起跳', 8.0, _slh_thrust),
        RewardTerm('hop_air',     '空中准备左脚接地，维持前跃姿态', 3.0, _slh_air),
        RewardTerm('hop_forward', '空中向前大步跃进，驱动连续向前跳跃10米', 10.0, _slh_forward),
        RewardTerm('hop_landing', '落地后恢复单脚平衡，头向前且后腿悬空', 6.0, _slh_landing),
        # hop_takeoff：Stage1 蓄势阶段奖励快速蹬伸动作，打破"站原地"局部最优
        RewardTerm('hop_takeoff', '蓄势阶段快速蹬伸膝关节，激励策略主动起跳', 5.0, _slh_takeoff),
        RewardTerm('one_leg_balance', '准备阶段调节整机重心', 1.,
                   lambda e: _slh_task(e)*_wc_balance(e) if e._slh['stage'] in (0, 1) else 0.),
        # raised_leg 权重大幅降低：1.0（原4.0），防止"站原地抬腿"成为主导策略
        RewardTerm('raised_leg',  '全过程保持右腿向后上方高高抬起平衡', 1.0, _slh_raised_leg),
        RewardTerm('head_forward', '头颈自由配重，视线转向前方', 1.5, _slh_gaze),
        RewardTerm('stay_straight', '减少转圈；侧移由独立验收限制', .5, _slh_straight_pen, True),
        RewardTerm('smooth_moves', '减少多余抖动，允许蹬伸', .15, _action_rate_pen, True),
        RewardTerm('save_energy', '限制舵机负担', .1, _torque_pen, True),
        RewardTerm('soft_landings', '减少落地冲击', .25, _soft_landing_pen, True),
    ), clip_name='white_crane_slow',
    curriculum=(
        CurriculumStage('先练左脚着地后的恢复', 1_500_000,
                        {'MICRODUCK_SPAWN_FAMILY_PROBS': '1.0',
                         'MICRODUCK_HOP_RECOVERY': '1',
                         'MICRODUCK_HOP_GOAL_HOPS': '0'},
                        detail='左脚已着地、轻微屈膝开始；稳定恢复0.5秒，不算完成跳跃。'),
        CurriculumStage('练习一次真实离地', 1_500_000,
                        {'MICRODUCK_SPAWN_FAMILY_PROBS': '1.0',
                         'MICRODUCK_HOP_RECOVERY': '0',
                         'MICRODUCK_HOP_GOAL_HOPS': '1',
                         'MICRODUCK_EPISODE_S': '6'},
                        detail='从左脚着地的准备姿态开始，学习深屈支撑腿、快速蹬伸并落回左脚。'),
        CurriculumStage('把落地接到下一跳', 2_500_000,
                        {'MICRODUCK_SPAWN_FAMILY_PROBS': '0.80',
                         'MICRODUCK_HOP_RECOVERY': '0',
                         'MICRODUCK_HOP_GOAL_HOPS': '2',
                         'MICRODUCK_EPISODE_S': '10'},
                        detail='落地后重新屈腿并再次起跳；右脚触地或身体倒地立即重来。'),
        CurriculumStage('连续向前跳满三秒', 4_000_000,
                        {'MICRODUCK_SPAWN_FAMILY_PROBS': '0.60',
                         'MICRODUCK_HOP_RECOVERY': '0',
                         'MICRODUCK_HOP_GOAL_HOPS': '0'},
                        detail='加入普通双脚起步，串联白鹤亮翅、连续单脚向前跳和稳定落地。'),
    ),
))

__all__ = [n for n in dir() if not n.startswith('__')]
