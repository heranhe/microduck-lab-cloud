"""双脚连续稳定跳跃（Continuous Bunny Hop）行为模块 V25（事件驱动与自适应平衡版）：
用户铁律：
1. 必须双脚同时起跳、双脚同时落地！
2. 单次跳跃 3 到 5 cm，落地极其稳定！
3. 连续跳跃 5 次不失去平衡！

物理架构：废除固定死板时钟，采用【接触与平衡事件驱动状态机】：
Stage 0 (平稳蓄势): 双足同时踩实地面，躯干直立平静，角速度平复归零，确认稳立后触发弹跳；
Stage 1 (双足起跳): 双腿严格毫秒级同时轻弹离地，赋予向上 vz~0.35m/s、向前 vx~0.35m/s；
Stage 2 (空中并排): 腾空 1cm 飞跃 3~5cm，双脚完全并排齐平，无高低差、无前后剪刀差；
Stage 3 (缓冲稳立): 双足严格同时触地，屈膝吸收冲击，将水平与垂向动能急刹归零，稳固立定！
认证成功判定：连续跳满 5 次且不失衡。
"""
from __future__ import annotations

import numpy as np

from .core import *  # noqa: F401,F403


# 轻巧微下潜蓄力姿态 (Stage 0)
HOP_CROUCH_POSE = np.array([
    0.0, -0.087, -0.55, 0.30, -0.10,   # left leg: 轻度微屈膝 (knee=+0.30)
    -0.10, -0.05, 0.0, 0.0,            # neck/head (微含胸)
    0.0, 0.087, 0.55, -0.30, 0.10,     # right leg: 对称微屈膝 (knee=-0.30)
], dtype=np.float32)

READY_STEPS = 3
LANDING_STABLE_STEPS = 10   # 0.20 s at 50 Hz
LANDING_TIMEOUT_STEPS = 25  # 0.50 s to recover before the landing is failed


def _lj_knob(env, key, default):
    try:
        return type(default)(_spawn_knob(env, key, str(default)))
    except (TypeError, ValueError):
        return default


def _lj_commands(env):
    """构建 61 维观测前调用：注入基于事件驱动的自适应状态信号。
    obs[48]: twist_cmd[0] forward 前向期望速度 (起跳 0.35m/s, 落地 0.0m/s)
    obs[55]: body_cmd[0] 当前 Stage 编码 (0.0: 蓄势, 0.33: 起跳, 0.66: 滞空, 1.0: 缓冲) (纯对称通道)
    obs[57]: body_cmd[2] 连续跳跃达成进度 min(hop_count / 5.0, 1.0) (纯对称通道)
    """
    _lj_update(env)
    s = env._lj
    env.twist_cmd[:] = 0.0

    stage = s.get("stage", 0)
    # Stage 编码注入
    stage_map = {0: 0.0, 1: 0.33, 2: 0.66, 3: 1.0}
    env.body_cmd[0] = float(stage_map.get(stage, 0.0))
    # 连跳进度注入
    env.body_cmd[2] = float(min(s.get("hop_count", 0) / 5.0, 1.0))

def _lj_reset(env):
    min_dist = max(0.0, _lj_knob(env, "MICRODUCK_LJ_MIN_DIST_M", 0.03))
    max_dist = max(min_dist, _lj_knob(env, "MICRODUCK_LJ_MAX_DIST_M", 0.05))
    env._lj = {
        "step": -1,
        "stage": 0,              # 0: 蓄势, 1: 起跳, 2: 腾空, 3: 落地缓冲
        "stage_steps": 0,        # 当前 stage 持续步数
        "hop_count": 0,          # 成功完成合规双足跳跃次数 (目标 >= 5)
        "air_steps": 0,          # 本次腾空步数
        "takeoff_x": None,       # 本跳起跳点 x
        "last_hop_dist": 0.0,    # 上一次单跳向前距离 (m)
        "pending_hop_dist": None,# 触地后待稳定认证的跳距
        "landed_fresh": False,   # 这一帧是否刚完成合规落地
        "stable_steps": 0,       # 双脚踩实且直立平静的连续步数
        "bad_fall": False,       # 是否跌倒或穿模
        "total_dist": 0.0,       # 累计有效位移 (m)
        "staggered_liftoff": False, # 是否出现单脚离地
        "staggered_landing": False, # 是否出现单脚触地
        "single_down_steps": 0,  # 连续单脚踩地步数
        "is_airborne": False,    # 是否真实处于空中腾空
        "land_absorb_steps": 0,  # 着地吸震步数计数
        "took_off_fresh": False, # 这一帧是否完成了真实双足离地
        "min_dist": min_dist,
        "max_dist": max_dist,
        "landing_stable_steps": max(1, _lj_knob(env, "MICRODUCK_LJ_STABLE_STEPS", LANDING_STABLE_STEPS)),
        "goal_hops": max(1, _lj_knob(env, "MICRODUCK_LJ_GOAL_HOPS", 5)),
    }


def _lj_spawn_landing(env):
    """从可落地的双足下蹲状态下落；出生姿态本身不计一次跳跃。"""
    d, m = env.data, env.model
    q = HOP_CROUCH_POSE.copy()
    d.qpos[env.joint_qpos_adr] = q
    d.qpos[2] += 0.04
    d.qvel[:] = 0.0
    d.ctrl[:] = q
    mujoco.mj_forward(m, d)
    for _ in range(400):
        if env._foot_contacts() == {"left": True, "right": True}:
            break
        d.qpos[2] -= 0.0002
        mujoco.mj_forward(m, d)
    else:
        raise RuntimeError("Long-jump landing spawn must reach both feet")
    d.qpos[2] += float(env._rng.uniform(0.004, 0.008))
    d.qvel[0] = float(env._rng.uniform(0.08, 0.18))
    d.qvel[2] = -float(env._rng.uniform(0.10, 0.22))
    d.qvel[4] = float(env._rng.uniform(-0.25, 0.25))
    mujoco.mj_forward(m, d)
    if env.bam is not None:
        env.bam.reset(q)
    env.prev_joint_vel[:] = 0.0
    env.last_action[:] = q - C.DEFAULT_POSE
    env.prev_action[:] = env.last_action
    env.step_count = 1
    _lj_reset(env)
    env._lj.update(stage=3, step=0, stage_steps=0)
    return env._get_obs()


def _lj_break_streak(s):
    s["hop_count"] = 0
    s["total_dist"] = 0.0
    s["last_hop_dist"] = 0.0
    s["pending_hop_dist"] = None


def _check_clean_airborne(env) -> bool:
    """物理检测：除双足外无任何部位触地"""
    feet = set(env.foot_geoms.values())
    floor = env.floor_geom
    for i in range(env.data.ncon):
        c = env.data.contact[i]
        if c.geom1 == floor and c.geom2 not in feet:
            return False
        if c.geom2 == floor and c.geom1 not in feet:
            return False
    return True


def _lj_update(env):
    if not hasattr(env, "_lj") or env.step_count == 0:
        _lj_reset(env)
    s = env._lj
    step = env.step_count
    if s["step"] == step:
        return
    s["step"] = step
    s["landed_fresh"] = False
    s["staggered_landing"] = False
    s["took_off_fresh"] = False
    s["stage_steps"] += 1

    contacts = env._foot_contacts()
    both_down = bool(contacts["left"] and contacts["right"])
    any_down = bool(contacts["left"] or contacts["right"])
    single_down = bool(contacts["left"] ^ contacts["right"])

    trunk_pos = env._trunk_xpos
    current_x = float(trunk_pos[0])
    current_z = float(trunk_pos[2])

    g = env._projected_gravity()
    # 严格直立端正性：前后俯仰 |g[0]| < 0.28 (约 16.2 度容差)，左右横滚 |g[1]| < 0.22，垂向 g[2] < -0.70
    is_upright = bool(g[2] < -0.70 and abs(g[0]) < 0.28 and abs(g[1]) < 0.22)

    # 物理双足真实腾空检测
    zl = float(env.data.geom_xpos[env.foot_geoms["left"]][2])
    zr = float(env.data.geom_xpos[env.foot_geoms["right"]][2])
    clearance = min(zl, zr)

    is_airborne = bool(
        (not any_down)
        and clearance > 0.003
        and current_z >= 0.075
        and _check_clean_airborne(env)
        and not s["bad_fall"]
    )
    s["is_airborne"] = is_airborne

    if is_airborne:
        s["land_absorb_steps"] = 0
    elif any_down:
        s["land_absorb_steps"] = s.get("land_absorb_steps", 0) + 1

    # 区分空中、落地吸震缓冲与地面稳立三态（严禁在空中与落地吸震瞬态误杀）：
    stage = s["stage"]
    is_landing_buffer = bool(s.get("land_absorb_steps", 0) <= 5 or (stage == 3 and s.get("stage_steps", 0) <= 5))

    if stage in (1, 2) or (is_airborne or (not any_down and current_z >= 0.070)):
        # 空中飞跃期：向前跳跃躯干具有向前动量，前倾 g[0] 在 0.15~0.60 属于正常飞行抛物线姿态！
        # 真正空中翻车是：倒立(g[2] > -0.25)、后空翻(g[0] < -0.48)、前空翻过度失控(g[0] > 0.75)、侧滚翻(|g[1]| > 0.35)
        severe_tilt = bool(g[2] > -0.25 or g[0] < -0.48 or g[0] > 0.75 or abs(g[1]) > 0.35)
    elif is_landing_buffer:
        # 落地刚踩地吸震缓冲期（前 5 步 / 0.10s）：
        # 迎地刹车由于地面反作用力冲量，躯干允许动态后倾吸能（放宽至 g[0] < -0.48，约 28.7 度）
        severe_tilt = bool(g[2] > -0.45 or g[0] < -0.48 or (g[0] > 0.60 and current_z < 0.060) or abs(g[1]) > 0.32)
    else:
        # 地面蓄势与落地稳立期：
        # 严禁向后仰翻倒(g[0] < -0.36)、严重横滚侧翻(|g[1]| > 0.28)、前倾倒伏贴地(g[0] > 0.52 且 z < 0.060)
        severe_tilt = bool(g[2] > -0.50 or g[0] < -0.36 or (g[0] > 0.52 and current_z < 0.060) or abs(g[1]) > 0.28)

    body_collapsed = bool(current_z < 0.052)
    if body_collapsed or severe_tilt:
        s["bad_fall"] = True
        _lj_break_streak(s)

    # 监测单脚起跳：仅在地面起跳转换期 (Stage 0, 1, 3) 检测
    # 若一脚已明显离地 (z > 1.2cm) 而另一脚仍在踩地，判定为异步单脚起跳！
    if stage in (0, 1, 3) and single_down and (zl > 0.012 or zr > 0.012):
        s["staggered_liftoff"] = True
        _lj_break_streak(s)

    # 铁律硬拦截：严禁单腿跨步走步（若连续 5 步即 100ms 仅单脚踩地支撑，视为非跳跃走步违规一票否决！）
    if single_down and not is_airborne:
        s["single_down_steps"] = s.get("single_down_steps", 0) + 1
        if s["single_down_steps"] >= 5:
            s["bad_fall"] = True
    else:
        s["single_down_steps"] = 0

    v_fwd, _, _ = env.heading_lin_vel()
    vz = float(env.data.qvel[2])
    gyro_calm = bool(abs(env._gyro[1]) < 1.5 and abs(env._gyro[0]) < 1.5)
    motion_calm = bool(abs(v_fwd) < 0.12 and abs(vz) < 0.15)

    # ------------------ 事件驱动状态机流转 ------------------
    next_stage = stage

    if stage == 0:
        # Stage 0: 蓄势平稳与二次起跳衔接
        if both_down:
            s["ground_x"] = current_x
        ready = both_down and is_upright and gyro_calm and motion_calm
        s["stable_steps"] = s["stable_steps"] + 1 if ready else 0

        # 重置后的自然下落不算起跳；先双脚站稳，才能进入已武装的起跳阶段。
        if s["stable_steps"] >= READY_STEPS and not s["bad_fall"]:
            next_stage = 1
            s["stable_steps"] = 0

    elif stage == 1:
        # Stage 1: 双足轻弹起跳。
        if both_down or any_down:
            s["ground_x"] = current_x

        if is_airborne:
            # 双脚完全腾空离地，记录起跳基准位置！
            s["takeoff_x"] = s.get("ground_x", current_x)
            s["air_steps"] = 1
            s["took_off_fresh"] = True
            next_stage = 2
        elif s["stage_steps"] >= 8:
            # 超时未弹起，退回蓄势重新调整
            next_stage = 0

    elif stage == 2:
        # Stage 2: 空中飞跃（滞空约 0.08~0.16s）。
        if is_airborne:
            s["air_steps"] += 1
            s["touch_steps"] = 0
        elif any_down:
            # 接触地面：允许 1 步 (0.02s) 内双足屈膝吸震踩实
            s["touch_steps"] = s.get("touch_steps", 0) + 1
            if both_down:
                # 触地只登记候选；连续稳定 0.20 s 后才认证，避免碰地即领奖。
                valid = (
                    s["takeoff_x"] is not None
                    and s["air_steps"] >= 2
                    and s["touch_steps"] == 1
                    and not s["staggered_liftoff"]
                    and not s["bad_fall"]
                    and is_upright
                )
                hop_dist = current_x - s["takeoff_x"] if s["takeoff_x"] is not None else 0.0
                if valid and s["min_dist"] <= hop_dist <= s["max_dist"]:
                    s["pending_hop_dist"] = hop_dist
                else:
                    _lj_break_streak(s)

                s["takeoff_x"] = None
                s["air_steps"] = 0
                s["touch_steps"] = 0
                s["staggered_liftoff"] = False
                s["ground_x"] = current_x
                next_stage = 3
            elif s["touch_steps"] >= 3:
                # 超过 3 步(60ms)依然只有单足触地，坚决一票否决！
                s["staggered_landing"] = True
                _lj_break_streak(s)
                s["takeoff_x"] = None
                s["air_steps"] = 0
                s["touch_steps"] = 0
                s["staggered_liftoff"] = False
                s["ground_x"] = current_x
                next_stage = 3

    elif stage == 3:
        # Stage 3: 落地屈膝缓冲吸能与平复动量。
        if both_down:
            s["ground_x"] = current_x
        stable = both_down and is_upright and gyro_calm and motion_calm
        s["stable_steps"] = s["stable_steps"] + 1 if stable else 0

        if s["stable_steps"] >= s["landing_stable_steps"]:
            if s["pending_hop_dist"] is not None:
                hop_dist = s["pending_hop_dist"]
                s["hop_count"] += 1
                s["last_hop_dist"] = hop_dist
                s["total_dist"] += hop_dist
                s["landed_fresh"] = True
            s["pending_hop_dist"] = None
            next_stage = 1
            s["stable_steps"] = 0
            s["staggered_liftoff"] = False
        elif s["stage_steps"] >= LANDING_TIMEOUT_STEPS:
            _lj_break_streak(s)
            s["bad_fall"] = True

    if next_stage != stage:
        s["stage"] = next_stage
        s["stage_steps"] = 0


# ------------------------------------------------------------- 核心奖励组

def _lj_hop_combo_award(env) -> float:
    """有界的连续跳跃进度奖励；落地稳定后每跳只结算一次。
    铁律条件：双足严格同时起跳、同时落地、跳跃 3~5cm 并站稳！
    1 到 5 跳线性增长到 1，避免一次抢跳奖励压过后续稳定性。
    """
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or not s["landed_fresh"]:
        return 0.0

    count = s["hop_count"]
    return float(min(count, 5) / 5.0)


def _lj_hop_window_reward(env) -> float:
    """单次跳跃距离精准窗口奖励（Target: 3.0 ~ 5.0 cm）。
    在 3.5cm ~ 4.5cm 处达到峰值 1.0，过小折损，超过 6.5cm 视为失控严重衰减！
    """
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or not s["landed_fresh"]:
        return 0.0

    dist = s["last_hop_dist"]
    center = 0.040
    score = float(np.exp(-0.5 * ((dist - center) / 0.012) ** 2))
    return score


def _lj_landing_settle(env) -> float:
    """每次双脚同时着地后的缓冲吸震与刹车稳立奖励。
    在 Stage 3 落地阶段，双脚踩实，强力鼓励屈膝吸收冲击、水平与垂向速度归零，角速度平静，保持直立！
    """
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or s["stage"] != 3:
        return 0.0

    contacts = env._foot_contacts()
    both_down = float(contacts["left"] and contacts["right"])
    if both_down < 0.5:
        return 0.0

    vx, _, _ = env.heading_lin_vel()
    vz = env.data.qvel[2]
    gyro = env._gyro

    # 水平急刹（vx -> 0）与垂向平缓（vz -> 0）与角速度平静
    brake_h = float(np.exp(-8.0 * (vx ** 2)))
    calm_v = float(np.exp(-10.0 * (vz ** 2)))
    calm_ang = float(np.exp(-3.0 * np.sum(gyro ** 2)))
    upright = _upright(env)

    # 落地后主动后仰垂直回正引导（彻底消除多跳前倾惯性累积）：
    # 落地踩实后积极把身体拉回绝对垂直 (g[0] <= 0.02)，重罚持续前倾！
    g = env._projected_gravity()
    pitch_restore = float(np.exp(-25.0 * (max(0.0, g[0] - 0.02) ** 2)))

    # 落地屈膝缓冲引导（吸收冲击，防止僵直猛蹬导致后仰翻倒）：
    q = env._joint_qpos()
    knee_err = (q[3] - HOP_CROUCH_POSE[3]) ** 2 + (q[12] - HOP_CROUCH_POSE[12]) ** 2
    knee_cushion = float(np.exp(-2.0 * knee_err))

    return float(both_down * brake_h * calm_v * calm_ang * upright * pitch_restore * knee_cushion * 4.0)


def _lj_pop_thrust(env) -> float:
    """起跳阶段（Stage 1）双脚同时反向轻弹离地。
    期望向上 vz 约 0.35m/s，向前 vx 约 0.35m/s，双腿发力严格对称，躯干绝对端正！
    """
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or (s["stage"] != 1 and not s["took_off_fresh"]):
        return 0.0

    v_fwd, _, _ = env.heading_lin_vel()
    vz = env.data.qvel[2]

    # 适度速度窗口（vx 在 0.25~0.45 最佳，vz 在 0.25~0.45 最佳）
    score_vx = float(np.exp(-0.5 * ((v_fwd - 0.35) / 0.15) ** 2))
    score_vz = float(np.exp(-0.5 * ((vz - 0.35) / 0.15) ** 2))

    # 双腿对称蹬伸（两膝角速度比值，1.0 为完全同步）
    qvel = env._joint_vel()
    ext_l = max(0.0, float(-qvel[3]))
    ext_r = max(0.0, float(qvel[12]))
    sync = min(ext_l, ext_r) / (max(ext_l, ext_r) + 1e-4)

    # 必须端正起跳，严禁前后俯仰或左右侧倾！
    g = env._projected_gravity()
    upright_pitch = float(np.exp(-25.0 * (g[0] ** 2)))
    upright_roll = float(np.exp(-25.0 * (g[1] ** 2)))

    return float(score_vx * score_vz * (sync ** 2) * upright_pitch * upright_roll)


def _lj_airborne_posture(env) -> float:
    """空中飞跃阶段（Stage 2）姿态保持端正与双脚向前探出迎地奖励。"""
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or s["stage"] != 2:
        return 0.0

    upright = _upright(env)
    foot_l = env.data.geom_xpos[env.foot_geoms["left"]]
    foot_r = env.data.geom_xpos[env.foot_geoms["right"]]
    trunk_x = env._trunk_xpos[0]
    trunk_z = env._trunk_xpos[2]

    # 双足在躯干下方迎地 (foot_z < trunk_z)
    feet_below = float(foot_l[2] < trunk_z and foot_r[2] < trunk_z)
    # 双足向前探出迎地 (avg_foot_x >= trunk_x - 0.015)，防止脚后拖导致扑倒！
    avg_foot_x = 0.5 * float(foot_l[0] + foot_r[0])
    feet_forward = float(avg_foot_x >= trunk_x - 0.015)
    # 空中抑制向前俯仰旋转角速度，保持飞行平稳不自转
    calm_spin = float(np.exp(-1.5 * (env._gyro[1] ** 2)))
    # 空中保持充沛前向滑翔速度 (0.15~0.40m/s)，坚决杜绝空中伸脚刹车倒退！
    v_fwd, _, _ = env.heading_lin_vel()
    forward_glide = float(np.clip(v_fwd / 0.25, 0.0, 1.2)) if v_fwd > 0.05 else 0.0

    return float(upright * feet_below * feet_forward * calm_spin * forward_glide)


def _lj_crouch_prep(env) -> float:
    """平稳蓄能阶段（Stage 0）双足平踏微下潜蓄力引导。"""
    _lj_update(env)
    s = env._lj
    if s["bad_fall"] or s["stage"] != 0:
        return 0.0

    contacts = env._foot_contacts()
    both_down = float(contacts["left"] and contacts["right"])
    q = env._joint_qpos()
    err2 = np.sum((q[[2, 3, 4, 11, 12, 13]] - HOP_CROUCH_POSE[[2, 3, 4, 11, 12, 13]]) ** 2)
    crouch_score = float(np.exp(-0.5 * err2 / (0.8 ** 2)))
    upright = _upright(env)

    return float(both_down * crouch_score * upright)


# ------------------------------------------------------------- 惩罚项（违规一票否决）

def _lj_stagger_touch_pen(env) -> float:
    """用户铁律：严惩单脚先着地或单脚起跳！(<= 0)
    一旦出现一脚踩地一脚悬空（contacts['left'] ^ contacts['right']），重罚！
    """
    _lj_update(env)
    contacts = env._foot_contacts()
    if contacts["left"] ^ contacts["right"]:
        return -3.0
    return 0.0


def _lj_feet_parallel_pen(env) -> float:
    """空中双足并排齐平考核：严惩高低差与前后剪刀差！(<= 0)
    1. 高低差 |zl - zr| 必须 < 1.0cm；
    2. 前后差 |xl - xr| 必须 < 1.5cm；
    """
    _lj_update(env)
    s = env._lj
    if s["stage"] not in (1, 2):
        return 0.0
    foot_l = env.data.geom_xpos[env.foot_geoms["left"]]
    foot_r = env.data.geom_xpos[env.foot_geoms["right"]]

    z_diff = abs(float(foot_l[2] - foot_r[2]))
    z_pen = float(np.clip((z_diff - 0.010) / 0.015, 0.0, 2.5)) if z_diff > 0.010 else 0.0

    x_diff = abs(float(foot_l[0] - foot_r[0]))
    x_pen = float(np.clip((x_diff - 0.015) / 0.020, 0.0, 2.5)) if x_diff > 0.015 else 0.0

    return -float(z_pen * 1.5 + x_pen * 1.0)


def _lj_overshoot_pen(env) -> float:
    """严厉惩罚单次跳跃冲刺过大（超过 6.5cm 视为失控大跳）(<= 0)"""
    _lj_update(env)
    s = env._lj
    if s["landed_fresh"] and s["last_hop_dist"] > 0.065:
        excess = s["last_hop_dist"] - 0.065
        return -float(np.clip(excess / 0.02, 0.0, 3.5))
    return 0.0


def _lj_tilt_pen(env) -> float:
    """全向倾斜失控严惩：杜绝后仰翻车、横滚侧翻与着地大前扑！(<= 0)"""
    _lj_update(env)
    s = env._lj
    g = env._projected_gravity()
    stage = s["stage"]

    if stage in (1, 2) or s.get("is_airborne", False):
        # 空中飞跃期：允许向前轻跃倾角 (g[0] 在 0.0~0.45 正常)，严惩后仰 (g[0] < -0.05) 与过大前冲 (g[0] > 0.52)
        pitch_back = max(0.0, -0.05 - float(g[0]))
        pitch_front = max(0.0, float(g[0]) - 0.52)
        pitch_err = pitch_back * 2.5 + pitch_front * 1.5
    else:
        # 地面稳立期：身体必须端正垂直 (|g[0]| < 0.10)
        pitch_err = max(0.0, abs(float(g[0])) - 0.10)

    roll_err = max(0.0, abs(float(g[1])) - 0.06)

    # 俯仰角速度严厉阻尼（惩罚旋转翻滚，引导平稳滑翔）：
    spin_pen = max(0.0, abs(float(env._gyro[1])) - 2.5) * 0.4

    total_tilt = 4.0 * (pitch_err ** 2) + 2.0 * pitch_err + 8.0 * (roll_err ** 2) + 4.0 * roll_err + spin_pen
    return -float(np.clip(total_tilt, 0.0, 4.0))


def _lj_straight_pen(env) -> float:
    """侧偏漂移惩罚 (<= 0)"""
    _, v_lat, _ = env.heading_lin_vel()
    pen = min(3.0 * (v_lat ** 2), 2.0)
    return -float(pen)


def _lj_smooth_pen(env) -> float:
    """动作平滑惩罚 (<= 0)"""
    raw_pen = -0.015 * float(((env.last_action - env.prev_action) ** 2).sum())
    return float(np.clip(raw_pen, -1.5, 0.0))


def _lj_fall_pen(env) -> float:
    """以吸收态补齐剩余时域，避免策略靠提前摔倒逃避后续代价。"""
    _lj_update(env)
    s = env._lj
    if s["bad_fall"]:
        return -2.0 * max(env.max_steps - env.step_count + 1, 1)
    return 0.0


# ------------------------------------------------------------- 行为注册

_register(Behavior(
    id="long_jump",
    emoji="🐇",
    title="双脚连续跳跃",
    description="双足同时起跳、同时落地：单次向前轻跳3~5cm，落地缓冲吸能刹停，连续稳定跳跃5次不失衡。",
    how_it_learns=(
        "事件驱动自适应平衡：双足踩实平静后自适应起跳，双足同时轻弹向前跳跃3~5cm，"
        "空中双足齐平并排，双足同时着地屈膝缓冲立定。连续完成5次双足稳定小跳获得通关大奖！"
    ),
    keywords=(
        "双脚连续跳", "连续跳", "兔子跳", "双脚跳跃", "小跳",
        "bunny hop", "continuous hop", "hop", "two foot hop", "long_jump",
    ),
    terms=(
        RewardTerm("hop_combo", "连续跳跃连击通关大奖（必须同时起跳且同时落地）", 30.0, _lj_hop_combo_award),
        RewardTerm("hop_window", "单次跳距精准窗口奖励（3~5cm高斯峰值）", 10.0, _lj_hop_window_reward),
        RewardTerm("landing_settle", "双脚同时落地缓冲吸震与速度刹停立定奖励", 10.0, _lj_landing_settle),
        RewardTerm("pop_thrust", "起跳双腿轻快对称弹射与端正冲量", 10.0, _lj_pop_thrust),
        RewardTerm("airborne_posture", "空中飞跃端正姿态与双足向下迎地", 5.0, _lj_airborne_posture),
        RewardTerm("crouch_prep", "周期蓄势阶段微屈膝蓄力", 2.0, _lj_crouch_prep),
        RewardTerm("stagger_pen", "严惩单脚先着地或单脚起跳（必须同时起跳同时落地）", 2.0,
                   _lj_stagger_touch_pen, is_penalty=True),
        RewardTerm("feet_parallel_pen", "空中双足并排高低差与前后剪刀差惩罚", 4.0,
                   _lj_feet_parallel_pen, is_penalty=True),
        RewardTerm("overshoot_pen", "单次冲刺过大超速惩罚（严禁超过6.5cm大跳）", 2.0,
                   _lj_overshoot_pen, is_penalty=True),
        RewardTerm("tilt_pen", "严禁全向倾斜（前后俯仰后倒/前扑与左右侧倾重罚）", 3.0,
                   _lj_tilt_pen, is_penalty=True),
        RewardTerm("straight_jump_pen", "抑制侧向滑移与走位漂移", 2.0,
                   _lj_straight_pen, is_penalty=True),
        RewardTerm("smooth_moves_pen", "动作平滑性安全截断", 0.1,
                   _lj_smooth_pen, is_penalty=True),
        RewardTerm("save_energy_pen", "电机能耗惩罚", 0.05,
                   _torque_pen, is_penalty=True),
        RewardTerm("fall_pen", "跌倒与身体塌陷失败惩罚", 2.0,
                   _lj_fall_pen, is_penalty=True),
    ),
    default_steps=30_000_000,
    episode_s=4.0,                  # 4.0 秒（200步），留足从容连跳5次及站稳展示的充裕时间！
    scene="walk",
    symmetric=True,
    terminate_on_fall=True,
    success_metric="双足同时起跳且同时落地，单次跳跃3~5cm，连续成功跳跃5次且保持平衡不跌倒",
    state_fn=_lj_update,
    spawn_families=((0.20, _lj_spawn_landing),),
    curriculum=(
        CurriculumStage("先练落地恢复和第一次起跳", 1_500_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "1.0",
                         "MICRODUCK_LJ_MIN_DIST_M": "0.02",
                         "MICRODUCK_LJ_MAX_DIST_M": "0.06",
                         "MICRODUCK_LJ_STABLE_STEPS": "5",
                         "MICRODUCK_LJ_GOAL_HOPS": "1",
                         "MICRODUCK_EPISODE_S": "2"},
                        detail="每次从双脚即将着地的低速下落开始，先吸震站稳，再接一次短跳。"),
        CurriculumStage("完成一次严格短跳", 2_000_000,
                        {"MICRODUCK_ACTUATOR": "xml",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.70",
                         "MICRODUCK_LJ_MIN_DIST_M": "0.03",
                         "MICRODUCK_LJ_MAX_DIST_M": "0.05",
                         "MICRODUCK_LJ_STABLE_STEPS": "10",
                         "MICRODUCK_LJ_GOAL_HOPS": "1",
                         "MICRODUCK_EPISODE_S": "3"},
                        detail="多数回合复习落地，普通站立回合学习双足同步完成一次3~5厘米短跳。"),
        CurriculumStage("把落地接成连续两跳", 3_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.40",
                         "MICRODUCK_LJ_MIN_DIST_M": "0.03",
                         "MICRODUCK_LJ_MAX_DIST_M": "0.05",
                         "MICRODUCK_LJ_STABLE_STEPS": "10",
                         "MICRODUCK_LJ_GOAL_HOPS": "2",
                         "MICRODUCK_EPISODE_S": "4"},
                        detail="切换真实执行器，落地站稳后立即重新蓄力，连续完成两跳。"),
        CurriculumStage("普通站立完成严格五连跳", 5_000_000,
                        {"MICRODUCK_ACTUATOR": "bam",
                         "MICRODUCK_SPAWN_FAMILY_PROBS": "0.0",
                         "MICRODUCK_LJ_MIN_DIST_M": "0.03",
                         "MICRODUCK_LJ_MAX_DIST_M": "0.05",
                         "MICRODUCK_LJ_STABLE_STEPS": "10",
                         "MICRODUCK_LJ_GOAL_HOPS": "5",
                         "MICRODUCK_EPISODE_S": "4"},
                        detail="全部从普通双脚站立开始，在真实执行器下完成五次严格短跳。"),
    ),
))

__all__ = [n for n in dir() if not n.startswith("__")]
