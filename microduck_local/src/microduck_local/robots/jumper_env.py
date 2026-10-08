"""Six-foot CPU prototype. Does not inherit the duck's two-foot rewards."""

from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np

from . import jumper_model as J

GAIT_STEPS = 16  # 3.125 Hz at 50 Hz; integer wrap gives equal tripod halves.
TRIPOD_A = np.array([True, False, False, True, True, False])  # LF, RM, LR


def phase_clock(step, command):
    if np.linalg.norm(command) <= 0.05:
        return np.zeros(2, np.float32)
    angle = 2 * math.pi * (step % GAIT_STEPS) / GAIT_STEPS
    return np.array([math.sin(angle), math.cos(angle)], np.float32)


class JumperWalkEnv(gym.Env):
    metadata = {"render_modes": []}
    standing = False

    def __init__(
        self,
        seed=None,
        robot=None,
        domain_rand=True,
        obs_noise=True,
        action_delay=False,
        random_yaw=True,
        actuator="xml",
        actuator_force=None,
        max_episode_s=20.0,
        push_robot=False,
    ):
        from .jumper import body

        if (actuator_force or actuator) != "xml":
            raise ValueError("Jumper uses lab XML PD; BAM describes MicroDuck XL330 only")
        if action_delay or push_robot:
            raise ValueError("Jumper v1 does not implement action_delay or push_robot")
        self.robot = robot or body()
        if self.robot.obs_dim != J.OBS_DIM:
            raise ValueError("Jumper observation contract must be 71D")
        self.model = mujoco.MjModel.from_xml_path(str(self.robot.scene_fn()))
        self.data = mujoco.MjData(self.model)
        self.trunk_body_id = self._need(mujoco.mjtObj.mjOBJ_BODY, self.robot.base_body)
        self._gyro_id = self._need(mujoco.mjtObj.mjOBJ_SENSOR, self.robot.gyro_sensor)
        self.floor_id = self._need(mujoco.mjtObj.mjOBJ_GEOM, self.robot.floor_geom)
        self.foot_ids = np.array(
            [
                self._need(mujoco.mjtObj.mjOBJ_GEOM, g)
                for geoms in self.robot.foot_geoms.values()
                for g in geoms
            ]
        )
        if len(self.foot_ids) != 6:
            raise ValueError("Jumper needs six foot geoms in LEGS order")
        self.joint_qpos_adr = np.array(
            [self.model.joint(n).qposadr[0] for n in self.robot.joint_names]
        )
        self.joint_qvel_adr = np.array(
            [self.model.joint(n).dofadr[0] for n in self.robot.joint_names]
        )
        self.ctrl_ids = np.array([self.model.actuator(n).id for n in self.robot.joint_names])
        self.nj = len(self.ctrl_ids)
        self.actuator_model = "xml"
        self._rng = np.random.default_rng(seed)
        self.domain_rand = domain_rand
        self.obs_noise = obs_noise
        self.random_yaw = random_yaw
        self._friction = self.model.geom_friction.copy()
        self.max_steps = int(max_episode_s / J.CTRL_DT)
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (J.OBS_DIM,), np.float32)
        self.action_space = gym.spaces.Box(-4.0, 4.0, (self.nj,), np.float32)
        self.twist_cmd = np.zeros(3, np.float32)
        self.last_action = np.zeros(self.nj, np.float32)
        self.prev_action = self.last_action.copy()
        self.prev_joint_vel = self.last_action.copy()
        self._terms = {}
        self.step_count = 0
        self.reset(seed=seed)

    def _need(self, kind, name):
        idx = mujoco.mj_name2id(self.model, kind, name)
        if idx < 0:
            raise KeyError(f"Jumper model has no {name!r}")
        return idx

    def _projected_gravity(self):
        return -(self.data.xmat[self.trunk_body_id].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0]))

    def body_lin_vel(self):
        out = np.empty(6)
        mujoco.mj_objectVelocity(
            self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.trunk_body_id, out, 1
        )
        return out[3:]

    def heading_lin_vel(self):
        return self.body_lin_vel()

    def _joint_pos_rel(self):
        return self.data.qpos[self.joint_qpos_adr] - self.robot.default_pose

    def _joint_vel(self):
        return self.data.qvel[self.joint_qvel_adr]

    def _get_obs(self):
        gyro = self.data.sensor(self.robot.gyro_sensor).data.copy()
        gravity = self._projected_gravity()
        pos = self._joint_pos_rel()
        vel = self._joint_vel().copy()
        if self.obs_noise:
            gyro += self._rng.uniform(-0.03, 0.03, 3)
            gravity += self._rng.uniform(-0.01, 0.01, 3)
            pos += self._rng.uniform(-0.001, 0.001, self.nj)
            vel += self._rng.uniform(-0.25, 0.25, self.nj)
        return np.concatenate(
            [
                gyro,
                gravity,
                pos,
                vel,
                self.last_action,
                self.twist_cmd,
                phase_clock(self.step_count, self.twist_cmd),
            ]
        ).astype(np.float32)

    def _sample_commands(self):
        if self.standing or self._rng.random() < 0.2:
            self.twist_cmd[:] = 0
        else:
            self.twist_cmd[:] = self._rng.uniform([-0.2, -0.2, -0.5], [0.2, 0.2, 0.5])

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        kid = self._need(mujoco.mjtObj.mjOBJ_KEY, self.robot.stand_keyframe)
        mujoco.mj_resetDataKeyframe(self.model, self.data, kid)
        self.model.geom_friction[:] = self._friction
        if self.domain_rand:
            self.model.geom_friction[self.foot_ids, 0] = self._friction[
                self.foot_ids, 0
            ] * self._rng.uniform(0.8, 1.2)
        if self.random_yaw:
            yaw = self._rng.uniform(-math.pi, math.pi)
            self.data.qpos[3:7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        self.step_count = 0
        self.last_action[:] = 0
        self.prev_action[:] = 0
        self._terms = {}
        self._sample_commands()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {}

    def _contacts(self):
        touched = set()
        for c in self.data.contact:
            if c.geom1 == self.floor_id:
                touched.add(c.geom2)
            elif c.geom2 == self.floor_id:
                touched.add(c.geom1)
        return np.array([int(i) in touched for i in self.foot_ids])

    def _reward(self):
        lin = self.body_lin_vel()
        gyro = self.data.sensor(self.robot.gyro_sensor).data
        gravity = self._projected_gravity()
        contacts = self._contacts()
        moving = np.linalg.norm(self.twist_cmd) > 0.05
        swing = TRIPOD_A if self.step_count % GAIT_STEPS < GAIT_STEPS // 2 else ~TRIPOD_A
        desired = ~swing if moving else np.ones(6, bool)
        terms = {
            "track_linear_velocity": 2.0
            * math.exp(-float(np.sum((lin[:2] - self.twist_cmd[:2]) ** 2)) / 0.04),
            "track_angular_velocity": math.exp(-float((gyro[2] - self.twist_cmd[2]) ** 2) / 0.25),
            "upright": 2.0 * math.exp(-float(np.sum(gravity[:2] ** 2)) / 0.05),
            "height": math.exp(
                -float((self.data.xpos[self.trunk_body_id, 2] - J.STAND_Z) ** 2) / 0.0004
            ),
            "pose": 0.5 * math.exp(-float(np.mean(self._joint_pos_rel() ** 2)) / 0.25),
            "tripod_gait": float(np.mean(contacts == desired)),
            "action_rate_penalty": -0.01
            * float(np.sum((self.last_action - self.prev_action) ** 2)),
            "joint_vel_penalty": -0.0001 * float(np.sum(self._joint_vel() ** 2)),
        }
        return sum(terms.values()), terms

    def step(self, action):
        action = np.asarray(action, np.float32)
        if action.shape != (self.nj,) or not np.isfinite(action).all():
            raise ValueError(f"Jumper action must be finite shape {(self.nj,)}")
        self.prev_action[:] = self.last_action
        self.last_action[:] = np.clip(action, -4, 4)
        self.data.ctrl[self.ctrl_ids] = self.robot.scale_action(self.last_action)
        for gripper in J.GRIPPERS:
            self.data.ctrl[self.model.actuator(gripper).id] = J.HOME[gripper]
        for _ in range(round(J.CTRL_DT / self.model.opt.timestep)):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1
        mujoco.mj_forward(self.model, self.data)
        finite = np.isfinite(self.data.qpos).all() and np.isfinite(self.data.qvel).all()
        fell = (
            self._projected_gravity()[2] >= self.robot.fall_gravity_z
            or self.data.xpos[self.trunk_body_id, 2] < self.robot.fall_height
        )
        terminated = bool(not finite or fell)
        reward, terms = self._reward() if finite else (0.0, {})
        for k, v in terms.items():
            self._terms[k] = self._terms.get(k, 0.0) + v
        truncated = self.step_count >= self.max_steps
        info = {"reward_terms": terms}
        if terminated or truncated:
            info["episode_rewards"] = self._terms.copy()
        if not self.standing and self.step_count % 250 == 0:
            self._sample_commands()
        return self._get_obs(), float(reward), terminated, bool(truncated), info


class JumperStandEnv(JumperWalkEnv):
    standing = True


class JumperStandDriver:
    """Room integration baseline: hold HOME with PD, no locomotion claim."""

    def __init__(self, model, prefix):
        self.model = model
        self.prefix = prefix
        self.root = model.joint(prefix + "floating_base").qposadr[0]
        self.base = model.body(prefix + "base_link").id
        self.ids = np.array([model.actuator(prefix + n).id for n in J.ALL_JOINTS])
        self.qpos = np.array([model.joint(prefix + n).qposadr[0] for n in J.ALL_JOINTS])
        self.targets = dict(J.HOME)

    def set_cmd(self, vx, wz, now):
        # No learned walker has shipped yet; the room baseline stays at HOME.
        self.command = (vx, wz)

    def set_arm(self, targets):
        if targets:
            raise ValueError("Jumper room baseline has no arm skill")

    def arm_targets(self):
        return {}

    def step(self, data):
        data.ctrl[self.ids] = list(self.targets.values())

    def spawn(self, data, x, y, yaw):
        data.qpos[self.root : self.root + 7] = [
            x,
            y,
            J.STAND_Z + 0.002,
            math.cos(yaw / 2),
            0,
            0,
            math.sin(yaw / 2),
        ]
        data.qpos[self.qpos] = list(J.HOME.values())
        self.step(data)

    def pose(self, data):
        R = data.xmat[self.base].reshape(3, 3)
        return (*data.xpos[self.base, :2], math.atan2(R[1, 0], R[0, 0]))

    def velocity(self, data):
        out = np.empty(6)
        mujoco.mj_objectVelocity(self.model, data, mujoco.mjtObj.mjOBJ_BODY, self.base, out, 1)
        return float(out[3]), float(out[2])
