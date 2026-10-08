"""Jumper: 22 physical joints, 20 gait actions; independent lab contract."""

from functools import lru_cache

import numpy as np

from . import jumper_model as J
from .body import RobotFrames, measure_lab_spacing_m
from .policy_contract import declare
from .spec import Effector, RobotSpec


class JumperBody(RobotSpec):
    visual_group = 2

    def ready(self):
        return J.ready()

    def fetch(self):
        return J.fetch()

    def contract(self):
        return declare(
            self,
            id=J.FACTS["contract_id"],
            rate_hz=1 / J.CTRL_DT,
            slots=J.SLOTS,
            deploy=J.DEPLOY,
        )

    def env_class(self, task="walk"):
        from .jumper_env import JumperStandEnv, JumperWalkEnv

        if task in (None, "", "walk", "tripod"):
            return JumperWalkEnv
        if task == "stand":
            return JumperStandEnv
        raise ValueError(f"unsupported Jumper task {task!r}; choose tripod or stand")

    def train_env_kwargs(self, args):
        if getattr(args, "actuator", None) == "bam":
            raise ValueError("Jumper requires --actuator xml; BAM models the duck only")
        return {"actuator_force": "xml"}

    def visual_scene(self):
        return visual_scene()

    def attach(self, spec, prefix, frame, collision="walk"):
        spec.attach(J.robot_spec(), prefix=prefix, frame=frame)

    def frames(self):
        return RobotFrames(base="base_link")

    def driver(self, model, prefix):
        from .jumper_env import JumperStandDriver

        return JumperStandDriver(model, prefix)

    def robot_model(self):
        return J.robot_spec().compile()


@lru_cache(maxsize=1)
def visual_scene():
    from .g1 import extract_visual_scene

    return extract_visual_scene(J.robot_spec().compile(), vert_scale=0.0001)


@lru_cache(maxsize=1)
def body():
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(J.scene_xml()))
    return JumperBody(
        id="jumper",
        title="Jumper",
        noun="Jumper",
        kind="legged",
        joint_names=J.JOINT_NAMES,
        joint_groups=tuple(n.split("_", 1)[0] for n in J.JOINT_NAMES),
        default_pose=np.array([J.HOME[n] for n in J.JOINT_NAMES], np.float32),
        obs_dim=J.OBS_DIM,
        action_scale=np.full(20, J.ACTION_SCALE, np.float32),
        action_clip=4.0,
        scene_fn=J.scene_xml,
        stand_keyframe="STAND",
        base_body="base_link",
        gyro_sensor="imu_ang_vel",
        foot_geoms={leg: (geom,) for leg, geom in zip(J.LEGS, J.FOOT_GEOMS)},
        fall_height=0.04,
        fall_gravity_z=-0.643,
        lin_vel_x_range=(-0.2, 0.2),
        lin_vel_y_range=(-0.2, 0.2),
        ang_vel_z_range=(-0.5, 0.5),
        twist_obs_slice=(66, 69),
        lab_spacing_m=measure_lab_spacing_m(model, "STAND")[0],
        physics_dt=J.PHYSICS_DT,
        effectors=tuple(
            Effector(id=leg, label=leg, body=foot, point="sole")
            for leg, foot in zip(J.LEGS, J.FEET)
        ),
        extra={
            "model_joints": J.ALL_JOINTS,
            "revision": J.REVISION,
            "open_loop_stable": True,
            "world_driver": "standing only",
        },
    )


class _LazyJumper:
    id = "jumper"
    title = "Jumper"
    noun = "Jumper"
    kind = "legged"

    def _resolve(self):
        return body()

    def ready(self):
        return J.ready()

    def fetch(self):
        return J.fetch()

    def setup_hint(self):
        return "uv run fetch-robot jumper"

    def __getattr__(self, name):
        return getattr(self._resolve(), name)


JUMPER = _LazyJumper()
