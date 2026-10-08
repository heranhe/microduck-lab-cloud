"""Pinned Jumper assets and a lab PD model, shared verbatim with the other harness.

Hardware facts come from KingKongRobotics/jumper at the revision in the JSON.
The constant torque clamp is a lab approximation, not the upstream thermal model.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

FACTS = json.loads(Path(__file__).with_name("jumper_assets.json").read_text())
REVISION = FACTS["revision"]
HOME = FACTS["HOME"]
ALL_JOINTS = tuple(HOME)
GRIPPERS = tuple(FACTS["GRIPPER_JOINTS"])
JOINT_NAMES = tuple(j for j in HOME if j not in GRIPPERS)
LEGS = tuple(FACTS["LEGS"])
FEET = tuple(FACTS["FEET"])
FOOT_GEOMS = tuple(f"{f}_meshcol" for f in FEET)
OBS_DIM = 6 + 3 * len(JOINT_NAMES) + 3 + 2
PHYSICS_DT = FACTS["physics_dt"]
CTRL_DT = FACTS["control_dt"]
ACTION_SCALE = FACTS["action_scale"]
STAND_Z = FACTS["stand_z"]
# Both projects keep their own cache unless the user explicitly shares one.
CACHE_DIR = Path(
    os.environ.get("MICRODUCK_JUMPER_DIR")
    or Path(__file__).resolve().parents[3] / ".cache" / "jumper"
)
SLOTS = (
    ("base_ang_vel", 0, 3),
    ("projected_gravity", 3, 6),
    ("joint_pos_rel", 6, 26),
    ("joint_vel", 26, 46),
    ("last_action", 46, 66),
    ("twist_cmd", 66, 69),
    ("gait_phase", 69, 71),
)


DEPLOY = "lab prototype: constant-clamp PD, no observation history; incompatible with upstream Jumper policies; untested on hardware"


def policy_contract() -> dict:
    return {
        "id": FACTS["contract_id"],
        "robot": "jumper",
        "obs_dim": OBS_DIM,
        "act_dim": len(JOINT_NAMES),
        "rate_hz": 1 / CTRL_DT,
        "slots": [list(slot) for slot in SLOTS],
        "deploy": DEPLOY,
    }


def stamp_policy(path: Path) -> None:
    """Carry the lab contract alongside mjlab's existing ONNX metadata."""
    import onnx

    model = onnx.load(str(path))
    # Reject upstream/history policies rather than labelling an incompatible graph.
    inp, out = model.graph.input[0], model.graph.output[0]
    if inp.type.tensor_type.shape.dim[-1].dim_value != OBS_DIM or out.type.tensor_type.shape.dim[
        -1
    ].dim_value != len(JOINT_NAMES):
        raise ValueError("Jumper lab policy must have 71 observations and 20 actions")
    keep = [p for p in model.metadata_props if p.key not in ("contract_id", "contract")]
    del model.metadata_props[:]
    for p in keep:
        model.metadata_props.add().CopyFrom(p)
    for key, value in (
        ("contract_id", FACTS["contract_id"]),
        ("contract", json.dumps(policy_contract(), sort_keys=True)),
    ):
        entry = model.metadata_props.add()
        entry.key, entry.value = key, value
    _atomic(Path(path), model.SerializeToString())


def _atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".jumper-")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(payload)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def ready() -> bool:
    try:
        return (CACHE_DIR / "revision.txt").read_text().strip() == REVISION and all(
            (CACHE_DIR / p).is_file() for p in FACTS["files"]
        )
    except OSError:
        return False


def fetch(source: Path | None = None) -> Path:
    """Fetch only referenced meshes plus licences; verify SHA256 before replacing.

    `source` is an already-cloned upstream checkout for offline installation.
    A failed or partial transfer never creates the revision marker.
    """
    for relative, expected in FACTS["files"].items():
        dest = CACHE_DIR / relative
        if dest.is_file() and hashlib.sha256(dest.read_bytes()).hexdigest() == expected:
            continue
        if source is not None:
            payload = (Path(source) / relative).read_bytes()
        else:
            url = f"https://raw.githubusercontent.com/KingKongRobotics/jumper/{REVISION}/{relative}"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Jumper SHA256 mismatch for {relative}: {actual}")
        _atomic(dest, payload)
    _atomic(CACHE_DIR / "revision.txt", (REVISION + "\n").encode())
    print(f"Jumper {REVISION}: {len(FACTS['files'])} verified files; Apache-2.0, KingKong Robotics")
    return CACHE_DIR


def robot_spec():
    """Read the unchanged source and apply the lab's adaptation in memory."""
    import mujoco

    if not ready():
        raise FileNotFoundError(
            "Jumper assets missing; run `fetch-robot jumper` in microduck_local or `python scripts/fetch_jumper.py` in microduck_rl"
        )
    src = CACHE_DIR / "assets/jumper/jumper.xml"
    tree = ET.parse(src)
    root = tree.getroot()
    root.find("compiler").set("meshdir", str(src.parent))
    # Same-leg meshes are adjacent rigid parts; collide across legs and with
    # ground, following upstream HYBRID_COLLISION's bit convention.
    bits = {leg: 1 << (i + 1) for i, leg in enumerate(LEGS)}
    all_bits = sum(bits.values())
    for geom in root.findall(".//geom"):
        name = geom.get("name", "")
        if not name.endswith("_meshcol"):
            continue
        leg = name.split("_", 1)[0]
        geom.set("contype", str(1 | bits.get(leg, 0)))
        geom.set("conaffinity", str(all_bits & ~bits[leg] if leg in bits else 0))
        foot = name in FOOT_GEOMS
        geom.set("condim", "3" if foot else "1")
        geom.set("priority", "1" if foot else "0")
        geom.set("solref", "0.02 1")
        geom.set("solimp", "0.8 0.9 0.002 0.5 2")
        if foot:
            geom.set("friction", "1.2 0.01 0.01")
    actuator = ET.SubElement(root, "actuator")
    for name in ALL_JOINTS:
        ET.SubElement(
            actuator,
            "position",
            name=name,
            joint=name,
            kp=str(FACTS["stiffness"]),
            kv=str(FACTS["damping"]),
            forcerange=f"-{FACTS['effort_limit']} {FACTS['effort_limit']}",
        )
    return mujoco.MjSpec.from_string(ET.tostring(root, encoding="unicode"))


def scene_xml() -> Path:
    import mujoco
    import numpy as np

    spec = robot_spec()
    floor = spec.worldbody.add_geom(
        name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[10, 10, 0.05], rgba=[0.3, 0.3, 0.3, 1]
    )
    floor.contype, floor.conaffinity = 1, 1
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1])
    model = robot_spec().compile()
    qpos = model.qpos0.copy()
    root = model.joint("floating_base").qposadr[0]
    qpos[root : root + 7] = [0, 0, STAND_Z + 0.002, 1, 0, 0, 0]
    for name, value in HOME.items():
        qpos[model.joint(name).qposadr[0]] = value
    key = spec.add_key(name="STAND")
    key.qpos = qpos
    key.qvel = np.zeros(model.nv)
    key.ctrl = np.array(list(HOME.values()))
    out = CACHE_DIR / "scene_lab.xml"
    payload = spec.to_xml().encode()
    if not out.is_file() or out.read_bytes() != payload:
        _atomic(out, payload)
    return out
