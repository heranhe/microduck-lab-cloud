"""The slot between the hull's top and the bin's floor, and what falls in it.

His collision model tops the `hull` box out at base z 0.100 and starts
`bin_floor` at 0.1082 — an 8.2 mm void running under the whole basket, open
on every side. The `v04_*` shell meshes close it to the eye and to nothing
else: they are `contype 0`. Measured cost before sealing: 4 of 64 moss-yard
seeds lose a prop into it, where it rides for 123-207 s of a 300 s run and is
never collected (`scripts/probe_moss_pin.py`).
"""
import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from microduck_local.robots import moss

pytestmark = pytest.mark.skipif(
    not moss.moss_ready(),
    reason="MOSS assets missing — uv run fetch-robot moss")


def _rover_boxes(m, d):
    """Every colliding box on the rover, as base-frame z spans."""
    root = m.body("m0/rover").id
    bp, bR = d.xpos[root], d.xmat[root].reshape(3, 3)
    out = {}
    for g in range(m.ngeom):
        name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or ""
        if not name.startswith("m0/") or not m.geom_contype[g]:
            continue
        c, h = m.geom_aabb[g][:3], m.geom_aabb[g][3:]
        gp, gR = d.geom_xpos[g], d.geom_xmat[g].reshape(3, 3)
        lo, hi = np.full(3, 1e9), np.full(3, -1e9)
        for sx in (-1, 1):
            for sy in (-1, 1):
                for sz in (-1, 1):
                    q = bR.T @ ((gp + gR @ (c + np.array([sx, sy, sz]) * h)) - bp)
                    lo, hi = np.minimum(lo, q), np.maximum(hi, q)
        out[name] = (lo, hi)
    return out


def _world(seed=0):
    from microduck_local.viz_server import load_policy_infer
    from microduck_local.world import scenario as S
    from microduck_local.world_server import WorldState
    # Resolved from THIS file, not the cwd: the suite runs from the repo
    # root and these two tests failed there while passing from
    # `microduck_local/` — the rest of tests/test_moss.py already does this.
    sc = S.Scenario.from_dict(json.loads(
        Path(__file__).resolve().parents[1]
        .joinpath("scenarios/moss-yard.json").read_text()))
    st = WorldState(load_infer=load_policy_infer)
    st.world, st.scenario = st.build(sc, seed=seed), sc
    return st


def test_the_void_under_the_bin_floor_is_sealed():
    st = _world()
    m, d = st.world.model, st.world.data
    boxes = _rover_boxes(m, d)
    filler = "m0/" + moss.BIN_VOID_GEOM
    assert filler in boxes, "nothing collides where the slot is"
    lo, hi = boxes[filler]
    # It fills the slot exactly, top and bottom.
    assert abs(lo[2] - moss.BIN_VOID_Z[0]) < 1e-3, lo[2]
    assert abs(hi[2] - moss.BIN_VOID_Z[1]) < 1e-3, hi[2]
    # ...and it is flush under the bin floor, not proud of it anywhere.
    flo, fhi = boxes["m0/bin_floor"]
    assert lo[0] >= flo[0] - 1e-6 and hi[0] <= fhi[0] + 1e-6
    assert lo[1] >= flo[1] - 1e-6 and hi[1] <= fhi[1] + 1e-6
    assert abs(hi[2] - flo[2]) < 1e-3, "a gap is left between filler and floor"
    assert abs(lo[2] - boxes["m0/hull"][1][2]) < 1e-3, "...or above the hull"


def test_the_filler_weighs_nothing():
    """The rover carries an EXPLICIT <inertial> (6.139 kg). A geom with mass
    here would be double counting — see "the mass" in robots/moss.py."""
    spec = moss.robot_spec()
    g = spec.geom(moss.BIN_VOID_GEOM)
    assert g is not None, "seal_bin_void did not add it"
    assert float(g.mass) == 0.0


def test_a_cap_pushed_down_onto_the_bin_floor_stays_above_it():
    """The measured event, reproduced. In seed 31 `cap0` is at rest on the
    bin floor when `tall0` (30 g) is released onto it; the cap goes down 9 mm
    in one 20 ms tick — 0.47 m/s, far too fast for gravity — through the 4 mm
    floor, and lands on `m0/hull` at base z 0.103 where it rides for 200 s.

    Here the push is applied directly: the cap is put where it rests in the
    bin and given that same downward speed. It has to end up ABOVE the floor.
    """
    st = _world()
    m, d = st.world.model, st.world.data
    bid = m.body("cap0").id
    jnt = int(m.body_jntadr[bid])
    qadr, vadr = int(m.jnt_qposadr[jnt]), int(m.jnt_dofadr[jnt])
    root = m.body("m0/rover").id
    bp, bR = d.xpos[root].copy(), d.xmat[root].reshape(3, 3).copy()
    rest_z = 0.1182                    # measured: on the floor, in the bin
    d.qpos[qadr:qadr + 3] = bp + bR @ np.array([-0.138, 0.012, rest_z])
    d.qpos[qadr + 3:qadr + 7] = [1.0, 0.0, 0.0, 0.0]
    d.qvel[vadr:vadr + 6] = 0.0
    d.qvel[vadr + 2] = -0.47           # measured: what tall0 does to it
    for _ in range(400):               # 0.8 s to settle
        mujoco.mj_step(m, d)
    z = float((bR.T @ (d.xpos[bid] - bp))[2])
    assert z > moss.BIN_VOID_Z[1], (
        f"the cap ended at base z {z:.4f}, in the sealed slot "
        f"{moss.BIN_VOID_Z} — it went through the floor")
