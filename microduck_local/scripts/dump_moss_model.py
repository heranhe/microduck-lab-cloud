"""Write the exact patched MOSS MJCF, and the numbers we measured off it.

    uv run python scripts/dump_moss_model.py [outdir]

Laurent asked for "the exact patched MJCF and reproduction script so we can
compare the same model". This is that: it writes the XML this repo actually
compiles (his file plus our edits, in order), and prints every geometry
number we have quoted, using the transform HE corrected us on — mesh
vertices rotated by `geom_xmat` and then expressed in the rover frame.
"""
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local.robots import moss

EDITS = ("add_planar_base", "add_camera", "couple_fingers",
         "widen_arm_proxies", "tune_contacts", "drop_base_servo",
         "set_base_inertial", "rewrite_home_key")


def mesh_bounds_in_rover(m, d, geom_name):
    """Mesh vertices in the ROVER frame — the transform Laurent supplied.

    `mesh_vert + geom_pos` is NOT the geometry: MuJoCo recentres and rotates
    vertices at compile time. Getting this wrong reported the bin 35 mm
    taller than it is and sent us building around a false obstacle.
    """
    gid = next(i for i in range(m.ngeom) if (m.geom(i).name or "") == geom_name)
    mid = int(m.geom_dataid[gid])
    a, n = int(m.mesh_vertadr[mid]), int(m.mesh_vertnum[mid])
    v = m.mesh_vert[a:a + n].reshape(-1, 3)
    world = v @ d.geom_xmat[gid].reshape(3, 3).T + d.geom_xpos[gid]
    rover = m.body(moss.BASE_BODY).id
    return (world - d.xpos[rover]) @ d.xmat[rover].reshape(3, 3), n


def main() -> None:
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/moss-model")
    out.mkdir(parents=True, exist_ok=True)
    spec = moss.robot_spec()
    (out / "moss_patched.xml").write_text(spec.to_xml())
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)

    print(f"wrote {out / 'moss_patched.xml'}")
    print(f"  his file at sha {moss.MOSS_JEV_SHA}")
    print(f"  our edits, in order: {', '.join(EDITS)}")
    print()
    verts, n = mesh_bounds_in_rover(m, d, "v04_030_bin")
    print(f"BIN, V0.4 visual mesh ({n} vertices), rover frame:")
    for ax, k in (("X", 0), ("Y", 1), ("Z", 2)):
        print(f"   {ax} {verts[:, k].min():+.6f} .. {verts[:, k].max():+.6f}")
    print("COLLISION walls:")
    for nm in ("bin_x1", "bin_x-1", "bin_y1", "bin_y-1"):
        g = next(i for i in range(m.ngeom) if (m.geom(i).name or "") == nm)
        top = float(m.geom_pos[g][2] + m.geom_size[g][2])
        print(f"   {nm:<10} top z {top:.4f}")
    print()
    print("ARM collision proxies (capsules):")
    for nm in ("upper_arm_link_proxy", "lower_arm_link_proxy"):
        g = next(i for i in range(m.ngeom) if (m.geom(i).name or "") == nm)
        print(f"   {nm:<24} radius {float(m.geom_size[g][0]):.4f} "
              f"half-len {float(m.geom_size[g][1]):.4f}  "
              f"contype {int(m.geom_contype[g])} conaffinity "
              f"{int(m.geom_conaffinity[g])}")
    print("   (contype 2 / conaffinity 1 on both: capsule-capsule contacts "
          "are FILTERED,")
    print("    so a self-collision between arm links is not reported at all.)")
    print()
    print("TRACKS — the unreconciled pair:")
    for nm in ("track_1", "track_-1"):
        g = next(i for i in range(m.ngeom) if (m.geom(i).name or "") == nm)
        print(f"   {nm:<10} y {float(m.geom_pos[g][1]):+.4f}")
    print(f"   collision spacing {moss.TRACK_CENTRES_COLLISION_M} m, "
          f"V0.4 CAD {moss.TRACK_CENTRES_M} m — both declared, neither adopted")
    print()
    print(f"TUCK_POSE {moss.TUCK_POSE}")
    print(f"CONTRACT  {moss.CONTRACT_ID}: {moss.OBS_DIM} obs -> "
          f"{moss.NUM_ACTIONS} actions at {moss.CONTROL_HZ} Hz")


if __name__ == "__main__":
    main()
