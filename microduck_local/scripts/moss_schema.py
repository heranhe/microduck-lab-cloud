"""Print MOSS's policy I/O schema from the body itself — the source docs/moss-policy-schema.md is written from.

    uv run python scripts/moss_schema.py

Exists so the document and the code cannot drift silently: every number below
is read off `robots/moss.py` and `robots/moss_env.py` at run time.
"""
from microduck_local.robots import moss
from microduck_local.robots import moss_env as env


def main() -> None:
    c = moss.MOSS.contract()
    print(f"contract {c.id}: obs[1,{c.obs_dim}] -> actions[1,{c.act_dim}] "
          f"at {c.rate_hz:g} Hz")
    print("\nobservation:")
    for slot in c.slots:
        print(f"  [{slot.start:2d}:{slot.stop:2d}] {slot.name}")
    print("\naction scaling:")
    print(f"  [0:5] arm delta   +-{env.ARM_DELTA_RAD} rad per step")
    print(f"  [5:6] gripper     +-{env.JAW_DELTA_M} m per step, clamp 0..0.041")
    print(f"  [6:7] vx          +-{env.MAX_VX} m/s")
    print(f"  [7:8] wz          +-{env.MAX_WZ} rad/s")
    print("\nkey constants:")
    print(f"  DEFAULT_POSE        {list(moss.DEFAULT_POSE)}")
    print(f"  grasp standoff      {moss.GRASP_STANDOFF_M} m, height {moss.GRASP_HEIGHT_M} m")
    print(f"  gripper grip cmd    {moss.GRASP_JAW_CTRL_M} m")
    print(f"  camera front face   {moss.CAMERA_POS} m, fov {moss.CAMERA_HFOV_DEG}x{moss.CAMERA_VFOV_DEG} deg")
    print(f"  depth unusable below {moss.DEPTH_MIN_RANGE_M} m (nothing here uses depth)")
    print(f"  track spacing drive {moss.TRACK_CENTRES_M} m, collision {moss.TRACK_CENTRES_COLLISION_M} m")
    print(f"  mass                {[(n, kg) for n, kg, _ in moss.BASE_COMPONENTS]} + arm {moss.ARM_MASS_KG}")

if __name__ == "__main__":
    main()
