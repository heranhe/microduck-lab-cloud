"""Measure a NEW object's two grasp numbers on MOSS's jaw: the drill rung,
a jaw sweep, and a scripted close+lift. Same method as the can's table."""
import numpy as np

from microduck_local.robots import moss
from microduck_local.robots.moss_env import ARM_DELTA_RAD, GraspProp, MossPickEnv

# the playroom block: world/scenario.PICKABLE_KINDS["block"], 4 cm, 20 g
SHAPE, SIZE, MASS = "box", (0.02, 0.02, 0.02), 0.02
lift_dir = np.clip((np.array(moss.LIFT_POSE) - np.array(moss.GRASP_POSE))
                   / ARM_DELTA_RAD, -1.0, 1.0)
print(f"{'jaw ctrl':>9} {'inner mm':>9} {'lifted':>8} {'mean lift':>10}")
for ctrl in (0.010, 0.012, 0.014, 0.016, 0.018):
    lifts, held = [], 0
    for seed in range(4):
        prop = GraspProp(id="block", shape=SHAPE, size=SIZE, mass=MASS,
                         jaw_ctrl_m=ctrl, grasp_height_m=0.020,
                         rgba=(0.95, 0.75, 0.2, 1.0))
        env = MossPickEnv(seed=seed, pick_rung=0, prop=prop)
        env.reset()
        a = np.zeros(9, np.float32)
        # close to THIS jaw value, then lift
        close_a = (ctrl - 0.041) / 25.0 / 0.004
        info = {}
        for i in range(70):
            a[:] = 0.0
            a[5] = float(np.clip(close_a, -1, 1)) if i < 25 else 0.0
            if i >= 25:
                a[:5] = lift_dir
            _o, _r, term, trunc, info = env.step(a)
            if term or trunc:
                break
        lifts.append(info.get("lift", 0.0))
        held += int(info.get("lift", 0.0) > 0.05)
    inner = 16.0 + 2.0 * ctrl * 1e3 - 8.0
    print(f"{ctrl:>9.3f} {inner:>9.1f} {held:>6}/4 {np.mean(lifts):>10.3f}")
