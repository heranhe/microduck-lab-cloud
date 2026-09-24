"""Roadmap 12a: LOOK at a replayed play swing (AGENTS.md verification #2).

    uv run python scripts/render_kick_swing.py --swings swings.jsonl \
        --which 33 --cell 4 --out /tmp/rr        # the swing that misses
    uv run python scripts/render_kick_swing.py --swings swings.jsonl \
        --which 33 --cell 2 --out /tmp/rr        # …the same state, ball on the spot

Renders one recorded play swing, replayed on the kick bench under one cell of
`replay12a.LADDER`, to a captioned contact sheet — the same `build_sheet` the
render-rollout skill writes, since render-rollout can only start an episode
from a behavior's own reset and the whole question here is what happens when
it starts from somewhere else.

Captions carry the two things a whiff turns on: how far the ball has moved
from where the swing found it, and whether anything is touching it.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C
from microduck_local.render_rollout import build_sheet, make_camera, sheet_indices

sys.path.insert(0, str(Path(__file__).resolve().parent))
import replay_kick_swings as R  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--swings", required=True)
    ap.add_argument("--which", type=int, nargs="+", default=[0])
    ap.add_argument("--cell", default="4")
    ap.add_argument("--seconds", type=float, default=1.2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--camera", default="three-quarter")
    ap.add_argument("--frames", type=int, default=12)
    args = ap.parse_args()

    label, cell = next((lb, c) for lb, c in R.LADDER if lb.split()[0] == args.cell)
    rows = [json.loads(x) for x in open(args.swings) if x.strip()]
    rows = [r for r in rows if r.get("state") and r["foot"] in ("kick_left", "kick_right")]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    steps = int(round(args.seconds / C.CTRL_DT))
    walker = R.onnx_infer(R.POLICIES_DIR / "alpha_walking.onnx")

    for idx in args.which:
        row = rows[idx]
        foot = row["foot"].split("_")[1]
        from microduck_local.world import World
        kick = R.onnx_infer(World.skill_path(f"kick_{foot}"))
        env, gid, qadr, _ = R.setup(row, cell, foot, steps)
        renderer = mujoco.Renderer(env.model, 360, 480)
        cam = make_camera(args.camera, 0.9)
        n_kick = steps if cell["window"] is None else int(round(cell["window"] / C.CTRL_DT))
        obs = env._get_obs()
        x0, y0 = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
        frames, caps = [], []
        for k in range(steps):
            obs, _, term, _, _ = env.step((kick if k < n_kick else walker)(obs))
            cam.lookat[:] = env.data.qpos[0:3]
            renderer.update_scene(env.data, cam)
            frames.append(renderer.render().copy())
            bx, by = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
            touch = _touching(env, gid)
            caps.append((f"t {k * C.CTRL_DT:.2f}s  ball {math.hypot(bx - x0, by - y0) * 100:5.1f}cm",
                         f"trunk z {float(env.data.qpos[2]) * 100:.1f}cm",
                         f"touch {touch or '-'}"))
            if term:
                break
        sel = sheet_indices(len(frames), args.frames)
        bx, by = float(env.data.qpos[qadr]), float(env.data.qpos[qadr + 1])
        moved = math.hypot(bx - x0, by - y0)
        build_sheet([frames[i] for i in sel], [caps[i] for i in sel],
                    [f"swing #{idx} seed {row['seed']} t {row['t']}s {row['foot']} — cell '{label}'",
                     f"ball at the swing: {row['ahead']:+.3f} m ahead, {row['side']:+.3f} m to the side; "
                     f"speed {np.hypot(*row['state']['ball_qvel'][:2]):.3f} m/s"],
                    [f"REPLAY: ball moved {moved * 100:.1f} cm in {len(frames) * C.CTRL_DT:.2f} s "
                     f"({'WHIFF' if moved < 0.10 else 'connected'});  "
                     f"IN PLAY the same swing moved {row['dist'] * 100:.1f} cm in 2.0 s "
                     f"({'WHIFF' if row['dist'] < 0.10 else 'connected'})"],
                    [False] * len(sel), out / f"swing{idx}_cell{args.cell}.png")
        print(f"{out}/swing{idx}_cell{args.cell}.png  replay {moved * 100:.1f} cm, play {row['dist'] * 100:.1f} cm")


def _touching(env, gid: int) -> str:
    d, m = env.data, env.model
    names = []
    for c in range(int(d.ncon)):
        g1, g2 = int(d.contact.geom1[c]), int(d.contact.geom2[c])
        oth = g2 if g1 == gid else (g1 if g2 == gid else None)
        if oth is not None:
            names.append(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, oth) or str(oth))
    return ",".join(n.replace("_collision", "").replace("microduck_", "") for n in names)[:24]


if __name__ == "__main__":
    main()
