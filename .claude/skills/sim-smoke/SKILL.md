---
name: sim-smoke
description: Look at the /sim world page the way a user would — bring up the lab in world mode and the viewer, open the page in headless Chromium, press keys, screenshot it, and READ the screenshot and console. Use after touching world_server / world / sensors / brain or the viewer's sim files, whenever "does the page actually render/stream" is the question, and to capture a picture of a scenario for a PR. Trigger on: "smoke test /sim", "screenshot the sim page", "is the world page working", "show me the room".
---

Two scripts, both from the repo root:

1. **Bring the stack up** (idempotent; `--restart` relaunches the lab, which
   you need after editing any lab Python — the running process holds stale
   code):

   ```
   bash .claude/skills/sim-smoke/bringup.sh living-room --restart
   ```

   Every launch goes through the `restart-servers` scripts (`restart.sh`,
   `viewer.sh`), which start each server in its own session so it outlives
   the agent shell — **never start the lab with a bare `nohup` from an agent
   shell.** This script used to, with `setsid` as the detach; macOS has none,
   so the lab stayed in the tool call's process group, and on 2026-09-10 the
   harness reaped a backgrounded call of it and took the lab down with it,
   hours after "lab: up". The scenario is then loaded over the lab's own API
   (`POST /world/load`), so the lab is the user's — its roster, its logs.
   `restart.sh` refuses while a teach job is training.

   Built-in scenarios: `empty-floor`, `wall-test`, `living-room`, `playroom`,
   `pitch`, `pitch-2v2`, `pitch-3v3` (`GET /scenarios` lists saved ones too).
   Logs: `microduck_local/lab-server.log`, `duck-viewer/viewer-server.log`.

2. **Screenshot the page** (needs Playwright; on the web runner it is in the
   global node modules and Chromium is at `/opt/pw-browsers/chromium`; on a
   Mac `npm i -g playwright && npx playwright install chromium` and unset
   `CHROMIUM_PATH`):

   ```
   node .claude/skills/sim-smoke/shot.mjs --out /tmp/sim.png --keys "Escape"
   node .claude/skills/sim-smoke/shot.mjs --keys "1" --out /tmp/sim-d0.png   # select duck 1
   ```

   Then **Read the PNG** and check: the `● live` badge (frames flowing),
   walls/boxes/ducks present, ToF dots on the surfaces the ducks face, the
   inspector heatmap painted, RTF ≈ 1.00 in the top bar. The script prints
   the browser console minus dev-server noise — a `pageerror` line is a bug.

Numbers to trust over the picture: `curl -s :8788/world` for per-duck falls
and presets, and `curl -s :8788/replay/ring | tail -c 600` for the last
recorded frames (once record/replay exists).

Gotchas: never `pkill -f duck-lab` from an agent shell (the pattern matches
your own command line and kills your shell — and every other lab on the
machine, including another session's scratch lab on its own port;
`restart.sh` stops only the process listening on :8788); the container has no EGL, so do not set
`MUJOCO_GL=egl` here; headless software GL is slower than a Mac GPU, so give
the page a few seconds. If a bring-up call ever overruns the tool timeout and
is backgrounded, the servers now survive its reaping — that is the point of
the delegation above.

A screenshot is one instant. For *what happened over a minute* — a fall, a
stall, a scrum — use the `record-world` skill instead: it runs the same
scenario headless under a seed and writes an mp4, a contact sheet and an
events log; no browser or running lab needed.

3. **Film the page**, when the thing you want is the PAGE and not the physics
   — the room's textures, the camera insets, the LiDAR fan, the brain's map
   panel. `record-world` renders raw MuJoCo and none of that is in it, so a
   clip for the README comes from here:

   ```
   node .claude/skills/sim-smoke/film.mjs --url http://localhost:63317/sim \
        --out /tmp/film --w 1280 --h 863 --reset moss-yard --skip 37000 --seconds 21
   ffmpeg -framerate 60 -t 19 -i /tmp/film/f%05d.jpg \
        -vf "setpts=PTS/4,fps=15,scale=720:-1,palettegen=max_colors=64" -y /tmp/p.png
   ffmpeg -framerate 60 -t 19 -i /tmp/film/f%05d.jpg -i /tmp/p.png \
        -lavfi "setpts=PTS/4,fps=15,scale=720:-1[x];[x][1:v]paletteuse" -y out.gif
   ```

   CDP `Page.startScreencast` at ~60 fps, so `setpts=PTS/4` is a true 4x and
   not dropped frames. `--reset <scenario>` POSTs `/world/load` AFTER the page
   is up, so `--skip` is measured from t = 0 and a window scouted with
   `record-world` on the same seed lands in the same place. **`--h` is the
   WINDOW, not the viewport**: Chrome's own chrome eats 143 px, so 863 gives
   720 — ask for 720 and you get a 2.11:1 clip where the rest of the README
   is 16:9. `docs/media/sim-moss-tidy.gif` was made with exactly the above.
