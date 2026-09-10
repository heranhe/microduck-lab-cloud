#!/usr/bin/env bash
# Bring up the lab and the viewer for a smoke test, and load a WORLD scenario.
# Usage: bash .claude/skills/sim-smoke/bringup.sh [scenario] [--restart]
# Idempotent: a healthy pair is left alone; --restart relaunches the lab (and,
# with it, the viewer) - needed after editing any lab Python, since the
# running process holds stale code.
#
# Every launch goes through the restart-servers scripts (restart.sh /
# viewer.sh), never a bare `nohup` from here. This script used to launch the
# lab itself with `setsid`-or-`nohup`; macOS has no setsid, so the lab stayed
# in the agent shell's process group, and on 2026-09-10 the harness reaped a
# backgrounded call of this script and took the lab down with it (SIGTERM,
# hours after "lab: up"). The restart-servers launch is the one proven to
# outlive the session that started it.
#
# The scenario is loaded over the lab's API (POST /world/load) once the lab
# is up, so the lab is the user's own (its roster, its logs) rather than a
# second one on the same port: microduck_local/lab-server.log and
# duck-viewer/viewer-server.log.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
RS="$ROOT/.claude/skills/restart-servers"
SCENARIO="${1:-living-room}"
[[ "$SCENARIO" == "--restart" ]] && SCENARIO="living-room"
RESTART=0
for a in "$@"; do [[ "$a" == "--restart" ]] && RESTART=1; done

lab_up() { curl -sf -m 3 -o /dev/null http://127.0.0.1:8788/world; }
viewer_up() { curl -sf -m 3 -o /dev/null http://localhost:63317/sim; }

if [[ "$RESTART" == 1 ]] || ! lab_up; then
  # restart.sh refuses while a teach job is training, restarts the lab
  # detached, then the viewer (--force). Its own output says which.
  bash "$RS/restart.sh"
elif ! viewer_up; then
  bash "$RS/viewer.sh"
fi
lab_up && echo "lab: up" || { echo "lab failed:"; tail -20 "$ROOT/microduck_local/lab-server.log"; exit 1; }
for _ in $(seq 1 60); do viewer_up && break; sleep 1; done
viewer_up && echo "viewer: up" || { echo "viewer failed:"; tail -20 "$ROOT/duck-viewer/viewer-server.log"; exit 1; }

# The world: a builtin (living-room, playroom, pitch-2v2, ...) or a scenario
# saved from the editor. A lab already on it is reloaded, which is a reset.
if ! curl -sf -m 120 -X POST http://127.0.0.1:8788/world/load \
      -H 'content-type: application/json' -d "{\"scenario\":\"$SCENARIO\"}" -o /dev/null; then
  echo "could not load '$SCENARIO' (GET /scenarios lists the names)"; exit 1
fi
echo "world: $(curl -s -m 5 http://127.0.0.1:8788/world | head -c 70)…"
