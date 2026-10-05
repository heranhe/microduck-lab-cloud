"""BLOCK until the lab's teach job finishes, then REPORT on what it produced.

    uv run python scripts/watch_training.py [host:port] [seeds]

Run it in the background: it exits when the run is done and the report is
printed, which is what re-invokes the agent. That closes the loop — launch,
wait, measure, adjust, relaunch — instead of a human having to ask "is it
done yet" and an agent re-deriving the same numbers by hand each time.

It exports the finished run (the ONNX bakes the normalizer in; a raw
checkpoint is never the thing to judge) and then prints the standard report
plus the per-term reward budget, because on 2026-09-25 a reward term silently
fell to 8% of its former value and two full runs trained on it unnoticed.
"""
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ADDR = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8788"
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 40
#: Optional: follow ONE job, by run-name prefix (a chain's base name, without
#: -sN). The lab runs jobs CONCURRENTLY and `/teach/status`'s "job" is only the
#: newest — a watcher following it silently switched to a second job launched
#: later and would never have reported the first (2026-09-26).
PREFIX = sys.argv[3] if len(sys.argv) > 3 else None
POLL_S = 20.0
HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs"


def status() -> dict:
    with urllib.request.urlopen(f"http://{ADDR}/teach/status", timeout=8) as r:
        return json.loads(r.read().decode())


def main() -> None:
    last = None
    seen_running = False
    while True:
        try:
            st = status()
        except Exception as e:                       # lab restarted mid-run
            print(f"[watch] lab unreachable ({e}); retrying", flush=True)
            time.sleep(POLL_S)
            continue
        job = st.get("job") or {}
        running = st.get("running")
        if PREFIX:
            mine = [j for j in (st.get("jobs") or [])
                    if str(j.get("runName", "")).startswith(PREFIX)]
            job = mine[-1] if mine else {}
            running = job.get("status") == "training"
        run = job.get("runName")
        if running:
            seen_running = True
            cur = (run, job.get("steps"))
            if cur != last:
                print(f"[watch] {run}  {job.get('steps')}/{job.get('total')}"
                      f"  stage {job.get('stage')}/{job.get('stages')}", flush=True)
                last = cur
            time.sleep(POLL_S)
            continue
        if not seen_running:
            print("[watch] nothing is training", flush=True)
            return
        print(f"[watch] FINISHED: {run} ({st.get('status')})", flush=True)
        break

    # The chain's last stage is what the run produced; a staged job names it
    # <base>-sN and `runName` already points at the one that just ended.
    d = RUNS / str(run)
    if not d.is_dir():
        print(f"[watch] no run dir at {d}")
        return
    subprocess.run(["uv", "run", "export-walk", str(d)], cwd=HERE.parent,
                   check=False)
    print()
    # WHICH REPORT depends on the TASK. Pointing `pick_report.py` at a stow
    # policy evaluates it in the pick env and prints "0/40 picked", which is
    # not a result — it is the wrong environment, and it is what this loop did
    # on 2026-09-25 until someone read it carefully.
    task = "pick"
    try:
        task = (json.loads((d / "run.json").read_text()).get("task") or "pick")
    except (OSError, ValueError):
        pass
    reports = (("stow_report.py",) if task == "stow"
               else ("pick_report.py", "reward_budget.py"))
    for script in reports:
        p = HERE / script
        if not p.is_file():
            continue
        print(f"----- {script} -----", flush=True)
        subprocess.run(["uv", "run", "python", str(p),
                        str(d / "policy.onnx"), str(SEEDS)],
                       cwd=HERE.parent, check=False)
        print(flush=True)


if __name__ == "__main__":
    main()
