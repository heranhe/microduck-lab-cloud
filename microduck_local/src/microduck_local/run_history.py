"""A readable history of past training runs, built from the files each run leaves.

Nothing here is stored separately: a run's own `progress.jsonl`, `behavior.json`
and `teach-job.json` already say how long it trained, how far it got and what it
scored. A curriculum chain (`…-s1` … `…-s5`) is one training to the person who
asked for it, so its stages are folded into one row.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

CHAIN_RE = re.compile(r"^(?P<base>.+)-s(?P<stage>\d+)$")
TAIL_BYTES = 65536


def _tail_records(path: Path) -> tuple[dict, bool, int | float | None]:
    """(the last full progress record, whether the run said it finished, its final step count).

    A finished trainer closes the log with a bare `{"steps", "total", "done": true}` line that
    carries no time or score, so the numbers come from the last record that has them. A
    half-written last line is skipped.
    """
    try:
        with path.open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - TAIL_BYTES))
            lines = f.read().splitlines()
    except OSError:
        return {}, False, None
    full: dict = {}
    done, final_steps = False, None
    for line in reversed(lines):
        try:
            value = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            continue
        if not isinstance(value, dict):
            continue
        if value.get("done") is True and not done:
            done, final_steps = True, value.get("steps")
        if "elapsed_s" in value or "ep_rew" in value:
            full = value
            break
    return full, done, final_steps


def _read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _crashed(run: Path) -> bool:
    """A trainer that died with a traceback, as opposed to one that was stopped."""
    try:
        with (run / "train.log").open("rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 4096))
            return b"Traceback (most recent call last)" in f.read()
    except OSError:
        return False


def _started_at(run: Path) -> float:
    st = run.stat()
    return float(getattr(st, "st_birthtime", None) or st.st_ctime)


def describe_run(run: Path, live: frozenset[str] | set[str] = frozenset()) -> dict | None:
    """One run's row, or None when it never wrote progress (nothing to report)."""
    progress = run / "progress.jsonl"
    if not progress.is_file():
        return None
    last, finished, final_steps = _tail_records(progress)
    job = _read_json(run / "teach-job.json")
    behavior = _read_json(run / "behavior.json")
    servo = (job.get("extraEnv") or {}).get("MICRODUCK_SERVO") or (job.get("stageEnv") or {}).get("MICRODUCK_SERVO")
    if run.name in live:
        status = "training"
    elif finished:
        status = "done"
    elif _crashed(run):
        status = "failed"
    else:
        status = "stopped"
    elapsed = last.get("elapsed_s")
    return {
        "run": run.name,
        "behavior": behavior.get("behavior"),
        "status": status,
        "startedAt": _started_at(run),
        "finishedAt": progress.stat().st_mtime,
        "durationS": float(elapsed) if isinstance(elapsed, (int, float)) and elapsed >= 0 else None,
        "steps": (final_steps if finished else last.get("steps")) if isinstance(final_steps if finished else last.get("steps"), (int, float)) else None,
        "total": last.get("total") if isinstance(last.get("total"), (int, float)) else None,
        "score": last.get("ep_rew") if isinstance(last.get("ep_rew"), (int, float)) else None,
        "sps": last.get("sps") if isinstance(last.get("sps"), (int, float)) else None,
        "servo": servo if servo in ("feetech", "official") else None,
        "hasPolicy": (run / "policy.onnx").is_file(),
    }


def _fold_chain(stages: list[dict]) -> dict:
    """A curriculum chain as one row: summed time and steps, the last stage's score and outcome."""
    stages = sorted(stages, key=lambda r: r["_stage"])
    last = stages[-1]
    durations = [s["durationS"] for s in stages]
    status = last["status"]
    if any(s["status"] == "failed" for s in stages) and status != "training":
        status = "failed"
    row = {k: v for k, v in last.items() if not k.startswith("_")}
    row.update(
        status=status,
        startedAt=min(s["startedAt"] for s in stages),
        durationS=None if any(d is None for d in durations) else float(sum(durations)),
        steps=None if any(s["steps"] is None for s in stages) else sum(s["steps"] for s in stages),
        stages=len(stages),
        chain=last["_base"],
    )
    return row


def run_history(runs_dir: Path, *, live: frozenset[str] | set[str] = frozenset(), limit: int = 40) -> list[dict]:
    """Newest first. Chains fold into one row; a missing number stays None, never 0."""
    try:
        dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
    except OSError:
        return []
    singles: list[dict] = []
    chains: dict[str, list[dict]] = {}
    for d in dirs:
        row = describe_run(d, live)
        if row is None:
            continue
        m = CHAIN_RE.match(d.name)
        if m and (runs_dir / f"{m['base']}-s1").is_dir() or (m and any(
                (runs_dir / f"{m['base']}-s{n}").is_dir() for n in range(1, 9) if n != int(m["stage"]))):
            row["_base"], row["_stage"] = m["base"], int(m["stage"])
            chains.setdefault(m["base"], []).append(row)
        else:
            singles.append(row)
    rows = singles + [_fold_chain(v) for v in chains.values()]
    rows.sort(key=lambda r: r["finishedAt"], reverse=True)
    return rows[: max(1, min(int(limit), 200))]
