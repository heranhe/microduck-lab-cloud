"""Where the bit-exact rollout goldens live, and for which machine.

The parity tests (test_step_perf_parity.py, test_bam_perf_parity.py) pin
rollouts to the last bit of float64. That is a property of ONE machine's
MuJoCo build, libm and numba code paths, so goldens are stored per
platform in tests/goldens/<name>-<system>-<machine>.json and a platform
without a recording SKIPS (with the command to make one) rather than
failing on numbers it never produced. Record on the platform in question:

    MICRODUCK_RECORD_GOLDENS=1 uv run --with pytest pytest tests/test_step_perf_parity.py tests/test_bam_perf_parity.py

2026-09-06: the physics-audit fixes (fresh IMU obs after the substep loop,
implicitfast / 10 / 20, mass+inertia / CoM / armature DR and velocity
pushes — walk_env.py) moved every obs byte and every DR draw, so the
recordings in tests/goldens/ predated the trajectory they pin and CI's
Linux runner failed on them (11 tests) from the moment the audit reached
main. Re-recorded 2026-09-10 on a hosted ubuntu runner by
.github/workflows/record-goldens.yml (`gh workflow run record-goldens.yml`,
then commit the artifact): a Mac cannot record the Linux file, and the
bit / tolerance split below is what makes a hosted runner a valid
recorder. Do that the day the physics moves next, not when someone pushes.

The file records the upstream model sha and the library versions it was
made against; a mismatch on either is reported first, because a golden
that moved with a model re-export (2026-09: the CAD re-export moved every
per-term sum in the 5th digit) is a recapture, not a regression.

The bits are a property of the HOST, and the CPU model string does not
name one. 2026-09-11: `ubuntu-latest` runs of the SAME tree disagreed —
run 34548996413 (7979484, a commit whose whole diff is 38 lines of
docs/roadmap.md) failed the 11 golden-bit assertions that runs
34548711420 (36a6b3d) and 34548413291 (5a2414c) had just passed, and the
differences were last-ulp: qpos off by max |Δ| 4.8e-15 (3.5e-14
relative, 1-2 ulp), the torque sum off by 1 ulp
(-0.42598059432454155 vs …094), the per-term reward sums an episode
accumulates off by up to ~2500 ulp (5.6e-13 relative) where the
trajectory amplified it. Every one of those runs reported the recorder's
own CPU model — "AMD EPYC 9V74 80-Core Processor" — so `same_cpu` was
True and the bit branch ran anyway: hosted runners share a model name
across hosts that do not share a reduction order. The tolerance branch
(`close`) passed on the same runs, which is what the bits were worth.

So the EXACT comparison — digest and float.hex — runs only where the bits
are a contract: a platform whose hardware the lab can pin, which today
means Apple Silicon (`BIT_EXACT_PLATFORMS`). Everywhere else the same
rollout is compared by TOLERANCE, in two tiers: `NEAR_RTOL` when the
machine at least reports the recorder's CPU model (1e3 above the measured
cross-host noise, 1e4 below the 5th-digit move a model re-export makes),
`RTOL` when it does not. Both are the same regression test; one of them is
pinned to the bit. `MICRODUCK_GOLDEN_BITS=1` forces the exact branch on
(re-verifying on the machine that just recorded), `=0` forces it off
(checking that the tolerant branch still catches a real change).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from microduck_local import contract as C

GOLDENS = Path(__file__).parent / "goldens"
RECORD = os.environ.get("MICRODUCK_RECORD_GOLDENS", "") not in ("", "0")
RTOL = 1e-7          # relative tolerance for the cross-CPU comparison
ATOL = 1e-9
# Same CPU model, another host: measured cross-host noise on the hosted
# ubuntu runners is <= 6e-13 relative (module docstring), a model re-export
# moves the same numbers in the 5th digit. 1e-9 sits between them.
NEAR_RTOL = 1e-9
NEAR_ATOL = 1e-9

# Platforms whose hardware the lab can pin, so the recorded BITS are a
# contract rather than one host's reduction order. Apple Silicon only:
# hosted x86 runners share a CPU model string across hosts that disagree in
# the last ulp, so a Linux recording is checked by tolerance, never by bit.
BIT_EXACT_PLATFORMS = ("darwin-arm64",)


def cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.lower().startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True, timeout=5)
            if out.stdout.strip():
                return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def platform_key() -> str:
    return f"{platform.system()}-{platform.machine()}".lower()


def provenance() -> dict:
    sha = "unknown"
    try:
        sha = subprocess.run(["git", "-C", str(C.MICRODUCK_RL_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10).stdout.strip() or "unknown"
    except Exception:
        pass
    return {"microduck_rl": sha, "mujoco": mujoco.__version__, "numpy": np.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "cpu": cpu_model()}


def same_cpu(golden: dict) -> bool:
    """True when this machine's CPU model is the one that recorded the golden.

    A model name, not a host: it is NOT enough for bit equality (see
    `bit_exact`), only enough to tighten the tolerance (see `tol`)."""
    return golden.get("provenance", {}).get("cpu", "?") == cpu_model()


def bits_are_a_contract() -> bool:
    """True where this platform's recorded bits are reproducible at all."""
    forced = os.environ.get("MICRODUCK_GOLDEN_BITS", "")
    if forced != "":
        return forced not in ("0", "no", "false")
    return platform_key() in BIT_EXACT_PLATFORMS


def bit_exact(golden: dict) -> bool:
    """True when this machine is expected to reproduce the golden's BITS —
    the recording machine's CPU, on a platform whose bits are a contract."""
    return same_cpu(golden) and bits_are_a_contract()


def tol(golden: dict) -> tuple[float, float]:
    """(rtol, atol) for the comparison that is not pinned to the bit: tight
    when this machine reports the recorder's CPU model, the cross-CPU
    tolerance when it does not."""
    return (NEAR_RTOL, NEAR_ATOL) if same_cpu(golden) else (RTOL, ATOL)


def close(a, b, what: str = "", rtol: float = RTOL, atol: float = ATOL) -> None:
    """assert a ≈ b (floats, hex strings, lists, dicts of those) within rtol/atol."""
    if isinstance(a, str):
        a = float.fromhex(a)
    if isinstance(b, str):
        b = float.fromhex(b)
    if isinstance(a, dict):
        assert set(a) == set(b), f"{what}: keys {sorted(a)} vs {sorted(b)}"
        for k in a:
            close(a[k], b[k], f"{what}.{k}", rtol, atol)
        return
    if isinstance(a, (list, tuple)):
        assert len(a) == len(b), f"{what}: length {len(a)} vs {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            close(x, y, f"{what}[{i}]", rtol, atol)
        return
    if isinstance(a, bool) or a is None:
        assert a == b, f"{what}: {a} vs {b}"
        return
    np.testing.assert_allclose(float(a), float(b), rtol=rtol, atol=atol, err_msg=what)


def path(name: str) -> Path:
    return GOLDENS / f"{name}-{platform_key()}.json"


def load(name: str) -> dict | None:
    p = path(name)
    if not p.exists():
        return None
    return json.loads(p.read_text())


def save(name: str, data: dict) -> Path:
    p = path(name)
    GOLDENS.mkdir(exist_ok=True)
    p.write_text(json.dumps({"provenance": provenance(), "data": data}, indent=1, sort_keys=True) + "\n")
    return p


def skip_reason(name: str) -> str:
    return (f"no golden for this platform ({path(name).name}); record one here with "
            f"MICRODUCK_RECORD_GOLDENS=1 pytest tests/test_{name}.py")


def check_provenance(golden: dict) -> str:
    """'' if the golden was made against what is installed now, else what differs."""
    want, have = golden.get("provenance", {}), provenance()
    diff = [f"{k}: golden {want.get(k)} / now {have[k]}" for k in ("microduck_rl", "mujoco", "numpy")
            if want.get(k) != have[k]]
    return "; ".join(diff)
