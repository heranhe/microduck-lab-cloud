"""Which knobs need ANOTHER knob set before they can do anything.

This module exists because the same mistake has now cost two batteries in
this repo, from two different sessions and two different mechanisms:

* `contest_margin=0.15` (2026-09-09) ran a full 16-seed contested gym and
  reproduced the baseline EPISODE FOR EPISODE. The rule is gated on
  `use_color`, which ships `False`, so the arm silently ran the shipped path
  and reported a clean, meaningless null.
* `gaze_bearing_max` (2026-09-07) was shadowed by a hard-coded `0.6` in the
  very gate it names, and cost a whole arm the same way.

Both are the same failure in the end: **an arm that measured nothing and said
so in the language of a result.** `kick_gym.is_identical` catches it after the
fact, empirically, whatever the mechanism — but only once the compute is
already spent. This catches the commonest form of it in about a second, from
the source, BEFORE the battery runs.

It reads `controllers.py` with `ast` and looks for `and`-conditions that
mention two knobs: if acting on X requires Y, and Y's default is falsy, then
setting X alone is a no-op. That is a real gate, not a guess — `use_color`
falls out of it without being named here, and so does anything added later.

It is deliberately a WARNING and not an error. A conjunction is evidence, not
proof (a knob may act on several paths, only one of which is gated), and this
must never block a legitimate arm. `is_identical` remains the ground truth.
"""

from __future__ import annotations

import ast
import functools
import pathlib
from dataclasses import fields

from .controllers import ChaseParams

_SOURCE = pathlib.Path(__file__).with_name("controllers.py")
# The receivers a `ChaseParams` is bound to in this package: `p.x`, `self.p.x`,
# `params.x`.  A name outside this set is some other object's attribute.
_RECEIVERS = frozenset({"p", "params"})


def _knob_names() -> frozenset[str]:
    return frozenset(f.name for f in fields(ChaseParams))


def _knob_of(node: ast.AST, knobs: frozenset[str]) -> str | None:
    """`p.X` / `self.p.X` / `params.X` -> `"X"`, else None."""
    if not isinstance(node, ast.Attribute) or node.attr not in knobs:
        return None
    v = node.value
    if isinstance(v, ast.Name) and v.id in _RECEIVERS:
        return node.attr
    if isinstance(v, ast.Attribute) and v.attr in _RECEIVERS:
        return node.attr
    return None


def _mentioned(node: ast.AST, knobs: frozenset[str]) -> set[str]:
    return {k for n in ast.walk(node) if (k := _knob_of(n, knobs))}


@functools.lru_cache(maxsize=4)
def co_gates(source: str | None = None) -> dict[str, frozenset[str]]:
    """knob -> the knobs that gate it, anywhere in `controllers.py`.

    Two things make this a real answer rather than a co-occurrence count:

    * **Gating is directional.** In `p.use_color and p.opp_keepout > 0`, the
      GATE is the bare truthiness test (`p.use_color`); `p.opp_keepout > 0` is
      a comparison and is the thing being gated. Reading the pair
      symmetrically would warn that `use_color` needs `opp_keepout`, which is
      backwards — `use_color` has plenty of effects of its own.
    * **A gate must hold on EVERY path.** So the gates of a knob are the
      INTERSECTION over every `and`-condition mentioning it, not the union. A
      knob used in one gated place and one ungated place is not disabled by
      leaving the gate off, and must not be warned about. This is what keeps
      `lost_s` — read all over the file — out of the report while keeping
      `contest_margin`, whose only conjunction carries `use_color`."""
    tree = ast.parse(source if source is not None else _SOURCE.read_text())
    knobs = _knob_names()
    seen_any: set[str] = set()
    gates: dict[str, set[str]] = {}
    for n in ast.walk(tree):
        if not (isinstance(n, ast.BoolOp) and isinstance(n.op, ast.And)):
            continue
        bare = {k for v in n.values if (k := _knob_of(v, knobs))}
        subjects: set[str] = set()
        for v in n.values:
            subjects |= _mentioned(v, knobs)
        for k in subjects:
            here = bare - {k}
            seen_any.add(k)
            gates[k] = here if k not in gates else (gates[k] & here)
    return {k: frozenset(v) for k, v in gates.items() if v}


def missing_gates(spec: str, source: str | None = None) -> dict[str, frozenset[str]]:
    """For a `MICRODUCK_CHASE` spec, the knobs it sets that are gated behind a
    knob it did NOT set and whose default is falsy — i.e. the arms that will
    measure the shipped path and report a null about nothing.

    Only falsy defaults are reported: a gate that ships ON (`lost_s`, say) is
    already satisfied and is not a trap."""
    spoken = ChaseParams.env_names(spec)
    if not spoken:
        return {}
    base = ChaseParams()
    gates = co_gates(source)
    out: dict[str, frozenset[str]] = {}
    for knob in sorted(spoken):
        need = {g for g in gates.get(knob, ())
                if g not in spoken and not getattr(base, g, True)}
        if need:
            out[knob] = frozenset(need)
    return out


def warning_for(spec: str, source: str | None = None) -> str | None:
    """The line to print before spending a battery, or None if the spec is
    clean. Phrased as the fix, because the reader is about to run something."""
    miss = missing_gates(spec, source)
    if not miss:
        return None
    lines = ["!! this arm may measure NOTHING: knob(s) gated behind a knob it does not set."]
    for knob, need in miss.items():
        gates = ", ".join(f"{g}={_enabling(g)}" for g in sorted(need))
        lines.append(f"   {knob} is gated on {', '.join(sorted(need))} "
                     f"(ships off) — set {gates} alongside it, or the shipped path runs.")
    lines.append("   A knob that changes nothing is BROKEN, not null (playbook rule 0).")
    return "\n".join(lines)


def _enabling(gate: str) -> str:
    """What to set a falsy-default gate to, in `MICRODUCK_CHASE` syntax."""
    return "1" if isinstance(getattr(ChaseParams(), gate, False), bool) else "<non-zero>"
