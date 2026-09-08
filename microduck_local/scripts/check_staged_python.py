#!/usr/bin/env python
"""Every Python file ABOUT TO BE COMMITTED must parse.

`precommit.sh` compiles the WORKING TREE, which is the file you just edited
and the one the tests import. It has never checked what is actually going
into the commit, and on 2026-09-08 that gap was measured: four files in
`development`'s committed history — `world/arena.py`, `brain/team.py`,
`brain/controllers.py`, `walk_env.py` — had not parsed for at least twelve
commits. Every one of them was fine in the working tree, so every test run,
every battery and every `precommit.sh` was green while `git show HEAD:` gave
an `IndentationError`. A clone of the branch could not import the package.

How it happens: an anchored text edit is applied to the file AND, separately,
to the committed version so that only that hunk is staged (the shared-checkout
rule — another session edits this tree). When the two texts have drifted, the
anchor lands somewhere else in the committed copy and strands a fragment: a
`return` tail after a class constant, an indented block whose `if` header is
gone. The result stages cleanly, diffs plausibly, and never runs.

So this reads the INDEX — what `git commit` will write — not HEAD and not the
working tree, which is also what lets the commit that FIXES a broken file
through. Exit 1 and name the file, the line and the error.

    uv run python scripts/check_staged_python.py
"""

from __future__ import annotations

import ast
import subprocess
import sys


def staged_python(cwd: str | None = None) -> list[str]:
    """Paths of the .py files in the index (staged for the next commit)."""
    # --full-name: paths from the REPO ROOT, so `git show :path` resolves the
    # same entry however deep in the tree this is run from (it is run from
    # microduck_local/, where a bare ls-files path is relative to the cwd and
    # `git show :` then cannot find it).
    out = subprocess.run(["git", "ls-files", "--cached", "--full-name", "--", "*.py"],
                         capture_output=True, text=True, check=True, cwd=cwd).stdout
    return [ln for ln in out.splitlines() if ln.strip()]


def blob(path: str, cwd: str | None = None) -> str:
    """The file's content AS STAGED — `git show :path`, not the working tree."""
    return subprocess.run(["git", "show", ":" + path], capture_output=True, text=True,
                          check=True, cwd=cwd).stdout


def offenders(paths: list[str], read) -> list[tuple[str, int | None, str]]:
    """(path, line, message) for every staged file that does not parse."""
    bad = []
    for p in paths:
        try:
            ast.parse(read(p), filename=p)
        except SyntaxError as e:
            bad.append((p, e.lineno, e.msg))
        except UnicodeDecodeError as e:                       # a binary blob under a .py name
            bad.append((p, None, f"not decodable as text: {e}"))
    return bad


def main() -> int:
    paths = staged_python()
    bad = offenders(paths, blob)
    if not bad:
        print(f"staged python ok ({len(paths)} files)")
        return 0
    print(f"{len(bad)} staged Python file(s) do NOT parse — this commit would break the branch:",
          file=sys.stderr)
    for p, line, msg in bad:
        print(f"  {p}:{line if line is not None else '?'}  {msg}", file=sys.stderr)
    print("\nThe working tree may well be fine; it is the STAGED copy that is broken. "
          "Re-stage the file from the tree you actually tested (`git add <path>`), or fix the hunk.",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
