"""The gate that would have caught twelve commits of broken history.

`development` carried four Python files that did not parse — `world/arena.py`,
`brain/team.py`, `brain/controllers.py`, `walk_env.py` — for at least twelve
commits (measured 2026-09-08). Every one was fine in the working tree, so the
tests, the batteries and `precommit.sh` were all green while `git show HEAD:`
gave an IndentationError and a fresh clone could not import the package.

`scripts/check_staged_python.py` closes that by reading the INDEX — what the
next commit will actually write — rather than the working tree.
"""

import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from check_staged_python import blob, main, offenders, staged_python  # noqa: E402

BROKEN = '''\
class World:
    def kickoff(self):
        q = 1
        self.data.qpos[q] = 0.0
            self.kickoff_team = self.team_defending(side)
        return q
'''
FINE = "def f():\n    return 1\n"


def test_offenders_names_the_file_the_line_and_the_error():
    """The real shape of the corruption: a stray line at the wrong indent in
    the middle of an otherwise plausible function."""
    files = {"a.py": FINE, "world/arena.py": BROKEN, "c.py": "def g():\n    pass\n"}
    bad = offenders(sorted(files), files.__getitem__)
    assert [p for p, _, _ in bad] == ["world/arena.py"]
    path, line, msg = bad[0]
    assert line == 5 and "indent" in msg.lower()
    # …and a clean set is silent.
    assert offenders(["a.py", "c.py"], files.__getitem__) == []


def test_a_file_that_is_not_decodable_is_reported_not_raised():
    def read(_):
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
    bad = offenders(["x.py"], read)
    assert len(bad) == 1 and bad[0][1] is None and "decodable" in bad[0][2]


def _git(repo, *args, **kw):
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True,
                          check=True, **kw).stdout


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "r"
    (r / "pkg").mkdir(parents=True)
    _git(r.parent, "init", "-q", str(r))
    _git(r, "config", "user.email", "t@t")
    _git(r, "config", "user.name", "t")
    (r / "pkg" / "ok.py").write_text(FINE)
    _git(r, "add", "-A")
    _git(r, "commit", "-qm", "first")
    return r


def test_the_gate_reads_the_index_and_not_the_working_tree(repo):
    """The whole point: a file that is FINE on disk and BROKEN as staged is
    what slipped through for twelve commits, and it is caught here."""
    f = repo / "pkg" / "ok.py"
    f.write_text(BROKEN)
    _git(repo, "add", "pkg/ok.py")                       # the broken version is staged…
    f.write_text(FINE)                                   # …and the tree is put back: tests would pass
    assert f.read_text() == FINE
    bad = offenders(staged_python(cwd=repo), lambda p: blob(p, cwd=repo))
    assert [p for p, _, _ in bad] == ["pkg/ok.py"]
    # Staging the tree that was actually tested clears it — a fixing commit is
    # never blocked by the state of HEAD.
    _git(repo, "add", "pkg/ok.py")
    assert offenders(staged_python(cwd=repo), lambda p: blob(p, cwd=repo)) == []


def test_paths_resolve_from_the_repo_root_however_deep_it_runs(repo):
    """`git ls-files` prints paths relative to the CWD and `git show :path`
    wants them from the root, so the gate asks for --full-name — without it it
    died on the first file when run from a subdirectory, which is where
    precommit.sh runs it."""
    got = staged_python(cwd=repo / "pkg")
    assert got == ["pkg/ok.py"]
    assert blob(got[0], cwd=repo / "pkg") == FINE


def test_the_gate_is_wired_into_precommit():
    pre = (Path(__file__).resolve().parents[1] / "scripts" / "precommit.sh").read_text()
    assert "check_staged_python.py" in pre


def test_main_exits_zero_on_this_repo_or_names_what_is_broken(capsys):
    """Run for real. Green means every staged .py parses; red is a true report
    about this checkout, and the message has to say so unambiguously."""
    rc = main()
    out = capsys.readouterr()
    assert rc in (0, 1)
    assert ("staged python ok" in out.out) if rc == 0 else ("do NOT parse" in out.err)
