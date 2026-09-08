"""record-world: the world-mode recorder runs a builtin scenario headlessly,
under the lab's own WorldState, and leaves the three files an agent reads."""

import sys

import pytest

from microduck_local import record_world


@pytest.mark.parametrize("scenario,camera,brain", [
    ("living-room", "top", []),
    ("playroom", "follow:d0", ["d0=tidy"]),
])
def test_record_world_writes_video_sheet_and_events(tmp_path, monkeypatch, scenario, camera, brain):
    out = tmp_path / "rw"
    argv = ["record-world", scenario, "--seconds", "1.0", "--out", str(out), "--camera", camera,
            "--width", "160", "--height", "120", "--sheet-frames", "4", "--stride", "10"]
    for b in brain:
        argv += ["--brain", b]
    monkeypatch.setattr(sys, "argv", argv)
    record_world.main()
    assert (out / "world.mp4").stat().st_size > 0
    assert (out / "sheet.png").stat().st_size > 0
    events = (out / "events.txt").read_text()
    assert events.startswith(f"# {scenario} seed 0")
    # the first tick is a transition into each brain's initial state
    assert "d0 ->" in events
    if brain:
        assert "d0=tidy" in events.splitlines()[0]


def test_record_world_rejects_unknown_duck_and_camera(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["record-world", "living-room", "--seconds", "0.2",
                                      "--out", str(tmp_path), "--brain", "d9=wander"])
    with pytest.raises(SystemExit, match="d9"):
        record_world.main()
    monkeypatch.setattr(sys, "argv", ["record-world", "living-room", "--seconds", "0.2",
                                      "--out", str(tmp_path), "--camera", "follow:d9"])
    with pytest.raises(SystemExit, match="follow:d9"):
        record_world.main()
