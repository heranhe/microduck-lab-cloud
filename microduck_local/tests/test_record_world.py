"""record-world: the world-mode recorder runs a builtin scenario headlessly,
under the lab's own WorldState, and leaves the three files an agent reads."""

import os
import platform
import sys

import pytest

from microduck_local import record_world


def _offscreen_gl() -> str | None:
    """Why this machine cannot render offscreen, or None if it can. The
    recorder builds a `mujoco.Renderer`, and on a headless runner that is
    not a test failure: macOS raises CGLError("invalid pixel format"), and
    Linux with no display and no MUJOCO_GL ABORTS the interpreter from
    inside GLFW (2026-09-10, both CI runners) - so the Linux check must
    run before mujoco is asked for a context at all."""
    if platform.system() == "Linux" and not os.environ.get("DISPLAY") \
            and os.environ.get("MUJOCO_GL", "") not in ("egl", "osmesa"):
        return "Linux with no DISPLAY and MUJOCO_GL unset (set MUJOCO_GL=egl for offscreen)"
    try:
        import mujoco
        m = mujoco.MjModel.from_xml_string(
            '<mujoco><worldbody><geom type="plane" size="1 1 0.1"/></worldbody></mujoco>')
        r = mujoco.Renderer(m, 16, 16)
        r.update_scene(mujoco.MjData(m))
        r.render()
        r.close()
    except Exception as e:  # noqa: BLE001 - any GL failure means "no offscreen context here"
        return f"no offscreen GL context: {type(e).__name__}: {e}"
    return None


_NO_GL = _offscreen_gl()
pytestmark = pytest.mark.skipif(_NO_GL is not None, reason=f"record-world renders video; {_NO_GL}")


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
