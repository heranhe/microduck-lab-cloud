"""What floor can each camera actually see, in the BASE frame, at rest?

    uv run python scripts/probe_moss_camera_cover.py

Ray-casts each frustum onto z = 0 and reports the patch, plus the covered
half-width at a series of distances, against the rover's own track width.
Geometry only — no detector, no occlusion — so every number here is an UPPER
bound on what a camera change could buy.

MEASURED (2026-09-29), and it decides the "pitch the head camera down"
question:

    base x |  covered |y| at head pitch 0 / 10 / 20 / 30 deg   (tracks: 0.212)
     0.19  |  0.000   0.000   0.000   0.067
     0.22  |  0.000   0.076   0.085   0.092
     0.25  |  0.094   0.105   0.112   0.117
     0.30  |  0.141   0.151   0.157   0.157
     0.40  |  0.235   0.244   0.243   0.239

Pitching down does move the near EDGE in (x 0.247 -> 0.184 at 30 deg, and the
far field survives: the floor is still covered past 3 m until about 40 deg).
But the binding limit in the run-over zone is LATERAL, not vertical: a camera
0.156 m back from the base origin with an 87 deg lens covers a strip only
0.09 m wide at x = 0.25, against tracks 0.42 m wide, and pitch moves that by
2 cm. The rover cannot see its own front corners at any pitch. See
`probe_moss_blindspot.py` for the same conclusion measured on real contacts.

The WRIST camera at rest sits at base (+0.087, +0.052, +0.302) aiming
(-0.17, +0.90, -0.41) — out to the robot's LEFT. Its floor patch is
y +0.207..+2.960: it covers NONE of the run-over zone, so using it there is a
re-aim of the rest pose, not a free win.
"""
import math
import numpy as np
from microduck_local.robots import moss
from microduck_local.robots.moss_pinch import MossKinematics

kin = MossKinematics()
rest = dict(zip(moss.ARM_JOINTS, moss.tuck_pose()))
try:
    from microduck_local.brain.tidy_moss import TidyMossParams
    rp = TidyMossParams().rest_pose
    if rp is not None:
        rest = dict(zip(moss.ARM_JOINTS, rp))
except Exception as e:
    print("rest_pose:", e)
pc, Rc = kin.body_in_base(rest, 0.041, moss.ARM_CAMERA_BODY)
print(f"wrist camera at base ({pc[0]:+.3f}, {pc[1]:+.3f}, {pc[2]:+.3f})  "
      f"axis {Rc[:,0].round(3)}")
print(f"head  camera at base ({moss.CAMERA_POS[0]:+.3f}, {moss.CAMERA_POS[1]:+.3f}, "
      f"{moss.CAMERA_POS[2]:+.3f})  pitch {math.degrees(moss.CAMERA_PITCH_RAD):.0f} deg")

def footprint(p, R, hfov, vfov, label):
    hh, hv = math.radians(hfov) / 2, math.radians(vfov) / 2
    pts = []
    for i in range(41):
        for j in range(41):
            b = -hh + 2 * hh * i / 40
            e = -hv + 2 * hv * j / 40
            v = R @ np.array([math.cos(e) * math.cos(b),
                              math.cos(e) * math.sin(b), math.sin(e)])
            if v[2] >= -1e-6:
                continue
            s = -p[2] / v[2]
            if s > 3.0:
                continue
            q = p + s * v
            pts.append((q[0], q[1]))
    if not pts:
        print(f"  {label}: sees no floor within 3 m"); return
    xs = [q[0] for q in pts]; ys = [q[1] for q in pts]
    print(f"  {label}: floor x {min(xs):+.3f}..{max(xs):+.3f}  "
          f"y {min(ys):+.3f}..{max(ys):+.3f}  ({len(pts)} rays land)")
    # the zone the contacts come from: just ahead of the bumper, within the tracks
    hit = [q for q in pts if 0.10 < q[0] < 0.32 and abs(q[1]) < moss.HALF_WIDTH_M]
    print(f"        of which inside the RUN-OVER zone "
          f"(x 0.10-0.32, |y|<{moss.HALF_WIDTH_M}): {len(hit)}")

footprint(np.array(moss.CAMERA_POS, float),
          np.array([[math.cos(moss.CAMERA_PITCH_RAD), 0, math.sin(moss.CAMERA_PITCH_RAD)],
                    [0, 1, 0],
                    [-math.sin(moss.CAMERA_PITCH_RAD), 0, math.cos(moss.CAMERA_PITCH_RAD)]]),
          moss.CAMERA_HFOV_DEG, moss.CAMERA_VFOV_DEG, "head @ 0 deg")
for deg in (10, 20, 30, 40, 50):
    th = math.radians(deg)
    footprint(np.array(moss.CAMERA_POS, float),
              np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                        [-math.sin(th), 0, math.cos(th)]]),
              moss.CAMERA_HFOV_DEG, moss.CAMERA_VFOV_DEG, f"head @ {deg} deg down")
footprint(np.asarray(pc, float), np.asarray(Rc, float),
          moss.ARM_CAMERA_HFOV_DEG, moss.ARM_CAMERA_VFOV_DEG, "wrist at rest")

print("\nHOW WIDE a strip of floor the head camera covers at each distance —")
print("the tracks are", f"{2*moss.HALF_WIDTH_M:.3f} m wide.")
print("   base x |  covered |y| at head pitch 0 / 10 / 20 / 30 deg")
for x in (0.19, 0.22, 0.25, 0.30, 0.40, 0.60, 1.00):
    row = []
    for deg in (0, 10, 20, 30):
        th = math.radians(deg)
        R = np.array([[math.cos(th), 0, math.sin(th)], [0, 1, 0],
                      [-math.sin(th), 0, math.cos(th)]])
        hh, hv = math.radians(moss.CAMERA_HFOV_DEG)/2, math.radians(moss.CAMERA_VFOV_DEG)/2
        best = 0.0
        p = np.array(moss.CAMERA_POS, float)
        for i in range(201):
            b = -hh + 2*hh*i/200
            for j in range(201):
                e = -hv + 2*hv*j/200
                v = R @ np.array([math.cos(e)*math.cos(b), math.cos(e)*math.sin(b), math.sin(e)])
                if v[2] >= -1e-9: continue
                s = -p[2]/v[2]
                q = p + s*v
                if abs(q[0] - x) < 0.005:
                    best = max(best, abs(q[1]))
        row.append(best)
    print(f"   {x:5.2f} m | " + "  ".join(f"{v:.3f}" for v in row))
