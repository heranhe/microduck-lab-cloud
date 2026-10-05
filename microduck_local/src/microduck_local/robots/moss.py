"""MOSS — Show Robotics' litter-picking rover, as a body this lab can drive.

MOSS is a ~28 cm tracked rover with a bin on its back and an SO-101-derived
arm carrying a NormaCore parallel gripper, built in public by Laurent Genoud
(https://github.com/metrox-eth/moss). It is the FOURTH body here and the
first that arrived as a finished MJCF rather than a URDF: on 2026-09-24 he
published `live/` in
[metrox-eth/moss-jev](https://github.com/metrox-eth/moss-jev) — the CPU
MuJoCo simulator his recorded pick-and-drop missions run in, `moss_robot.xml`
and 57 meshes included.

**What is his, and what is ours.** Everything in `ASSETS` is downloaded
byte-for-byte at the pinned sha and never edited on disk, so the cache stays
verifiable against the manifest. The rewrites this lab needs happen in
MEMORY, on the way into a spec (`robot_spec`), and there are three:

1. **A heading.** His base is one `base_x` slide on a rail 0.8 m long, which
   he says himself: "Base movement is currently simplified, so it isn't a
   full tracked-drive simulation yet." A room needs (x, y, yaw), so
   `add_planar_base` adds the other two. They take MARS's joint NAMES and
   order (`robots/mars.BASE_JOINTS`) because `robots/mars_drive.py`'s
   velocity PD is the controller that drives them, and a body that spelled
   them differently would need a driver of its own for no reason.
2. **No base servo.** His `base_x` position actuator (kp 3000, ctrlrange
   0-0.8) is dropped. A planar base here is PUSHED through `xfrc_applied`,
   and a position servo left on one of the three DoFs would fight the drive
   on that axis alone — the robot would steer, and then creep back onto his
   rail.
3. **No rail, no floor drag.** `range="0 .8"` and `damping="40"` go with the
   actuator. Both were properties of his RAIL rather than of the rover (40
   N*s/m is 24 N of drag at 0.6 m/s, his own top track speed), and gains
   tuned against them would be tuned against furniture.

Nothing here touches the arm, the gripper, the contact model or the meshes:
his grasp is the half that already works, and this file is the half he asked
for help with.

**Measured, on this MJCF at the pinned sha** (`scripts/probe_moss.py`):

    nq 8 -> 10 with the planar base    nbody 11   ngeom 209   nmesh 57
    total mass 6.959 kg, of which the rover BODY is 6.139 kg  <- see below
    widest horizontal extent at HOME 0.519 m -> stage pitch 1.83 m
    tcp reach: 17.6% of random arm poses get below 7 cm (a can on the floor),
      3.5% put the tcp over the bin above the rim with a held can clearing
      it, self-collision-free — the drop target is wide, not a needle
    jaws 16 mm shut, 98 mm open; his can is 66 x 115 mm, 18 g
    ~24k-41k physics steps/s single env on an M-series Mac (2 ms step)

**The shipped base mass is a placeholder and it is load-bearing here.** The
arm's seven links carry real per-link `fullinertia` (0.820 kg all told,
SO-101's own). The chassis ships an explicit `<inertial>` of 6.138672 kg,
which his exporter froze from a density composite of unmassed boxes — solid
PETG, which a printed shell is nothing like. It matters more for DRIVING
than for grasping: `mars_drive`'s velocity PD is an explicit force loop,
stable only while `KP*dt/M < 2`, so every gain this lab picks is picked
against that number. Laurent weighed the rover at 3.5 kg on 2026-09-24 and
`set_base_inertial` is where it goes in.

Licences travel with the download (`MOSS_LICENCE`): Laurent's rover geometry
is CC BY 4.0, the SO-101 extract and the NormaCore gripper are Apache-2.0
(TheRobotStudio and norma-core respectively), and his simulator code is
Apache-2.0. Attribution does not imply endorsement.

**Untested on hardware, and differently so than MARS.** Nothing in this repo
has ever driven a MOSS, and the rover in the photographs has driven for two
days. This model is his September 2026 demonstration model — his own README
says it is "not the V0.4 manufacturing CAD or a calibrated digital twin".
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import urllib.request
from functools import lru_cache
from pathlib import Path

import mujoco
import numpy as np

from .body import BodyBase, RobotFrames
from .policy_contract import declare

_ROOT = Path(__file__).resolve().parents[3]

# ------------------------------------------------------------- provenance
#
# metrox-eth/moss-jev at the sha `live/` was published under (2026-09-24). A
# revision is a different robot: the hull boxes, the bin walls, the finger
# pads and the home keyframe are all his, hand-built for those missions, so
# the sha travels with the manifest.
MOSS_JEV_SHA = "be395a8076b156d1983dbf847601cd63b836b19c"
MOSS_LICENCE = ("CC BY 4.0 (MOSS rover geometry, Laurent Genoud / Show "
                "Robotics) + Apache-2.0 (SO-101 extract, NormaCore gripper, "
                "simulator code)")
RAW_BASE = ("https://raw.githubusercontent.com/metrox-eth/moss-jev/"
            f"{MOSS_JEV_SHA}/live/model")

#: HOW FAR THE ROVER STICKS OUT BEHIND ITS OWN ORIGIN, m. MEASURED off the
#: MJCF (2026-09-29): the true AABB corners of every hull, bin and track geom,
#: arm excluded, in the base frame — x from -0.231 to +0.186, y +-0.212. The
#: BIN is what makes it 0.231 rather than the hull's 0.179: it overhangs the
#: back. Needed by anything that reverses, because MOSS's scanner looks
#: FORWARD and the rear is checked against the map rather than seen.
#: (`geom_rbound` reads 0.320 here — a bounding SPHERE around a long track is
#: not a rear extent.)
REAR_EXTENT_M = 0.231

#: ...AND IN FRONT, m, from the same AABB sweep: the front bumper is +0.186
#: and the widest point (the tracks) is +-0.212. Anything the camera reports
#: nearer than `FRONT_EXTENT_M` is the robot itself; anything past it inside
#: `HALF_WIDTH_M` of the centre line is in the way of the tracks. Needed by
#: `brain/tidy_moss.py`'s `near_min_x`, which has to tell "litter I am about
#: to drive over" from "my own chassis" and had been throwing away both.
FRONT_EXTENT_M = 0.186
HALF_WIDTH_M = 0.212

#: Where `fetch()` puts them. `MICRODUCK_MOSS_DIR` moves the cache, the way
#: `MICRODUCK_MARS_DIR` and `MICRODUCK_G1_DIR` do, so a scratch checkout can
#: verify without touching the main one's download.
CACHE_DIR = Path(os.environ.get("MICRODUCK_MOSS_DIR")
                 or _ROOT / ".cache" / "moss")
#: His own layout, kept: `moss_robot.xml` declares `meshdir="meshes"`, so the
#: STLs must sit in a `meshes/` directory beside it or nothing resolves.
ASSET_SUBDIR = "model"

#: (path under `model/`, sha256). 119 files, 20.4 MB — his robot plus the
#: **V0.4 visual overlay** he published on 2026-09-24 (`visuals_v04/`).
#: Measured from the pinned revision, so a moved branch, a truncated transfer
#: or a proxy's error page is refused rather than compiled.
ASSETS: tuple[tuple[str, str], ...] = (
    ("moss_robot.xml",
     "da29c056fd34295a300a67ff327c6292b891cea36ee4c1b137e64b78da5ee52c"),
    ("mesh_manifest.json",
     "6ee5013f4303b95957e30ecc95b176369ae4463c6292b76f74d58bb87ea9397a"),
    ("meshes/base_link_0.stl",
     "76cc960718a7f53f82ba6e90c229f87c2c8677102c37a40b3402b0c844bfea89"),
    ("meshes/base_link_1.stl",
     "208cae1647fb90a80fb161bca94418b0bb67c826f4ab1dab5a5c879552ecda98"),
    ("meshes/base_link_2.stl",
     "819fbee3b0a2eb6a64845fae8b4dae6363e608f94aabbba46ec8215c6ba2bfdf"),
    ("meshes/base_link_3.stl",
     "9c9b601f79b5b31f60458fc118c720baf573251c305081c5dd2fd1a67ad6bf9b"),
    ("meshes/body_1.stl",
     "9eca0c193a4057454723d6ed38b6e406b02e7da24c71fe28b89451b8513c1273"),
    ("meshes/body_10.stl",
     "41d352d97520b9ec17123d68b6882333ebdb795dcf8239d69a811f82dc8486bb"),
    ("meshes/body_11.stl",
     "f1d4574e0336c5a77b30e9fdec684f346ebbdb26a93af244b8f890d5a8916d96"),
    ("meshes/body_12.stl",
     "ee852deed7cfff2892cc3757e047374d6ecc00594826e4e9051496a02f947d28"),
    ("meshes/body_13.stl",
     "873cccc39bfa1f41209c38b2c0553542649b4da1b42698af68a338d087830d61"),
    ("meshes/body_14.stl",
     "3b9a37dd4340a6472737e7da90f9e74aa92d3b080e6bbda7e627f96833c9abce"),
    ("meshes/body_15.stl",
     "b0a62f5897a0d6c2bd6b0be07966bff1cda101f27110f4944e54221654ffe630"),
    ("meshes/body_16.stl",
     "9514d7ea7ac0e33f5a779077eaab7b826b0b0f6ffabe5690330a37c821fbeca1"),
    ("meshes/body_17.stl",
     "97696533f426409e5376c15c498eb6a96612b98bdd8eb61a4789932a2799a4d1"),
    ("meshes/body_18.stl",
     "55f41fff8b1fa1a8e86d4a993d44064f75edec0d912f838a577121621c0ad518"),
    ("meshes/body_19.stl",
     "176a0367bc0355b9bdb3abe627a4f63d3cde5b420b985202dbb10f2225c6159e"),
    ("meshes/body_2.stl",
     "c3f47372af99844c761df5f4192ab50f4a204b60afb251e16b752167fcae0a8a"),
    ("meshes/body_20.stl",
     "0ea82a178e41220efc54d3f7b05af93e18b0b947317b003a8031a2221ce94cc4"),
    ("meshes/body_21.stl",
     "2aae39992379c2fef358696e0b35400622aa463a9cf8c5263834d8910285eaa2"),
    ("meshes/body_22.stl",
     "46b1ba13e54eee0277997546546eddb57723b997a486875c7b7f6a55180a043e"),
    ("meshes/body_23.stl",
     "20e790984a8b16c3858bd569e6c6092ef937fcba745ae1ee7535ec8c5699b9a5"),
    ("meshes/body_24.stl",
     "6cb1af800b80af96243825eafb9fdad7fb01994bad0eb92fa83e57eecdeb836a"),
    ("meshes/body_25.stl",
     "f408104107f884da65a1acc1622a5840511daa9a3b100ec6aebfa7e8ff94802e"),
    ("meshes/body_26.stl",
     "2ce572f99afc6a5b8b190e7cf90ef104b9e2b666e62710b1beb06a0231ddf49f"),
    ("meshes/body_27.stl",
     "a9a57fcbd43ef4c440fa85a05785c7d7433e2e7bebaa6068152594c950f00d54"),
    ("meshes/body_28.stl",
     "1a4a56ffa713cf40075d33a937ad19f599ba3ab9392dc9f907e30ef23e1b0ce3"),
    ("meshes/body_31.stl",
     "7e8f964fcdf76bf2e2f507931f8c949ac5b011836ef6e49a7dc1a4b8ef30015f"),
    ("meshes/body_32.stl",
     "1b41e69a966d6ab8f064f746d706b11499a704baad2e01ee16475ba0fb713a7f"),
    ("meshes/body_35.stl",
     "0f1175fc9284010fee3be52caaab9a0188e7e52b79270b2016ba897466daa36d"),
    ("meshes/body_36.stl",
     "4c70962a7b026073b83c3628eb89677b88b4deb3c2b2705bcb807b1f60da15f2"),
    ("meshes/body_4.stl",
     "7b10bf86b4d82f64ee226c9986947bd742171cd869967c641333168e60e7463d"),
    ("meshes/body_44.stl",
     "0bbbac71de4183ba5300ca21bbc9d836d8fef4a4bfc9b334cc6f86546cb054d0"),
    ("meshes/body_45.stl",
     "89c9b208db97f72f4d4d5c4708f34c4a6f66a7861ddf3e70669416b945d6b43a"),
    ("meshes/body_50.stl",
     "fd29c18bf9c0e64eb422156efd2ea9133d228429724be42d632d231d1a1d0b40"),
    ("meshes/body_51.stl",
     "e0d35c7de84ee0e81415abac55ef5ab7a9f02f3e29d9b27febc3c72fde05b3b0"),
    ("meshes/body_55.stl",
     "db2da5e7e9591fb78869142ce3baffe1b4b03e87363b87d9e47f08ceae06f9ff"),
    ("meshes/body_56.stl",
     "ecfbe73948f2c386de9dd1ad705af898811db45182cef1febaac1bf344bbc877"),
    ("meshes/body_9.stl",
     "e02180267001a1133770c45d3c60f2eee520b2551a80442554729d0e63f76eba"),
    ("meshes/grip_132.stl",
     "a7f6a87686bd80261ae1febdae3fbf50eaee2a308a6b81dc9926fce94a76cde7"),
    ("meshes/grip_133.stl",
     "55fc492f75eb688aa461864038b72bf896e5175857268d41bff298b53bb6494c"),
    ("meshes/grip_134.stl",
     "48d0bdb0b3d7c05be692e36e0c380504be6325c3556093a75b209a7ceac6788d"),
    ("meshes/grip_135.stl",
     "8cc8472644e34b31fdd3f706914410d64e39cc1d95d70a050abf12c2b2ab6fc0"),
    ("meshes/grip_136.stl",
     "8cf7c5bed87079f0e107a7777f0e4f5784a54a2da364ad2962d7ba77abe750b4"),
    ("meshes/grip_137.stl",
     "4e9ab8caad43a281a3b58e2c65f18ddebe10d061138cdaf9f07639a5300f9dc5"),
    ("meshes/grip_141.stl",
     "7491783a85eba8c6d1ed3ae56849ffdc7abfdc126c7c21c8b32371fda1b9c3d4"),
    ("meshes/grip_144.stl",
     "76b93cf3ba386a813065116004e386b8d01c8d03aebdc19191ca864f2f1880c2"),
    ("meshes/grip_159.stl",
     "c94426ea4857a1a0ddda3e9941994d2be548b4080b3c439148ab9d1246d07da5"),
    ("meshes/grip_160.stl",
     "4d6d15ff101539de870ccf057194a3c4b089747fa4d92f7c916a74abc0867c68"),
    ("meshes/lower_arm_link_0.stl",
     "59f4215e03c430515904fab9cc857f144ef2bd74341bee012f4e1ec62821159d"),
    ("meshes/lower_arm_link_1.stl",
     "2cbb01be1ffef0949b0c2dbbd7d462f7010aac78cf0ee5313a4e6a9b582ef7c0"),
    ("meshes/lower_arm_link_2.stl",
     "4c314076cf647589d6f919f871e847507893d0168f7b5ebdc80208248009c128"),
    ("meshes/shoulder_link_0.stl",
     "06aba91a426e25f004050bfe79658ff8ef15415c7678083b9be53f42c982231b"),
    ("meshes/shoulder_link_1.stl",
     "474a087ea8149c3e373b3ad8f583887470eee522ef65a8b9bea5871d856831dc"),
    ("meshes/shoulder_link_2.stl",
     "65c29cb880f59b6e042888bdc1990b078ee9aa35b30be2d94a65251d3e5c1d70"),
    ("meshes/upper_arm_link_0.stl",
     "3b92835d809d775703c66a4e23c3107074d3ae1c75fa875e6ae2119e108b2ecf"),
    ("meshes/upper_arm_link_1.stl",
     "33f263290418bfef20b9d32eeaa0e2eacc74e1e0dbc7dd8b58bb636674223c99"),
    ("meshes/wrist_link_0.stl",
     "4d143a7fff523877d21de7ad9d85ef9e10210f35c5444870083b967aa41ebb83"),
    ("meshes/wrist_link_1.stl",
     "e37ee4ef0718a21229b176780f828d79df24d568e0a9a7799cff6add0d133104"),
    # --- the V0.4 visual overlay (`visuals_v04/README.md`) ------------------
    # His own patcher and the hull it draws. It replaces ONLY the zero-mass,
    # zero-contact `visual_body_*` and `tread_*` geoms and refuses anything
    # else, so the inertial, the joints, the camera, the actuators, the
    # keyframe, the solver block and every collision proxy survive it. That
    # is why this lab can take a visual update without re-measuring a thing.
    ("visuals_v04/apply_visuals.py",
     "edaf27e872074fbf2430158a24d3da9ff961d3984cfcebe99ea5dff01acf30e4"),
    ("visuals_v04/manifest.json",
     "1cc614294b73dad04e9b024066197414d18ff8b1e8b50d884554e8e37e0dae97"),
    ("visuals_v04/assets_fragment.xml",
     "4c09fda7543da752220002018e444e3870381fb614c590544f0fee15d7db4afe"),
    ("visuals_v04/rover_geoms_fragment.xml",
     "dda34f6f7c591022f485f278e1d465033661a6503346fad46c00258f01bedaf0"),
    ("visuals_v04/validation.json",
     "6f22d3d4f871eb3dd3e440b5eaee5a571b1a1313e839a01240bfaf75c0ddf483"),
    ("visuals_v04/NOTICE",
     "b5b385e3018427e13df18c7dd1ad2f035eb4c32ae3d33ccb40273643f35d7020"),
    ("visuals_v04/LICENSE",
     "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"),
    ("visuals_v04/meshes/v04_000_underbody.stl",
     "e71934e24cdd9174fffec32de4bad46f425dfd7397895532f9f30d2ccc11667d"),
    ("visuals_v04/meshes/v04_001_underbody.stl",
     "bd98c77d887cdfbd64276252f06e5497a749ee0adaf3a776cd1fbd61bbf08492"),
    ("visuals_v04/meshes/v04_002_cover.stl",
     "3e088d633d161a31a18797ef29b0275d62b2ef40614ae8c63e0a269bb0ff48de"),
    ("visuals_v04/meshes/v04_003_track_left.stl",
     "256434871eebada6ec2beb9099ed877a10698427589d732b6d358a9d8d78165d"),
    ("visuals_v04/meshes/v04_004_track_left.stl",
     "dbd27dc21397b492ebaf9bb6bdc5fe3785284cdfc0205812cd7a2939c803c859"),
    ("visuals_v04/meshes/v04_005_track_left.stl",
     "060a0b8d35a05bacc2b114fe8511063e9b85506a113e4708e94570c822ff7a5c"),
    ("visuals_v04/meshes/v04_006_track_left.stl",
     "f253d675f3e4597fc67b7f409a2cb9281a8257984b40e8d6c68ed87c180ac311"),
    ("visuals_v04/meshes/v04_007_track_left.stl",
     "15db958bc3a4ed911308d812f65e6eabf534892d17f4aa0239d4bc8cae58a450"),
    ("visuals_v04/meshes/v04_008_track_left.stl",
     "0fad27aa8eedea2160397b92b920c5165ad4fa03907f359913c978ea9f644ec7"),
    ("visuals_v04/meshes/v04_009_track_left.stl",
     "4efca30319ae17d5d84efc6329a6c3700132fdd1013128bc0162224a8010a3e9"),
    ("visuals_v04/meshes/v04_010_track_left.stl",
     "d1cb4a791f1276a112a63c2cced3dbcbaf9327be59d9728fc97e45be01cadd4c"),
    ("visuals_v04/meshes/v04_011_track_left.stl",
     "8149b73ee68699784b2f28288275f46de95532326eb98fd76be47e8b90bab94f"),
    ("visuals_v04/meshes/v04_012_track_left.stl",
     "4986ddb62d621c8d800700a5168a31e80cd42107db417766e105b4c17c4ea47e"),
    ("visuals_v04/meshes/v04_013_track_left.stl",
     "b0e12b050cc7b35981e3228e35e5972580ecb00099a42e34e68cd9787a94273b"),
    ("visuals_v04/meshes/v04_014_track_right.stl",
     "5b88300734365e20bbe2fded0cf056b62ddacf8b1eea977beee257bd8f68bdcf"),
    ("visuals_v04/meshes/v04_015_track_right.stl",
     "104c2537e1db0454a8c81896c0ea53a6f554cf1017e609d863ca7e050aa781f1"),
    ("visuals_v04/meshes/v04_016_track_right.stl",
     "6322195a458af2d3553156ee351f76a1eda83c61f6c98867288d84711ee424f3"),
    ("visuals_v04/meshes/v04_017_track_right.stl",
     "fbcc97d9a91fa8b648dbf68688493f843b27cb96484c1ebdebdd5140e98dae35"),
    ("visuals_v04/meshes/v04_018_track_right.stl",
     "ccc2b943d48a62fec3605f072a52879b37f7522e79b33c7dd5633d21a0ba4a8a"),
    ("visuals_v04/meshes/v04_019_track_right.stl",
     "a4d0f5ab41300ab4d34506e7c4e2882da6dd156ddf4eda4ab13e1ce53472ceaf"),
    ("visuals_v04/meshes/v04_020_track_right.stl",
     "c6ba9e5a7780b540948e13da4fadae25a9639da2bb1c104225564e7a3ac7b0c5"),
    ("visuals_v04/meshes/v04_021_track_right.stl",
     "6535d113facc3fbf9d4735ee60e4ff521ff542668dd117b2967e6e30958f1425"),
    ("visuals_v04/meshes/v04_022_track_right.stl",
     "68a26b8583d9e577a3f4d35403388b47dcd98f628fd891e1166c024e23323fe0"),
    ("visuals_v04/meshes/v04_023_track_right.stl",
     "bd4ca353683f985b7779968be3f26fc9188a26cec9e106f8b1ca0a256d253a21"),
    ("visuals_v04/meshes/v04_024_track_right.stl",
     "23d118ada9af3fc109085f3ebc6148d0efa93c08ebd31e09d4e8d00c194694b0"),
    ("visuals_v04/meshes/v04_025_track_left.stl",
     "bece30d0952f848941d41cc7453e3b26a3dc9fdcce17ce95725934901659f090"),
    ("visuals_v04/meshes/v04_026_track_left.stl",
     "caf313bd6943aecedbb1746aad98244051605f322f187886d5ee35ce92aad815"),
    ("visuals_v04/meshes/v04_027_track_right.stl",
     "165e3ead4aa1c303d7736e762da8725df82b471bed13897c8d8dc07698da5ea2"),
    ("visuals_v04/meshes/v04_028_track_right.stl",
     "5037d290cd872dd68b5d729f541b4b2c7d604533307f99feb44834803a9a6bbe"),
    ("visuals_v04/meshes/v04_029_arm_plate.stl",
     "56afdfb15f4fe821c3e1b8da4425abcb806d50b536f2207f9ffc751248271f7a"),
    ("visuals_v04/meshes/v04_030_bin.stl",
     "01ed2618f9c0c847f1dbc5d28cfa3d99d83387d799ffb9bc6dff047afcbe3ad5"),
    ("visuals_v04/meshes/v04_battery.stl",
     "70136b89a680fb6e9c78a96f8615391898b03cb757fb113a871164646e030834"),
    ("visuals_v04/meshes/v04_belt_left.stl",
     "2a4baa8738fe2fcd5a18af8b24baeea8269c60c6bd26a320038937977dc594c0"),
    ("visuals_v04/meshes/v04_belt_right.stl",
     "253b37204f23a6189ed880b3cbc88d1d17dbd0f857e5b7e9284a154af15027c5"),
    ("visuals_v04/meshes/v04_bff.stl",
     "0030e1c80ebd8f2a487f890bc1c6ddf83fbaeffa4ff8a0cd937d67f147646b07"),
    ("visuals_v04/meshes/v04_bracket_left.stl",
     "a49c598edaf5be6a20f8dc58e5f5e7c325c3a0281db71c191638965815d1ce92"),
    ("visuals_v04/meshes/v04_bracket_right.stl",
     "dff00c2e87d2fd16246827075399ff31df2bc6256ec62022a8ac1deb632de8fd"),
    ("visuals_v04/meshes/v04_camera_front.stl",
     "7829314e9b5a24f81b459371911ff13e196a066709bd5d9721b4b3eaa00c590b"),
    ("visuals_v04/meshes/v04_camera_lens_0.stl",
     "41683b79bcf1b24ad4d9719b86182972629fe1225c0a278bd4b6ec7e338feff1"),
    ("visuals_v04/meshes/v04_camera_lens_1.stl",
     "33a915bc9fd4ee2501ec4f2033768220074ae0aed8a3185d5292f72ec071c46a"),
    ("visuals_v04/meshes/v04_camera_lens_2.stl",
     "26e973acb2dd87bd0ffc9273259be7c8bbae0bf6a03134f981cfb6fcd924c7c7"),
    ("visuals_v04/meshes/v04_cytron.stl",
     "bfc1a1c7ddafb16d28b03551221c9eed6b60702a1ffcd0f88f2e2d543b1dbbaf"),
    ("visuals_v04/meshes/v04_hull.stl",
     "afc2cd5ab21c6ed8b0ff967e294ee5fa6e57f1fd9a4f14341dd10741ceac12cf"),
    ("visuals_v04/meshes/v04_ina.stl",
     "24789d3303f657f810e2971b6cfc8c2625077e190412354b0ef5d75530db8e48"),
    ("visuals_v04/meshes/v04_jetson.stl",
     "262993cb39f538a2810983c4af95d374428faf439a8f73b85fd3c5d083366dbd"),
    ("visuals_v04/meshes/v04_motor_left.stl",
     "a29c2f7f16090e591beb6adc3e46c45c7dd8b9285c1193555621308e7768964c"),
    ("visuals_v04/meshes/v04_motor_right.stl",
     "5123ab949591fb86781400808ad8eeefab87aaf5e36c52414115edc488a7c29a"),
    ("visuals_v04/meshes/v04_qtpy.stl",
     "d6f95771b16b3a8c7f5f9c587756394b5511511aba3a2646bd9853d4390b593b"),
    ("visuals_v04/meshes/v04_realsense.stl",
     "5f4734087157e021e3c8409cf9bd9d0737d504ce0d533d42c230fd03235f6e6b"),
    ("visuals_v04/meshes/v04_waveshare.stl",
     "67eff62e437477ff2c0479b891cd4f1e0494122151f196790e712b81b7ff0d85"),
    ("visuals_v04/meshes/v04_wire_black.stl",
     "154cba5e5688a06f5c934dfbbac01c706217cfe73867bccb1e6d9f041d328025"),
    ("visuals_v04/meshes/v04_wire_red.stl",
     "47aa7b959de38f0bdc9a3a4dce2879a66481e31aee1f4f202868328852eefedc"),
    ("visuals_v04/meshes/v04_xt.stl",
     "b3ecc4640f03b9f2f02634553cfda4919ddfcdf0a0b56498e15bbab741e55620"),)


# The collision package was published separately from the pinned robot.
MOSS_COLLISION_SHA = "39c6de8fa26a7c7f0ebd332e126c620550087a7d"
COLLISION_RAW_BASE = ("https://raw.githubusercontent.com/metrox-eth/moss-jev/"
                      f"{MOSS_COLLISION_SHA}/live/model")
COLLISION_ASSETS: tuple[tuple[str, str], ...] = (
    ('collision_diagnostics/INTEGRATION.md',
     '6f14178363ff86699abf6ff024aa5539894c78a41c9e04156044c4b9a7da7d23'),
    ('collision_diagnostics/SHA256SUMS',
     '5b33daccb6de049e0a59da314f7442c8e2fb7284e4fa4138bee53f4378112d61'),
    ('collision_diagnostics/audit.py',
     '7324ed188a4f71bb46ffcacaff1ec546c8b2778b24181d2e4f59514fea6e2fbd'),
    ('collision_diagnostics/candidate_self_validation.json',
     '8ed979bd5aa56b0da8e803560fae36c60afad4faf4cd9d12901b68156170f93b'),
    ('collision_diagnostics/candidate_validation.json',
     '86f83043e1ab9451613d7f81dc6e80a1aae7102fa7f1ed65cb89481830955f12'),
    ('collision_diagnostics/prepare_candidate.py',
     '9ad0bbcb31450223239250eb0b94fd2cde5283eb47227d01140697d3b7493121'),
    ('collision_diagnostics/render_candidate.py',
     '105ce2711ce07b24e132406b9fa57c6839fcd9fd34c5f126388e52bfb56e34ec'),
    ('collision_diagnostics/results.json',
     '888b66baed66c967aa369dfc756e5cca399ecf3dc48002e9eea0412a79224a1c'),
    ('collision_diagnostics/validation.json',
     '1896b114ba229e6efa407df7ac5ab89bc785a9ffb61a1e49354b6b03329ee27a'),
    ('collision_diagnostics/validation_linux.json',
     'a601597abebf23ae22219dcbf18c00f16d5f1f1e18e1148ecfe5d452ae6a1465'),
    ('collision_diagnostics/verify_candidate.py',
     '92a93da839dc64bd5ec2bec572433f7ef114d811994be1f11080ba2234cf134e'),
)
ASSETS = (*ASSETS, *COLLISION_ASSETS)

ROBOT_XML = "moss_robot.xml"
#: What his patcher writes, in the cache beside the download. DERIVED, so it
#: is not in `ASSETS` and never hashed against the manifest.
ROBOT_V04_XML = "moss_robot_v04_visual.xml"
#: His collision candidate, derived from the V0.4 visual XML by his own
#: `collision_diagnostics/prepare_candidate.py`.
ROBOT_COLLISION_XML = "moss_robot_v04_collision.xml"
#: Where that package lives once fetched, beside the model.
COLLISION_DIAG_DIR = "collision_diagnostics"
SCENE_NAME = "scene_moss.xml"
DOWNLOAD_TIMEOUT_S = 60.0

# ------------------------------------------------------------ what MOSS is
#
# The subtree root: his whole robot hangs off `rover`, and the planar base's
# three DoFs go on it. (His arm's own root is `base_link`, which is the
# SO-101's name for the shoulder mount and NOT the chassis — the one name in
# this model that reads like the opposite of what it is.)
BASE_BODY = "rover"
ARM_MOUNT_BODY = "base_link"
#: The gripper's own frame — the jaws hang off it, so it does NOT move when
#: they open. Where the wrist camera mounts.
GRIPPER_FRAME_BODY = "gripper_frame_link"
#: (x, y, yaw), spelled and ORDERED as `robots/mars.BASE_JOINTS` — see the
#: module docstring. `base_x` is his; the other two are added by
#: `add_planar_base`.
BASE_JOINTS: tuple[str, ...] = ("base_x", "base_y", "base_yaw")
#: The five arm joints, in his actuator order.
ARM_JOINTS: tuple[str, ...] = ("shoulder_pan", "shoulder_lift", "elbow_flex",
                               "wrist_flex", "wrist_roll")
#: The two finger SLIDES. His MJCF actuates them independently and the real
#: gripper does not: **one servo drives both jaws** (Laurent, 2026-09-24).
#: `couple_fingers` ties them with an equality constraint so the simulator
#: cannot produce a motion the hardware cannot, and the contract exposes ONE
#: gripper command rather than two — otherwise a policy is free to learn a
#: scissor action that no MOSS can perform.
FINGER_JOINTS: tuple[str, ...] = ("finger_left", "finger_right")
#: The one the single servo drives; the other follows it.
GRIPPER_JOINT = "finger_left"
#: What the lab poses and a policy would act on: the arm and the jaw. The
#: base is excluded for MARS's reason — a wheeled body's DoFs are driven by a
#: controller, not posed by an animator.
JOINT_NAMES: tuple[str, ...] = ARM_JOINTS + FINGER_JOINTS
#: Panel sections for the 🎬 editor, in `JOINT_NAMES` order.
JOINT_GROUPS: tuple[str, ...] = ("arm",) * 5 + ("gripper",) * 2

#: His `home` keyframe's CTRL row — the commanded pose, which is the one a
#: controller holds. The keyframe's `qpos` row beside it is where the arm
#: SETTLES under gravity, 7 mrad below at the shoulder lift; storing the
#: command and letting the servo find the sag is what every other body here
#: does, and it is why `HOME_QPOS` is not this vector.
ARM_HOME: dict[str, float] = {
    "shoulder_pan": 1.34,
    "shoulder_lift": -0.65,
    "elbow_flex": 0.4,
    "wrist_flex": 1.25,
    "wrist_roll": -0.05,
    "finger_left": 0.037,
    "finger_right": 0.037,
}
DEFAULT_POSE = np.array([ARM_HOME[j] for j in JOINT_NAMES], np.float32)
HOME_KEY = "home"                     # his own keyframe's name, kept
FLOOR_GEOM = "floor"

#: The jaw, measured off the compiled model (pad-centre separation):
#: 16 mm shut, 98 mm at the slide's 41 mm limit. His missions grasp at 37 mm
#: of travel, which is ~90 mm of jaw around a 66 mm can.
JAW_SHUT_MM = 16.0
JAW_OPEN_MM = 98.0
MISSION_OPEN_M = 0.037
#: What to command the fingers to GRASP his can with. MEASURED
#: (`scripts/probe_moss_grasp.py`), and the window is narrow enough that
#: guessing it would have cost a day: pad CENTRES are `16 + 2*ctrl` mm apart
#: and the pads are 8 mm boxes, so the inner faces sit 8 mm closer. A 66 mm
#: can wants a few mm of interference and nothing like a squeeze.
#:
#: **RE-MEASURED 2026-09-24 after the jaws were coupled**, and the first
#: table did not survive it. With one servo and a STIFF coupling the window
#: is not narrow at all — every closure that actually touches the can holds
#: it, and only a gap fails:
#:
#:     ctrl   jaw inner   on a 66 mm can    lifted   mean lift
#:     0.000      8 mm    58 mm squeeze      3/5      0.049 m
#:     0.010     28 mm    38 mm squeeze      3/5      0.049 m
#:     0.018     44 mm    22 mm squeeze      3/5      0.050 m
#:     0.022     52 mm    14 mm squeeze      3/5      0.049 m
#:     0.026     60 mm     6 mm squeeze      3/5      0.049 m
#:     0.030     68 mm     2 mm GAP          0/5      0.001 m
#:
#: The earlier table — "2-6 mm holds, 10 mm EJECTS" — was an artefact of the
#: soft default coupling, where contact impulses on the unactuated follower
#: could throw the can out. It is recorded here rather than deleted because
#: it is exactly the kind of number that gets quoted for a year: **it was
#: measured on a model that no longer exists.**
#:
#: 0.027 is kept as the scripted default: 6 mm of interference is inside the
#: working band and is the gentlest setting that still holds, which is what a
#: real gripper on a thin aluminium can wants even though this simulator no
#: longer punishes a crush.
#:
#: At 0.027 over a 3x3 spot grid in the standoff band (x 0.22-0.28,
#: y +-0.06, grasp height 0.050 m): **7/9 held**, lifts 0.163-0.193 m. Both
#: failures are the x = 0.22 row — the near edge, where a POSITION-only IK
#: solution puts the jaw at an angle the can slips out of. That is the open
#: problem for a pick brain, and it is orientation, not grip.
GRASP_JAW_CTRL_M = 0.027
#: Where on the can to put the jaws: 50 mm up a 115 mm can. MEASURED in the
#: same sweep — 0.065 and 0.080 never lifted it at any jaw command, for the
#: same orientation reason.
GRASP_HEIGHT_M = 0.050

# ------------------------------------------------------- the four arm poses
#
# MOSS's pick needs no run-time IK, and that is not a shortcut — it is the
# shape of the problem. The grasp point is FIXED in the robot's own frame
# (0.26 m ahead, 0.05 m up), so what varies is where the BASE is, and the
# base is what the brain already steers. Each pose below was solved once with
# damped least squares onto the `tcp` site and is contact-free at solve time;
# `tests/test_moss.py` re-derives the tcp each one puts the jaws at.
#
# **Why the jaws are pre-positioned and the base drives the can in, rather
# than the arm descending onto it.** MEASURED: his palm sits ~60 mm up the
# approach axis from the tcp and his can is 115 mm tall, so a descent onto a
# standing can lands the palm on the lid and stops 35 mm short of the grasp
# (tcp 0.085 against a commanded 0.050). Moving the arm out of the tuck with
# a can already in front is worse — the links sweep it 20 cm away in 0.4 s.
# What works, 5/5 including the stow, is a litter-picker's own order: open
# the jaws at can height while the can is still 0.55 m off, then creep.
# (The can TOPPLES into the jaws as it enters — it is picked up lying down,
# not standing. That is what the 5/5 is; nothing here grasps an upright can.)
#: Folded inside the rover's own footprint, which is what Laurent asked for
#: on 2026-09-24: "keep the gripper within the rover's footprint, clear of
#: the bin". MEASURED over 72k samples with the footprint as a HARD
#: constraint, height the objective, and THREE tests: reachable, inside the
#: shell, and clear of ITSELF.
#:
#: The self-collision test is why this value moved three times, and it is
#: only possible because Laurent pointed out that the arm capsules carry
#: contype=2 / conaffinity=1 with no explicit pairs — capsule-to-capsule
#: contacts are FILTERED, so MuJoCo reports `ncon == 0` for a pose that
#: drives the gripper through the shoulder. Opening those masks (his
#: `--self-contacts`) is what made the question answerable at all.
#:
#: Measured with them open: the first value put the palm 38.5 mm into the
#: shoulder (10 penetrating pairs; his own fixture said 8), and the second —
#: found while a 35 mm phantom bin was in the way — was worse at 45.3 mm and
#: 13 pairs. This one arrives (0.0056 rad after settling), sits 2.9 mm inside
#: the shell, tops out at 0.2531 m under the real 0.261 rim, and has ZERO
#: penetrating pairs. 3 of 500,000 sampled poses qualify.
#:
#: It is measured against HIS 13 mm capsules, which underrepresent a 70 mm
#: arm — so "self-clear" here is as good as that collision model, and his
#: per-part hulls are what would settle it. Worth a physical check either way.
TUCK_POSE: tuple[float, ...] = (-0.9129, 0.6491, 0.5893, 1.6044, -0.7775)
#: The same pose for HIS V0.4 collision hulls, which need their own: the
#: line above stands 13.31 mm outside the shell and 0.2962 m high on them,
#: against a 0.261 m rim. Found by asking whether the ARM touches anything
#: rather than whether the MODEL does — see `apply_collision_candidate` for
#: why that distinction cost a wrong conclusion. Verified on his hulls:
#: reachable to 0.010 rad, 0.00 mm outside the shell, top 0.2583, zero arm
#: contacts. On the legacy proxies it is 4.28 mm proud, so this is not a
#: replacement for `TUCK_POSE` but the other half of a pair.
TUCK_POSE_V04: tuple[float, ...] = (-0.5014, -1.7374, 1.5055, 1.3084, 1.9457)
#: Jaws open at can height, 0.26 m ahead — the middle of the measured band.
GRASP_POSE: tuple[float, ...] = (1.3534, 0.3752, 0.7008, 0.5032, -0.05)
#: Straight up from the grasp, 0.25 m, before anything swings.
LIFT_POSE: tuple[float, ...] = (1.3534, -0.8146, 0.3999, 1.4345, -0.05)
#: Over its own bin: tcp at (-0.13, -0.03, 0.36), which is one of 27 points
#: over the bin interior that are reachable AND contact-free. The obvious
#: one — the bin's centre at rim + a can — is NOT: the arm grazes the front
#: wall `bin_x1` by 3 mm, and the centre at 0.336 is 87 mm out of reach.
DROP_POSE: tuple[float, ...] = (1.1446, -1.7453, -1.432, 1.55, -0.05)
#: Where the can must be, in the base frame, for the jaws to be around it.
GRASP_STANDOFF_M = 0.26
#: Where the arm comes out. Far enough that the deploy sweep misses the can.
DEPLOY_STANDOFF_M = 0.55

#: The bin on its back, read off the `bin_*` geoms rather than typed: the
#: interior, the floor and the rim's top edge, all in the rover's frame.
#: What a "put it in the bin" task aims at, and what `probe_reach.py`
#: measured the reachable set against.
BIN_INTERIOR_X = (-0.164, -0.010)
BIN_INTERIOR_Y = (-0.096, 0.096)
BIN_FLOOR_Z = 0.110
#: The bin's rim, from the V0.4 mesh transformed CORRECTLY (see
#: `raise_bin_walls` for the error this replaces): the 3276 vertices reach
#: z 0.260000 in the rover frame, and his tapered collision walls top out at
#: 0.261 — agreement, not the 35 mm discrepancy previously reported.
BIN_MESH_RIM_Z = 0.260
BIN_RIM_Z = 0.261

#: The chassis geoms. Named because a planar base cannot sink: anything of
#: his that touched the floor would answer with a hard constraint rather than
#: a wheel, which is the trap `robots/mars.tune_contacts` exists for. MEASURED
#: on this model: the tracks' underside sits 4 mm clear at the spawn height,
#: so nothing does, and none of these is re-tuned. Kept as a list because the
#: first thing a tracked-drive model will do is give them friction.
TRACK_GEOMS: tuple[str, ...] = ("track_-1", "track_1")
#: The pads a grasp is made of, and the contact model they impose on
#: everything they touch (`tune_contacts`). The numbers are HIS — what
#: `moss_robot.xml` already gives them — and what this lab changes is only
#: the PRIORITY, so the jaw wins the pair instead of the litter.
FINGER_PADS: tuple[str, ...] = ("pad_left", "pad_right")
FINGER_CONDIM = 4
#: Sliding / TORSIONAL / rolling for the jaw pads. His value for the middle
#: one was 0.01, and with it a gripped can rotates inside the jaws: MEASURED
#: over the swing round to the bin, the can turns 10 deg relative to the
#: gripper and then breaks loose altogether (124 deg, grip lost). Swept over
#: five seeds, 0.01 holds 4/5 with 10.0 deg of spin, 0.15 holds 5/5 with
#: 0.7 deg, and 0.40 buys almost nothing more (5/5, 0.5 deg). 0.15 is the
#: least that makes the can turn WITH the hand, which is what a gripper does.
#:
#: Honest about what this is: MuJoCo resolves a grip through a handful of
#: contact points, and torsional friction is the only thing resisting spin
#: about the line between the two pads. A naive single-patch estimate for a
#: 32 mm rubber pad is nearer 0.03, so 0.15 is compensating for the contact
#: model as much as for the rubber. WORTH ASKING LAURENT: does the real
#: gripper hold a can through a 180 deg wrist rotation without it turning in
#: the jaws? That measurement, not this one, should set the number.
FINGER_FRICTION: tuple[float, float, float] = (1.4, 0.15, 0.001)
HULL_GEOMS: tuple[str, ...] = ("hull", "bin_floor", "bin_x-1", "bin_x1",
                               "bin_y-1", "bin_y1")
#: Added by `seal_bin_void`, not his: the filler under the bin floor.
BIN_VOID_GEOM = "bin_void"

# --------------------------------------------------------------- the mass
#
# The `rover` body ships with an EXPLICIT `<inertial>` of 6.138672 kg
# (`moss_robot.xml` line 86, `moss_visual.xml` line 87). It sits AFTER the
# arm subtree rather than before it, which is legal MJCF and is how a first
# reading here missed it and wrongly called the number a density default —
# Laurent corrected that on 2026-09-24. The distinction matters exactly as he
# said it does: with an explicit inertial present, changing geom masses would
# not replace it, and only another explicit one will.
#
# Where the number came from is the other half. His hull, track and bin boxes
# carry no `mass=`, and his INTEGRATION.md says the exporter freezes "body
# inertia to the original compiled model before visuals are added" — so 6.139
# kg BEGAN as MuJoCo's 1000 kg/m^3 composite of those boxes and was then
# written out as a fact. Solid PETG, which a printed shell is not.
#
# Laurent weighed the real rover and sent the numbers below; they REPLACE the
# shipped inertial through `set_base_inertial`, which is why the download
# stays byte-identical while the lab's model is 3.5 kg.
SHIPPED_BASE_MASS_KG = 6.138672
#: [Laurent Genoud, 2026-09-24] the rover without the SO-101 — and it is a
#: SUM, not a lump. He broke it down: **3.18 kg equipped base** (no arm, no
#: cover, no bin) + **140 g cover** + **180 g empty bin** = the 3.5 kg he
#: weighed. That matters because the balance point he measured — over the
#: central roller axle — belongs to the **3.18 kg configuration**, with the
#: cover, arm and bin off. Lumping all 3.5 kg there would put the cover's and
#: the bin's mass at the axle, which is precisely where they are not.
#:
#: So the three are carried separately (`BASE_COMPONENTS`) and composed. The
#: bin sits 87 mm behind the axle and 110 mm up; the cover is 25 mm behind
#: and 124 mm up — both read off HIS V0.4 manifest's mesh bounds rather than
#: guessed. The composite COM therefore lands where his geometry puts it, and
#: it moves rearward when the bin is loaded, which is his own point about
#: stability during a pickup with the arm extended.
BASE_MASS_KG = 3.5
#: [his] the SO-101 upstream totals ~632 g and the custom gripper adds ~50 g,
#: so the whole robot is ~4.18 kg. MEASURED on this MJCF the arm links sum to
#: 0.820 kg. The 138 g is not a discrepancy to reconcile: he confirmed
#: (2026-09-24) that 820 g is what the supplied model contains and 682 g is
#: their REVISED estimate, not yet applied to these files. So the lab's total
#: is 4.32 kg today and will be ~4.18 kg at whichever revision lands it.
ROBOT_MASS_KG_ESTIMATE = 4.18
#: [his, 2026-09-24] weighed separately and ALREADY INSIDE `BASE_MASS_KG`:
#: the blue top cover and the bin. Recorded because a payload model needs to
#: know what is structure and what is cargo — and because "the bin weighs
#: 180 g" is the kind of number nobody can recover later.
COVER_MASS_KG = 0.140
BIN_MASS_KG = 0.180
#: The equipped base alone — the configuration the balance check was made in.
EQUIPPED_BASE_MASS_KG = 3.18
#: (mass kg, centre in the rover frame). The base's centre is his measured
#: balance point; the cover's and the bin's are the centroids of their own
#: V0.4 meshes (`visuals_v04/manifest.json` bounds_m).
BASE_COMPONENTS: tuple[tuple[str, float, tuple[float, float, float]], ...] = (
    ("equipped_base", EQUIPPED_BASE_MASS_KG, (0.0, 0.0, 0.075)),
    ("cover", COVER_MASS_KG, (-0.025, 0.0, 0.124)),
    ("bin", BIN_MASS_KG, (-0.087, 0.0, 0.185)),
)
#: **The 3.5 kg and its balance point are the UNLOADED rover**, and the
#: fore/aft check was made with the cover, arm and bin removed. So nothing
#: here claims a loaded rover stays balanced: a can in the bin sits 87 mm
#: behind the axle and shifts the combined centre of mass rearward.
#:
#: In this simulator that happens by itself and correctly — the can is a free
#: body with its own mass resting on the bin floor, so its weight and its
#: moment arrive through contact rather than through an assumed inertia. What
#: is NOT modelled is a revised inertia tensor for a loaded bin, because
#: nobody has measured one on the real rover. `PAYLOAD_UNMEASURED` is that
#: caveat with a name, and it stays True: the numbers below are the MODEL's,
#: not a weighing.
#:
#: What the model does say, composed over every body and reported about the
#: composite centre (2026-09-24), answering his stability question directly —
#: three cans in the bin, by can mass:
#:
#:     per can   total kg   COM x mm   d(COM x)   Ixx     Iyy     Izz
#:      15 g       4.365      +19.7      -1.1    0.0448  0.1039  0.1114
#:     200 g       4.920       +7.6     -13.1    0.0516  0.1073  0.1177
#:     350 g       5.370       -0.3     -21.1    0.0561  0.1092  0.1227
#:
#: About **-20 mm of COM travel per kg** in the bin, and +7.5 mm up. Two
#: things follow. For the litter this robot is built for — empty aluminium
#: cans at ~15 g — the payload is NEGLIGIBLE: three of them move the centre
#: about a millimetre, so a pickup with the arm out is not a loaded-stability
#: problem. But his prediction is right at the other end: three FULL cans put
#: the composite centre at x = -0.3 mm, i.e. across the balance point he
#: measured, and this is the regime where a revised tensor would matter.
#:
#: The unloaded composite sits +20.8 mm FORWARD of the axle rather than at
#: his x = 0, because the SO-101 hangs ahead of it; his balance check had the
#: arm off. That is the same 20 mm disagreement recorded under `BASE_COM_M`,
#: seen from the other side.
PAYLOAD_UNMEASURED = True
ARM_MASS_KG = 0.820
#: [his] the balance point with cover, arm and bin removed sits directly over
#: the central roller axle: x = 0 in his frame, lateral symmetry assumed. The
#: vertical is his estimate from the geometry, 75 mm, to be swept over
#: 50-100 mm for sensitivity (`set_base_inertial(com_z=)` takes it).
#:
#: The shipped inertial's own COM is (-0.02015, 0, 0.07507). The z agreeing
#: with his 75 mm to a tenth of a millimetre is NOT a validation and this
#: file said otherwise for one draft: both numbers come from the same
#: geometry, so they are one measurement counted twice — his correction,
#: 2026-09-24. What IS informative is the disagreement in x: the exported
#: centroid sits 20 mm behind the axle, and his balance test had the cover,
#: arm and bin off. His x = 0 is what is declared here; the 20 mm is an open
#: question for the rover with its bin on.
BASE_COM_M: tuple[float, float, float] = (0.0, 0.0, 0.075)
BASE_COM_Z_SWEEP = (0.050, 0.100)
#: The rover's principal moments at `BASE_MASS_KG`, and the frame they are
#: principal in. SHAPE from HIS EXPORTED TENSOR — the `diaginertia` and
#: `quat` on that line-86 inertial, which is the only statement anybody has
#: about how his chassis distributes mass — SCALED by 3.5/6.138672 to his
#: weighed figure. Not a diagonal in body axes on purpose:
#: the tensor has a real Ixz product (12.5% of the smallest diagonal — the
#: bin is high and behind), and dropping it to force a tidy diagonal would be
#: inventing a symmetry the shape does not have.
BASE_INERTIA_KGM2: tuple[float, float, float] = (0.048650, 0.032301, 0.031116)
BASE_INERTIA_QUAT: tuple[float, float, float, float] = (
    0.557788, 0.557788, -0.434594, 0.434594)

# ---------------------------------------------------------------- the drive
#
# ---------------------------------------------------------------- the eye
#
# **His MJCF has no camera, and the real rover does.** The Jev missions read
# MuJoCo state directly (his INTEGRATION.md says so), so nothing in the model
# marks where the RealSense sits. The hardware does: a **RealSense D455**,
# $466 on his BOM, and his build log for 18 September says "the RealSense
# sits up front, with access from above". That sentence is the whole basis
# for the mount below.
#
# **FROM HIS CAD** (Laurent Genoud, 2026-09-24), after this file spent a day
# on a guess taken from that one sentence: y = 0 (centred — the guess was
# right), **z = 0.075** above the nominal ground plane, **x = +0.156** at the
# centre of the camera's front face, and a downward tilt of **zero**, looking
# straight along +X. The 0.095 this file first used was wrong twice over: it
# was picked off the hull's top, and 95 mm is his REAR MOUNTING-HOLE SPACING.
#
# His own MJCF snippet for the same pose, kept because it is the thing to
# paste if a `<camera>` element is ever wanted instead of this frame:
#
#     <camera name="front_camera" pos="0.156 0 0.075" xyaxes="0 -1 0 0 0 1"/>
#
# **Still not calibrated.** These are CAD front-face-centre coordinates, not
# the RGB or depth optical centre; he says the stream-specific offsets and
# the field of view remain to be configured. So the MOUNT is his and the
# LENS below is still a datasheet.
CAMERA_BODY = "moss_camera"
CAMERA_POS: tuple[float, float, float] = (0.156, 0.0, 0.075)
#: Level, which is now TWICE settled: this lab drove a lap of `moss-yard` at
#: four pitches and found 0/10/20 deg a null (278/278/282 frames with a can
#: in view, all three cans at every angle) and 30 deg a 6% loss — and his CAD
#: then gave a downward tilt of zero. A level lens at 75 mm sees the floor
#: from 0.124 m out against a 65 deg vertical field, so the question was
#: never "can it see the floor".
CAMERA_PITCH_RAD = 0.0
# [datasheet, NOT measured on his camera] The rover carries a **D455f**, and
# Intel's own documents disagree about its RGB field: the D455f product page
# lists 87 x 62 deg, the D400-family datasheet groups D455/D455f at 90 x 65.
# The narrower, more specific figure is used and the other is recorded, and
# both are placeholders — Laurent's own conclusion is the right one: the
# actual stream intrinsics off the camera replace this.
CAMERA_HFOV_DEG = 87.0
CAMERA_VFOV_DEG = 62.0
CAMERA_HFOV_DEG_FAMILY = 90.0      # the family datasheet's figure
CAMERA_VFOV_DEG_FAMILY = 65.0
CAMERA_MAX_RANGE_M = 6.0
#: **RGB visibility and valid DEPTH are different questions near the gripper,
#: and this lab only models the first.** What the detector reports is a
#: bearing and a width — a colour-pipeline measurement, which is also what
#: his BOM says the RGB camera is for ("colour litter detection"). Depth on a
#: D455f has a published minimum around **0.52 m at full depth resolution**,
#: configuration-dependent, so there is NO valid depth at this robot's grasp
#: standoff of 0.26 m. Nothing in the pick may be built on depth at that
#: range, and the near cutoff below is a geometric RGB limit, not a depth one.
DEPTH_MIN_RANGE_M = 0.52
#: 10 Hz, the lab's detector default. NOT his: nothing here has measured what
#: his Jetson gets out of a D455 through dimOS.
CAMERA_RATE_HZ = 10.0
#: **THE DEPTH, as a scan** (2026-09-28). The D455f's depth stream read the
#: way ROS's `depthimage_to_laserscan` reads one: the image row through the
#: optical axis, one range per column. The lens is level, so that row is a
#: horizontal slice of the room 75 mm off the floor — it sees walls, boxes
#: and an upright can, and passes over a lying can (66 mm) and a cap. Toys
#: stay the RGB detector's; this is for the room. Mounted as a `LidarSensor`
#: on `CAMERA_BODY` (a forward fan, `centred=True`), so the lab's scan
#: plumbing — the brain's `senses.lidar`, the frame's `sensors.lidar`, the
#: /sim overlay — carries it with nothing new.
#: [datasheet, NOT measured on his camera] depth field 87 x 58 deg, ideal
#: range 0.6-6 m; the near floor is `DEPTH_MIN_RANGE_M` above.
DEPTH_SCAN_RAYS = 88
DEPTH_MAX_RANGE_M = 6.0

# ------------------------------------------------------------ the ARM camera
#
# **Laurent confirmed on 2026-09-25 that the build will carry one**, and his
# own BOM lists it as the one vision item excluded from every column ("A
# wrist camera, gamepad, tools... are not included in any column"). So it is
# coming and it is unspecified, and EVERY NUMBER BELOW IS OURS until he sends
# his. They are labelled rather than blended into the file, because the last
# unlabelled assumption here was a 0.12 m depth cutoff that he had to correct.
#
# What it is for is not more coverage. It is ATTITUDE: the front RealSense
# sits 0.156 m up on the chassis and reads a can at 40 cm as a silhouette,
# from which the axis of a lying cylinder is an inference off a bounding box.
# A camera on the gripper housing looks down the approach from 12 cm behind
# the tool point and sees it.
# MEASURED on the shipped leg with no attitude in the observation at all:
# the correlation between the can's axis and `wrist_roll` is -0.030 and the
# wrist travels 3.2 degrees — the jaw never turns to meet a perpendicular
# can, which is what a human watching the lab reported.
#: OURS: on the SIDE of the gripper housing, at its back edge, looking down
#: the approach past the housing's corner. A wrist camera that sits AT the
#: jaws is occluded by the object at exactly the moment it matters.
ARM_CAMERA_BODY = "moss_arm_camera"
#: **The gripper frame's +z is the APPROACH, not "up".** MEASURED 2026-09-27
#: over GRASP, LIFT, DROP, TUCK_POSE_V04 and the home pose: the tool point sits
#: at (-0.008, 0, -0.014) in `gripper_frame_link` in every one, the fingers
#: reach z +0.005, and the housing the jaws come out of spans z -0.098..-0.042
#: and about 12 x 12 cm across (x -0.070..+0.054, y -0.058..+0.059). The
#: jaws slide along the 41 deg / 221 deg diagonal of the frame's x-y plane.
#:
#: The first mount, (-0.060, 0, +0.065) pitched 0.90 rad about +y, was read in
#: the opposite convention (+x the approach, +z above the jaws): it put the
#: lens 6.5 cm PAST the jaw tips, under the floor in 32% of the frames inside
#: 10 cm of a grasp (world z -0.029 m at GRASP_POSE), looking UP at the tool
#: point. The detector ignores occlusion, so every wrist reading before this
#: came from a lens no bracket could hold — and the mount sweep that picked it
#: searched only the half-space past the jaws.
#:
#: RE-SWEPT with occlusion (1,564 frames of the 478dad pick, in its eval env
#: and its trained settings; a frame counts when the object's centre is in the
#: 70 x 55 field AND 3 of 5 points on it are not hidden by the robot's own
#: visual meshes or the floor), within 10 cm of a grasp:
#:
#:     mount                                   seen   housing  forearm
#:     old, past the jaws                       61%      --     (under floor 32%)
#:     on the jaw-travel diagonal (45/225)   27-46%
#:     behind the housing corner, z -0.13       85%    3.2 cm   0.5 cm  <- fouls
#:     side face, back edge (THIS)              88%    2.2 cm   2.1 cm
#:
#: A plateau, not a peak: 88-90% anywhere from 290 to 340 deg round the axis,
#: 8.5-9.5 cm out, z -0.09..-0.11, aimed 8-15 cm ahead of the housing — so
#: a real bracket a centimetre off still sees the grasp. Taken off the jaw-
#: travel diagonal (the fingers block that side) and 2 cm clear of the housing
#: box and of the forearm/wrist meshes over the WHOLE wrist_flex x wrist_roll
#: range (the stow passes flex 1.55; the corner behind the housing came within
#: 0.5 cm there). Over all 3,127 frames it sees 89% inside 10 cm and 94%
#: inside 5 cm. At GRASP_POSE the lens is 13.6 cm off the floor and the tool
#: point 26 deg off its axis, 12 cm away.
ARM_CAMERA_POS: tuple[float, float, float] = (0.021, -0.080, -0.100)
#: Where the optical axis points: a spot on the approach axis 15 cm past the
#: gripper frame, so the axis crosses under the jaws at a 19 deg toe-in. The
#: image's up is AWAY from the approach axis, so the jaws sit at the bottom
#: edge of the picture, as on any side-mounted wrist camera.
ARM_CAMERA_AIM: tuple[float, float, float] = (-0.008, 0.0, 0.150)
#: WHICH wrist mount this model carries, recorded in every MOSS run's
#: `env_kwargs` (`Body.train_env_kwargs`) so a policy is checked against the
#: sensor it trained behind (`moss_env.arm_camera_mount_of`). 1 is the
#: retired mount past the jaw tips: every run before 2026-09-27, whose
#: run.json has no such key. 2 is this side mount. BUMP IT whenever
#: `ARM_CAMERA_POS` or `ARM_CAMERA_AIM` moves: a leg's wrist slots (26,
#: 28-30) are only the readings it learned on the mount it learned them on.
ARM_CAMERA_MOUNT = 2
ARM_CAMERA_MOUNT_LEGACY = 1
#: **The SO-101's own default wrist camera**, which is what this arm is
#: derived from. `TheRobotStudio/SO-ARM100` ships six wrist mounts — a
#: 32x32 UVC module (hex-nut, integrated and plug-on variants), a RealSense
#: D405, a D435/D435I and a Vinmooog webcam — and its BOM names no camera at
#: all. The 32x32 UVC module is taken as the default here: it is listed
#: first, it is the only one whose mount is labelled for the SO101
#: specifically, and it is the cheap end, which matches the $25 USB colour
#: cameras on Laurent's own Pi builds.
#:
#: The FIELD is ours — a 32x32 UVC board module is typically 65-75 degrees
#: horizontal and no datasheet is pinned by either repo. At 30 cm a 70
#: degree field spans 0.42 m, so a 66 mm can crosses about a sixth of the
#: frame: plenty to read an axis from, which is the whole reason for it.
ARM_CAMERA_HFOV_DEG = 70.0
ARM_CAMERA_VFOV_DEG = 55.0
#: OURS: it is a close-work sensor, not a room scanner. Past this the front
#: camera is the better answer anyway and this one adds only noise.
ARM_CAMERA_MAX_RANGE_M = 0.60
#: OURS. The module itself does 30 fps, but the number that matters is the
#: DETECTOR's rate, and a second stream shares the same NPU as the front
#: camera's 10 Hz. 15 Hz is a guess at that split and is the first thing to
#: replace with a measurement on the Jetson.
ARM_CAMERA_RATE_HZ = 15.0

# There are THREE numbers for this robot's track width and only one of them
# is the robot's. Recorded together because picking one silently is how a
# yaw error becomes somebody's week:
#
#   0.300 m   `moss_dimos/diffdrive.py`'s default — uncalibrated, he said so
#   0.244 m   the legacy simulation COLLISION proxies, y = +-0.122
#   0.266 m   the V0.4 CAD wheels and visual belts, y = +-0.133
#
# **The drive uses the CAD figure**, because the kinematics belong to the
# vehicle and not to a proxy box: the belts at +-0.133 are where the real
# machine touches the floor. The collision geoms still sit at +-0.122 and he
# flagged that discrepancy himself when he shipped the V0.4 overlay — it is
# explicit now rather than hidden, and reconciling it is a PHYSICS change
# (contacts move, skid-steer effective width changes with the surface) that
# has to be validated by turning the real robot, not by editing a constant.
#: **The box the pickup policy is competent in**, in the base frame:
#: (x_lo, x_hi, |y| max) metres. Declared HERE rather than in the env because
#: it is a contract between two halves that never import each other — the env
#: SPAWNS the can in it (`moss_env.RUNG_BOX[2]`) and the mission loop must
#: DELIVER the can into it before handing over (`tidy_moss`'s deploy).
#:
#: MEASURED 2026-09-24, and it is the pickup leg's whole room/env gap: the
#: handover tested the ARM's pose and never asked where the can was, so only
#: 13 of 47 room handovers (28%) started inside this box — 38% too far, 23%
#: too near — while the same policy scores 12/12 when its env starts it
#: inside. A policy cannot be blamed for a state it was never shown.
#: **WITHIN THE ARM'S REACH, which the old far edge was not.** MEASURED
#: 2026-09-25 by sampling 60,000 arm configurations with the base held still:
#: the TCP reaches x = 0.508 m at graspable height, and covers the full +-0.12
#: lateral box only out to x = 0.48 —
#:
#:     handover x    lateral half-width the arm can cover
#:       0.36-0.48        0.14 - 0.32 m     covers the box
#:       0.50             0.101 m           short
#:       0.53-0.55        0.000 m           CANNOT REACH AT ALL
#:
#: The far edge was 0.55. So on roughly a third of hand-overs the brain put a
#: can where the arm physically could not get it, and the ONLY way to succeed
#: was to drive — which is why every pick policy trained here drives at the
#: can, saturates its base command and spins, and why no penalty on the base
#: ever stopped it: the base was doing necessary work. A penalty cannot price
#: away a motion the task requires. 0.47 keeps margin for the base shifting
#: when the arm extends (an undriven planar base gets shoved by its own arm).
PICK_HANDOVER_BOX: tuple[float, float, float] = (0.36, 0.47, 0.120)

#: The visual/CAD belt centres — what the drive divides by.
TRACK_CENTRES_M = 0.266
#: Sliding friction for the V0.4 belt hulls once they are on the floor, and
#: the contact priority that makes it WIN over the floor's own. Swept against
#: a 0.2 m/s command held for 2 s, which should travel 0.400 m:
#:
#:     mu    0.90   0.30   0.10   0.05   0.02   0.01
#:     m     0.003  0.016  0.103  0.230  0.355  0.385
#:
#: 0.01 is the working end of that, and it is deliberately NOT a rubber
#: track's coefficient: see `tracks_as_support` for why a number that low is
#: the honest one here and what it must not be used for.
TRACK_SUPPORT_FRICTION = 0.01
#: The belts must out-rank the floor, or the contact takes the floor's
#: friction and the sweep above does nothing.
TRACK_CONTACT_PRIORITY = 5
#: The yaw gain to use ONCE THE BELTS ARE ON THE FLOOR. Even at mu 0.01 they
#: leave about 1.69 N m s/rad of rotational drag, and a proportional velocity
#: loop against drag has a standing error: measured, `KP_YAW` = 3 delivers a
#: CONSTANT 64% of every commanded rate (0.2, 0.4, 0.6 and 1.0 rad/s all read
#: 0.64), which is the signature of a gain, not of saturation.
#:
#:     kp_yaw   3.0    20     40     45.2 (the stability bound)
#:     ratio    0.64   0.92   0.96   0.96, and it starts to overshoot
#:
#: 40 is 96% with no overshoot, matching what the forward axis already
#: delivers, and it stays under `MossDriver._stable_gain`'s bound rather than
#: sitting on it. This compensates the SUPPORT surface's residual drag; it is
#: not a friction model and does not make one — see `tracks_as_support`.
TRACK_SUPPORT_KP_YAW = 40.0
#: What the collision proxies still use. MEASURED off the geoms.
TRACK_CENTRES_COLLISION_M = 0.244
#: His `diffdrive.py` default, kept only so nobody re-adopts it by accident.
TRACK_CENTRES_DIFFDRIVE_DEFAULT_M = 0.30
#: [his `diffdrive.DiffDriveConfig`] the per-track speed limit the firmware
#: scales to. Kept because it IS his envelope; only the width was a default.
MAX_TRACK_SPEED_MPS = 0.6

#: His `<option>`, which `MjSpec.attach` would otherwise drop on the floor of
#: whatever room he lands in. A MOSS room adopts these (`world/compose`), and
#: that is the whole "give it its own environment" decision of 2026-09-24:
#: rather than one shared scene with two sets of settings, a MOSS room is a
#: MOSS room. The timestep is `GRASP_PHYSICS_DT` and travels separately,
#: because `Scenario.physics_dt` already carries a room's clock.
SOLVER_OPTIONS: dict[str, object] = {
    "integrator": "implicitfast",
    "cone": "elliptic",
    "iterations": 80,
}

#: His own timestep, and the reason a MOSS room is its own room. 2 ms is what
#: `moss_robot.xml` declares and what his missions grasp at; `robots/mars.py`
#: measured on the other claw in this lab that 5 ms does not grasp at all (it
#: EJECTS: 4/16 scripted picks against 14/16), and that the elliptic cone was
#: a null while the step was everything. `world/scenario.robot_physics_dt`
#: asks the body for this, so a MOSS room compiles at 2 ms without any
#: builder naming this robot.
GRASP_PHYSICS_DT = 0.002

# ---------------------------------------------------------- the contract v0
#
# Declared before any env exists, for the reason MARS's was: a body has to
# speak one contract from the day it is listed, and the width is what the
# exporter shapes a graph from. This is the layout a drive-then-pick task
# will fill, and every slot in it is something this model can already answer.
OBS_JOINT_POS = slice(0, 7)       # arm + both jaws, about DEFAULT_POSE, rad/m
OBS_JOINT_VEL = slice(7, 14)      # rad/s and m/s
OBS_LAST_ACTION = slice(14, 22)   # the previous action, all 8 slots, +-1
OBS_BASE_TWIST = slice(22, 24)    # measured (vx m/s, wz rad/s) of the base
OBS_TARGET_BASE = slice(24, 27)   # xyz of the target in the BASE frame, m
OBS_TARGET_SEEN = slice(27, 28)   # 1.0 when the fix is fresh, else 0.0
#: **Room for a TASK's own slots**, and the pickup uses three of them.
#:
#: `target_base` is a POSITION and nothing in the shared layout says which
#: way the object is lying — so a policy cannot rotate the jaw to close
#: across a cylinder's axis, because it cannot see the axis. MEASURED on the
#: shipped leg over 21 lying cans: the correlation between the can's axis
#: yaw and `wrist_roll` at the grasp is -0.030, and the wrist moves 3.2
#: degrees of standard deviation in total. It is not choosing the base over
#: the arm; it has nothing to choose with, and a reward for aligning the jaw
#: would have been the unobservable-target mistake this repo has already
#: paid for once.
#:
#: So slots 28-30 carry the object's attitude in the BASE frame, and only a
#: task that needs them fills them — `approach` and `stow` leave the whole
#: block at zero, which is what every leg trained on before 2026-09-25 and
#: what the contract has always promised a reader.
OBS_RESERVED = slice(28, 32)      # zeros unless the TASK fills them
#: cos and sin of TWICE the object's axis yaw in the base frame. Doubled
#: because a cylinder has no head: an axis at 10 degrees and one at 190 are
#: the same grasp, and a single angle would ask the policy to learn that
#: seam. Zero for an object with no meaningful axis.
OBS_TARGET_AXIS = slice(28, 30)
#: How UPRIGHT it is: |cos(tilt)|, 1 standing and 0 flat on its side. The
#: jaw closes across the diameter either way, but from above for one and
#: from the side for the other.
OBS_TARGET_UPRIGHT = slice(30, 31)
#: THE STOW's use of the same pair, when it runs with `publish_drop`: where
#: in the bin to put the object, as an offset from the bin's centre in the
#: base frame (`moss_bin.drop_obs`, +-1 at 5 cm). The stow never publishes
#: attitude, and the env refuses both at once.
OBS_DROP = slice(28, 30)
#: Still free.
OBS_SPARE = slice(31, 32)
#: THE GRIP, from the wrist depth camera (`publish_grip`): (distance of the
#: object from the tool point, m; 1 when that fix is fresh) — (0.15, 0) when
#: it is not. In the two slots that only ever DUPLICATED another: the second
#: jaw is tied to the first (one servo, an equality constraint), so its
#: position (6) and velocity (13) track slots 5 and 12 to within a fraction
#: of a millimetre — 478dad's normaliser has the same mean and sd for each
#: pair to four places. A leg trained without the flag sees those duplicates
#: as before; one trained with it sees the grip there. The run records which.
OBS_GRIP = (6, 13)
#: THE GRIP IN FULL (`grip_xyz`): where the object sits in the TOOL frame —
#: x, y across the jaws, z along the approach, m — and 1 when that depth fix
#: is fresh; (0, 0, 0.15, 0) when not. Whether it is CENTRED between the jaws
#: is the part a distance throws away. The two duplicate jaw slots plus the
#: base twist (22, 23), which says nothing to a pick that trains with its
#: base locked — so only for a leg trained from scratch with the flag.
OBS_GRIP_XYZ = (6, 13, 22, 23)
#: 32 = 28 used + 4 reserved.
OBS_DIM = 32
#: Five arm targets, ONE gripper command, and the base twist.
#:
#: **Eight and not nine, and that is Laurent's correction (2026-09-24).** His
#: exported MJCF actuates the two fingers separately; the physical gripper
#: has a single servo driving both jaws. A nine-action contract would let a
#: policy learn an asymmetric pinch that the robot cannot reproduce, and the
#: ONNX would emit it to a consumer that has nowhere to send it. So the
#: fingers are coupled in the MODEL (`couple_fingers`) and the contract
#: carries one gripper number.
ACT_ARM = slice(0, 5)             # joint targets, rad
ACT_GRIPPER = slice(5, 6)         # one command, metres of finger travel
ACT_BASE = slice(6, 8)            # (vx, wz)
NUM_ACTIONS = 8
#: Innate's policy-defined skills tick at 25 Hz, and so does his replay.
CONTROL_HZ = 25.0
#: v1: nine actions became eight when the jaws were coupled. The id changes
#: because a consumer that fed a v0 graph eight floats would be silently
#: wrong, and the whole point of a contract id is that it cannot be.
CONTRACT_ID = "moss-arm-32-v1"

#: His visual meshes are group 2 (`contype=0`, `conaffinity=0`, `mass=0`);
#: the collision proxies are group 3. Stated by his INTEGRATION.md and
#: confirmed on the compiled model.
VISUAL_GROUP = 2
COLLISION_GROUP = 3

#: 3.52x the widest horizontal extent at HOME, the arithmetic every body's
#: stage pitch uses (`body.measure_lab_spacing_m`). MEASURED on the compiled
#: scene, and it MOVED with the V0.4 overlay: his newer hull is 0.5688 m
#: across against the old presentation meshes' 0.5189 m, so the pitch went
#: 1.83 -> 2.00 m. A visual update that changes the drawn extent changes
#: where the lab stands the next robot, which is the sort of consequence a
#: conformance suite exists to notice.
LAB_SPACING_M = 2.00


# ----------------------------------------------------------------- the fetch

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path) -> None:
    with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_S) as r:
        dest.write_bytes(r.read())


def asset_dir(root: Path | None = None) -> Path:
    """Where the downloaded files live: `<cache>/model/`, his own layout."""
    return (root or CACHE_DIR) / ASSET_SUBDIR


def robot_xml_path(root: Path | None = None) -> Path:
    return asset_dir(root) / ROBOT_XML


def moss_ready(root: Path | None = None) -> bool:
    """Every manifest file present. Presence, not hashes: `fetch` verifies on
    the way in, and re-hashing 12.7 MB is not something a registry listing or
    a `--robot` choices= should pay for."""
    d = asset_dir(root)
    return all((d / rel).is_file() for rel, _ in ASSETS)


def require_moss(root: Path | None = None) -> None:
    if not moss_ready(root):
        raise FileNotFoundError(
            f"MOSS's assets are not in {asset_dir(root)} — run "
            "`uv run fetch-robot moss` (12.7 MB from metrox-eth/moss-jev, "
            f"{MOSS_LICENCE})")


def fetch(dest: Path | None = None) -> Path:
    """Download his model into the cache, verifying every sha256.

    Idempotent: a file that is present and hashes correctly is skipped, so a
    second run checks the complete manifest without downloading again. Each download lands on a temp
    name in the target directory and is `os.replace`d into place only after
    its hash matches — the repo's atomic-write rule, which here also means a
    half-written STL can never be imported by a parallel worker (AGENTS.md,
    "Atomic writes and live imports").

    A hash that does not match is a RuntimeError naming both. It is the one
    check that tells a moved branch, a truncated transfer and a captive
    portal's login page apart from the robot.
    """
    root = dest or CACHE_DIR
    d = asset_dir(root)
    fresh = 0
    for rel, want in ASSETS:
        out = d / rel
        if out.is_file() and _sha256(out) == want:
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f".{out.name}.{os.getpid()}.part")
        base = COLLISION_RAW_BASE if rel.startswith("collision_diagnostics/") else RAW_BASE
        url = f"{base}/{rel}"
        print(f"[moss] {rel} <- {url}")
        try:
            _download(url, tmp)
            got = _sha256(tmp)
            if got != want:
                raise RuntimeError(
                    f"{rel} from moss-jev {MOSS_JEV_SHA[:12]} hashes "
                    f"{got[:12]}, the manifest says {want[:12]} — refusing it "
                    "(robots/moss.ASSETS pins the revision)")
            os.replace(tmp, out)
        finally:
            tmp.unlink(missing_ok=True)
        fresh += 1
    if not moss_ready(root):
        missing = [rel for rel, _ in ASSETS if not (d / rel).is_file()]
        raise RuntimeError(f"MOSS fetch finished but {missing} are missing")
    print(f"[moss] {fresh} file(s) downloaded, {len(ASSETS) - fresh} already "
          f"verified — {d} ({MOSS_LICENCE})")
    return root


# ---------------------------------------------------------------- the model

def apply_v04_visuals(root: Path | None = None) -> Path:
    """Run HIS patcher to produce the V0.4-skinned robot XML, once, cached.

    `visuals_v04/apply_visuals.py` is Laurent's own script and is executed as
    published: it removes the zero-mass, zero-contact `visual_body_*` and
    `tread_*` geoms and refuses to touch anything else — his check, not ours
    — then adds the V0.4 hull, cover, side assemblies, component envelopes
    and the continuous belts. The inertial, the joints, the camera, the
    actuators, the keyframe, the solver block and every collision proxy come
    through untouched, which is what makes a visual update cheap here.

    The output is DERIVED and lives beside the download; the manifest-hashed
    files are never written to. Idempotent: an existing output newer than
    both inputs is reused, because compiling 53 STLs is not free.
    """
    d = asset_dir(root)
    src, out = d / ROBOT_XML, d / ROBOT_V04_XML
    patcher = d / "visuals_v04" / "apply_visuals.py"
    if not patcher.is_file():
        require_moss(root)
    if out.is_file() and out.stat().st_mtime >= max(src.stat().st_mtime,
                                                    patcher.stat().st_mtime):
        return out
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    spec = importlib.util.spec_from_file_location("_moss_apply_visuals", patcher)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.apply(str(src), str(tmp))
    os.replace(tmp, out)
    return out


#: Our fully-edited robot, written out so HIS collision patcher can read it.
ROBOT_LAB_XML = "moss_robot_v04_lab.xml"


def _edited_spec() -> mujoco.MjSpec:
    """His V0.4 robot plus the lab's edits, before any collision work."""
    spec = load_robot_spec()
    add_planar_base(spec)
    add_camera(spec)
    add_arm_camera(spec)
    couple_fingers(spec)
    tune_contacts(spec)
    drop_base_servo(spec)
    set_base_inertial(spec)
    rewrite_home_key(spec)
    return spec


def apply_collision_candidate(root: Path | None = None) -> Path | None:
    """Run HIS collision patcher over our edited robot, once, cached.

    `collision_diagnostics/prepare_candidate.py` is Laurent's script, run as
    published. It does three things we each tried ourselves and got wrong:

    * **The bin walls.** It keeps his four tapered walls, topping out near
      0.261 m against a real rim of 0.260. We had raised them to 0.2957 off
      a mis-transformed mesh reading — a 35 mm phantom obstacle.
    * **The arm.** It reuses each visual part as a convex collision hull in
      its ORIGINAL POSE rather than one capsule per link. We had widened his
      13 mm capsules to a uniform 35 mm; his point is that the links are
      offset from the capsule axes, so one radius leaves ~18-24 mm outside
      while adding bulk opposite. MEASURED with self-contacts open: our
      fattened capsules admit ZERO tucks that are inside the footprint,
      below the rim and self-clear out of 500,000 sampled poses.

      "His hulls admit some" stood here unmeasured, then a measurement
      appeared to refute it, and THAT measurement was the broken one. The
      record of both, because the second mistake is the instructive one:

      A 400,000-pose search reported ZERO tucks inside the shell and
      contact-free under his hulls, and it was an artefact of a change made
      an hour earlier in this same file. `tracks_as_support` puts the belts
      ON THE FLOOR, so every pose now carries two track-floor contacts, and
      the probe's test was `ncon == 0` — written when the legacy boxes sat
      4 mm proud and the robot touched nothing at all. The test could no
      longer pass for any pose whatsoever. The contact census gave it away:
      `floor|track_-1` and `floor|track_1` appeared in 8,977 of 8,977
      in-shell poses.

      Asking the question the tuck actually asks — does the ARM touch
      anything — his hulls admit a tuck at
      `(-0.5014, -1.7374, 1.5055, 1.3084, 1.9457)`: reachable to 0.010 rad,
      0.00 mm outside the shell, top 0.2583 against the 0.261 rim, zero arm
      contacts. It is `TUCK_POSE_V04`, and it is a DIFFERENT pose from the
      legacy one — each geometry needs its own, because the shipped
      `TUCK_POSE` stands 13.31 mm proud and 0.2962 high on his hulls, and
      this one stands 4.28 mm proud on the old proxies.

    * **The tracks.** It replaces the two legacy boxes with convex hulls of
      the V0.4 belts, at the CAD's 266 mm rather than the proxies' 244 mm.
      He asked for exactly that on 2026-09-25 — "the collision geometry
      needs to match the V0.4 CAD" — with skid-steer width to be calibrated
      on hardware separately. His call, taken.

    It runs AFTER our edits because it requires an explicit rover inertia,
    which `set_base_inertial` supplies; it preserves masses, joints,
    actuators, equalities, keyframes and the camera.

    **IT ALSO PUTS THE ROBOT ON THE GROUND FOR THE FIRST TIME.** His original
    track boxes sit with their underside at z +0.0040 — 4 mm PROUD of the
    floor — so MOSS has never made a floor contact in this simulator at all:
    measured at rest, 0 contacts with the box proxies and 2 with the belt
    hulls. The planar base is driven kinematically, so with nothing to push
    against it tracked every command almost exactly (-1.5%), and the 266 vs
    244 spacing measured as having no effect on turning because the tracks
    were touching nothing to turn against. Every drive number this repo has
    is from a hovering robot.

    That USED TO stop the rover dead: commanded 0.2 m/s for 2 s it travelled
    3 mm instead of 400, because ground friction opposed a velocity servo
    that was never sized against it. **Fixed 2026-09-24** — the belts are now
    declared a SUPPORT surface (`tracks_as_support`) rather than a traction
    one, and the yaw loop's gain is raised to match on that geometry alone
    (`TRACK_SUPPORT_KP_YAW`). On his hulls the rover now holds station to
    0.0 mm, travels 0.385 m of a commanded 0.400 (96%) and turns 1.152 rad of
    a commanded 1.200 (96%), against 0.003 m and 0.766 rad before.

    What that does NOT do is make a friction model. The tracks still bear
    0.00 N at rest because the planar joint carries the weight, so skid-steer
    width remains uncalibratable here and mu = 0.01 is not a rubber track's
    coefficient. It buys the honest half: the robot stands on the floor,
    collides with furniture, cannot hover, and its collision geometry is his
    CAD's 266 mm rather than a legacy proxy's 244.

    Historical adoption measurements (the collision model is now ON by
    default; `MICRODUCK_MOSS_COLLISION_V04=0` selects the legacy model):
    The candidate replaces the two track boxes
    with belt MESHES, which changes what the robot stands on, and he says so
    himself — "updating contacts intentionally changes motion outcomes;
    passing preservation checks does not mean previous pickup replays still
    work." MEASURED on adoption: 7 of 116 tests fail, including the drive
    tests (`it_drives_where_it_is_told`, `a_stale_command_stops_the_robot`),
    the composed-room test and can detection. Every trained policy would
    need retraining against it, and the mission loop re-measured.

    So it ships available and unadopted, but the reason has changed and
    shrunk. It is no longer "the rover cannot move"; it is five tests that
    encode the legacy geometry and three policies trained against it:

        test_it_drives_where_it_is_told          0.869 against 0.9 +- 0.02
        test_a_stale_command_stops_the_robot     0.389 against 0.4 +- 0.01
        test_the_track_width_is_measured_off_his_geoms_not_his_default
        test_the_tuck_pose_keeps_the_arm_inside_the_shell
        test_the_home_keyframe_is_widened_from_his_numbers_not_retyped

    The first two are the 96% tracking above and are CORRECT behaviour for a
    robot whose belts touch the ground — their tolerances describe a hovering
    one. The third asserts the unreconciled 244 mm on purpose. The last two
    are real: the tuck pose and the keyframe were fitted against the old arm
    proxies and have to be re-fitted against his hulls. Adopting it is now a
    bounded piece of work plus a retrain, rather than a blocked one.

    Returns None when the package has not been fetched, and the caller falls
    back to the unpatched build.
    """
    d = asset_dir(root)
    patcher = d / COLLISION_DIAG_DIR / "prepare_candidate.py"
    if not patcher.is_file():
        return None
    src, out = d / ROBOT_LAB_XML, d / ROBOT_COLLISION_XML
    v04 = apply_v04_visuals(root)
    if not (src.is_file() and src.stat().st_mtime >= v04.stat().st_mtime):
        src.write_text(_edited_spec().to_xml())
    if out.is_file() and out.stat().st_mtime >= max(src.stat().st_mtime,
                                                    patcher.stat().st_mtime):
        return out
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    tmp.unlink(missing_ok=True)        # his patcher refuses existing outputs
    spec = importlib.util.spec_from_file_location("_moss_prepare_candidate",
                                                  patcher)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    # prefix '' not None: his code interpolates it into the body
    # query, so None searches for a body literally named "Nonerover".
    mod.prepare(str(src), str(tmp), "", False)
    os.replace(tmp, out)
    return out


def load_robot_spec(xml: Path | None = None) -> mujoco.MjSpec:
    """His robot as a spec, wearing the V0.4 hull.

    No rewrites on the way in — unlike `robots/mars.load_robot_spec`, which
    has three, because this is already an MJCF and he already kept the
    visuals. What DOES happen first is his own V0.4 patcher
    (`apply_v04_visuals`), because the shipped `moss_robot.xml` carries the
    older Jev presentation meshes and he asked for the newer hull to be used.
    `meshdir="meshes"` resolves relative to the file, which is why `fetch`
    reproduces his directory layout instead of flattening it.
    """
    path = xml or apply_v04_visuals()
    if not path.is_file():
        require_moss()
    return mujoco.MjSpec.from_file(str(path))


def add_planar_base(spec: mujoco.MjSpec) -> None:
    """Give the rover (x, y, yaw) where he shipped (x).

    His `base_x` is KEPT rather than deleted and re-added: it is the joint
    his recorded missions move along, and re-creating it would put it after
    the two new ones in qpos order for no gain. What changes on it is the
    rail — `range` and `damping` — both of which belonged to the position
    servo this function's caller drops.

    The order is `BASE_JOINTS`, which is MARS's, which is Innate's, because
    `robots/mars_drive.MarsDriver` indexes them by that name tuple.
    """
    base = spec.body(BASE_BODY)
    x = spec.joint(BASE_JOINTS[0])
    x.limited = mujoco.mjtLimited.mjLIMITED_FALSE
    x.range = [0.0, 0.0]
    x.damping = [0.0, 0.0, 0.0]
    for name, jtype, axis in (
        (BASE_JOINTS[1], mujoco.mjtJoint.mjJNT_SLIDE, (0, 1, 0)),
        (BASE_JOINTS[2], mujoco.mjtJoint.mjJNT_HINGE, (0, 0, 1)),
    ):
        joint = base.add_joint()
        joint.name = name
        joint.type = jtype
        joint.axis = axis
        joint.damping = [0.0, 0.0, 0.0]


def add_arm_camera(spec: mujoco.MjSpec) -> None:
    """Mount the WRIST camera as a massless frame on the gripper frame.

    Same shape as `add_camera`: a body, not a site, because `Detector`
    resolves a mount by body and reads its x axis as the optical axis.

    On the gripper FRAME rather than a finger, so it does not move when the
    jaws do — a sensor whose pose changes with the grip would make the
    attitude estimate depend on how open the hand is, which is the sort of
    coupling nobody finds for a month.
    """
    grip = spec.body(GRIPPER_FRAME_BODY)
    cam = spec.body(ARM_CAMERA_BODY)
    if cam is None:
        cam = grip.add_body()
        cam.name = ARM_CAMERA_BODY
    cam.pos = list(ARM_CAMERA_POS)
    cam.quat = list(arm_camera_quat())


def arm_camera_quat() -> np.ndarray:
    """The wrist camera's orientation in the gripper frame: x at
    `ARM_CAMERA_AIM`, z (image up) pointing away from the approach axis."""
    p, aim = np.asarray(ARM_CAMERA_POS, float), np.asarray(ARM_CAMERA_AIM, float)
    x = aim - p
    x /= np.linalg.norm(x)
    out = np.array([p[0] - aim[0], p[1] - aim[1], 0.0])
    z = out - x * (out @ x)
    if np.linalg.norm(z) < 1e-6:
        # On the approach axis there is no "away from it" to call up, and
        # normalising a zero vector would compile a NaN frame that silently
        # detects nothing.
        raise ValueError("ARM_CAMERA_POS lies on the approach axis through "
                         "ARM_CAMERA_AIM; the image's up is undefined")
    z /= np.linalg.norm(z)
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([x, np.cross(z, x), z]).ravel())
    return q


def add_camera(spec: mujoco.MjSpec) -> None:
    """Mount the RealSense as a massless frame on the chassis.

    A BODY and not a site, because `sensors/detector.Detector` resolves a
    mount by body when given one and reads its x axis as the optical axis —
    the URDF convention MARS's head camera already arrives in. Massless and
    geomless: it is a frame, so it changes nothing about the dynamics, and
    `set_base_inertial` is upstream of it anyway.

    The pose is `CAMERA_POS` / `CAMERA_PITCH_RAD`, both of which carry their
    provenance up there — the position is from one sentence in his build log
    and the pitch is measured in this lab, not on his robot.
    """
    cam = spec.body(BASE_BODY).add_body()
    cam.name = CAMERA_BODY
    cam.pos = list(CAMERA_POS)
    # Pitch DOWN about +y: MuJoCo's y is left, so a positive rotation about
    # it tips the x axis toward the floor.
    cam.quat = [math.cos(CAMERA_PITCH_RAD / 2), 0.0,
                math.sin(CAMERA_PITCH_RAD / 2), 0.0]


#: The forearm's visual mesh is 70 mm across at its widest; his collision
#: capsule for it is 26 mm (radius 13 mm). Same for the upper arm. So physics
#: stops a thin rod inside a fat arm, and up to 22 mm of visible forearm rides
#: through whatever it is next to.
ARM_PROXY_RADIUS_M = 0.035


def collision_v04() -> bool:
    """Is the lab using HIS V0.4 collision model? One place asks the env."""
    return os.environ.get("MICRODUCK_MOSS_COLLISION_V04", "1") == "1"


#: **THE WAY BACK TO THE TUCK POSE, which is not a straight line.**
#:
#: The tuck pose and the post-delivery pose are each collision-free, but the
#: BIN SITS BETWEEN THEM: commanding the arm straight to tuck and holding it
#: for 600 steps leaves it jammed on `bin_x1`, 0.65 rad short — which is the
#: same jam `brain/tidy_moss.py`'s tuck state already records as "0.74 rad
#: short and riding 75 mm proud". Ten hand-built routes scored 0-4/24 and 3M
#: steps of training plateaued, because there is no monotone path for either
#: a human or an optimiser to find.
#:
#: Found by running a one-waypoint planner (sample configurations, keep those
#: joined to BOTH ends by a collision-free straight segment) against 14 real
#: post-delivery poses and asking for one waypoint that connects all of them.
#: It connects 14/14, and flown in physics it folds the arm home on 14/24
#: seeds at a 0.154 rad residual, bringing the arm's forward reach to 114 mm —
#: the tuck pose's own value — against 178 mm for a direct attempt.
#:
#: Route through this FIRST, then to `tuck_pose()`.
RETRACT_WAYPOINT: tuple[float, ...] = (0.5259, -0.8036, -1.5515, -1.6033, 1.7983)


def tuck_pose() -> tuple[float, ...]:
    """The folded pose for whichever collision model is in use.

    They are not interchangeable and that is measured, not cautious: the
    legacy pose stands 13.31 mm outside the shell and 0.2962 m high on his
    hulls against a 0.261 m rim, and his pose stands 4.28 mm proud on the
    legacy proxies. A tuck is a statement about clearance, so it belongs to
    the geometry it was fitted against.
    """
    return TUCK_POSE_V04 if collision_v04() else TUCK_POSE


def tracks_as_support(spec: mujoco.MjSpec,
                      mu: float = TRACK_SUPPORT_FRICTION) -> None:
    """Make the V0.4 belt hulls hold the robot UP without fighting the drive.

    Adopting his collision candidate puts MOSS on the ground for the first
    time and stops it dead: commanded 0.2 m/s for 2 s it travels 3 mm instead
    of 400. The cause is not a gain. `base_x/base_y/base_yaw` pin z, so the
    rover's 42.4 N of weight is carried by the JOINT and the tracks bear
    0.00 N at rest — an unloaded track cannot generate traction, so friction
    there is pure parasitic drag whose normal force only appears when the
    chassis is pushed into the contact. The drive then pays the resistance
    twice and collects the propulsion never.

    Two coherent ways out. Give the rover a free root under gravity so the
    weight actually reaches the belts and friction becomes traction — that is
    a different robot and a different drive. Or say plainly that the belts
    are a SUPPORT surface under a frankly kinematic drive, which is what this
    does: they hold the robot at its ride height, collide with furniture, and
    stop it hovering, while the twist stays the controller's to deliver.

    The second is not a fudge as long as it is declared, and it is strictly
    better than the legacy boxes, which sit 4 mm PROUD of the floor and make
    zero contacts at all — every drive figure in this repo for this body was
    measured on a robot touching nothing. What it is NOT is a friction model:
    nobody should calibrate skid-steer width against hardware from it.
    """
    for name in TRACK_GEOMS:
        g = spec.geom(name)
        f = list(g.friction)
        f[0] = float(mu)
        g.friction = f
        g.priority = TRACK_CONTACT_PRIORITY


def reconcile_track_spacing(spec: mujoco.MjSpec) -> None:
    """Put the track collision boxes where the V0.4 CAD puts the belts.

    **NOT CALLED.** Available for whoever reconciles this WITH Laurent, which
    is what he asked for: "we should reconcile those together before calling
    it a physically updated V0.4 model." Applying it unilaterally is exactly
    what `test_the_track_width_is_measured_off_his_geoms_not_his_default`
    exists to stop, and running it anyway broke a second test — the rover saw
    2 of the 3 cans in its own yard instead of 3, because moving the contact
    patches moves how it sits and therefore where the camera points. That is
    the validation-on-hardware this needs, demonstrated on the way past.

    Laurent, 2026-09-24: "the CAD wheel/belt centre spacing is 266 mm,
    whereas the old collision boxes are 244 mm apart... we should reconcile
    those together before calling it a physically updated V0.4 model."

    His boxes sit at y +/-0.122 and the V0.4 belt meshes centre on +/-0.1330
    (read off their vertices), so each box is 11 mm inboard of the belt it
    stands for. This moves them out to the CAD.

    MEASURED, and the answer is not what it looks like: the spacing does NOT
    affect turning in this simulation. Commanding 0.3, 0.6 and 1.0 rad/s
    yields the same achieved yaw (-1.5% tracking error) at either spacing,
    because MOSS's base here is a KINEMATIC planar joint — `MossDriver` sets
    the body velocity directly and the tracks never propel anything. The
    spacing sets where the robot TOUCHES THE FLOOR, so it matters for
    tipping, for the footprint against furniture, and for the shell tests;
    it does not stand in for the drive.

    On the real robot the tracks do propel it, so there the spacing IS the
    turn rate — which is why `docs/moss-policy-schema.md` tells the deploy
    side to use 0.266 in its own `twist_to_tracks`, and why the -1.5% here
    is our servo's tracking error rather than a geometry error.
    """
    for name, sgn in (("track_1", +1.0), ("track_-1", -1.0)):
        g = spec.geom(name)
        pos = list(g.pos)
        pos[1] = sgn * TRACK_CENTRES_M / 2.0
        g.pos = pos


def widen_arm_proxies(spec: mujoco.MjSpec) -> None:
    """**NOT CALLED.** A uniform radius was the wrong fix; kept as the record.

    His 13 mm capsules do underrepresent a 70 mm arm, and this widened them
    to 35 mm. Laurent's correction is sharper than the problem: the LINKS ARE
    OFFSET FROM THE CAPSULE AXES, so a uniform radius still leaves lower and
    upper-arm vertices outside by ~17.6 and ~24.0 mm while adding bulk on the
    opposite side — it trades one wrong shape for another, fatter one.

    MEASURED, and it is not cosmetic: with these 35 mm capsules and the
    self-collision masks opened, 500,000 sampled poses contain ZERO that are
    inside the shell, below the bin rim and clear of the arm itself. The
    phantom bulk makes a legal tuck impossible. Against his per-part convex
    hulls (`live/model/collision_diagnostics/prepare_candidate.py`, which
    reuses each visual part in its original pose) such poses do exist — 2 in
    400,000, still a narrow corner, but not empty.

    The real fix is his hulls, not a radius. This stays as the reason why.
    """


def raise_bin_walls(spec: mujoco.MjSpec) -> None:
    """**WRONG, AND NOT CALLED.** Kept as the record of a measurement error.

    This raised the bin's four collision walls from z 0.261 to 0.2957 on the
    grounds that the V0.4 visual mesh reached 0.2957 and his boxes stopped
    35 mm short. The mesh does not reach 0.2957. Laurent found the error
    (2026-09-24): MuJoCo RECENTRES AND ROTATES mesh vertices at compile
    time, so `mesh_vert + geom_pos` is not the geometry — `geom_xmat` has to
    be applied, and then the result put into the rover's frame:

        world = v @ data.geom_xmat[gid].reshape(3, 3).T + data.geom_xpos[gid]
        rover = (world - data.xpos[rover]) @ data.xmat[rover].reshape(3, 3)

    Done properly the same 3276 vertices span z 0.108998 to 0.260000 and x
    -0.179 to +0.005. His tapered walls top out at 0.261 against a real rim
    of 0.260 — they were right all along, and this "fix" built a 35 mm FALSE
    OBSTACLE that the arm then had to work around. Every reach, jam and tuck
    number measured while it was in place is void.

    Reproducing the bad answer exactly: `mesh_vert + geom_pos` gives z
    0.063729..0.295730 and x -0.157744..-0.006675, which is what was
    reported. A number that looks decisive is worth nothing if the transform
    under it is wrong, and nothing in the sim complains.
    """


#: Above a room prop's 1, as `FINGER_PADS` are: the bin floor's contact
#: model governs a landing (`bin_floor_governs`).
BIN_FLOOR_PRIORITY = 2


def bin_floor_governs(spec: mujoco.MjSpec) -> None:
    """Let the BIN FLOOR's contact model govern what lands on it, as the jaw
    pads' governs what they grip (`tune_contacts`).

    A room gives its props priority 1 and a soft solref (0.02 s), and MuJoCo
    resolves a pair with the higher-priority geom's parameters, so a landing
    in the bin was the PROP's soft contact against a 4 mm floor. MEASURED
    2026-09-28: the cigarette butt (0.6 g, 8 mm) released at the stow's
    height meets the floor at ~2 m/s, sinks past its middle before the soft
    contact catches it, and is pushed out of the BOTTOM — into the 8 mm gap
    over the hull, where it rode ~5 min counted as "in the bin" and then slid
    out of the back while the robot drove: the one object lost from the bin
    in every moss-yard run (12/12). Under the floor's own (0.008 s) it lands
    on top. His geometry is unchanged.
    """
    try:
        floor = spec.geom("bin_floor")
    except Exception:
        return
    if floor is not None:
        floor.priority = BIN_FLOOR_PRIORITY


#: The VOID under the bin floor, sealed by `seal_bin_void`.
#:
#: MEASURED off the collision model (2026-09-30): his `hull` box tops out at
#: base z 0.100 and his `bin_floor` box starts at 0.1082, so there is an
#: **8.2 mm slot** running the whole way under the basket, open on every
#: side. The `v04_*` shell meshes close it visually — they are `contype 0`
#: and collide with nothing — so it exists only for physics.
BIN_VOID_Z = (0.100, 0.1082)


def seal_bin_void(spec: mujoco.MjSpec) -> None:
    """Fill the slot between the hull's top and the bin's floor.

    WHAT IT COSTS UNSEALED, measured over 64 moss-yard seeds (2026-09-30):
    in 4 of them a prop is driven through the 4 mm `bin_floor` and lands in
    the void, where it RIDES for the rest of the run — 123 to 207 s of a
    300 s run, pressed on `m0/hull` at 1.5 to 2.1 N. Every case is `cap0`,
    the 2 g bottle cap; every one sits at base x -0.13..-0.14, z 0.103 with a
    2 mm standard deviation, which is the slot. The head camera never reports
    it again (0 ticks of 10 000) because it is inside the robot.

    The mechanism is not a landing from height — `bin_floor_governs` already
    fixed that one for the cigarette butt. The cap is ALREADY AT REST on the
    bin floor when a second, heavier prop is released on top of it: in seed
    31 `tall0` (30 g, 170 mm) touches it at 0.53 N and it goes down 9 mm in
    one 20 ms tick, 0.47 m/s, far too fast for gravity. Squeezed between a
    heavy prop above and a 4 mm floor below, a 2 g object pops through.

    Sealing the void is the fix rather than stiffening the floor because it
    removes the TRAP instead of guessing at every way in — the butt reached
    the same slot by a different route. The filler exactly matches the bin
    floor's own footprint and sits directly beneath it, so nothing outside
    the robot can reach a surface it could not reach before, and it shares
    the `rover` body with the hull, so the two never collide with each other.

    His geometry is unchanged: this ADDS a geom, it edits none.
    """
    try:
        floor = spec.geom("bin_floor")
    except Exception:
        return
    if floor is None:
        return
    lo, hi = BIN_VOID_Z
    half = (hi - lo) / 2.0
    body = floor.parent
    g = body.add_geom()
    g.name = BIN_VOID_GEOM
    g.type = mujoco.mjtGeom.mjGEOM_BOX
    g.size = [float(floor.size[0]), float(floor.size[1]), half]
    g.pos = [float(floor.pos[0]), float(floor.pos[1]), lo + half]
    g.group = int(floor.group)
    g.rgba = [0.2, 0.3, 0.4, 0.0]
    # Mass would be double-counted against the explicit `<inertial>` the
    # rover carries (see "the mass" above); it must weigh nothing.
    g.mass = 0.0
    g.priority = BIN_FLOOR_PRIORITY


def tune_contacts(spec: mujoco.MjSpec) -> None:
    """Let the JAW's contact model govern whatever it grips.

    His pads already carry the right parameters — `condim 4` and friction
    1.4/0.01/0.001, which is what his own missions grasp on — but they carry
    them at the default priority, and MuJoCo resolves a pair by taking the
    HIGHER-priority geom's parameters wholesale.

    **MEASURED, and it is the difference between a working pick and a
    pointless one.** `world/compose.py` gives a room's props `priority 1` and
    `condim 3` with friction 0.8/0.005/0.0001. So in MOSS's own scene the
    scripted pick stowed **5/5** cans, and the identical state machine in a
    composed room stowed **0/3**: the can's own frictionless-in-torsion model
    was winning every contact, and a smooth cylinder between two flat pads
    simply spins out. Nothing about the brain or the arm was wrong.

    `robots/mars.tune_contacts` sets its blades to priority 2 for this exact
    reason, with the same one-line justification ("the finger's params govern
    every pair"). This is that, for his jaw.
    """
    for name in FINGER_PADS:
        pad = spec.geom(name)
        pad.priority = 2
        pad.condim = FINGER_CONDIM
        pad.friction = list(FINGER_FRICTION)


def couple_fingers(spec: mujoco.MjSpec) -> None:
    """One servo, two jaws — tied in the MODEL, not by convention.

    His exported MJCF gives each finger its own slide and its own position
    actuator; the physical gripper has a single servo driving both (Laurent,
    2026-09-24). An equality constraint makes the simulator obey that, so no
    policy, brain or hand-written script can produce a scissor motion the
    robot cannot perform — and the contract can carry ONE gripper command
    instead of two (`NUM_ACTIONS`).

    A constraint rather than deleting the second actuator: the follower still
    has its own servo and its own contact forces, which is what a real jaw
    linkage has too. What it does not have is an independent command.
    """
    eq = spec.add_equality()
    eq.name = "gripper_coupling"
    eq.type = mujoco.mjtEq.mjEQ_JOINT
    eq.objtype = mujoco.mjtObj.mjOBJ_JOINT
    eq.name1, eq.name2 = FINGER_JOINTS[1], FINGER_JOINTS[0]
    # follower = 0 + 1.0 * leader: the jaws mirror, they do not scissor.
    eq.data[:5] = [0.0, 1.0, 0.0, 0.0, 0.0]
    eq.active = True
    # STIFF. At MuJoCo's default softness the pair ratchets: contact impulses
    # on the unactuated follower push it out a little each grasp and the
    # constraint drags the leader with it, so over a 300 s room run the
    # leader's reported position climbed 0.042 -> 0.054 m, past its own
    # 0.041 limit. A grip test that reads "commanded minus achieved" then
    # sees a permanent 40 mm stall and calls every empty jaw a catch.
    eq.solref = [0.002, 1.0]
    eq.solimp = [0.99, 0.999, 0.001, 0.5, 2.0]
    # ...and the follower loses its SERVO, because an equality constraint is
    # soft and two opposing position servos simply fight it: MEASURED, with
    # both actuators present and commanded 31 mm apart, the jaws settled
    # 11 mm apart and the coupling bought nothing. One servo is also the
    # literal truth of the hardware. `nu` is then 6 — five arm joints and one
    # gripper — which is exactly the joint half of the 8-action contract.
    spec.delete(spec.actuator(FINGER_JOINTS[1]))


def drop_base_servo(spec: mujoco.MjSpec) -> None:
    """Remove his `base_x` position actuator.

    The base is pushed through `xfrc_applied` here (`robots/mars_drive.py`),
    and a kp-3000 position servo on one of the three planar DoFs would not
    merely be redundant: it would hold x while the drive steered in y and
    yaw, so a commanded arc would come out as a sideways crab back onto his
    rail. Everything else in his `<actuator>` block — the five arm servos and
    the two finger servos — is untouched.
    """
    spec.delete(spec.actuator(BASE_JOINTS[0]))


def set_base_inertial(spec: mujoco.MjSpec, *, mass: float = BASE_MASS_KG,
                      com_z: float = BASE_COM_M[2]) -> None:
    """Give the rover Laurent's weighed mass and balance point.

    Without this the chassis is `SHIPPED_BASE_MASS_KG` of solid PETG,
    and the number is load-bearing for the half of this robot the lab is
    building: `mars_drive`'s velocity PD is an EXPLICIT force loop, stable
    only while `KP*dt/M < 2`, so every gain is picked against the base's
    apparent inertia. A drive tuned at 6.139 kg would be tuned for a robot
    that does not exist.

    His model already carries an explicit `<inertial>` on `rover` (line 86),
    so this REPLACES one rather than overriding the geoms — which is the
    distinction he drew: with an explicit inertial in the file, editing geom
    masses would change nothing at all. The shape stays his exported tensor,
    the scale becomes his weighed one, and no bin wall has to pretend to be
    2 kg of plastic.

    `com_z` is a knob because he asked for one: 75 mm is his estimate from
    the geometry, and 50-100 mm is the sensitivity sweep
    (`BASE_COM_Z_SWEEP`). The arm's links are untouched, so the COMBINED
    centre of mass still moves with the arm — his other request, and it comes
    free from not lumping them together.
    """
    base = spec.body(BASE_BODY)
    # The COM is COMPOSED from his three weighed components, not asserted:
    # the equipped base at the balance point he measured, the cover and the
    # bin at their own V0.4 centroids. `com_z` shifts the BASE component,
    # which is the part that is an estimate; the other two are geometry.
    total = sum(m for _n, m, _c in BASE_COMPONENTS)
    scale = float(mass) / total
    cx = sum(m * c[0] for _n, m, c in BASE_COMPONENTS) / total
    cz = sum(m * (float(com_z) if _n == "equipped_base" else c[2])
             for _n, m, c in BASE_COMPONENTS) / total
    base.mass = float(mass)
    base.ipos = [cx, 0.0, cz]
    base.iquat = list(BASE_INERTIA_QUAT)
    base.inertia = [v * scale for v in BASE_INERTIA_KGM2]
    base.explicitinertial = True


def apply_solver_options(spec: mujoco.MjSpec) -> None:
    """Write his `<option>` onto `spec` — `SOLVER_OPTIONS`.

    Called on the ROBOT's own spec (where it is a no-op, since these are the
    values his file already carries) and on a ROOM's, where it is the point:
    `MjSpec.attach` keeps the parent's option block, so a MOSS attached into
    a scene compiled with the defaults silently loses his integrator and
    cone. The timestep is not here — `Scenario.physics_dt` carries a room's
    clock, and a room is entitled to know its own.
    """
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.iterations = int(SOLVER_OPTIONS["iterations"])


def rewrite_home_key(spec: mujoco.MjSpec) -> None:
    """Widen his `home` keyframe to the planar base, from his own numbers.

    His key is 8 qpos (base_x first, then the arm) and 8 ctrl (the arm first,
    base_x last). After `add_planar_base` there are two more DoFs and after
    `drop_base_servo` one less actuator, so the rows have to move — and they
    are moved by SLICING his, never by typing `ARM_HOME` in again. The
    module-level constant is for readers and for `DEFAULT_POSE`; the spec's
    truth is the file's.
    """
    key = spec.key(HOME_KEY)
    qpos, ctrl = np.asarray(key.qpos, float), np.asarray(key.ctrl, float)
    if qpos.size != 8 or ctrl.size != 8:
        raise RuntimeError(
            f"{HOME_KEY!r} has {qpos.size} qpos and {ctrl.size} ctrl, "
            "expected 8 and 8 — moss_robot.xml changed shape, so this "
            "rewrite is guessing (robots/moss.ASSETS pins the revision)")
    # base_x keeps his value (~0); base_y and base_yaw spawn at zero.
    key.qpos = [qpos[0], 0.0, 0.0, *qpos[1:]]
    # His trailing base_x command goes with `drop_base_servo`, and one finger
    # command goes with `couple_fingers` — six actuators remain: five arm
    # joints and the single gripper servo the real robot has.
    key.ctrl = list(ctrl[:6])


def robot_spec(xml: Path | None = None) -> mujoco.MjSpec:
    """His robot with the lab's three edits — the model this body IS.

    Never the scene: a scene carries a floor and a light, and attaching those
    into a room would give it two floors.

    Not cached. `MjSpec` is mutable and `attach` consumes one, so a shared
    instance would let a room with two MOSSes in it edit the other's robot.
    """
    if xml is None and collision_v04():
        cand = apply_collision_candidate()
        if cand is not None:
            # HIS collision model, with our edits already baked in — see
            # `apply_collision_candidate`. Nothing more to do to it: the
            # patcher preserves masses, joints, actuators, equalities, the
            # keyframe and the camera, and replaces exactly the geometry we
            # had each got wrong (bin walls, arm proxies, track boxes).
            cspec = mujoco.MjSpec.from_file(str(cand))
            # His hulls REACH THE FLOOR, which the legacy boxes never did.
            # Without this the rover is on the ground and cannot move; see
            # `tracks_as_support` for why the answer is the belts' friction
            # rather than the drive's gain.
            tracks_as_support(cspec)
            add_arm_camera(cspec)
            bin_floor_governs(cspec)
            seal_bin_void(cspec)
            return cspec
    spec = load_robot_spec(xml)
    add_planar_base(spec)
    add_camera(spec)
    add_arm_camera(spec)
    couple_fingers(spec)
    tune_contacts(spec)
    bin_floor_governs(spec)
    seal_bin_void(spec)
    drop_base_servo(spec)
    set_base_inertial(spec)
    rewrite_home_key(spec)
    return spec


def _scene_spec(timestep: float | None = None) -> mujoco.MjSpec:
    """The standalone scene: his robot, a floor, a light, his keyframe.

    The floor is `FLOOR_GEOM` at the world origin, and the robot is NOT
    attached — it is already the spec's own worldbody child, so this only
    adds the two things a standalone scene needs. The timestep defaults to
    his (`GRASP_PHYSICS_DT`); a task that needs another passes it, the way
    `robots/mars_env` does.
    """
    spec = robot_spec()
    apply_solver_options(spec)
    spec.option.timestep = float(timestep or GRASP_PHYSICS_DT)
    spec.worldbody.add_light(pos=[0, 0, 3.0], dir=[0, 0, -1],
                             type=mujoco.mjtLightType.mjLIGHT_DIRECTIONAL)
    spec.worldbody.add_geom(
        name=FLOOR_GEOM, type=mujoco.mjtGeom.mjGEOM_PLANE, size=[4, 4, 0.05],
        rgba=[0.05, 0.09, 0.13, 1.0])
    return spec


def scene_spec(timestep: float | None = None) -> mujoco.MjSpec:
    """The standalone scene as an editable spec, for a task that adds to it
    (a can, a bin target). A caller that adds a free-jointed body must also
    widen the keyframe's qpos — `MjSpec.key(HOME_KEY)`."""
    return _scene_spec(timestep)


def write_scene_xml(spec: mujoco.MjSpec, path: Path) -> Path:
    """Write `spec` to `path` atomically, and only when the content differs.

    Beside the assets, so `meshdir` still resolves, and never over a
    downloaded file. A vec-env worker must never import a half-written scene
    (AGENTS.md, "Atomic writes and live imports").
    """
    xml = spec.to_xml()
    if not path.exists() or path.read_text() != xml:
        tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        tmp.write_text(xml)
        os.replace(tmp, path)
    return path


def scene_xml() -> Path:
    """Path to the generated scene, written under the cache."""
    require_moss()
    return write_scene_xml(_scene_spec(), asset_dir() / SCENE_NAME)


@lru_cache(maxsize=1)
def model() -> mujoco.MjModel:
    """The compiled standalone scene. Cached: the 57 meshes cost ~0.3 s."""
    return mujoco.MjModel.from_xml_path(str(scene_xml()))


def visual_scene() -> dict:
    """The viewer's mesh dump for one MOSS — his 57 visual meshes.

    Built from the robot alone (no floor, no light) so the body list is
    exactly what an `attach` into a room produces. `g1.extract_visual_scene`
    is the body-agnostic dump; the per-body things in it are the group (his
    is 2) and the lattice.

    **0.1 mm, not the G1's 1 mm.** His front camera plate is 0.4 mm thick
    and straddles the RealSense's front face at x = 0.156 m; on a millimetre
    lattice both of its faces round onto that face and the browser z-fights
    the dark plate against the grey box. At 0.1 mm the plate keeps its
    0.2 mm standoff, for 6.24 → 6.65 MB (MEASURED 2026-09-26).
    """
    from .g1 import extract_visual_scene
    return extract_visual_scene(robot_spec().compile(), group=VISUAL_GROUP,
                                vert_scale=0.0001)


def home_qpos(model: mujoco.MjModel, prefix: str = "") -> np.ndarray:
    """A full-width qpos vector with THIS robot's joints at HOME.

    Full width so a caller can index its own addresses out of it: assigning
    the whole vector in a composed world would zero every other body.
    """
    qpos = np.array(model.qpos0, float)
    for name, value in ARM_HOME.items():
        qpos[model.joint(prefix + name).qposadr[0]] = value
    for name in BASE_JOINTS:
        qpos[model.joint(prefix + name).qposadr[0]] = 0.0
    return qpos


# ------------------------------------------------------------------ the body

class MossBody(BodyBase):
    """MOSS's half of the lab's `Body` contract — a wheeled body that DRIVES.

    `BodyBase`, like MARS and for the same reason: no `foot_geoms`, no
    `fall_height`, no gyro, no twist ranges. A planar base cannot fall, and
    inheriting a walker's fields is how a new body ends up sagging into
    another robot's proxy.

    **Level 2 on `docs/mars-roadmap.md` §7.1's ladder, reached from the other
    end.** Level 1 is "it trains" and level 2 is "it lives in a room", and
    MOSS arrives with the second and not the first: what it has is a
    CONTROLLER (`robots/moss_drive.MossDriver`) and a room to drive in, while
    the task that would earn level 1 is still ahead. That is the right order
    for this robot specifically — his grasp already works in his own
    simulator, and the base is the half he published asking for help with.

    **Senses: none declared, deliberately.** The real rover carries a
    RealSense D455 on a Jetson, but `moss_robot.xml` has no camera site and
    no scanner — the Jev missions read MuJoCo state directly, which his
    INTEGRATION.md says in as many words. So `make_sensors` is inherited and
    answers `{}`: a body with no apertures in its model declares none, and a
    MOSS in a room falls its brain back to `script` exactly as a blind duck
    does. Mounting a camera is a measurement (his lens, his mount height),
    not a line to type here.
    """

    #: What a jaw READING means on this robot — the pair `world/arena` looks
    #: for with `getattr` defaults, so a body without a claw simply has no
    #: `gripper` block in its frame. MOSS's finger servos are `position`
    #: actuators with `forcerange` +-8 N; the hold threshold is NOT declared
    #: because nothing here has measured what "holding" reads as on his pads.
    gripper_limit_nm = 8.0

    @property
    def num_actions(self) -> int:
        return NUM_ACTIONS

    def contract(self):
        """MOSS's contract v0 — and it says what it does not have.

        Declared before the env exists for MARS's reason (a body speaks one
        contract from the day it is listed), and `deploy` is where the honesty
        goes: his Jetson side takes a Twist over dimOS and his ESP32 takes
        `drive LEFT RIGHT` at 115200 baud, so a policy trained here would
        have to leave as a twist source and a jaw command, not as a joint
        stream. Nothing in this repo has ever driven a MOSS, and the rover
        that exists has driven for two days.
        """
        return declare(
            self, id=CONTRACT_ID, rate_hz=CONTROL_HZ,
            slots=tuple((name, sl.start, sl.stop) for name, sl in (
                ("joint_pos", OBS_JOINT_POS), ("joint_vel", OBS_JOINT_VEL),
                ("last_action", OBS_LAST_ACTION),
                ("base_twist", OBS_BASE_TWIST),
                ("target_base", OBS_TARGET_BASE),
                ("target_seen", OBS_TARGET_SEEN),
                ("reserved", OBS_RESERVED))),
            deploy=("five arm-joint INCREMENTS (+-0.03 rad/step) and ONE "
                    "gripper increment (+-0.004 m/step, one servo for both "
                    "jaws), plus a twist (vx, wz) for the ESP32's `drive "
                    "LEFT RIGHT` through his diffdrive mapping at 0.266 m "
                    "track spacing. Full schema with units: "
                    "docs/moss-policy-schema.md. ONE LEG of the mission "
                    "loop: approach, pick and stow are each their own "
                    "policy under this same contract, loaded per state by "
                    "brain/tidy_moss.py, and only search is scripted. The "
                    "run's own record.json says which leg this file is. "
                    "Untested on hardware."))

    def ready(self) -> bool:
        return moss_ready()

    def fetch(self) -> Path:
        return fetch()

    # -------------------------------------------------------------- viewer

    def visual_scene(self) -> dict:
        return visual_scene()

    def look(self) -> str:
        """The viewer's `generic` material table: paint the body by its own
        MJCF rgba. His meshes carry their colours (he recoloured them for the
        browser), so a MOSS-specific table would be inventing a second
        opinion about a robot whose own is already in the file."""
        return "generic"

    # ------------------------------------------------------------ training

    def env_class(self, task: str = "pick") -> type:
        """The env that trains `task` on MOSS — `robots/moss_env.py`.

        THREE of the mission loop's four legs, each its own env because each
        fails its own way: `approach` drives up to a can without toppling it,
        `pick` brings it between the pads and lifts it, `stow` carries it to
        the bin and lets go INSIDE. Only `search` is still scripted in
        `brain/tidy_moss.py` — a spin-and-scan with nothing to learn — which
        is the same split this lab uses on MARS.

        Imported lazily, like the G1's and MARS's: a machine with no MOSS
        assets must still be able to train the duck.
        """
        from .moss_env import TASKS, MossApproachEnv, MossPickEnv, MossStowEnv
        by_task = {"pick": MossPickEnv, "approach": MossApproachEnv,
                   "stow": MossStowEnv}
        if task in by_task:
            return by_task[task]
        raise SystemExit(
            f"unknown --task {task!r} for {self.id} (have: "
            f"{', '.join(TASKS)}) — driving is a CONTROLLER here "
            "(robots/moss_drive.py), not something trained")

    # `tasks()` is INHERITED: `BodyBase.tasks` answers
    # `behaviors.for_robot(self.id)`, which is the Behavior RECIPES the lab
    # lists -- not the env's task strings. Overriding it with
    # `moss_env.TASKS` handed the palette a tuple of `str` and four lab tests
    # died on `b.trainer`. The env's task names are `env_class`'s business.

    def repurposed_obs_dims(self, env_kwargs: dict, donor_dir) -> list[int]:
        """Slots whose MEANING a warm start changes: the donor's normaliser
        holds statistics of something else there, so they must start as
        pass-through (`train.py`). `publish_grip` puts the grip in the
        duplicate jaw slots (`OBS_GRIP`) of a donor that had not."""
        import json as _json
        from pathlib import Path as _P
        try:
            kw = _json.loads((_P(donor_dir) / "run.json").read_text()
                             ).get("env_kwargs") or {}
        except (OSError, ValueError):
            kw = {}
        dims: list[int] = []
        if env_kwargs.get("grip_xyz") and not kw.get("grip_xyz"):
            dims += list(OBS_GRIP_XYZ)
        elif env_kwargs.get("publish_grip") and not kw.get("publish_grip"):
            dims += list(OBS_GRIP)
        # `publish_proximity` turns the target's HEIGHT (slot 26) into the
        # wrist camera's range: the same slot, a different quantity
        if env_kwargs.get("publish_proximity") and not kw.get("publish_proximity"):
            dims.append(OBS_TARGET_BASE.start + 2)
        return dims

    def train_env_kwargs(self, args) -> dict:
        """MOSS's per-body knob: which rung of the current task's ladder.

        One variable per task — `MICRODUCK_MOSS_PICK_RUNG`,
        `MICRODUCK_MOSS_APPROACH_RUNG`, `MICRODUCK_MOSS_STOW_RUNG` — the way
        MARS carries its own, so a curriculum chain is a sequence of runs
        with the rung recorded in each run's `env_kwargs` rather than a flag
        nobody can read back. The env var is read HERE rather than inside the
        env for exactly that reason: an env that reads its own environment
        trains fine and leaves no record of what it trained on.
        """
        from .moss_env import APPROACH_RUNG_BOX, RUNGS, STOW_RUNG_GRIP
        ladders = {
            "pick": ("MICRODUCK_MOSS_PICK_RUNG", "pick_rung", tuple(RUNGS)),
            "approach": ("MICRODUCK_MOSS_APPROACH_RUNG", "approach_rung",
                         tuple(APPROACH_RUNG_BOX)),
            "stow": ("MICRODUCK_MOSS_STOW_RUNG", "stow_rung",
                     tuple(STOW_RUNG_GRIP)),
        }
        task = getattr(args, "task", None) or self.default_task
        out: dict = {}
        # WHICH WRIST CAMERA this run trains behind — every task, because the
        # model carries it whatever the task. Without it a policy trained on
        # the retired mount is scored on the new one with nothing to say so
        # (478dad: -4.4 points, 2026-09-27). See `ARM_CAMERA_MOUNT`.
        out["arm_camera_mount"] = ARM_CAMERA_MOUNT
        # WHICH OBSERVATION THIS RUN TRAINS ON, as a recorded kwarg rather than
        # a process environment variable. A policy trained with the can's axis
        # in slots 28-30 and one trained without are NOT interchangeable: the
        # normalizer baked into an axis-blind export has var 3e-10 there, so
        # handing it a live axis clips three inputs at the bound and it stops
        # picking up cans (10/12 -> 0/12, and 59/60 -> 0/60 in a probe that
        # made this exact mistake). Recording it in run.json is what lets
        # `eval_moss_pick.py` and the brain pick the right env per policy
        # instead of relying on whoever runs them remembering.
        if task == "pick":
            # READ THE ENVIRONMENT HERE, not module constants. `_body_env_kwargs`
            # stages these variables around this call so the lab's in-process
            # PREVIEW runs the same physics as the trainer subprocess — but a
            # module constant was evaluated at IMPORT time and cannot see that
            # staging. Every knob added on 2026-09-25 was a module constant, so
            # the stage showed a plain upright can while the trainer ran varied
            # shapes with the base locked: the watcher was told "here is your
            # training" and shown something else. The rung was always read this
            # way, which is why it was the only one that worked.
            def _f(name, default=0.0):
                return float(os.environ.get(name, default) or default)

            def _b(name):
                return os.environ.get(name, "0") not in ("", "0")

            out["publish_attitude"] = _b("MICRODUCK_MOSS_ATTITUDE")
            out["publish_size"] = _b("MICRODUCK_MOSS_SIZE_OBS")
            out["publish_proximity"] = _b("MICRODUCK_MOSS_PROXIMITY")
            out["prop_variety"] = _b("MICRODUCK_MOSS_PROP_VARIETY")
            out["wrist_drill"] = _b("MICRODUCK_MOSS_WRIST_DRILL")
            out["base_lock"] = _b("MICRODUCK_MOSS_BASE_LOCK")
            out["wrist_free"] = _b("MICRODUCK_MOSS_WRIST_FREE")
            out["wrist_start_rand"] = _f("MICRODUCK_MOSS_WRIST_START")
            out["jaw_align"] = _f("MICRODUCK_MOSS_JAW_ALIGN")
            out["align_hold"] = _f("MICRODUCK_MOSS_ALIGN_HOLD")
            out["gap_scale"] = _f("MICRODUCK_MOSS_GAP_SCALE", 1.0)
            out["can_topple"] = _f("MICRODUCK_MOSS_TOPPLE")
            out["torque_sat"] = _f("MICRODUCK_MOSS_TORQUE_SAT")
            out["overspeed"] = _f("MICRODUCK_MOSS_OVERSPEED")
            out["arm_floor"] = _f("MICRODUCK_MOSS_ARM_FLOOR")
            out["low_approach"] = _f("MICRODUCK_MOSS_LOW_APPROACH")
            out["handover_bank"] = _b("MICRODUCK_MOSS_HANDOVER_BANK")
            out["deep_grip_m"] = _f("MICRODUCK_MOSS_DEEP_GRIP")
            out["pick_box"] = os.environ.get("MICRODUCK_MOSS_PICK_BOX", "")
            out["gap_from_tcp"] = _b("MICRODUCK_MOSS_GAP_TCP")
            out["litter"] = _b("MICRODUCK_MOSS_LITTER")
            out["sphere_centre_m"] = _f("MICRODUCK_MOSS_SPHERE_CENTRE")
            out["publish_grip"] = _b("MICRODUCK_MOSS_GRIP_OBS")
            out["handover_states"] = _b("MICRODUCK_MOSS_HANDOVER_STATES")
            out["ball_frac"] = _f("MICRODUCK_MOSS_BALL_FRAC")
            out["grip_xyz"] = _b("MICRODUCK_MOSS_GRIP_XYZ")
            out["handover_frac"] = _f("MICRODUCK_MOSS_HANDOVER_FRAC", 1.0)
            out["max_episode_s"] = _f("MICRODUCK_MOSS_PICK_EPISODE_S", 8.0)
        if task == "stow":
            # The stow leg's own knobs. Same rule as the pick's: read the
            # ENVIRONMENT here so the lab's in-process preview shows the same
            # episode the trainer runs — including the RETRACT phase, without
            # which the watched arm ends the episode parked over the bin.
            def _f2(name, default=0.0):
                return float(os.environ.get(name, default) or default)

            def _b2(name):
                return os.environ.get(name, "0") not in ("", "0")

            out["retract"] = _f2("MICRODUCK_MOSS_RETRACT")
            out["bin_scrape"] = _f2("MICRODUCK_MOSS_BIN_SCRAPE")
            out["home_bonus"] = _f2("MICRODUCK_MOSS_HOME_BONUS")
            out["retract_staged"] = _b2("MICRODUCK_MOSS_RETRACT_STAGED")
            out["retract_wp_bonus"] = _f2("MICRODUCK_MOSS_RETRACT_WP_BONUS")
            out["retract_drill"] = _b2("MICRODUCK_MOSS_RETRACT_DRILL")
            out["retract_drill_wp"] = _f2("MICRODUCK_MOSS_RETRACT_DRILL_WP")
            out["bin_clutter"] = int(_f2("MICRODUCK_MOSS_BIN_CLUTTER"))
            out["drop_target"] = _b2("MICRODUCK_MOSS_DROP_TARGET")
            out["drop_accuracy"] = _f2("MICRODUCK_MOSS_DROP_ACCURACY")
            out["valid_start"] = _b2("MICRODUCK_MOSS_VALID_START")
            out["retract_drill_path"] = _f2("MICRODUCK_MOSS_RETRACT_DRILL_PATH")
            # "1" = the default bank; any other non-"0" value names a bank
            # FILE under robots/data/ (recorded in run.json as that name)
            _bk = os.environ.get("MICRODUCK_MOSS_RETRACT_DRILL_BANK", "0")
            out["retract_drill_bank"] = (False if _bk in ("", "0") else
                                         True if _bk == "1" else _bk)
            out["cmd_leash"] = _f2("MICRODUCK_MOSS_CMD_LEASH")
            out["overspeed"] = _f2("MICRODUCK_MOSS_OVERSPEED")
            out["prop_variety"] = _b2("MICRODUCK_MOSS_PROP_VARIETY")
        if task not in ladders:
            return out
        var, kwarg, rungs = ladders[task]
        raw = os.environ.get(var)
        if not raw:
            return out
        try:
            rung = int(raw)
        except ValueError as e:
            raise SystemExit(
                f"{var}={raw!r} is not an integer; the rungs are "
                f"{', '.join(str(r) for r in rungs)}") from e
        if rung not in rungs:
            raise SystemExit(
                f"{var}={rung} is not a rung of the {task} ladder "
                f"({', '.join(str(r) for r in rungs)}) — a spawn box nobody "
                "measured is not a curriculum")
        out[kwarg] = rung
        return out

    def shipped_policies(self) -> tuple[dict, ...]:
        """None. His repo ships recorded MISSIONS — Jev's per-step choices
        replayed against his planner — not a network: `live/recordings/*.json`
        are decisions and observations, and `live/policy.py` is the planner
        that executes them. There is nothing here to load as an ONNX."""
        return ()

    # ----------------------------------------------------------- /sim world

    def attach(self, spec, prefix: str, frame) -> None:
        """Put one MOSS in a world model under `prefix`.

        The child's `<option>` is written to the PARENT's timestep first.
        `MjSpec.attach` keeps the parent's block either way, but it WARNS
        when the two disagree, and his differs from the world's in three
        fields (timestep, integrator, cone). Saying it explicitly is also the
        more honest spelling: the room's clock is the robot's clock, and a
        MOSS room is 2 ms because `Scenario.physics_dt` asked this body.
        """
        robot = robot_spec()
        robot.option.timestep = spec.option.timestep
        spec.attach(robot, prefix=prefix, frame=frame)

    def apply_solver_options(self, spec) -> None:
        """Write his `<option>` onto a ROOM's spec — `SOLVER_OPTIONS`.

        The seam for "give it its own environment" (2026-09-24). A room asks
        every body it holds for this hook before it attaches anything
        (`world/compose`), because `MjSpec.attach` keeps the parent's option
        block and a MOSS compiled into a default scene would silently lose
        the integrator and the friction cone his grasp was built under. A
        body that does not declare the hook changes nothing, so every duck
        room compiles to the model it always did.
        """
        apply_solver_options(spec)

    def driver(self, model, prefix: str):
        """One `MossDriver` on this model — `robots/moss_drive.py`.

        A velocity PD on the planar base, with his own drive envelope
        (`moss_dimos/diffdrive.py`: 0.30 m track width, 0.6 m/s, scale rather
        than clip) between the command and the loop, and his arm held at HOME
        by the servos already in his MJCF.

        Imported here rather than at module level: `moss_drive` imports this
        module for the names, and a top-level import would be a cycle.
        """
        from .moss_drive import MossDriver
        return MossDriver(model, prefix)

    def make_sensors(self, model, prefix: str, *, presets, targets, seed) -> dict:
        """MOSS's one aperture: the RealSense up front, as a detector —
        and, when the scenario gives it a range preset, its DEPTH as a scan.

        There is no lidar — his chassis carries none, and the Pi variants on
        his BOM (a Mighty camera, an ST time-of-flight board) are experiments
        he has not built. The range sensor is the RealSense's own depth
        stream, mounted as `lidar` (`DEPTH_SCAN_RAYS`' comment) when the
        scenario's `tof` field names a preset, which is the field's meaning
        on every body (the range-sensor preset). `tof: null` is a MOSS with
        the colour stream only: the lab falls the brain back to `script`
        unless the scenario asks for a detector.

        The lens is the D455's datasheet RGB field, NOT
        `DetectorSpec.from_env()`: `MICRODUCK_CAMERA` is a knob for the
        DUCK's camera variants, and a battery that sets it is asking a
        question about a wide M12 board on a 25 cm duck, not about a
        RealSense on a rover.
        """
        from ..sensors import Detector, DetectorNoise, DetectorSpec
        out: dict = {}
        preset = presets.get("detector")
        if preset is not None:
            out["detector"] = Detector(
                model, body=prefix + CAMERA_BODY,
                spec=DetectorSpec(fov_h_deg=CAMERA_HFOV_DEG,
                                  fov_v_deg=CAMERA_VFOV_DEG,
                                  max_range_m=CAMERA_MAX_RANGE_M,
                                  rate_hz=CAMERA_RATE_HZ),
                noise=DetectorNoise.preset(preset), targets=targets,
                seed=seed())
            # THE WRIST CAMERA, as its own detector. Not more coverage — the
            # front RealSense already sees the room — but ATTITUDE: it looks
            # down the approach from 12 cm behind the jaws, where a lying can is a shape
            # rather than the silhouette the chassis camera reads at 40.
            # Same class of sensor, different question, so it is a second
            # `Detector` rather than a wider spec on the first.
            out["arm_detector"] = Detector(
                model, body=prefix + ARM_CAMERA_BODY,
                spec=DetectorSpec(fov_h_deg=ARM_CAMERA_HFOV_DEG,
                                  fov_v_deg=ARM_CAMERA_VFOV_DEG,
                                  max_range_m=ARM_CAMERA_MAX_RANGE_M,
                                  rate_hz=ARM_CAMERA_RATE_HZ),
                noise=DetectorNoise.preset(preset), targets=targets,
                seed=seed())
        # The depth, AFTER both detectors: `seed()` draws off the world's RNG
        # and a draw before theirs would reseed every detector in every
        # measured moss-yard run.
        depth = presets.get("tof")
        if depth is not None:
            from ..sensors import LidarNoise, LidarSensor
            out["lidar"] = LidarSensor(
                model, prefix + CAMERA_BODY, n_rays=DEPTH_SCAN_RAYS,
                fov_deg=CAMERA_HFOV_DEG, max_range=DEPTH_MAX_RANGE_M,
                min_range=DEPTH_MIN_RANGE_M, rate_hz=CAMERA_RATE_HZ,
                noise=LidarNoise.preset(depth), seed=seed(),
                base_body=prefix + BASE_BODY, centred=True)
        return out

    def frames(self) -> RobotFrames:
        """`rover` is the subtree root — every body of a MOSS hangs off it,
        and it carries the planar base's three DoFs.

        No `head_pitch_joint`: MOSS has no neck. A gaze intent reaches
        nothing on this robot, which `world/arena.WorldRobot.set_cmd` drops
        the way it drops the duck's three extra gaze slots on MARS. What MOSS
        would point at a target is the arm, and that is a task, not a frame.
        """
        return RobotFrames(base=BASE_BODY)


MOSS = MossBody(
    id="moss",
    title="MOSS (Show Robotics)",
    noun="MOSS",
    kind="wheeled",
    joint_names=JOINT_NAMES,
    joint_groups=JOINT_GROUPS,
    default_pose=DEFAULT_POSE,
    obs_dim=OBS_DIM,
    lab_spacing_m=LAB_SPACING_M,
    scene_fn=scene_xml,
    stand_keyframe=HOME_KEY,
    # A lab slot with no task named DRIVES: there is no trainable task yet,
    # and `BodyBase`'s "walk" default would reach `env_class`'s raise for
    # every roster slot rather than for a typo.
    default_task="pick",
    # A room with a MOSS in it runs at his 2 ms, not the world's 5 ms —
    # `world/scenario.robot_physics_dt` asks the body, so no builder names
    # this robot to get it.
    physics_dt=GRASP_PHYSICS_DT,
)


__all__ = ["ARM_HOME", "ARM_JOINTS", "ASSETS", "BASE_BODY", "BASE_JOINTS",
           "BASE_COMPONENTS", "BIN_FLOOR_Z", "BIN_INTERIOR_X",
           "BIN_INTERIOR_Y", "BIN_MASS_KG", "BIN_RIM_Z", "COVER_MASS_KG",
           "EQUIPPED_BASE_MASS_KG",
           "SHIPPED_BASE_MASS_KG",
           "CACHE_DIR", "CONTRACT_ID", "CONTROL_HZ", "FINGER_JOINTS",
           "REAR_EXTENT_M", "FRONT_EXTENT_M", "HALF_WIDTH_M",
           "BIN_VOID_GEOM", "BIN_VOID_Z", "seal_bin_void",
           "DEPLOY_STANDOFF_M", "DROP_POSE", "GRASP_HEIGHT_M",
           "GRASP_JAW_CTRL_M", "GRASP_PHYSICS_DT", "GRASP_POSE",
           "GRASP_STANDOFF_M", "LIFT_POSE", "TUCK_POSE",
           "JOINT_NAMES", "MOSS", "MOSS_JEV_SHA",
           "MOSS_LICENCE", "NUM_ACTIONS", "OBS_DIM", "MossBody",
           "CAMERA_BODY", "CAMERA_POS", "GRIPPER_JOINT", "add_camera", "add_planar_base",
           "apply_v04_visuals", "couple_fingers", "PICK_HANDOVER_BOX",
           "ARM_CAMERA_BODY", "ARM_CAMERA_POS", "ARM_CAMERA_AIM",
           "ARM_CAMERA_MOUNT", "ARM_CAMERA_MOUNT_LEGACY", "arm_camera_quat",
           "add_arm_camera",
           "OBS_TARGET_AXIS", "OBS_TARGET_UPRIGHT", "OBS_SPARE", "OBS_DROP",
           "tracks_as_support", "TRACK_SUPPORT_FRICTION", "TUCK_POSE_V04",
           "tuck_pose", "collision_v04",
           "TRACK_SUPPORT_KP_YAW",
           "tune_contacts",
           "apply_solver_options", "asset_dir",
           "drop_base_servo", "fetch",
           "home_qpos", "load_robot_spec", "model", "moss_ready",
           "require_moss", "rewrite_home_key", "robot_spec", "robot_xml_path",
           "set_base_inertial",
           "scene_spec", "scene_xml", "visual_scene", "write_scene_xml"]
