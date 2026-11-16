"""The mirror signs derived from the joint axes equal BlinkMimic's rule for the G1: roll and yaw joints are negated.

    JAX_PLATFORMS=cpu python tests/test_mirror.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from xmimic.algorithm import Mirror  # noqa: E402
from xmimic.model import load_meta  # noqa: E402
from xmimic.sim import XiLayout  # noqa: E402

meta = load_meta()
mirror = Mirror(meta, XiLayout(meta))
names = meta["joint_names"]
for n, p, s in zip(names, mirror.perm, mirror.sign):
    assert names[p] == n.replace("left_", "@").replace("right_", "left_").replace("@", "right_")
    assert s == (-1.0 if any(f"_{t}_" in n for t in ("roll", "yaw")) else 1.0), (n, s)
print(f"mirror signs of {len(names)} joints match the roll / yaw rule")
