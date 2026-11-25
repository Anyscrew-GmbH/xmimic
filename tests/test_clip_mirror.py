"""--mirror_clips: MirroredMotion reads frame f >= N as the mirror image of frame f - N. Every per-frame field, the goal and
the command equal those of the mirror-image clip loaded from its own CSV (root y negated, quaternion mirrored, joints
swapped with their signs), mirroring twice gives the original back, a clip and its mirror share their bins, a share of
the reset frames moves to the mirror, it passes through jit, and the device memory does not grow.

    JAX_PLATFORMS=cpu python tests/test_clip_mirror.py      # XMIMIC_ROBOT_DIR=<robot directory>: another robot than the G1
"""

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from xmimic.algorithm import Mirror  # noqa: E402
from xmimic.kin import Kinematics  # noqa: E402
from xmimic.model import feet_of, load_meta  # noqa: E402
from xmimic.motion import load_motion, mirrored  # noqa: E402
from xmimic.sim import XiLayout  # noqa: E402

meta = load_meta(os.environ.get("XMIMIC_ROBOT_DIR"))
J, K = len(meta["joint_names"]), 10
q0 = np.asarray(meta["default_joint_pos"], np.float32)
kin, feet = Kinematics(meta), feet_of(meta)
mirror = Mirror(meta, XiLayout(meta))
M = mirror.M
rng = np.random.default_rng(0)


def clip(seconds, speed, psi, turn):
    """30 fps: a walk along heading psi that turns at ``turn`` rad/s, swaying sideways, rolling and pitching, joints moving."""
    t = np.arange(int(seconds * 30)) / 30.0
    yaw = psi + turn * t + 0.2 * np.sin(2 * np.pi * t)
    side = 0.03 * np.sin(2 * np.pi * t + 1.0)
    p = np.stack((speed * t * np.cos(yaw) - side * np.sin(yaw), speed * t * np.sin(yaw) + side * np.cos(yaw), 0.75 + 0.02 * np.sin(4 * np.pi * t)), -1)
    roll, pitch = 0.05 * np.sin(2 * np.pi * t), 0.05 + 0 * t
    cr, sr, cp, sp, cy, sy = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2), np.cos(yaw / 2), np.sin(yaw / 2)
    quat = np.stack((sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy, cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy), -1)  # xyzw, ZYX
    q = q0 + 0.2 * np.sin(rng.uniform(1, 6, J) * t[:, None] + rng.uniform(0, 6, J))
    return p, quat, q


with tempfile.TemporaryDirectory() as d:
    os.mkdir(os.path.join(d, "a")), os.mkdir(os.path.join(d, "b"))
    for name, args in (("turn", (6, 0.8, 3.0, 0.5)), ("walk", (4.5, 1.2, -1.0, 0.0))):
        p, quat, q = clip(*args)
        np.savetxt(os.path.join(d, "a", name + ".csv"), np.concatenate((p, quat, q), 1), delimiter=",")
        np.savetxt(os.path.join(d, "b", name + ".csv"), np.concatenate((p * M, quat * [-1, 1, -1, 1], np.asarray(mirror.joints(q))), 1), delimiter=",")
    mo, mo_b = load_motion(os.path.join(d, "a"), kin, feet), load_motion(os.path.join(d, "b"), kin, feet)
N, C = mo.num_frames, len(mo.names)
mm, mm_b = mirrored(mo, mirror, kin, feet, 0.5), mirrored(mo_b, mirror, kin, feet, 0.5)
f = jnp.arange(N)
FIELDS = ("root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R", "body_v", "body_w", "root_v_local", "root_w_local",
          "cmd_yaw_R", "cmd_w", "cmd_v", "ref_contact")  # fmt: skip

# the mirror image read on the fly equals the mirror-image clip loaded from its CSV; twice mirrored is the original
for name in FIELDS:
    a, b = getattr(mo, name), getattr(mo_b, name)
    assert jnp.array_equal(getattr(mm, name)[f], a), name  # f < N: the original, unchanged
    err, err2 = float(jnp.abs(getattr(mm, name)[f + N] - b).max()), float(jnp.abs(getattr(mm_b, name)[f + N] - a).max())
    assert err < 1e-3 and err2 < 1e-3, (name, err, err2)
for g in ("goal", "goal_cmd"):
    err = float(jnp.abs(getattr(mm, g)(f + N, K) - getattr(mo_b, g)(f, K)).max())
    assert err < 1e-3, (g, err)
print(f"{len(FIELDS)} fields, goal and command of the mirror image match the mirrored clip's own (twice mirrored: the original)")

# clips and bins: mirror clip c + C ends where clip c ends, N later; bins are shared; the state the trainer resets to
assert mm.num_frames == 2 * N and mm.num_bins == mo.num_bins and mm.names == mo.names
assert jnp.array_equal(mm.clip_id[f + N], mo.clip_id + C) and jnp.array_equal(mm.frame_bin[f + N], mo.frame_bin)
assert jnp.array_equal(mm.clip_end[mm.clip_id[f + N]], mo.clip_end[mo.clip_id] + N)
assert jnp.array_equal(mm.clip_start[mm.clip_id[f + N]], mo.clip_start[mo.clip_id] + N)
assert jnp.array_equal(mm.future_frames(f + N, K), mo.future_frames(f, K) + N)
for x, y in zip(mm.ref_state(f + N), mo_b.ref_state(f)):
    assert float(jnp.abs(x - y).max()) < 1e-3
print(f"clips: {C} originals and their mirrors, {mo.num_bins} shared bins, futures held at the mirror clip's end")

# reset frames: share 0.5 moves half of them to the mirror; a plain Motion leaves them alone
k = jax.random.PRNGKey(0)
fs = jnp.asarray(rng.integers(0, N, 20000), jnp.int32)
moved = mm.sample_mirror(k, fs)
share = float((moved >= N).mean())
assert abs(share - 0.5) < 0.02 and jnp.array_equal(moved % N, fs) and jnp.array_equal(mo.sample_mirror(k, fs), fs), share
print(f"resets: {share:.3f} of 20,000 sampled frames moved to the mirror image (share 0.5)")

# a pytree argument of jitted functions; on the device only the original clips plus the permutations and body frames
g_jit = jax.jit(lambda m, i: m.goal(i, K))(mm, f + N)
assert jnp.allclose(g_jit, mm.goal(f + N, K), atol=1e-6)
size = lambda t: sum(x.nbytes for x in jax.tree.leaves(t))  # noqa: E731
extra = size(mm) - size(mo)
assert extra == sum(x.nbytes for x in (mm.perm, mm.sign, mm.body_perm, mm.body_C, mm.body_d, mm.foot_perm)) and extra < 4096, extra
print(f"memory: {size(mo) / 1e6:.2f} MB of clips, mirroring adds {extra} bytes")
