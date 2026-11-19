"""Goal masking (--goal_mask_share): a flagged episode's policy sees only a command as a joystick gives it. Its goal keeps
the reference root's smoothed planar velocity and yaw rate, every other entry is neutral and the flag is appended; its
observation relates to the smoothed heading instead of the reference orientation. Without it, nothing changes.

    JAX_PLATFORMS=cpu python tests/test_goal_mask.py
"""

import os
import sys
import tempfile

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from xmimic import nets  # noqa: E402
from xmimic.algorithm import Mirror  # noqa: E402
from xmimic.kin import Kinematics  # noqa: E402
from xmimic.mdp import MDP, obs_dim  # noqa: E402
from xmimic.model import feet_of, load_meta  # noqa: E402
from xmimic.motion import CMD_WINDOW, CONTROL_DT, GOAL_Q_OFFSET, load_motion  # noqa: E402
from xmimic.rot import mat_to_6d, mm3, so3_exp, tr3  # noqa: E402
from xmimic.sim import XiLayout  # noqa: E402

meta = load_meta()
J, K, N = len(meta["joint_names"]), 10, 6
q0, alpha = np.asarray(meta["default_joint_pos"], np.float32), np.asarray(meta["action_scale"], np.float32)
kin, feet = Kinematics(meta), feet_of(meta)
L0, L = XiLayout(meta), XiLayout(meta, 0.5)
mirror = Mirror(meta, L)
M = mirror.M
rng = np.random.default_rng(0)
z = lambda *s: jnp.asarray(rng.normal(size=s), jnp.float32)  # noqa: E731

# a synthetic clip (8 s at 30 fps): a straight walk at SPEED along the heading PSI, the root swaying sideways and wobbling
# in yaw at 1 Hz (through +-pi, so the yaw must be unwrapped), a constant pitch, small joint motions; and its mirror image
SPEED, PSI, WOBBLE, SWAY = 1.0, 3.0, 0.2, 0.03
t = np.arange(240) / 30.0
yaw = PSI + WOBBLE * np.sin(2 * np.pi * t)
side = SWAY * np.sin(2 * np.pi * t + 1.0)
p = np.stack((SPEED * t * np.cos(PSI) - side * np.sin(PSI), SPEED * t * np.sin(PSI) + side * np.cos(PSI), 0.78 + 0 * t), -1)
cy, sy, cp, sp = np.cos(yaw / 2), np.sin(yaw / 2), np.cos(0.05), np.sin(0.05)
quat = np.stack((-sy * sp, cy * sp, sy * cp, cy * cp), -1)  # xyzw of Rz(yaw) Ry(0.1)
q = q0 + 0.05 * np.sin(rng.uniform(1, 6, J) * t[:, None] + rng.uniform(0, 6, J))
with tempfile.TemporaryDirectory() as d:
    os.mkdir(os.path.join(d, "a")), os.mkdir(os.path.join(d, "b"))
    np.savetxt(os.path.join(d, "a", "walk.csv"), np.concatenate((p, quat, q), 1), delimiter=",")
    np.savetxt(os.path.join(d, "b", "walk.csv"), np.concatenate((p * M, quat * [-1, 1, -1, 1], np.asarray(mirror.joints(q))), 1), delimiter=",")
    mo, mo2 = load_motion(os.path.join(d, "a"), kin, feet), load_motion(os.path.join(d, "b"), kin, feet)
T = mo.num_frames
f = jnp.asarray([0, 7, 120, 233, T - 12, T - 5], jnp.int32)  # the last frames' goals run into the end of the clip
g = mo.goal(f, K)
fut = mo.future_frames(f, K)
S = (z(N, 3), so3_exp(z(N, 3)), z(N, 3), z(N, 3), z(N, J), z(N, J))
a_prev = z(N, J)
flag = jnp.arange(N) % 2  # worlds 1, 3, 5 are flagged
xi = z(N, L.dim).at[:, L.mask].set(flag)
rel = lambda R_ref: mat_to_6d(mm3(tr3(S[1]), R_ref))  # noqa: E731

# share 0: no flag in xi, the policy's goal is the reference goal itself and the observation relates to the reference
mdp0 = MDP(kin, q0, feet, L0.dim, meta["contact_z"])
o0 = mdp0.obs(mo, S, a_prev, xi[:, : L0.dim], f)
assert L0.mask is None and L0.dim == 1 + J + 3 and L0.masked(g, None, None, None) is g and g.shape[-1] == mo.goal_dim(K) == K * (15 + 2 * J)
assert o0.shape[-1] == obs_dim(J) and jnp.array_equal(o0[:, -6:], rel(mo.root_R[f]))
print(f"share 0: {L0.dim} context values and {g.shape[-1]} goal values, unchanged")

# the command: away from the clip's ends (there the window is cut) the wobble is nearly gone and the speed is the walk's
h = CMD_WINDOW // 2
inner = slice(h, T - h)
head = np.unwrap(np.arctan2(mo.cmd_yaw_R[:, 1, 0], mo.cmd_yaw_R[:, 0, 0]))
cmd_w, cmd_v = np.asarray(mo.cmd_w), np.asarray(mo.cmd_v)
w_left, yaw_left = cmd_w[inner].std() / np.asarray(mo.root_w)[inner, 2].std(), np.abs(head[inner] - PSI).max() / WOBBLE
assert w_left < 0.05 and yaw_left < 0.05, (w_left, yaw_left)
assert np.abs(cmd_v[inner, 0] - SPEED).max() < 0.01 and np.abs(cmd_v[inner, 1]).max() < 0.01, np.abs(cmd_v[inner] - [SPEED, 0]).max(0)
assert jnp.array_equal(mo.cmd_yaw_R[:, 2], jnp.broadcast_to(jnp.asarray([0.0, 0.0, 1.0]), (T, 3)))  # yaw only
turn = ((cmd_w[1:] + cmd_w[:-1]) / 2).sum() * CONTROL_DT  # trapezoids undo the central differences
assert abs(turn - (head[-1] - head[0])) < 1e-3 and abs(turn) > 0.02, (turn, head[-1] - head[0])
print(f"command: {w_left:.3f} of the yaw-rate wobble and {yaw_left:.3f} of the yaw wobble left, speed {cmd_v[inner, 0].mean():.3f} of {SPEED}, "
      f"integrated yaw rate {turn:.4f} rad = heading change {head[-1] - head[0]:.4f} rad")

# share > 0: one more context value
assert L.dim == L0.dim + 1 and L.center.shape == L.scale.shape == (L.dim,) and L.normalize(jnp.ones(L.dim))[L.mask] == 1
gm = L.masked(g, mo, f, xi)
assert gm.shape == (N, mo.goal_dim(K) + 1) and jnp.array_equal(gm[:, -1], flag)
assert jnp.array_equal(gm[::2, :-1], g[::2])  # unflagged: the full goal
m = gm[1::2, :-1].reshape(N // 2, K, -1)
keep = np.isin(np.arange(15 + 2 * J), (9, 10, 14))
neutral = np.concatenate((np.zeros(3), np.eye(3)[:, :2].reshape(6), np.zeros(6), q0, np.zeros(J))).astype(np.float32)
assert jnp.array_equal(m[..., 9:11], mo.cmd_v[fut[1::2]]) and jnp.array_equal(m[..., 14], mo.cmd_w[fut[1::2]])
assert jnp.array_equal(m[..., ~keep], jnp.broadcast_to(neutral[~keep], m[..., ~keep].shape))
assert not jnp.allclose(m[..., keep], g[1::2].reshape(N // 2, K, -1)[..., keep], atol=0.05)  # not the raw reference velocities
F = jnp.minimum(f[None] + jnp.arange(3)[:, None], T - 1)
assert jnp.array_equal(L.masked(mo.goal(F, K), mo, F, xi)[1], L.masked(mo.goal(F[1], K), mo, F[1], xi))  # (H, B) frames
print(f"masked goal: {int(keep.sum())} of {keep.size} values per frame are the command, the others neutral, the flag appended")

# the policy takes the longer goal, and its reference skip term is zero for a masked goal (decoder output set to zero)
pi, pi_b = nets.pi_init(jax.random.PRNGKey(0), obs_dim(J), gm.shape[-1], J, 2, enc_hidden=(8,), dec_hidden=(8,))
pi["dec"][-1]["w"] = jnp.zeros_like(pi["dec"][-1]["w"])
a = nets.pi_apply(pi, pi_b, z(N, 2, obs_dim(J)), gm, (GOAL_Q_OFFSET, q0, alpha))
assert jnp.array_equal(a[1::2], jnp.zeros((N // 2, J))) and jnp.array_equal(a[::2], (mo.q[f[::2] + 1] - q0) / alpha)
print("skip term: zero for a masked goal, (q_ref - q0) / alpha otherwise")

# observation: a flagged world's last 6 values relate to the command's heading, everything else is as without the flag
mdp = MDP(kin, q0, feet, L.dim, meta["contact_z"], L.mask)
o = mdp.obs(mo, S, a_prev, xi, f)
assert jnp.array_equal(o[:, :-6], o0[:, :-6]) and jnp.array_equal(o[::2, -6:], o0[::2, -6:])
assert jnp.array_equal(o[1::2, -6:], rel(mo.cmd_yaw_R[f])[1::2]) and not jnp.allclose(o[1::2, -6:], o0[1::2, -6:], atol=0.01)
F1 = F[1:]
assert jnp.array_equal(mdp.ref_R(mo, xi, F1)[:, 1::2], mo.cmd_yaw_R[F1][:, 1::2]) and jnp.array_equal(mdp.ref_R(mo, xi, F1)[:, ::2], mo.root_R[F1][:, ::2])
print("observation: the last 6 values of a masked episode are the 6d of R^T cmd_yaw_R[f]")


# mirror: the sagittal mirror of a goal (Mirror.state's rules per frame) commutes with the masking; the flag passes through
def mirror_goal(x):
    n = K * (15 + 2 * J)
    y = x[..., :n].reshape(x.shape[:-1] + (K, -1))
    y = jnp.concatenate((y[..., :3] * M, y[..., 3:9] * (M[:, None] * M[None, :2]).reshape(6), y[..., 9:12] * M, -(y[..., 12:15] * M),
                         mirror.joints(y[..., 15 : 15 + J]), mirror.joints(y[..., 15 + J :])), -1)  # fmt: skip
    return jnp.concatenate((y.reshape(x.shape[:-1] + (n,)), x[..., n:]), -1)


# the mirrored clip's command is the command mirrored as Mirror.state mirrors a root velocity (v x, v y, 0) and (0, 0, w z)
o3 = jnp.zeros((T, 1))
_, cR2, cv2, cw2, _, _ = mirror.state((o3, mo.cmd_yaw_R, jnp.concatenate((mo.cmd_v, o3), -1), jnp.concatenate((o3, o3, mo.cmd_w[:, None]), -1), mo.q, mo.qd))
assert jnp.allclose(mo2.cmd_v, cv2[:, :2], atol=1e-5) and jnp.allclose(mo2.cmd_w, cw2[:, 2], atol=1e-4) and jnp.allclose(mo2.cmd_yaw_R, cR2, atol=1e-5)
assert jnp.allclose(mo2.cmd_v, mo.cmd_v * jnp.asarray([1.0, -1.0]), atol=1e-5) and jnp.allclose(mo2.cmd_w, -mo.cmd_w, atol=1e-4)
assert jnp.allclose(mo2.goal(f, K), mirror_goal(g), atol=1e-3)  # mirror_goal is the goal of the mirrored clip
assert jnp.array_equal(mirror.xi(xi)[:, L.mask], flag)
a_, b_ = mirror_goal(gm), L.masked(mirror_goal(g), mo2, f, mirror.xi(xi))
assert jnp.array_equal(a_[::2], b_[::2]) and jnp.allclose(a_[1::2], b_[1::2], atol=1e-4) and not jnp.allclose(a_, gm, atol=1e-3)
print("mirror: mirror(mask(g)) == mask(mirror(g)) with the mirrored clip's command (v y and w z flip), the flag unchanged")
