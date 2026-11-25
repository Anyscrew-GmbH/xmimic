"""Reference motion library (a port of blinkmimic/motion.py).

The G1-retargeted LAFAN1 clips (30 fps CSV) are resampled to the 50 Hz control rate. Root and joint velocities come from
finite differences, the tracked bodies from forward kinematics. Clips are cut into 1-second reset bins, and the policy's
goal is a window of future reference frames (with --goal_mask_share, in a share of the episodes only a command as a
joystick gives it, the root's planar velocity and yaw rate smoothed over CMD_WINDOW, plus a flag: XiLayout.masked in
sim.py). The library is a pytree of device arrays: it is passed into jitted functions as an argument and never closed
over as a constant.
"""

from __future__ import annotations

import dataclasses
import glob
import os
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .rot import mat_to_6d, mm3, mv3, tr3, yaw_mat

CONTROL_DT = 0.02
BIN_FRAMES = 50
GOAL_Q_OFFSET = 15  # index of q(f+1) in the goal vector (first future frame: dp 3, rot 6, v 3, w 3, then q)
CMD_WINDOW = 51  # frames averaged into the command (1 s, about one gait cycle): the reference's gait phase must not show


def load_lafan1_csv(path: str):
    """30 fps rows of root position, root quaternion (xyzw) and the joint angles (29 for the G1) -> (p, quat wxyz, q, fps)."""
    raw = np.loadtxt(path, delimiter=",")
    p, q_xyzw, j = raw[:, :3], raw[:, 3:7], raw[:, 7:]
    return p, np.concatenate((q_xyzw[:, 3:4], q_xyzw[:, :3]), -1), j, 30.0


def _slerp_np(q0, q1, t):
    """blinkmimic.rotations.quat_slerp in float32 numpy."""
    dot = (q0 * q1).sum(-1, keepdims=True)
    q1 = np.where(dot < 0, -q1, q1)
    dot = np.minimum(np.abs(dot), np.float32(1.0))
    theta = np.arccos(dot)
    s = np.sin(theta)
    t = t[..., None]
    w0 = np.where(s < 1e-6, 1 - t, np.sin((1 - t) * theta) / (s + np.float32(1e-12)))
    w1 = np.where(s < 1e-6, t, np.sin(t * theta) / (s + np.float32(1e-12)))
    q = w0 * q0 + w1 * q1
    return (q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), np.float32(1e-12))).astype(np.float32)


def _quat_to_mat_np(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.stack((1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
                     2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
                     2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)), -1).reshape(q.shape[:-1] + (3, 3))  # fmt: skip


def _so3_log_np(R):
    """blinkmimic.rotations.so3_log (largest-diagonal quaternion branch, then the rotation vector) in numpy."""
    m = R
    qw = np.sqrt(np.maximum(1 + m[..., 0, 0] + m[..., 1, 1] + m[..., 2, 2], 1e-10)) / 2
    qx = np.sqrt(np.maximum(1 + m[..., 0, 0] - m[..., 1, 1] - m[..., 2, 2], 1e-10)) / 2
    qy = np.sqrt(np.maximum(1 - m[..., 0, 0] + m[..., 1, 1] - m[..., 2, 2], 1e-10)) / 2
    qz = np.sqrt(np.maximum(1 - m[..., 0, 0] - m[..., 1, 1] + m[..., 2, 2], 1e-10)) / 2
    m01, m02, m10, m12, m20, m21 = m[..., 0, 1], m[..., 0, 2], m[..., 1, 0], m[..., 1, 2], m[..., 2, 0], m[..., 2, 1]
    cands = np.stack((np.stack((qw, (m21 - m12) / (4 * qw), (m02 - m20) / (4 * qw), (m10 - m01) / (4 * qw)), -1),
                      np.stack(((m21 - m12) / (4 * qx), qx, (m01 + m10) / (4 * qx), (m02 + m20) / (4 * qx)), -1),
                      np.stack(((m02 - m20) / (4 * qy), (m01 + m10) / (4 * qy), qy, (m12 + m21) / (4 * qy)), -1),
                      np.stack(((m10 - m01) / (4 * qz), (m02 + m20) / (4 * qz), (m12 + m21) / (4 * qz), qz), -1)), -2)  # fmt: skip
    idx = np.argmax(np.stack((qw, qx, qy, qz), -1), -1)
    q = np.take_along_axis(cands, idx[..., None, None], axis=-2)[..., 0, :]
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = q * np.where(q[..., :1] < 0, -1.0, 1.0)
    v = q[..., 1:]
    sin_half = np.linalg.norm(v, axis=-1, keepdims=True)
    half = np.arctan2(sin_half, q[..., :1])
    scale = np.where(sin_half < 1e-6, 2.0 / np.maximum(q[..., :1], 1e-6), 2 * half / (sin_half + 1e-12))
    return (v * scale).astype(np.float32)


def _gradient(x, dt):
    """Central differences inside, one-sided at the ends (numpy.gradient in float32)."""
    g = np.zeros_like(x)
    g[1:-1] = (x[2:] - x[:-2]) / np.float32(2 * dt)
    g[0] = (x[1] - x[0]) / np.float32(dt)
    g[-1] = (x[-1] - x[-2]) / np.float32(dt)
    return g


def _smooth(x):
    """Centered moving average over CMD_WINDOW frames along axis 0; at the ends, over the frames that exist."""
    n, h = len(x), CMD_WINDOW // 2
    c = np.concatenate((np.zeros_like(x[:1]), np.cumsum(x, 0)))
    lo, hi = np.maximum(np.arange(n) - h, 0), np.minimum(np.arange(n) + h + 1, n)
    return (c[hi] - c[lo]) / (hi - lo).reshape((-1,) + (1,) * (x.ndim - 1))


@partial(jax.tree_util.register_dataclass,
         data_fields=["root_p", "root_R", "root_v", "root_w", "q", "qd", "body_p", "body_R", "body_v", "body_w", "root_v_local",
                      "root_w_local", "cmd_yaw_R", "cmd_w", "cmd_v", "clip_id", "clip_start", "clip_end", "bin_clip", "bin_start",
                      "frame_bin", "ref_contact"],
         meta_fields=["names", "num_frames", "num_bins", "num_joints"])  # fmt: skip
@dataclasses.dataclass
class Motion:
    """All clips concatenated along one frame axis. Per frame: root and joint state, tracked-body poses and velocities, the
    command (smoothed heading as a yaw rotation, its yaw rate, the smoothed planar velocity in its frame), the clip it
    belongs to and its reset bin, and the reference foot contacts."""

    root_p: jax.Array
    root_R: jax.Array
    root_v: jax.Array
    root_w: jax.Array
    q: jax.Array
    qd: jax.Array
    body_p: jax.Array
    body_R: jax.Array
    body_v: jax.Array
    body_w: jax.Array
    root_v_local: jax.Array
    root_w_local: jax.Array
    cmd_yaw_R: jax.Array
    cmd_w: jax.Array
    cmd_v: jax.Array
    clip_id: jax.Array
    clip_start: jax.Array
    clip_end: jax.Array
    bin_clip: jax.Array
    bin_start: jax.Array
    frame_bin: jax.Array
    ref_contact: jax.Array
    names: tuple
    num_frames: int
    num_bins: int
    num_joints: int

    def goal_dim(self, K: int) -> int:
        return K * (3 + 6 + 3 + 3 + 2 * self.num_joints)

    def future_frames(self, f, K: int):
        """The K frames after f, held at the clip's last frame."""
        end = self.clip_end[self.clip_id[f]]
        fut = f[..., None] + jnp.arange(1, K + 1, dtype=f.dtype)
        return jnp.minimum(fut, end[..., None])

    def goal(self, f, K: int):
        """The policy's goal, from the reference only: K future frames relative to the current reference frame (..., K*(15+2J))."""
        fut = self.future_frames(f, K)
        Ht = tr3(yaw_mat(self.root_R[f]))[..., None, :, :]
        dp = mv3(Ht, self.root_p[fut] - self.root_p[f][..., None, :])
        rot = mat_to_6d(mm3(Ht, self.root_R[fut]))
        feats = jnp.concatenate((dp, rot, self.root_v_local[fut], self.root_w_local[fut], self.q[fut], self.qd[fut] * 0.1), -1)
        return feats.reshape(f.shape + (-1,))

    def goal_cmd(self, f, K: int):
        """What a masked goal keeps: the command's planar velocity and yaw rate at the K future frames (..., K, 3)."""
        fut = self.future_frames(f, K)
        return jnp.concatenate((self.cmd_v[fut], self.cmd_w[fut][..., None]), -1)

    def ref_state(self, f):
        return self.root_p[f], self.root_R[f], self.root_v[f], self.root_w[f], self.q[f], self.qd[f]

    def sample_mirror(self, key, f):
        """Reset frames f as they are (MirroredMotion moves a share of them to the mirror image)."""
        return f


class _Frames:
    """``motion.<field>[i]`` of a MirroredMotion: index i of the original, or i - n of it mirrored by ``fn(x, i - n)`` where
    i >= n."""

    def __init__(self, base, n, fn):
        self.base, self.n, self.fn = base, n, fn

    def __getitem__(self, i):
        i = jnp.asarray(i)
        m = i >= self.n
        g = jnp.where(m, i - self.n, i)
        x = self.base[g]
        return jnp.where(m.reshape(m.shape + (1,) * (x.ndim - m.ndim)), self.fn(x, g), x)


@partial(jax.tree_util.register_dataclass, data_fields=["base", "perm", "sign", "body_perm", "body_C", "body_d", "foot_perm"],
         meta_fields=["share"])  # fmt: skip
@dataclasses.dataclass
class MirroredMotion:
    """A Motion and its left/right mirror image as one library of twice the frames, without a second copy on the device
    (--mirror_clips). Frame f >= N (N = base.num_frames) is frame f - N mirrored at read time, by the rules of
    algorithm.Mirror.state (M = diag(1, -1, 1); joints and tracked bodies swap sides); clip c >= C is clip c - C's mirror.
    A clip and its mirror share their reset bins and bin statistics. ``sample_mirror`` moves each reset frame to its mirror
    image with probability ``share``. Read ``motion.<field>[f]`` as for a Motion; whole arrays are only in ``base``."""

    base: Motion
    perm: jax.Array
    sign: jax.Array
    body_perm: jax.Array
    body_C: jax.Array  # per tracked body: the constant rotation from the mirrored partner's frame to the body's own
    body_d: jax.Array  # ... and its origin's constant offset from the mirrored partner's origin, in its own frame
    foot_perm: jax.Array
    share: float

    goal_dim, future_frames, goal, goal_cmd, ref_state = Motion.goal_dim, Motion.future_frames, Motion.goal, Motion.goal_cmd, Motion.ref_state

    @property
    def num_frames(self):
        return 2 * self.base.num_frames

    def sample_mirror(self, key, f):
        return f + self.base.num_frames * (jax.random.uniform(jax.random.fold_in(key, 1), f.shape) < self.share)

    def __getattr__(self, name):
        if name.startswith("_") or name == "base":
            raise AttributeError(name)
        b, n, C = self.base, self.base.num_frames, len(self.base.names)
        Mv = jnp.asarray([1.0, -1.0, 1.0])
        vec, pseudo, rot = (lambda x, g=None: x * Mv), (lambda x, g=None: -(x * Mv)), (lambda x, g=None: x * Mv[:, None] * Mv[None, :])  # noqa: E731
        bodies = lambda x: x[..., self.body_perm, :]  # noqa: E731
        R = lambda g: rot(b.body_R[g][..., self.body_perm, :, :]) @ self.body_C  # noqa: E731
        d = lambda g: (R(g) @ self.body_d[:, :, None])[..., 0]  # noqa: E731
        w = lambda g: pseudo(bodies(b.body_w[g]))  # noqa: E731
        fn = {"root_p": vec, "root_v": vec, "root_v_local": vec, "root_w": pseudo, "root_w_local": pseudo, "root_R": rot, "cmd_yaw_R": rot,
              "q": lambda x, g: x[..., self.perm] * self.sign, "qd": lambda x, g: x[..., self.perm] * self.sign,
              "body_p": lambda x, g: vec(bodies(x)) + d(g), "body_v": lambda x, g: vec(bodies(x)) + jnp.cross(w(g), d(g)),
              "body_w": lambda x, g: w(g), "body_R": lambda x, g: R(g), "cmd_v": lambda x, g: x * jnp.asarray([1.0, -1.0]),
              "cmd_w": lambda x, g: -x, "ref_contact": lambda x, g: x[..., self.foot_perm], "frame_bin": lambda x, g: x,
              "clip_id": lambda x, g: x + C}  # fmt: skip
        if name in fn:
            return _Frames(getattr(b, name), n, fn[name])
        if name in ("clip_start", "clip_end"):
            return _Frames(getattr(b, name), C, lambda x, g: x + n)
        return getattr(b, name)  # bins, names, sizes: those of the original clips


def mirrored(motion: Motion, mirror, kin, feet: list[int], share: float) -> MirroredMotion:
    """``motion`` and its mirror image (``mirror``: algorithm.Mirror of the robot) for --mirror_clips ``share``."""
    assert mirror.body_perm is not None, "--mirror_clips: every tracked body needs its left / right partner among the tracked bodies"
    o = jnp.zeros((1, 3))
    pz, Rz, _, _ = kin.forward(o, jnp.eye(3)[None], o, o, jnp.zeros((1, len(kin.q_lo))), jnp.zeros((1, len(kin.q_lo))))
    Mm, bp = np.asarray([1.0, -1.0, 1.0], np.float32), list(mirror.body_perm)
    pz, Rz = np.asarray(pz[0]), np.asarray(Rz[0])
    # body b in the mirrored pose: rotation M R_partner M C_b, origin M p_partner + R_b d_b (a frame origin off the robot's
    # plane of symmetry); the zero pose is its own mirror image, which fixes C_b and d_b
    C = np.swapaxes(Rz[bp] * Mm[:, None] * Mm[None, :], -1, -2) @ Rz
    d = (np.swapaxes(Rz, -1, -2) @ (pz - pz[bp] * Mm)[:, :, None])[..., 0]
    return MirroredMotion(base=motion, perm=jnp.asarray(mirror.perm), sign=jnp.asarray(mirror.sign), body_perm=jnp.asarray(bp),
                          body_C=jnp.asarray(C, jnp.float32), body_d=jnp.asarray(d, jnp.float32),
                          foot_perm=jnp.asarray([feet.index(bp[i]) for i in feet]), share=float(share))  # fmt: skip


def load_motion(motion_dir: str, kin, feet: list[int], margin: float = 0.015, vmax: float = 0.5, quant: float = 0.05) -> Motion:
    """All *.csv clips of ``motion_dir`` in sorted order (blinkmimic.motion.MotionLib.from_dir plus set_ref_contact(feet))."""
    files = sorted(glob.glob(os.path.join(motion_dir, "*.csv")))
    assert files, f"no clips in {motion_dir}: run tools/download_data.sh first"
    names = tuple(os.path.splitext(os.path.basename(f))[0] for f in files)
    ps, Rs, vs, ws, js, jds, lens, cRs, cws, cvs = [], [], [], [], [], [], [], [], [], []
    for f in files:
        p, quat, j, fps = load_lafan1_csv(f)
        assert j.shape[1] == len(kin.q_lo), f"{f}: {j.shape[1]} joint columns, the robot has {len(kin.q_lo)} joints"
        p, quat, j = (x.astype(np.float32) for x in (p, quat, j))
        T_src = p.shape[0]
        dur = (T_src - 1) / fps
        n = int(np.floor(dur / CONTROL_DT)) + 1
        t = np.arange(n, dtype=np.float32) * np.float32(CONTROL_DT) * np.float32(fps)
        i0 = np.minimum(np.floor(t).astype(np.int64), T_src - 2)
        a = np.clip(t - i0.astype(np.float32), 0, 1)
        p = p[i0] * (1 - a[:, None]) + p[i0 + 1] * a[:, None]
        j = j[i0] * (1 - a[:, None]) + j[i0 + 1] * a[:, None]
        quat = _slerp_np(quat[i0], quat[i0 + 1], a)
        R = _quat_to_mat_np(quat).astype(np.float32)
        w = np.zeros_like(p)
        w[1:-1] = _so3_log_np(R[2:] @ np.swapaxes(R[:-2], -1, -2)) / np.float32(2 * CONTROL_DT)
        w[0] = _so3_log_np(R[1] @ R[0].T) / np.float32(CONTROL_DT)
        w[-1] = _so3_log_np(R[-1] @ R[-2].T) / np.float32(CONTROL_DT)
        v = _gradient(p, CONTROL_DT)
        ps.append(p), Rs.append(R), vs.append(v), ws.append(w), js.append(j), jds.append(_gradient(j, CONTROL_DT))
        head = _smooth(np.unwrap(np.arctan2(R[:, 1, 0], R[:, 0, 0]).astype(np.float64)))
        c, s, o, vm = np.cos(head), np.sin(head), np.zeros(n), _smooth(v[:, :2].astype(np.float64))
        cRs.append(np.stack((c, -s, o, s, c, o, o, o, o + 1), -1).reshape(n, 3, 3))
        cws.append(_gradient(head, CONTROL_DT))
        cvs.append(np.stack((c * vm[:, 0] + s * vm[:, 1], c * vm[:, 1] - s * vm[:, 0]), -1))
        lens.append(n)
    lens = np.asarray(lens)
    clip_start = np.concatenate(([0], np.cumsum(lens[:-1]))).astype(np.int32)
    clip_end = (clip_start + lens - 1).astype(np.int32)
    clip_id = np.concatenate([np.full(n, i, np.int32) for i, n in enumerate(lens)])
    cat = lambda xs: jnp.asarray(np.concatenate(xs).astype(np.float32))  # noqa: E731
    root_p, root_R, root_v, root_w, q, qd = cat(ps), cat(Rs), cat(vs), cat(ws), cat(js), cat(jds)
    fk = jax.jit(kin.forward)
    out = [fk(root_p[s:s + 65536], root_R[s:s + 65536], root_v[s:s + 65536], root_w[s:s + 65536], q[s:s + 65536], qd[s:s + 65536])
           for s in range(0, len(clip_id), 65536)]  # fmt: skip
    body_p, body_R, body_v, body_w = (jnp.concatenate([o[i] for o in out]) for i in range(4))
    Rt = tr3(root_R)
    bin_clip, bin_start = [], []
    for c in range(len(lens)):
        for b in range(int(np.ceil(lens[c] / BIN_FRAMES))):
            bin_clip.append(c)
            bin_start.append(int(clip_start[c]) + b * BIN_FRAMES)
    frame_bin = np.zeros(len(clip_id), np.int32)
    for b, (c, s) in enumerate(zip(bin_clip, bin_start)):
        frame_bin[s : min(s + BIN_FRAMES, int(clip_end[c]) + 1)] = b
    # reference foot contact (BlinkMimic --ref_contact_fix): below the clip's own floor (quantile) + margin, slower than vmax
    z = np.asarray(body_p[:, feet, 2])
    vh = np.linalg.norm(np.asarray(body_v[:, feet, :2]), axis=-1)
    floor = np.zeros_like(z)
    for c in range(len(lens)):
        s = clip_id == c
        floor[s] = np.quantile(z[s], quant, axis=0)
    ref_contact = ((z < floor + margin) & (vh < vmax)).astype(np.float32)
    return Motion(root_p=root_p, root_R=root_R, root_v=root_v, root_w=root_w, q=q, qd=qd, body_p=body_p, body_R=body_R,
                  body_v=body_v, body_w=body_w, root_v_local=mv3(Rt, root_v), root_w_local=mv3(Rt, root_w),
                  cmd_yaw_R=cat(cRs), cmd_w=cat(cws), cmd_v=cat(cvs),
                  clip_id=jnp.asarray(clip_id), clip_start=jnp.asarray(clip_start), clip_end=jnp.asarray(clip_end),
                  bin_clip=jnp.asarray(bin_clip, jnp.int32), bin_start=jnp.asarray(bin_start, jnp.int32),
                  frame_bin=jnp.asarray(frame_bin), ref_contact=jnp.asarray(ref_contact), names=names,
                  num_frames=int(len(clip_id)), num_bins=len(bin_start), num_joints=int(q.shape[-1]))  # fmt: skip
