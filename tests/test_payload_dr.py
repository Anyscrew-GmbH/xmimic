"""Payload randomization (meta["dr"]["payload"]): the sim context grows by 4 values (mass, center of mass) per listed body,
the mirror swaps a pair and flips the lateral component, the evaluation takes the middle of each range, and (on a GPU) the
drawn values reach the model fields and a heavier payload changes the physics.

    JAX_PLATFORMS=cpu python tests/test_payload_dr.py    # layout and mirror
    python tests/test_payload_dr.py                      # also the simulator; the robot and its clips can be set with
                                                         # XMIMIC_ROBOT_DIR and XMIMIC_MOTION_DIR (default: the G1)
"""

import copy
import os
import sys

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xmimic.algorithm import Mirror  # noqa: E402
from xmimic.model import feet_of, load_meta, load_model  # noqa: E402
from xmimic.sim import ACTUATOR_FIELDS, Sim, XiLayout  # noqa: E402

ROBOT = os.environ.get("XMIMIC_ROBOT_DIR")
meta0 = load_meta(ROBOT)
meta = copy.deepcopy(meta0)
# a made-up payload: the last tracked left / right pair (an asymmetric lateral range) and the first tracked body
ids = dict(zip(meta["tracked_bodies"], meta["tracked_body_ids"]))
left = [n for n in ids if n.startswith("left_") and ids[n] != meta["com_body_id"]][-1]
PAIR = {"bodies": [ids[left], ids[left.replace("left_", "right_", 1)]], "mass": [0.4, 1.6],
        "com_lo": [-0.01, 0.0, -0.12], "com_hi": [0.03, 0.04, -0.03]}  # fmt: skip
ONE = {"bodies": [meta["tracked_body_ids"][0]], "mass": [1.0, 3.0], "com_lo": [-0.02, -0.03, 0.0], "com_hi": [0.02, 0.03, 0.1]}
MID = np.float32([[1.0, 0.01, 0.02, -0.075], [1.0, 0.01, -0.02, -0.075], [2.0, 0.0, 0.0, 0.05]])  # the pair, the single body
meta["dr"]["payload"] = [PAIR, ONE]
meta0["dr"].pop("payload", None)
J = len(meta["joint_names"])
L, L0 = XiLayout(meta), XiLayout(meta0)
(pair, sp, lo, hi), (one, s1, _, _) = L.payload
assert L.dim == L0.dim + 12 and L.center.shape == L.scale.shape == (L.dim,) and not L0.payload
assert (sp.start, s1.stop) == (1 + J + 3 + J * len(L.factors), L0.dim + 12 - (L.delay is not None))  # after the factors, before the flags
assert np.array_equal(L.center[: sp.start], L0.center[: sp.start]) and np.array_equal(L.scale[: sp.start], L0.scale[: sp.start])
assert np.array_equal(lo[1], np.float32([0.4, -0.01, -0.04, -0.12])) and np.array_equal(hi[1], np.float32([1.6, 0.03, 0.0, -0.03]))
np.testing.assert_allclose(L.center[sp.start : s1.stop], MID.ravel(), atol=1e-7)  # the evaluation's values: the middles
np.testing.assert_allclose(L.scale[sp], [0.6, 0.02, 0.02, 0.045] * 2, atol=1e-7)

mirror = Mirror(meta, L)
xi = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, (3, L.dim)), jnp.float32)
m = mirror.xi(xi)
assert jnp.allclose(mirror.xi(m), xi)
a, b, flip = np.asarray(xi[:, sp]).reshape(3, 2, 4), np.asarray(m[:, sp]).reshape(3, 2, 4), np.float32([1, 1, -1, 1])
assert np.array_equal(b[:, 0], a[:, 1] * flip) and np.array_equal(b[:, 1], a[:, 0] * flip)
assert np.array_equal(np.asarray(m[:, s1]), np.asarray(xi[:, s1]) * flip)
np.testing.assert_allclose(mirror.xi(jnp.asarray(L.center))[sp], L.center[sp], atol=1e-7)  # the pair's ranges are mirror images
print(f"layout: {L.dim} context values ({L0.dim} without the payload), mirror swaps the pair and flips its lateral component")

if jax.default_backend() == "cpu":
    sys.exit(print("no GPU: simulator part skipped"))

from xmimic.kin import Kinematics  # noqa: E402
from xmimic.mdp import MDP  # noqa: E402
from xmimic.motion import load_motion  # noqa: E402

N = 8
kin = Kinematics(meta)
motion = load_motion(os.environ.get("XMIMIC_MOTION_DIR", os.path.join(ROOT, "data", "lafan1_g1")), kin, feet_of(meta))
mj = load_model(ROBOT)[0]
bodies = np.concatenate((pair, one))
for eval_mode in (True, False):
    sim = Sim(mj, meta, N, None, eval_mode=eval_mode)
    sim.mdp = MDP(kin, meta["default_joint_pos"], feet_of(meta), L.dim, meta["contact_z"])
    model_b, st = sim.init(jax.random.PRNGKey(0), motion)
    v = np.asarray(st["xi"][:, sp.start : s1.stop]).reshape(N, 3, 4)
    if eval_mode:
        np.testing.assert_allclose(v, np.broadcast_to(MID, v.shape), atol=1e-7)
    else:
        assert (v[:, :2] >= lo).all() and (v[:, :2] <= hi).all() and (v[..., 0].std(0) > 0.05).all()
    np.testing.assert_allclose(model_b.body_mass[:, bodies], v[..., 0], rtol=1e-6)
    np.testing.assert_allclose(model_b.body_ipos[:, bodies], v[..., 1:], atol=1e-7)
    np.testing.assert_allclose(model_b.body_inertia[:, bodies], mj.body_inertia[bodies] * (v[..., :1] / mj.body_mass[bodies, None]), rtol=1e-5)
    others = np.setdiff1d(np.arange(mj.nbody), bodies)
    np.testing.assert_allclose(model_b.body_mass[:, others], np.broadcast_to(mj.body_mass[others], (N, others.size)), rtol=1e-6)
print("simulator: mass, center of mass and inertia of the payload bodies are the drawn values (evaluation: the middles)")

# every world in world 0's state and model: one trajectory; with the lightest pair in world 0 and the heaviest in the others: two
same = lambda x: jnp.broadcast_to(x[:1], x.shape)  # noqa: E731
d = st["data"]
data = d.replace(qpos=same(d.qpos), qvel=same(d.qvel), ctrl=same(d.ctrl))
target = jnp.broadcast_to(sim.default_q + 0.2, (N, J))
fields = ("geom_friction", "body_ipos", "body_mass", "body_inertia") + tuple({f for n in L.factors for f in ACTUATOR_FIELDS[n]})
model_same = model_b.replace(**{f: same(getattr(model_b, f)) for f in fields})
model_same = model_same.replace(body_mass=model_same.body_mass.at[:, pair].set(PAIR["mass"][0]))
physics = jax.jit(sim.physics)
q = np.asarray(physics(model_same, data, target).qpos)
assert np.abs(q - q[0]).max() < 1e-6, np.abs(q - q[0]).max()
heavy = model_same.replace(body_mass=model_same.body_mass.at[1:, pair].set(PAIR["mass"][1]))
q = np.asarray(physics(heavy, data, target).qpos)
spread = np.abs(q[1:] - q[0]).max(1).min()
assert spread > 1e-5 and np.abs(q[1:] - q[1]).max() < 1e-6, spread
print(f"a heavier payload changes the step (smallest difference to the lightest world: {spread:.1e} rad)")
