"""Per-joint actuator randomization (meta["dr"]["actuator"]): the sim context grows by J factors per constant, the mirror
swaps their sides, and (on a GPU) each constant's factors reach the model fields and change the physics.
Action delay (meta["dr"]["action_delay"]): a delayed episode's joint targets come from the previous action.

    JAX_PLATFORMS=cpu python tests/test_actuator_dr.py    # layout and mirror
    python tests/test_actuator_dr.py                      # also the simulator; the robot and its clips can be set with
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

RANGES = {"kp": [0.5, 1.5], "kd": [0.5, 1.5], "armature": [0.5, 2.0], "frictionloss": [0.2, 3.0]}
ROBOT = os.environ.get("XMIMIC_ROBOT_DIR")
meta0 = load_meta(ROBOT)
meta = copy.deepcopy(meta0)
meta["dr"]["actuator"] = RANGES
meta["dr"]["action_delay"] = 0.5
J = len(meta["joint_names"])
L = XiLayout(meta)
meta0["dr"].pop("actuator", None), meta0["dr"].pop("action_delay", None)
assert L.dim == XiLayout(meta0).dim + 4 * J + 1 and L.center.shape == L.scale.shape == (L.dim,)
assert XiLayout(meta0).applied(1, 2, None) == 1  # without a delay: the action itself, whatever the context

mirror = Mirror(meta, L)
xi = jnp.asarray(np.random.default_rng(0).uniform(0.5, 1.5, (3, L.dim)), jnp.float32)
assert jnp.allclose(mirror.xi(mirror.xi(xi)), xi)
i = next(k for k, n in enumerate(meta["joint_names"]) if n.startswith("left_"))
sl = L.factors["kp"][0]
assert jnp.array_equal(mirror.xi(xi)[:, sl][:, i], xi[:, sl][:, mirror.perm[i]])
flags = jnp.zeros((2, L.dim)).at[1, L.delay].set(1.0)
assert jnp.array_equal(L.applied(jnp.ones((2, J)), jnp.zeros((2, J)), flags)[:, 0], jnp.asarray([1.0, 0.0]))
print(f"layout: {L.dim} context values, mirror swaps the factors' sides")

if jax.default_backend() == "cpu":
    sys.exit(print("no GPU: simulator part skipped"))

from xmimic.kin import Kinematics  # noqa: E402
from xmimic.mdp import MDP  # noqa: E402
from xmimic.motion import load_motion  # noqa: E402

N = 8
kin = Kinematics(meta)
motion = load_motion(os.environ.get("XMIMIC_MOTION_DIR", os.path.join(ROOT, "data", "lafan1_g1")), kin, feet_of(meta))
mj = load_model(ROBOT)[0]
sim = Sim(mj, meta, N, None)
sim.mdp = MDP(kin, meta["default_joint_pos"], feet_of(meta), L.dim, meta["contact_z"])
model_b, st = sim.init(jax.random.PRNGKey(0), motion)
fac = {n: np.asarray(st["xi"][:, s]) for n, (s, lo, hi) in L.factors.items()}
assert all((f >= RANGES[n][0]).all() and (f <= RANGES[n][1]).all() and f.std() > 0.05 for n, f in fac.items())
kp, kd = mj.actuator_gainprm[sim.act, 0], -mj.actuator_biasprm[sim.act, 2]
np.testing.assert_allclose(model_b.actuator_gainprm[:, sim.act, 0], kp * fac["kp"], rtol=1e-5)
np.testing.assert_allclose(model_b.actuator_biasprm[:, sim.act, 1], -kp * fac["kp"], rtol=1e-5)
np.testing.assert_allclose(model_b.actuator_biasprm[:, sim.act, 2], -kd * fac["kd"], rtol=1e-5)
np.testing.assert_allclose(model_b.dof_armature[:, sim.dadr], mj.dof_armature[sim.dadr] * fac["armature"], rtol=1e-5)
np.testing.assert_allclose(model_b.dof_frictionloss[:, sim.dadr], mj.dof_frictionloss[sim.dadr] * fac["frictionloss"], rtol=1e-5)

# every world in world 0's state and model: one trajectory; with one constant's factors back in: N trajectories
same = lambda x: jnp.broadcast_to(x[:1], x.shape)  # noqa: E731
d = st["data"]
data = d.replace(qpos=same(d.qpos), qvel=same(d.qvel) + 1.0, ctrl=same(d.ctrl))  # moving joints: kd and friction loss act
target = jnp.broadcast_to(sim.default_q + 0.2, (N, J))
fields = ("geom_friction", "body_ipos") + tuple({f for fs in ACTUATOR_FIELDS.values() for f in fs})
fields += ("body_mass", "body_inertia") if L.payload else ()  # a robot with payload randomization: per world as well
model_same = model_b.replace(**{f: same(getattr(model_b, f)) for f in fields})
physics = jax.jit(sim.physics)
q = np.asarray(physics(model_same, data, target).qpos)
assert np.abs(q - q[0]).max() < 1e-6, np.abs(q - q[0]).max()
for name, fs in ACTUATOR_FIELDS.items():
    q = np.asarray(physics(model_same.replace(**{f: getattr(model_b, f) for f in fs}), data, target).qpos)
    spread = np.abs(q[1:] - q[0]).max(1).min()
    assert spread > 1e-5, (name, spread)
    print(f"{name}: per-world factors change the step (smallest difference to world 0: {spread:.1e} rad)")

# action delay: the flag is 0 or 1, a delayed episode starts from the action that holds its reset pose, and delayed
# worlds' joint targets come from the previous action
late = np.asarray(st["xi"][:, L.delay])
assert set(np.unique(late)) <= {0.0, 1.0}
hold = np.asarray(sim.default_q + st["xi"][:, L.joint_offset] + sim.alpha * st["a_prev"])  # after the reset in init
np.testing.assert_allclose(hold[late > 0.5], np.asarray(st["data"].ctrl[:, sim.act])[late > 0.5], atol=1e-5)
assert not np.asarray(st["a_prev"])[late < 0.5].any()
st = {**st, "xi": st["xi"].at[:, L.delay].set(jnp.arange(N) % 2), "a_prev": jnp.full((N, J), -0.5)}
st1, _ = jax.jit(sim.step_core)(model_b, motion, st, jnp.full((N, J), 0.5))
got = np.asarray(st1["data"].ctrl[:, sim.act] - sim.default_q - st["xi"][:, L.joint_offset]) / sim.alpha
np.testing.assert_allclose(got, np.where(np.arange(N)[:, None] % 2, -0.5, 0.5) * np.ones((1, J)), atol=1e-5)
print("action delay: delayed worlds apply the previous action")
