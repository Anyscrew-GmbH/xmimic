"""--init_policy / --pi_hold: a checkpoint written by runtime.export_policy loads back into a fresh policy exactly (other
network sizes are refused), and the policy LR multiplier is 0 for the first pi_hold iterations, unchanged after.

    JAX_PLATFORMS=cpu python tests/test_init_policy.py
"""

import os
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import numpy as np  # noqa: E402

from xmimic import nets  # noqa: E402
from xmimic.algorithm import XMimic, load_policy  # noqa: E402
from xmimic.config import Config  # noqa: E402
from xmimic.runtime import export_policy  # noqa: E402

O, G, J, H, W = 7, 11, 5, 3, (32, 16)
pi, pi_b = nets.pi_init(jax.random.PRNGKey(0), O, G, J, H, enc_hidden=W, dec_hidden=W)
pi_b = {**pi_b, "obs_norm": {**pi_b["obs_norm"], "mean": pi_b["obs_norm"]["mean"] + 0.5, "count": pi_b["obs_norm"]["count"] + 3.0}}
with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "ckpt_000001.npz")
    export_policy(path, (pi, pi_b), {"iteration": 1})
    fresh = nets.pi_init(jax.random.PRNGKey(1), O, G, J, H, enc_hidden=W, dec_hidden=W)
    got = load_policy(path, *fresh)
    for a, b in zip(jax.tree.leaves(got), jax.tree.leaves((pi, pi_b))):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    try:
        load_policy(path, *nets.pi_init(jax.random.PRNGKey(1), O, G, J, H, enc_hidden=(32, 8), dec_hidden=W))
        raise SystemExit("other sizes were not refused")
    except (AssertionError, ValueError):
        pass

cfg = SimpleNamespace(**{**vars(Config()), "pi_hold": 100})
for it, decay_at, want in ((0, -1, 0.0), (99, -1, 0.0), (100, -1, cfg.pi_lr_mult), (500, 100, 1.0)):
    got = float(XMimic.pi_mult(SimpleNamespace(cfg=cfg), float(it), float(decay_at)))
    assert abs(got - want) < 1e-6, (it, decay_at, got, want)
print("test_init_policy: ok")
