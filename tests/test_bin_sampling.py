"""Reset-bin sampling (Sim.sample_frames) follows the bin probabilities and never draws a bin of probability 0.

    JAX_PLATFORMS=cpu python tests/test_bin_sampling.py
"""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xmimic.motion import BIN_FRAMES  # noqa: E402
from xmimic.sim import Sim  # noqa: E402

nb, N = 40, 400_000
q = np.random.default_rng(0).random(nb)
q[[0, 7, nb - 1]] = 0.0  # first, inner and last bin impossible
q /= q.sum()
motion = SimpleNamespace(num_bins=nb, bin_start=jnp.arange(nb) * BIN_FRAMES, bin_clip=jnp.zeros(nb, jnp.int32),
                         clip_end=jnp.asarray([nb * BIN_FRAMES - 1]), sample_mirror=lambda k, f: f)  # fmt: skip
st = {"bin_G": jnp.zeros(nb), "bin_w_avg": jnp.ones(nb) / nb, "focus_share": 1.0, "focus_q": jnp.asarray(q, jnp.float32)}
sim = SimpleNamespace(N=N, gamma=0.99, fail_reset=0.0)
f, bins, _ = Sim.sample_frames(sim, motion, st, jax.random.PRNGKey(0), False)
freq = np.bincount(np.asarray(bins), minlength=nb) / N
assert freq[[0, 7, nb - 1]].sum() == 0, freq
assert np.abs(freq - q).max() < 5 * np.sqrt(q.max() / N), np.abs(freq - q).max()
assert (np.asarray(f) // BIN_FRAMES == np.asarray(bins)).all()

# ~200k bins, uniform but the last 10 impossible: float32 cumsum rounding leaves no gap at the top and no draw past it
nb = 200_000
q = np.r_[np.ones(nb - 10), np.zeros(10)] / (nb - 10)
big = SimpleNamespace(**{**vars(motion), "num_bins": nb, "bin_start": jnp.arange(nb) * BIN_FRAMES, "bin_clip": jnp.zeros(nb, jnp.int32),
                         "clip_end": jnp.asarray([nb * BIN_FRAMES - 1])})  # fmt: skip
st = {**st, "bin_G": jnp.zeros(nb), "bin_w_avg": jnp.ones(nb) / nb, "focus_q": jnp.asarray(q, jnp.float32)}
_, bins, _ = Sim.sample_frames(SimpleNamespace(N=1_000_000, gamma=0.99, fail_reset=0.0), big, st, jax.random.PRNGKey(1), False)
top = np.asarray(bins)
assert top.max() < nb - 10 and (top >= nb - 1000).mean() > 0.8 * 990 / (nb - 10), (top.max(), (top >= nb - 1000).mean())
print("test_bin_sampling: ok")
