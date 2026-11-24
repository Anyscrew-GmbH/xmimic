"""Weights & Biases logging (xmimic/wandb_log.py), with a stand-in for the wandb package; needs no wandb installation, no
network and no GPU. Training itself needs the GPU simulator, so the run log (runtime.RunLog, which every layout logs through)
is fed made-up statistics of a few log intervals.

  off      without --wandb_project (and WANDB_PROJECT) wandb is not started
  run      each train_log.csv row's numbers are sent at its iteration; the settings switch off console, code, git, metadata
           and system metrics; the flags reach wandb.init
  config   the configuration sent holds no option that names a file or directory (no key naming one, no value that looks
           like a path), and keeps the hyperparameters

    JAX_PLATFORMS=cpu python tests/test_wandb_log.py
"""

import csv
import dataclasses
import os
import sys
import tempfile
from types import SimpleNamespace
from unittest import mock

import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from xmimic import wandb_log  # noqa: E402
from xmimic.config import parse_args  # noqa: E402
from xmimic.runtime import RunLog  # noqa: E402

NEVER = str(10**9)
STATS = ("sim_steps", "wm", "wm8", "pi", "v", "ir", "gn", "surv", "r", "n_coll", "n_done", "ep_len_sum", "ep_ret_norm_sum", "term_sum",
         "lr", "pi_lr", "nonfinite", "decay_at")  # fmt: skip


class FakeWandb:
    """wandb.Settings, wandb.init and run.log, recorded."""

    def __init__(self):
        self.init_kw, self.logged, self.metrics = None, [], {}

    def Settings(self, **kw):
        return kw

    def init(self, **kw):
        self.init_kw = kw
        return self

    def define_metric(self, name, step_metric):
        self.metrics[name] = step_metric

    def log(self, data, step=None):
        self.logged.append((step, data))


def run(flags, intervals=3):
    """A RunLog of a config with `flags`, fed `intervals` log intervals: (the fake wandb, train_log.csv rows, the config)."""
    fake, wandb_log._run = FakeWandb(), None
    xm = SimpleNamespace(obs_dim=3, ref=None, motion=SimpleNamespace(goal=lambda f, k: jnp.zeros((4, 2))),
                         layout=SimpleNamespace(dim=0, masked=lambda g, *a: g))  # fmt: skip
    rng = np.random.default_rng(0)
    with mock.patch.dict(sys.modules, {"wandb": fake}), tempfile.TemporaryDirectory() as d:
        cfg = parse_args(["--log_every", "2", "--ckpt_every", NEVER, "--iterations", NEVER, "--log_dir", d] + flags)
        log = RunLog(xm, cfg, 0.0, {"w": np.zeros(3)}, {**dataclasses.asdict(cfg), "gpus": cfg.gpus})
        for it in range(1, 2 * intervals + 1):
            st = {k: rng.uniform(1, 2, (1,)) for k in STATS}
            log.add(it, {**st, "decay_at": -np.ones(1), "nonfinite": np.zeros(1, np.int32)}, None)
        log.close()
        assert fake.init_kw is None or fake.init_kw["dir"] == d
        rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
    wandb_log._run = None
    return fake, rows, cfg


def test_wandb_log():
    with mock.patch.dict(os.environ, {"WANDB_PROJECT": ""}):
        fake, rows, _ = run([])
    assert fake.init_kw is None and not fake.logged and len(rows) == 3
    print(f"off: {len(rows)} train-log rows, wandb not started")

    paths = ["--motion_dir", "clips/set_a", "--robot_dir", "robots/r"]
    fake, rows, cfg = run(["--wandb_project", "proj", "--wandb_entity", "team", "--wandb_name", "run-a", "--wandb_mode", "offline"] + paths)
    kw = fake.init_kw
    assert (kw["project"], kw["entity"], kw["name"], kw["mode"]) == ("proj", "team", "run-a", "offline")
    assert kw["settings"]["console"] == "off" and not kw["settings"]["save_code"] and kw["settings"]["disable_git"] and kw["settings"]["x_disable_meta"]
    assert kw["settings"]["x_disable_stats"] and not kw["settings"]["x_save_requirements"]
    assert [s for s, _ in fake.logged] == [int(r["iteration"]) for r in rows] == [2, 4, 6]
    for (_, sent), row in zip(fake.logged, rows):
        assert set(sent) == set(RunLog.COLS) and all(np.isclose(float(row[k]), v, rtol=1e-5) for k, v in sent.items()), (sent, row)
    print(f"run: {len(rows)} rows sent at iterations {[s for s, _ in fake.logged]}, {len(RunLog.COLS)} numbers each, equal to train_log.csv")

    config = kw["config"]
    names_file = lambda s: wandb_log.PATH_KEY.search(s) or wandb_log.PATH_VALUE.search(s)  # noqa: E731
    assert not any(names_file(k) or (isinstance(v, str) and names_file(v)) for k, v in config.items()), config
    assert not {"motion_dir", "robot_dir", "log_dir"} & set(config) and not any(k.startswith("wandb") for k in config)
    assert {"lr", "pi_clip_value", "horizon", "num_envs", "goal_mask_share", "gpus"} <= set(config) and config["num_envs"] == cfg.num_envs
    print(f"config: {len(config)} options sent, none naming a file or directory; dropped: "
          f"{sorted(set(dataclasses.asdict(cfg)) - set(config))}")


if __name__ == "__main__":
    test_wandb_log()
    print("\nall checks passed")
