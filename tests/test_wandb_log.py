"""Weights & Biases logging (xmimic/wandb_log.py), with a stand-in for the wandb package; needs no wandb installation, no
network and no GPU. Training itself needs the GPU simulator, so the run log (runtime.RunLog, which every layout logs through)
is fed made-up statistics of a few log intervals.

  off      without --wandb_project (and WANDB_PROJECT) wandb is not started
  run      each train_log.csv row's numbers are sent at its iteration; the settings switch off console, code, git, metadata
           and system metrics; the flags reach wandb.init
  config   the configuration sent holds no option that names a file or directory (no key naming one, no value that looks
           like a path), and keeps the hyperparameters
  fallback a wandb that cannot start online (login, network) leaves one warning and an offline run; one that cannot start at
           all, or fails while logging, leaves a warning and the run log goes on
  real     with the wandb package installed: an offline run's record (what `wandb sync` uploads) holds the numbers and the
           filtered configuration, not a path, console line, program or host name; no output, requirements, metadata or code file

    JAX_PLATFORMS=cpu python tests/test_wandb_log.py
"""

import contextlib
import csv
import dataclasses
import glob
import io
import os
import socket
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

    def __init__(self, fail=()):
        self.init_kw, self.logged, self.metrics, self.fail = None, [], {}, fail  # fail: modes whose init raises, or "log"

    def Settings(self, **kw):
        return kw

    def init(self, **kw):
        if kw["mode"] in self.fail:
            raise RuntimeError("stand-in: cannot start")
        self.init_kw = kw
        return self

    def define_metric(self, name, step_metric):
        self.metrics[name] = step_metric

    def log(self, data, step=None):
        if "log" in self.fail:
            raise RuntimeError("stand-in: cannot log")
        self.logged.append((step, data))


def run(flags, intervals=3, fail=()):
    """A RunLog of a config with `flags`, fed `intervals` log intervals: (the fake wandb, train_log.csv rows, the config, the output)."""
    fake, wandb_log._run, out = FakeWandb(fail), None, io.StringIO()
    xm = SimpleNamespace(obs_dim=3, ref=None, motion=SimpleNamespace(goal=lambda f, k: jnp.zeros((4, 2))),
                         layout=SimpleNamespace(dim=0, masked=lambda g, *a: g))  # fmt: skip
    rng = np.random.default_rng(0)
    with mock.patch.dict(sys.modules, {"wandb": fake}), tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(out):
        cfg = parse_args(["--log_every", "2", "--ckpt_every", NEVER, "--iterations", NEVER, "--log_dir", d] + flags)
        log = RunLog(xm, cfg, 0.0, {"w": np.zeros(3)}, {**dataclasses.asdict(cfg), "gpus": cfg.gpus})
        for it in range(1, 2 * intervals + 1):
            st = {k: rng.uniform(1, 2, (1,)) for k in STATS}
            log.add(it, {**st, "decay_at": -np.ones(1), "nonfinite": np.zeros(1, np.int32)}, None)
        log.close()
        assert fake.init_kw is None or fake.init_kw["dir"] == d
        rows = list(csv.DictReader(open(os.path.join(d, "train_log.csv"))))
    wandb_log._run = None
    return fake, rows, cfg, out.getvalue()


def test_wandb_log():
    with mock.patch.dict(os.environ, {"WANDB_PROJECT": ""}):
        fake, rows, _, _ = run([])
    assert fake.init_kw is None and not fake.logged and len(rows) == 3
    print(f"off: {len(rows)} train-log rows, wandb not started")

    paths = ["--motion_dir", "clips/set_a", "--robot_dir", "robots/r"]
    fake, rows, cfg, _ = run(["--wandb_project", "proj", "--wandb_entity", "team", "--wandb_name", "run-a", "--wandb_mode", "offline"] + paths)
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


def test_fallback():
    online = ["--wandb_project", "proj", "--wandb_mode", "online"]
    fake, rows, _, out = run(online, fail=("online",))
    assert fake.init_kw["mode"] == "offline" and out.count("[wandb] warning") == 1 and "logging offline to" in out
    assert len(fake.logged) == len(rows) == 3
    for fail in (("online", "offline"), ("log",)):
        fake, rows, _, out = run(online, fail=fail)
        assert not fake.logged and out.count("[wandb] warning") == len(fail) and len(rows) == 3, out
    print("fallback: online start failed -> one warning, offline run with every row; no start at all, or a failing log -> "
          "warnings, every train-log row written")


def test_real():
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("real: wandb not installed, skipped")
        return
    wandb_log._run = None
    with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"WANDB_SILENT": "true"}):
        config = {"wandb_project": "proj", "wandb_mode": "offline", "lr": 0.001, "data_dir": "x/y", "init_ckpt": "a.npz"}
        wandb_log.start(config, d)
        print("console output that must stay local")
        for it in (1, 2):
            wandb_log.log({"it": it, "loss": 1.0 / it, "name": "x/y"}, step=it)
        wandb_log._run.finish()
        wandb_log._run = None
        (record,) = glob.glob(os.path.join(d, "wandb", "offline-run-*", "run-*.wandb"))  # what `wandb sync` uploads
        run_dir = os.path.dirname(record)
        others = [f for f in glob.glob(os.path.join(run_dir, "**"), recursive=True) if os.path.isfile(f) and f != record]
        assert all(os.path.relpath(f, run_dir).startswith("logs" + os.sep) for f in others), others  # wandb's local debug logs only
        data = open(record, "rb").read()
        sent = [b"lr", b"loss", b"0.001"]
        kept = [b"x/y", b"a.npz", b"data_dir", b"init_ckpt", b"must stay local", b"test_wandb_log", socket.gethostname().encode()]
        assert all(x in data for x in sent) and not [x for x in kept if x in data], [x for x in kept if x in data]
    print(f"real: wandb {wandb.__version__} offline run: its record holds the numbers and the filtered configuration; no path, console line, "
          "program name or host name, and no output, requirements, metadata or code file")


if __name__ == "__main__":
    test_wandb_log()
    test_fallback()
    test_real()
    print("\nall checks passed")
