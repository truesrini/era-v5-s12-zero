"""
zero_sim.py - a small, readable simulation of DDP and ZeRO stages 1, 2 and 3.

Every "virtual GPU" is one Python thread that owns
  * its own rank and its own gloo communicator (real collectives over loopback TCP),
  * its own copy (or shard) of the model state.

Nothing is shared between ranks except through the collectives
(all-reduce, reduce-scatter, all-gather). The main thread only starts the
ranks and collects their reports at the end.

The model is a plain MLP whose forward and backward passes are written by
hand (no autograd), so you can see exactly *where* each stage gathers
parameters, reduces gradients and frees memory.

Mixed-precision bookkeeping follows the course notes (16 bytes / parameter):
    bf16 parameters ............ 2 bytes
    bf16 gradients ............. 2 bytes
    fp32 master copy ........... 4 bytes
    fp32 Adam m (1st moment) ... 4 bytes
    fp32 Adam v (2nd moment) ... 4 bytes

Usage from Python / a notebook:
    from zero_sim import Config, run
    res = run("zero2", Config(world_size=32))

Usage from the command line:
    python zero_sim.py --world-size 32 --steps 12
"""

from __future__ import annotations

import os

# Silence the harmless "client socket has failed to connect to <hostname>" warnings
# that c10d prints on Windows before it falls back to 127.0.0.1.
os.environ.setdefault("TORCH_CPP_LOG_LEVEL", "ERROR")

import argparse
import itertools
import json
import math
import threading
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import timedelta

import torch
import torch.distributed as dist

STAGES = ["ddp", "zero1", "zero2", "zero3"]
STAGE_LABEL = {"ddp": "DDP (no sharding)", "zero1": "ZeRO-1", "zero2": "ZeRO-2", "zero3": "ZeRO-3"}

PARAM_DTYPE = torch.bfloat16   # working copy used for compute: 2 bytes
GRAD_DTYPE = torch.bfloat16    # gradients: 2 bytes
MASTER_DTYPE = torch.float32   # master weights and Adam moments: 4 bytes each

# Every run gets a fresh port so sockets from the previous run never collide.
_PORTS = itertools.count(29700)


@dataclass
class Config:
    world_size: int = 32      # number of virtual GPUs (threads)
    hidden: int = 512         # width of every Linear layer
    layers: int = 6           # number of Linear(hidden, hidden) layers
    local_batch: int = 8      # samples per rank per step
    steps: int = 12           # optimizer steps
    warmup_steps: int = 2     # steps excluded from timing averages
    lr: float = 3e-4
    seed: int = 0
    track_timeline: bool = True   # record rank 0's memory after every alloc/free


# ----------------------------------------------------------------------------
# Analytic formulas (what the ZeRO paper / course notes predict)
# ----------------------------------------------------------------------------

def bytes_per_param(stage: str, n: int) -> dict:
    """Model-state bytes per parameter held by ONE rank, split by category."""
    shard = {"ddp": (1, 1, 1), "zero1": (1, 1, n), "zero2": (1, n, n), "zero3": (n, n, n)}[stage]
    p, g, o = shard
    return {"params": 2 / p, "grads": 2 / g, "optimizer": 12 / o}


def comm_volume_in_P(stage: str, n: int) -> float:
    """Bytes each rank sends per step, as a multiple of P (= 2 bytes x #params).

    Ring algorithms: all-reduce sends 2(n-1)/n, reduce-scatter and all-gather (n-1)/n.
    """
    f = (n - 1) / n
    return {"ddp": 2 * f, "zero1": 2 * f, "zero2": 2 * f, "zero3": 3 * f}[stage]


# ----------------------------------------------------------------------------
# Per-rank memory accounting
# ----------------------------------------------------------------------------

class MemTracker:
    """Counts the bytes of every tensor a rank holds, by category.

    All ranks live in one OS process, so process-level memory tools cannot tell
    ranks apart. Instead every state tensor is registered here when it is created
    and unregistered when it is dropped. Peak = the largest total ever live.
    """

    CATEGORIES = ["params", "gathered_params", "grads", "optimizer", "activations"]

    def __init__(self, record: bool):
        self.live: dict[str, tuple[str, int]] = {}
        self.by_cat = defaultdict(int)
        self.total = 0
        self.peak = 0
        self.peak_by_cat: dict[str, int] = {}
        self.peak_model_state = 0     # peak excluding activations
        self.record = record
        self.timeline: list[dict] = []
        self.step = -1                # -1 = initialisation, then 0, 1, 2, ...

    def alloc(self, key: str, cat: str, t: torch.Tensor, label: str = ""):
        nbytes = t.numel() * t.element_size()
        assert key not in self.live, key
        self.live[key] = (cat, nbytes)
        self.by_cat[cat] += nbytes
        self.total += nbytes
        self._mark(label or f"alloc {key}")
        return t

    def free(self, key: str, label: str = ""):
        cat, nbytes = self.live.pop(key)
        self.by_cat[cat] -= nbytes
        self.total -= nbytes
        self._mark(label or f"free {key}")

    def _mark(self, label: str):
        if self.total > self.peak:
            self.peak = self.total
            self.peak_by_cat = dict(self.by_cat)
        self.peak_model_state = max(self.peak_model_state, self.total - self.by_cat["activations"])
        if self.record:
            self.timeline.append({"step": self.step, "label": label,
                                  **{c: self.by_cat[c] for c in self.CATEGORIES}})

    def snapshot(self) -> dict:
        return {c: self.by_cat[c] for c in self.CATEGORIES}


# ----------------------------------------------------------------------------
# Communication wrapper: real gloo collectives + byte / time accounting
# ----------------------------------------------------------------------------

class Comm:
    def __init__(self, pg, rank: int, world_size: int):
        self.pg, self.rank, self.n = pg, rank, world_size
        self.log = defaultdict(lambda: {"calls": 0, "bytes_sent": 0.0, "time_s": 0.0})
        self.enabled = True   # turned off for warm-up steps

    def _record(self, tag: str, nbytes_sent: float, dt: float):
        if self.enabled:
            e = self.log[tag]
            e["calls"] += 1
            e["bytes_sent"] += nbytes_sent
            e["time_s"] += dt

    def all_reduce(self, t: torch.Tensor, tag: str):
        """Every rank ends with the SUM of everyone's t."""
        t0 = time.perf_counter()
        self.pg.allreduce([t]).wait()
        size = t.numel() * t.element_size()
        self._record(tag, 2 * (self.n - 1) / self.n * size, time.perf_counter() - t0)

    def reduce_scatter(self, out: torch.Tensor, inp: torch.Tensor, tag: str):
        """inp is a full tensor on every rank; out (1/n of it) gets this rank's slice of the SUM."""
        t0 = time.perf_counter()
        self.pg._reduce_scatter_base(out, inp).wait()
        size = inp.numel() * inp.element_size()
        self._record(tag, (self.n - 1) / self.n * size, time.perf_counter() - t0)

    def all_gather(self, out: torch.Tensor, inp: torch.Tensor, tag: str):
        """Each rank contributes its shard inp; out (n x bigger) receives all shards in rank order."""
        t0 = time.perf_counter()
        self.pg._allgather_base(out, inp).wait()
        size = out.numel() * out.element_size()
        self._record(tag, (self.n - 1) / self.n * size, time.perf_counter() - t0)

    def barrier(self):
        self.pg.barrier().wait()


def _make_process_group(port: int, rank: int, world_size: int):
    store = dist.TCPStore("127.0.0.1", port, world_size, False, use_libuv=False,
                          timeout=timedelta(seconds=300))
    opts = dist.ProcessGroupGloo._Options()
    opts._devices = [dist.ProcessGroupGloo.create_device(hostname="127.0.0.1")]
    opts._timeout = timedelta(seconds=300)
    return dist.ProcessGroupGloo(store, rank, world_size, opts)


# ----------------------------------------------------------------------------
# The model: `layers` x Linear(hidden, hidden), ReLU between them, MSE loss.
# Each layer's weight and bias live in ONE flat vector ("flat parameter"),
# padded so it splits evenly into world_size shards. Rank r owns slice r.
# ----------------------------------------------------------------------------

def layer_numel(cfg: Config) -> int:
    return cfg.hidden * cfg.hidden + cfg.hidden


def padded_numel(cfg: Config) -> int:
    n = cfg.world_size
    return math.ceil(layer_numel(cfg) / n) * n


def init_full_layer(cfg: Config, layer: int) -> torch.Tensor:
    """Same deterministic fp32 init on every rank (and for every stage)."""
    g = torch.Generator().manual_seed(cfg.seed * 1000 + layer)
    flat = torch.zeros(padded_numel(cfg), dtype=MASTER_DTYPE)
    h = cfg.hidden
    flat[: h * h] = torch.randn(h * h, generator=g) * math.sqrt(2.0 / h)   # He init for ReLU
    return flat   # bias starts at zero, padding stays zero


def batch_for(cfg: Config, step: int, rank: int, world_size: int):
    """A synthetic regression problem: y = sin(x @ A). Rank r takes rows r*B:(r+1)*B
    of the *global* batch, so every stage (and every world size) sees the same data."""
    h, b = cfg.hidden, cfg.local_batch
    g = torch.Generator().manual_seed(10_000 + cfg.seed)
    A = torch.randn(h, h, generator=g) / math.sqrt(h)
    g = torch.Generator().manual_seed(20_000 + cfg.seed * 1000 + step)
    x = torch.randn(world_size * b, h, generator=g)
    y = torch.sin(x @ A)
    return x[rank * b:(rank + 1) * b], y[rank * b:(rank + 1) * b]


def views(cfg: Config, flat: torch.Tensor):
    """Weight and bias views into a flat (padded) layer vector."""
    h = cfg.hidden
    return flat[: h * h].view(h, h), flat[h * h: h * h + h]


def adam_update(p32, g32, m, v, step, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Plain Adam on fp32 tensors (works the same on a full tensor or a shard)."""
    m.mul_(b1).add_(g32, alpha=1 - b1)
    v.mul_(b2).addcmul_(g32, g32, value=1 - b2)
    m_hat = m / (1 - b1 ** step)
    v_hat = v / (1 - b2 ** step)
    p32.addcdiv_(m_hat, v_hat.sqrt_().add_(eps), value=-lr)


# ----------------------------------------------------------------------------
# One rank's training loop
# ----------------------------------------------------------------------------

class Rank:
    def __init__(self, stage: str, cfg: Config, rank: int, comm: Comm):
        self.stage, self.cfg, self.rank, self.n = stage, cfg, rank, cfg.world_size
        self.comm = comm
        self.mem = MemTracker(record=cfg.track_timeline and rank == 0)
        self.L = cfg.layers
        self.numel = padded_numel(cfg)
        self.shard = self.numel // self.n
        self.lo, self.hi = rank * self.shard, (rank + 1) * self.shard
        self.timing = defaultdict(float)
        self.losses: list[float] = []
        self._init_state()

    # ---- what each stage keeps between steps --------------------------------
    def _init_state(self):
        st, mem = self.stage, self.mem
        self.params = [None] * self.L     # bf16: full (ddp/z1/z2) or shard (z3)
        self.master = [None] * self.L     # fp32: full (ddp) or shard (z1/z2/z3)
        self.m = [None] * self.L
        self.v = [None] * self.L
        self.grad_full = [None] * self.L  # bf16 full-size grads (ddp/z1, during backward)
        self.grad_shard = [None] * self.L  # bf16 shard grads (z1/z2/z3)
        for l in range(self.L):
            full32 = init_full_layer(self.cfg, l)
            if st == "zero3":
                p = full32[self.lo:self.hi].to(PARAM_DTYPE).clone()
            else:
                p = full32.to(PARAM_DTYPE)
            self.params[l] = mem.alloc(f"param{l}", "params", p, f"init param L{l}")
            owned32 = full32 if st == "ddp" else full32[self.lo:self.hi].clone()
            self.master[l] = mem.alloc(f"master{l}", "optimizer", owned32, f"init master L{l}")
            self.m[l] = mem.alloc(f"m{l}", "optimizer", torch.zeros_like(owned32), f"init adam_m L{l}")
            self.v[l] = mem.alloc(f"v{l}", "optimizer", torch.zeros_like(owned32), f"init adam_v L{l}")
            del full32

    # ---- ZeRO-3 helpers: materialise a full layer, then throw it away --------
    def _gather_layer(self, l: int, phase: str) -> torch.Tensor:
        full = torch.empty(self.numel, dtype=PARAM_DTYPE)
        self.mem.alloc(f"gathered{l}", "gathered_params", full, f"{phase}: all-gather L{l}")
        self.comm.all_gather(full, self.params[l], f"param all-gather ({phase})")
        return full

    def _release_layer(self, l: int, phase: str):
        self.mem.free(f"gathered{l}", f"{phase}: free gathered L{l}")

    def _full_params(self, l: int, phase: str) -> torch.Tensor:
        return self._gather_layer(l, phase) if self.stage == "zero3" else self.params[l]

    # ---- one optimizer step -------------------------------------------------
    def step(self, step_idx: int):
        cfg, mem, comm, st = self.cfg, self.mem, self.comm, self.stage
        timed = step_idx >= cfg.warmup_steps
        comm.enabled = timed
        T = self.timing if timed else defaultdict(float)
        mem.step = step_idx
        comm.barrier()
        t_step = time.perf_counter()

        x, y = batch_for(cfg, step_idx, self.rank, self.n)
        h = x.to(PARAM_DTYPE)

        # ---------------- forward ----------------
        saved_inputs = []
        for l in range(self.L):
            full = self._full_params(l, "fwd")
            t0 = time.perf_counter()
            W, b = views(cfg, full)
            mem.alloc(f"act{l}", "activations", h, f"fwd: save input L{l}")
            saved_inputs.append(h)
            out = h @ W.T + b
            if l < self.L - 1:
                out = torch.relu(out)
            T["forward_compute_s"] += time.perf_counter() - t0
            if st == "zero3":
                self._release_layer(l, "fwd")
            h = out

        out32 = h.float()
        loss = ((out32 - y) ** 2).mean()
        grad = (2.0 * (out32 - y) / out32.numel()).to(PARAM_DTYPE)   # dLoss/dOut
        self.losses.append(loss.item())

        # ---------------- backward (last layer first) ----------------
        for l in reversed(range(self.L)):
            full = self._full_params(l, "bwd")
            t0 = time.perf_counter()
            W, _ = views(cfg, full)
            inp = saved_inputs[l]
            g_flat = torch.zeros(self.numel, dtype=GRAD_DTYPE)
            gW, gb = views(cfg, g_flat)
            gW.copy_(grad.T @ inp)
            gb.copy_(grad.sum(0))
            grad_in = grad @ W if l > 0 else None
            T["backward_compute_s"] += time.perf_counter() - t0

            mem.alloc(f"gradfull{l}", "grads", g_flat, f"bwd: grad L{l}")
            if st == "zero3":
                self._release_layer(l, "bwd")

            if st == "ddp":
                # DDP: all-reduce this layer's gradient as soon as it is ready (bucket = 1 layer)
                comm.all_reduce(g_flat, "grad all-reduce")
                self.grad_full[l] = g_flat
            elif st == "zero1":
                # ZeRO-1: keep the FULL gradient until backward is done, reduce afterwards
                self.grad_full[l] = g_flat
            else:
                # ZeRO-2/3: reduce-scatter now, keep only my slice, drop the full gradient
                self._reduce_scatter_grad(l, g_flat, "bwd")
                mem.free(f"gradfull{l}", f"bwd: free full grad L{l}")
                del g_flat

            mem.free(f"act{l}", f"bwd: free input L{l}")
            if grad_in is not None:
                # this layer's input was ReLU(previous layer's output), so the
                # ReLU derivative is simply "was my input positive?"
                grad = grad_in * (inp > 0)
            saved_inputs[l] = None

        if st == "zero1":
            for l in range(self.L):
                self._reduce_scatter_grad(l, self.grad_full[l], "after bwd")
                mem.free(f"gradfull{l}", f"after bwd: free full grad L{l}")
                self.grad_full[l] = None

        # ---------------- optimizer ----------------
        t0 = time.perf_counter()
        for l in range(self.L):
            if st == "ddp":
                g32 = self.grad_full[l].float() / self.n          # average over ranks
            else:
                g32 = self.grad_shard[l].float() / self.n
            adam_update(self.master[l], g32, self.m[l], self.v[l], step_idx + 1, cfg.lr)
            if st == "ddp":
                self.params[l].copy_(self.master[l])              # full fp32 -> full bf16
                mem.free(f"gradfull{l}", f"step: free grad L{l}")
                self.grad_full[l] = None
            elif st in ("zero1", "zero2"):
                # write my updated slice into the full bf16 copy
                self.params[l][self.lo:self.hi].copy_(self.master[l])
            else:
                self.params[l].copy_(self.master[l])              # shard -> shard
            if st != "ddp":
                mem.free(f"gradshard{l}", f"step: free grad shard L{l}")
                self.grad_shard[l] = None
        T["optimizer_s"] += time.perf_counter() - t0

        if st in ("zero1", "zero2"):
            # everybody needs everybody's updated slice back: all-gather the bf16 params
            for l in range(self.L):
                mine = self.params[l][self.lo:self.hi].clone()
                comm.all_gather(self.params[l], mine, "param all-gather (after step)")

        T["step_s"] += time.perf_counter() - t_step

    def _reduce_scatter_grad(self, l: int, g_flat: torch.Tensor, phase: str):
        gs = torch.empty(self.shard, dtype=GRAD_DTYPE)
        self.mem.alloc(f"gradshard{l}", "grads", gs, f"{phase}: grad shard L{l}")
        self.comm.reduce_scatter(gs, g_flat, "grad reduce-scatter")
        self.grad_shard[l] = gs

    # ---- report -------------------------------------------------------------
    def report(self) -> dict:
        timed_steps = self.cfg.steps - self.cfg.warmup_steps
        return {
            "rank": self.rank,
            "losses": self.losses,
            "end_of_step_bytes": self.mem.snapshot(),
            "peak_bytes": self.mem.peak,
            "peak_by_cat": self.mem.peak_by_cat,
            "peak_model_state_bytes": self.mem.peak_model_state,
            "timeline": self.mem.timeline,
            "comm": {k: dict(v) for k, v in self.comm.log.items()},
            "timing_per_step": {k: v / timed_steps for k, v in self.timing.items()},
            # kept in memory only (not JSON) for the correctness check
            "_master": [t.clone() for t in self.master],
            "_params": [t.clone() for t in self.params],
        }


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------

def spawn(fn, world_size: int):
    """Start `world_size` virtual GPUs. Each one is a thread that builds its own gloo
    communicator and calls fn(rank, comm). Returns the list of fn's return values."""
    torch.set_num_threads(1)          # all ranks share the CPU; no nested threading
    port = next(_PORTS)
    master_store = dist.TCPStore("127.0.0.1", port, world_size, True,
                                 wait_for_workers=False, use_libuv=False,
                                 timeout=timedelta(seconds=300))
    results: list = [None] * world_size
    errors: list[BaseException] = []

    def worker(r: int):
        try:
            comm = Comm(_make_process_group(port, r, world_size), r, world_size)
            results[r] = fn(r, comm)
            comm.barrier()
        except BaseException as e:   # surface errors from threads
            errors.append(e)
            raise

    threads = [threading.Thread(target=worker, args=(r,), daemon=True) for r in range(world_size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    del master_store
    if errors:
        raise errors[0]
    return results


def demo_collectives(world_size: int = 4) -> list[dict]:
    """Tiny example: rank r holds [10r, 10r+1, ...]. Shows what each collective returns."""
    def fn(r, comm):
        x = torch.arange(world_size, dtype=torch.float32) + 10 * r
        ar = x.clone(); comm.all_reduce(ar, "demo")
        rs = torch.empty(1); comm.reduce_scatter(rs, x.clone(), "demo")
        ag = torch.empty(world_size); comm.all_gather(ag, torch.tensor([float(10 * r)]), "demo")
        return {"rank": r, "input": x.tolist(), "all_reduce": ar.tolist(),
                "reduce_scatter": rs.tolist(), "all_gather (input = [10r])": ag.tolist()}
    return spawn(fn, world_size)


def run(stage: str, cfg: Config | None = None) -> dict:
    """Train `cfg.steps` steps with `cfg.world_size` virtual GPUs and return a report."""
    assert stage in STAGES, stage
    cfg = cfg or Config()

    def fn(r, comm):
        rk = Rank(stage, cfg, r, comm)
        for s in range(cfg.steps):
            rk.step(s)
        return rk.report()

    t0 = time.perf_counter()
    reports = spawn(fn, cfg.world_size)
    return _summarise(stage, cfg, reports, time.perf_counter() - t0)


def _summarise(stage: str, cfg: Config, reports: list[dict], wall: float) -> dict:
    n = cfg.world_size
    psi = cfg.layers * padded_numel(cfg)           # parameters actually stored (incl. padding)
    psi_real = cfg.layers * layer_numel(cfg)
    r0 = reports[0]

    # rebuild the full fp32 master weights from what the ranks own (check only, no comm)
    if stage == "ddp":
        master_full = r0["_master"]
    else:
        master_full = [torch.cat([rep["_master"][l] for rep in reports]) for l in range(cfg.layers)]

    # do all replicas of the bf16 parameters agree? (not applicable to ZeRO-3: no replicas)
    if stage == "zero3":
        replica_max_diff = 0.0
    else:
        replica_max_diff = max(
            (rep["_params"][l].float() - r0["_params"][l].float()).abs().max().item()
            for rep in reports for l in range(cfg.layers))

    comm_tags = sorted({t for rep in reports for t in rep["comm"]})
    comm = {}
    timed_steps = cfg.steps - cfg.warmup_steps
    for tag in comm_tags:
        e = r0["comm"][tag]
        comm[tag] = {
            "calls_per_step": e["calls"] / timed_steps,
            "bytes_sent_per_step": e["bytes_sent"] / timed_steps,
            "time_s_per_step_mean_over_ranks": sum(rep["comm"][tag]["time_s"] for rep in reports) / n / timed_steps,
        }
    bytes_sent = sum(c["bytes_sent_per_step"] for c in comm.values())

    timing = {k: sum(rep["timing_per_step"].get(k, 0.0) for rep in reports) / n
              for k in ["forward_compute_s", "backward_compute_s", "optimizer_s", "step_s"]}
    timing["comm_s"] = sum(c["time_s_per_step_mean_over_ranks"] for c in comm.values())

    # average the per-rank losses -> global loss per step
    losses = [sum(rep["losses"][s] for rep in reports) / n for s in range(cfg.steps)]

    return {
        "stage": stage,
        "label": STAGE_LABEL[stage],
        "config": asdict(cfg),
        "num_params": psi_real,
        "num_params_padded": psi,
        "P_bytes": 2 * psi,
        "memory": {
            "end_of_step_bytes": r0["end_of_step_bytes"],
            "peak_bytes": max(rep["peak_bytes"] for rep in reports),
            "peak_by_cat": r0["peak_by_cat"],
            "peak_model_state_bytes": max(rep["peak_model_state_bytes"] for rep in reports),
            "predicted_bytes": {k: v * psi for k, v in bytes_per_param(stage, n).items()},
        },
        "comm": comm,
        "comm_bytes_sent_per_step": bytes_sent,
        "comm_in_P": bytes_sent / (2 * psi),
        "comm_in_P_predicted": comm_volume_in_P(stage, n),
        "timing_per_step": timing,
        "losses": losses,
        "replica_max_abs_diff": replica_max_diff,
        "timeline": r0["timeline"],
        "wall_s": wall,
        "_master_full": master_full,
    }


def run_all(cfg: Config | None = None, stages=STAGES, verbose=True) -> dict:
    cfg = cfg or Config()
    out = {}
    for s in stages:
        out[s] = run(s, cfg)
        if verbose:
            r = out[s]
            print(f"{r['label']:<18} peak/rank {r['memory']['peak_model_state_bytes']/2**20:7.2f} MiB  "
                  f"comm {r['comm_in_P']:.2f} P  step {r['timing_per_step']['step_s']*1e3:7.1f} ms  "
                  f"final loss {r['losses'][-1]:.5f}  ({r['wall_s']:.1f}s)")
    return out


def to_json(results: dict) -> dict:
    """Strip tensors so results can be saved."""
    return {s: {k: v for k, v in r.items() if not k.startswith("_")} for s, r in results.items()}


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--world-size", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--local-batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="results/results.json")
    a = ap.parse_args()
    cfg = Config(world_size=a.world_size, hidden=a.hidden, layers=a.layers,
                 local_batch=a.local_batch, steps=a.steps)
    res = run_all(cfg)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(to_json(res), f, indent=1)
    print("saved", a.out)
