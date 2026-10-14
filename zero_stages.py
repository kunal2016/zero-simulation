"""
Per-stage ZeRO training steps built on zero_sim.py.

Each stage has its OWN step function with its OWN collective sequence, operating on the
arrays the virtual GPUs actually own:

  ZeRO-0  every GPU owns full param + full grad + full optimizer (all replicated);
          gradients all-reduced; every GPU redundantly updates the full weights.
  ZeRO-1  optimizer sharded; gradients reduce-scattered, each GPU updates its optimizer
          shard, updated params all-gathered back to everyone.
  ZeRO-2  + gradients sharded (only the shard is kept resident after reduce-scatter).
  ZeRO-3  + parameters sharded; the full weights are all-gathered for the forward,
          freed, and all-gathered AGAIN for the backward (two real collectives).

Because the reduction is deterministic and Adam is elementwise, every stage reproduces
the ZeRO-0 parameter trajectory exactly -- they differ only in memory and communication.
"""

import numpy as np
from zero_sim import (VirtualGPU, FP16, FP32, make_shards,
                      all_reduce, reduce_scatter, all_gather,
                      adam_update, compute_grads, run_forward, run_backward)

STAGE_NAMES = {0: "ZeRO-0 (baseline DDP)", 1: "ZeRO-1 (Pos)",
               2: "ZeRO-2 (Pos+g)", 3: "ZeRO-3 (Pos+g+p)"}
EB = 2   # fp16 bytes per element -> communication is measured in fp16 bytes


def init_gpus(stage, model, shards):
    """Create N GPUs and give each the REAL arrays it owns for this stage."""
    n, P, init = len(shards), model.P, model.init_flat
    gpus = [VirtualGPU(i) for i in range(n)]
    for i, g in enumerate(gpus):
        s = shards[i]
        ss = s.stop - s.start
        if stage == 0:                                   # everything replicated
            g.put('param',  init.astype(FP16).copy())
            g.put('grad',   np.zeros(P, FP16))
            g.put('master', init.copy())
            g.put('m', np.zeros(P, FP32)); g.put('v', np.zeros(P, FP32))
        elif stage == 1:                                 # shard optimizer only
            g.put('param',  init.astype(FP16).copy())
            g.put('grad',   np.zeros(P, FP16))
            g.put('master', init[s].copy())
            g.put('m', np.zeros(ss, FP32)); g.put('v', np.zeros(ss, FP32))
        elif stage == 2:                                 # + shard gradients
            g.put('param',  init.astype(FP16).copy())
            g.put('grad',   np.zeros(ss, FP16))
            g.put('master', init[s].copy())
            g.put('m', np.zeros(ss, FP32)); g.put('v', np.zeros(ss, FP32))
        else:                                            # stage 3: shard everything
            g.put('param',  init[s].astype(FP16).copy())
            g.put('grad',   np.zeros(ss, FP16))
            g.put('master', init[s].copy())
            g.put('m', np.zeros(ss, FP32)); g.put('v', np.zeros(ss, FP32))
    return gpus


def full_master(stage, gpus):
    """Reconstruct the full fp32 weight vector from whatever the GPUs own."""
    if stage == 0:
        return gpus[0].store['master'].copy()            # replicated
    return np.concatenate([g.store['master'] for g in gpus])  # sharded


# ---------------------------------------------------------------------- per-stage steps
def step_z0(gpus, model, shards, batches, t, total):
    # Replicated params: each GPU uses its own copy (identical). Forward == backward copy.
    fwd = [g.store['param'] for g in gpus]
    sse, grads = compute_grads(gpus, model, fwd, fwd, batches)
    for i, g in enumerate(gpus):
        g.store['grad'][...] = grads[i]                  # full gradient resident
    reduced = all_reduce(gpus, 'grad', EB)               # 2*Psi; full grad on every GPU
    for g in gpus:                                       # every GPU redundantly updates
        adam_update(g.store['master'], g.store['m'], g.store['v'], reduced, t)
        g.store['param'][...] = g.store['master'].astype(FP16)
    return sum(sse) / total


def step_z1(gpus, model, shards, batches, t, total):
    fwd = [g.store['param'] for g in gpus]
    sse, grads = compute_grads(gpus, model, fwd, fwd, batches)
    for i, g in enumerate(gpus):
        g.store['grad'][...] = grads[i]                  # full gradient resident (2*Psi)
    gshard = reduce_scatter(grads, shards, gpus, EB)     # Psi; each GPU gets its shard
    for i, g in enumerate(gpus):                         # update owned optimizer shard
        adam_update(g.store['master'], g.store['m'], g.store['v'], gshard[i], t)
    new_slices = [g.store['master'].astype(FP16) for g in gpus]
    full = all_gather(new_slices, gpus, EB)              # Psi; refresh full params
    for g in gpus:
        g.store['param'][...] = full
    return sum(sse) / total


def step_z2(gpus, model, shards, batches, t, total):
    fwd = [g.store['param'] for g in gpus]
    sse, grads = compute_grads(gpus, model, fwd, fwd, batches)
    for g in gpus:
        g.touch_transient(model.P * EB)                  # full grad exists transiently
    gshard = reduce_scatter(grads, shards, gpus, EB)     # Psi; keep only the shard
    for i, g in enumerate(gpus):
        g.store['grad'][...] = gshard[i].astype(FP16)    # gradient shard resident
        adam_update(g.store['master'], g.store['m'], g.store['v'], gshard[i], t)
    new_slices = [g.store['master'].astype(FP16) for g in gpus]
    full = all_gather(new_slices, gpus, EB)              # Psi; refresh full params
    for g in gpus:
        g.store['param'][...] = full
    return sum(sse) / total


def step_z3(gpus, model, shards, batches, t, total):
    # Parameters are sharded, so the full weights exist only transiently. The real
    # ZeRO-3 order: gather -> forward -> FREE -> gather -> backward -> FREE, so only one
    # full copy is ever alive (hence a peak of one full buffer above the resident shards).
    pshards = [g.store['param'] for g in gpus]

    params = all_gather(pshards, gpus, EB)               # Psi  (gather for FORWARD)
    for g in gpus: g.touch_transient(model.P * EB)
    caches = run_forward(gpus, model, [params]*len(gpus), batches)
    del params                                           # free the full weights

    params = all_gather(pshards, gpus, EB)               # Psi  (gather AGAIN for BACKWARD)
    for g in gpus: g.touch_transient(model.P * EB)
    sse, grads = run_backward(gpus, model, [params]*len(gpus), caches, batches)
    del params                                           # free again

    for g in gpus: g.touch_transient(model.P * EB)       # full grad transient
    gshard = reduce_scatter(grads, shards, gpus, EB)     # Psi
    for i, g in enumerate(gpus):
        g.store['grad'][...] = gshard[i].astype(FP16)
        adam_update(g.store['master'], g.store['m'], g.store['v'], gshard[i], t)
        g.store['param'][...] = g.store['master'].astype(FP16)   # own param shard
    return sum(sse) / total


STEP_FN = {0: step_z0, 1: step_z1, 2: step_z2, 3: step_z3}


def make_batches(X, Y, n):
    idx = np.array_split(np.arange(X.shape[0]), n)
    return [(X[i], Y[i]) for i in idx]


def train(stage, model, X, Y, n_gpus, steps):
    """Full run for one stage. Returns (loss_hist, reconstructed_full_master, gpus)."""
    shards = make_shards(model.P, n_gpus)
    gpus = init_gpus(stage, model, shards)
    batches = make_batches(X, Y, n_gpus)
    step = STEP_FN[stage]
    hist = []
    for t in range(1, steps + 1):
        for g in gpus:
            g.reset_comm()                               # keep only the latest step
        hist.append(step(gpus, model, shards, batches, t, X.shape[0]))
    return np.array(hist), full_master(stage, gpus), gpus
