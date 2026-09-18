"""
Per-stage training steps for ZeRO-0/1/2/3 built on zero_sim.py.

Design contract that makes the demo trustworthy:
  * The NUMERIC update is identical for every stage (same fp16 casting, same
    fixed-order fp32 gradient accumulation, same elementwise Adam). Therefore
    ZeRO-1/2/3 must reproduce the ZeRO-0 (baseline DDP) parameter trajectory
    bit-for-bit -- they differ ONLY in where data lives and how much is moved.
  * Each GPU's resident_bytes is set by build_layout() to the arrays that stage
    actually keeps, so per-device memory equals the sum of real `.nbytes`.
  * Communication is charged with the ring-algorithm cost so the per-step volume
    reproduces the paper's 2*Psi / 2*Psi / 2*Psi / 3*Psi table.
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor
from zero_sim import VirtualGPU, FP16, FP32, make_shards, AdamShard, _ring_factor

STAGE_NAMES = {0: "ZeRO-0 (baseline DDP)", 1: "ZeRO-1 (Pos)",
               2: "ZeRO-2 (Pos+g)", 3: "ZeRO-3 (Pos+g+p)"}


def build_layout(stage, gpus, P, shards):
    """Materialize each GPU's PERSISTENT tensors so resident_bytes is byte-honest."""
    for i, g in enumerate(gpus):
        for name in list(g.store):     # clear
            g.free(name)
        s = shards[i]
        ss = s.stop - s.start
        opt = ss                        # optimizer states are sharded in 1/2/3
        if stage == 0:                  # everything replicated
            g.put('param_fp16', np.zeros(P, FP16))
            g.put('grad_fp16',  np.zeros(P, FP16))
            g.put('master_fp32', np.zeros(P, FP32))
            g.put('m_fp32', np.zeros(P, FP32))
            g.put('v_fp32', np.zeros(P, FP32))
        elif stage == 1:                # shard optimizer states only
            g.put('param_fp16', np.zeros(P, FP16))
            g.put('grad_fp16',  np.zeros(P, FP16))
            g.put('master_fp32', np.zeros(opt, FP32))
            g.put('m_fp32', np.zeros(opt, FP32))
            g.put('v_fp32', np.zeros(opt, FP32))
        elif stage == 2:                # + shard gradients
            g.put('param_fp16', np.zeros(P, FP16))
            g.put('grad_fp16',  np.zeros(ss, FP16))
            g.put('master_fp32', np.zeros(opt, FP32))
            g.put('m_fp32', np.zeros(opt, FP32))
            g.put('v_fp32', np.zeros(opt, FP32))
        elif stage == 3:                # + shard parameters -> everything sharded
            g.put('param_fp16', np.zeros(ss, FP16))
            g.put('grad_fp16',  np.zeros(ss, FP16))
            g.put('master_fp32', np.zeros(opt, FP32))
            g.put('m_fp32', np.zeros(opt, FP32))
            g.put('v_fp32', np.zeros(opt, FP32))


def _accum_fp32(grad_fp16_list):
    """Sum per-GPU fp16 gradients in fixed rank order, accumulating in fp32."""
    total = np.zeros_like(grad_fp16_list[0], dtype=FP32)
    for g in grad_fp16_list:
        total = total + g.astype(FP32)
    return total


def init_state(model, shards):
    return {
        'master': model.init_flat.copy(),                        # authoritative fp32
        'adam': [AdamShard(s.stop - s.start) for s in shards],   # sharded optimizer
    }


def run_step(stage, gpus, model, shards, state, batches, total_samples):
    n = len(gpus)
    P = model.P
    ELEM = 2                                  # fp16 bytes -> Psi is measured in fp16 bytes
    Psi = P * ELEM

    params_fp16 = state['master'].astype(FP16)

    # ZeRO-3 keeps only a param shard resident; reconstruct the full vector to compute.
    if stage == 3:
        param_shards = [params_fp16[s].copy() for s in shards]
        for g in gpus:                        # forward all-gather of params  (Psi)
            g.add_comm(_ring_factor(n) * P * ELEM)
            g.touch_transient(P * ELEM)       # transient full-param buffer
        for g in gpus:                        # backward re-gather of params  (Psi)
            g.add_comm(_ring_factor(n) * P * ELEM)
        compute_params = np.concatenate(param_shards)
    else:
        compute_params = params_fp16

    # Each virtual GPU computes its micro-batch gradient -- run on CPU threads.
    def work(i):
        Xi, Yi = batches[i]
        sse, grad = model.forward_backward(compute_params, Xi, Yi)
        return sse, grad.astype(FP16)         # gradients live in fp16 (2 bytes)

    with ThreadPoolExecutor(max_workers=min(n, 8)) as ex:
        results = list(ex.map(work, range(n)))
    sse_list = [r[0] for r in results]
    grad_fp16 = [r[1] for r in results]

    # Gradient reduction + communication charge (per stage)
    if stage == 0:                            # all-reduce full gradients        (2*Psi)
        for g in gpus:
            g.add_comm(2 * _ring_factor(n) * P * ELEM)
    elif stage == 1:                          # reduce-scatter grads             (Psi)
        for g in gpus:
            g.add_comm(_ring_factor(n) * P * ELEM)
    else:                                     # stage 2 & 3: grads sharded
        for g in gpus:
            g.touch_transient(P * ELEM)       # transient full grad during backward
            g.add_comm(_ring_factor(n) * P * ELEM)   # reduce-scatter grads      (Psi)

    reduced_full = _accum_fp32(grad_fp16)     # identical for every stage
    reduced_slices = [reduced_full[s] for s in shards]

    # Sharded Adam step: GPU i updates only its slice of the master weights.
    for i, s in enumerate(shards):
        state['adam'][i].step(state['master'][s], reduced_slices[i])

    # Parameter refresh after the update (all-gather).  Stage 0 needs none;
    # stage 3 already paid 2*Psi of param gather above.
    if stage in (1, 2):                       # all-gather updated params        (Psi)
        for g in gpus:
            g.add_comm(_ring_factor(n) * P * ELEM)

    return sum(sse_list) / total_samples


def make_batches(X, Y, n):
    """Split the global batch into n micro-batches, one per virtual GPU."""
    idx = np.array_split(np.arange(X.shape[0]), n)
    return [(X[i], Y[i]) for i in idx]


def train(stage, model, X, Y, n_gpus, steps):
    """Full training run for one ZeRO stage. Returns (loss_hist, final_master, gpus)."""
    shards = make_shards(model.P, n_gpus)
    gpus = [VirtualGPU(i) for i in range(n_gpus)]
    build_layout(stage, gpus, model.P, shards)
    state = init_state(model, shards)
    batches = make_batches(X, Y, n_gpus)
    loss_hist = []
    for step in range(steps):
        for g in gpus:
            g.reset_comm()                    # keep only the most recent step's volume
        loss = run_step(stage, gpus, model, shards, state, batches, X.shape[0])
        loss_hist.append(loss)
    return np.array(loss_hist), state['master'].copy(), gpus
