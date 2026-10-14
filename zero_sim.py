"""
ZeRO from scratch: a byte-honest NumPy simulation of ZeRO-0/1/2/3 across N virtual GPUs.

Each VirtualGPU genuinely *owns* the arrays it is responsible for -- its parameter
copy, its gradient, and its slice of the Adam optimizer state -- and runs its own
optimizer update on the data it holds. ZeRO-1/2/3 differ from the baseline precisely
because the arrays each GPU stores differ, so the per-device memory a stage reports is
the sum of `.nbytes` of the arrays that device actually holds.

Honest scope note: the *training state* is owned per GPU, but the *collectives*
(`all_reduce` / `reduce_scatter` / `all_gather`) are **simulated centrally** -- one
Python process computes the sum or concatenation and hands the result back -- and then
charged at ring-algorithm cost. This is a single-machine model of the communication,
not a real networked transfer; what it faithfully reproduces is the memory layout and
the communication *volume*, not bandwidth or latency.

Precision layout (mixed-precision Adam), per parameter Psi:
    fp16 parameter copy        2 bytes   (used for forward/backward compute)
    fp16 gradient              2 bytes
    fp32 master parameter      4 bytes   |
    fp32 Adam momentum (m)     4 bytes   |  optimizer states = 12 bytes = K*Psi, K=12
    fp32 Adam variance  (v)    4 bytes   |
    ---------------------------------------
    total                     16 bytes  (baseline / ZeRO-0, replicated on every GPU)
"""

import numpy as np
from concurrent.futures import ThreadPoolExecutor

FP16 = np.float16   # 2 bytes/elem
FP32 = np.float32   # 4 bytes/elem


# --------------------------------------------------------------------------------------
# A virtual GPU: owns real arrays + a byte-accurate ledger of what it stores
# --------------------------------------------------------------------------------------
class VirtualGPU:
    def __init__(self, rank):
        self.rank = rank
        self.store = {}          # name -> ndarray  (the REAL data this GPU owns)
        self.resident_bytes = 0  # bytes currently held
        self.peak_bytes = 0      # high-water mark incl. transient buffers
        self.comm_bytes = 0      # bytes moved in/out of this GPU per measured step

    def put(self, name, arr):
        if name in self.store:
            self.resident_bytes -= self.store[name].nbytes
        self.store[name] = arr
        self.resident_bytes += arr.nbytes
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes)

    def free(self, name):
        if name in self.store:
            self.resident_bytes -= self.store.pop(name).nbytes

    def touch_transient(self, nbytes):
        """Account for a temporary buffer (an all-gather / reduce-scatter staging area)."""
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes + nbytes)

    def reset_comm(self):
        self.comm_bytes = 0

    def add_comm(self, nbytes):
        self.comm_bytes += nbytes


# --------------------------------------------------------------------------------------
# Collective communication primitives -- they really move data between GPU-owned arrays
# --------------------------------------------------------------------------------------
# Ring-algorithm cost, PER rank, for a message of M elements across N ranks:
#   all_reduce      : 2 * (N-1)/N * M
#   reduce_scatter  :     (N-1)/N * M
#   all_gather      :     (N-1)/N * M
# The reduction is accumulated in fp32 in fixed rank order -> deterministic, so every
# stage gets bit-identical reduced values (this is what lets us prove exact equality).

def _ring(n):
    return (n - 1) / n


def all_reduce(gpus, name, elem_bytes):
    """Sum `name` across all GPUs; leave the full fp32 sum resident on every GPU's
    `name` buffer (cast back to that buffer's dtype). Returns the fp32 sum."""
    total = np.zeros_like(gpus[0].store[name], dtype=FP32)
    for g in gpus:                                  # fixed rank order
        total = total + g.store[name].astype(FP32)
    m = total.size
    for g in gpus:
        g.add_comm(2 * _ring(len(gpus)) * m * elem_bytes)
        g.store[name][...] = total.astype(g.store[name].dtype)
    return total


def reduce_scatter(full_arrays, shards, gpus, elem_bytes):
    """Sum per-GPU full arrays, then hand each GPU only its shard slice (fp32).
    `full_arrays` is one full-length array per GPU (e.g. the just-computed gradient)."""
    total = np.zeros_like(full_arrays[0], dtype=FP32)
    for a in full_arrays:                           # fixed rank order
        total = total + a.astype(FP32)
    m = total.size
    for g in gpus:
        g.add_comm(_ring(len(gpus)) * m * elem_bytes)
    return [total[s] for s in shards]               # one fp32 shard per GPU


def all_gather(shard_arrays, gpus, elem_bytes):
    """Concatenate each GPU's shard into the full vector; every GPU receives it."""
    full = np.concatenate(shard_arrays)
    m = full.size
    for g in gpus:
        g.add_comm(_ring(len(gpus)) * m * elem_bytes)
    return full


def make_shards(P, n):
    """Contiguous, load-balanced shard slices of a length-P flat vector over n GPUs."""
    sizes = [P // n + (1 if i < P % n else 0) for i in range(n)]
    bounds, start = [], 0
    for s in sizes:
        bounds.append(slice(start, start + s))
        start += s
    return bounds


# --------------------------------------------------------------------------------------
# Demo model: a flat-parameter MLP with SEPARATE forward and backward
# (separate so ZeRO-3 can gather params for the forward, free them, and gather AGAIN
#  for the backward -- two real collective operations, as the real algorithm does.)
# --------------------------------------------------------------------------------------
class FlatMLP:
    def __init__(self, sizes, seed=0):
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.shapes = []
        params = []
        for a, b in zip(sizes[:-1], sizes[1:]):
            W = (rng.standard_normal((a, b)) * np.sqrt(2.0 / a)).astype(FP32)
            bvec = np.zeros(b, dtype=FP32)
            params += [W, bvec]
            self.shapes += [W.shape, bvec.shape]
        self.P = int(sum(int(np.prod(s)) for s in self.shapes))
        self.init_flat = self._pack(params)

    def _pack(self, arrays):
        return np.concatenate([a.ravel() for a in arrays]).astype(FP32)

    def _unpack(self, flat):
        out, i = [], 0
        for s in self.shapes:
            n = int(np.prod(s))
            out.append(flat[i:i + n].reshape(s))
            i += n
        return out

    def forward(self, flat_params, X):
        """Run the forward pass; return a cache the backward pass will consume."""
        p = self._unpack(flat_params.astype(FP32))
        Ws, bs = p[0::2], p[1::2]
        acts = [X.astype(FP32)]
        pre, h, L = [], X.astype(FP32), len(Ws)
        for l in range(L):
            z = h @ Ws[l] + bs[l]
            pre.append(z)
            h = np.maximum(z, 0.0) if l < L - 1 else z   # ReLU except last layer
            acts.append(h)
        return {"acts": acts, "pre": pre, "pred": acts[-1]}

    def backward(self, flat_params, cache, Y):
        """Backward pass -- genuinely needs the parameters (to propagate deltas), so
        ZeRO-3 must have re-gathered them. Returns (sum_of_squared_error, flat_grad)."""
        p = self._unpack(flat_params.astype(FP32))
        Ws = p[0::2]
        acts, pre = cache["acts"], cache["pre"]
        L = len(Ws)
        diff = acts[-1] - Y.astype(FP32)
        sse = float(np.sum(diff ** 2))
        gW, gb, delta = [None] * L, [None] * L, 2.0 * diff
        for l in reversed(range(L)):
            gW[l] = acts[l].T @ delta
            gb[l] = delta.sum(axis=0)
            if l > 0:
                delta = (delta @ Ws[l].T) * (pre[l - 1] > 0)
        packed = []
        for l in range(L):
            packed += [gW[l], gb[l]]
        return sse, self._pack(packed)

    def forward_backward(self, flat_params, X, Y):
        """Convenience wrapper (one param copy for both passes)."""
        cache = self.forward(flat_params, X)
        return self.backward(flat_params, cache, Y)


# --------------------------------------------------------------------------------------
# Adam update -- a pure, in-place step that each GPU runs on the slice IT owns.
# Elementwise, so a per-shard update equals the global update restricted to that slice.
# --------------------------------------------------------------------------------------
def adam_update(master, m, v, grad_fp32, t, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    m[...] = b1 * m + (1 - b1) * grad_fp32
    v[...] = b2 * v + (1 - b2) * (grad_fp32 ** 2)
    mhat = m / (1 - b1 ** t)
    vhat = v / (1 - b2 ** t)
    master -= lr * mhat / (np.sqrt(vhat) + eps)


def run_forward(gpus, model, fwd_params, batches):
    """Forward phase only: each GPU runs its micro-batch forward on its own thread and
    keeps a cache for the backward. `fwd_params[i]` is GPU i's parameter copy."""
    def work(i):
        return model.forward(fwd_params[i], batches[i][0])
    with ThreadPoolExecutor(max_workers=min(len(gpus), 8)) as ex:
        return list(ex.map(work, range(len(gpus))))


def run_backward(gpus, model, bwd_params, caches, batches):
    """Backward phase only: each GPU runs its backward using the caches from the forward
    and `bwd_params[i]` (which ZeRO-3 has re-gathered). Returns (sse_list, grad_list)."""
    def work(i):
        sse, grad = model.backward(bwd_params[i], caches[i], batches[i][1])
        return sse, grad.astype(FP16)
    with ThreadPoolExecutor(max_workers=min(len(gpus), 8)) as ex:
        res = list(ex.map(work, range(len(gpus))))
    return [r[0] for r in res], [r[1] for r in res]


def compute_grads(gpus, model, fwd_params, bwd_params, batches):
    """Convenience: forward then backward (one param copy held throughout). Used by the
    stages that keep the full weights resident; ZeRO-3 instead calls the two phases
    separately so it can free the gathered weights between them."""
    caches = run_forward(gpus, model, fwd_params, batches)
    return run_backward(gpus, model, bwd_params, caches, batches)
