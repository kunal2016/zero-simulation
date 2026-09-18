"""
ZeRO from scratch: a byte-honest NumPy simulation of ZeRO-0/1/2/3 across N virtual GPUs.

The point of this module is *not* speed. It is to make the memory that each ZeRO
stage keeps resident on a device literally equal to the sum of `.nbytes` of the
arrays that device stores -- so the famous 16-byte-per-parameter accounting from
the ZeRO paper falls out of real allocations instead of a formula.

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
# Memory + communication ledgers
# --------------------------------------------------------------------------------------
class VirtualGPU:
    """One simulated GPU. Holds arrays and a byte-accurate ledger of what it stores."""

    def __init__(self, rank):
        self.rank = rank
        self.store = {}          # name -> ndarray (the real data)
        self.resident_bytes = 0  # bytes currently held
        self.peak_bytes = 0      # high-water mark incl. transient buffers
        self.comm_bytes = 0      # bytes moved in/out of this GPU per measured window

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
        """Account for a temporary buffer (e.g. an all-gather reconstruction)."""
        self.peak_bytes = max(self.peak_bytes, self.resident_bytes + nbytes)

    def reset_comm(self):
        self.comm_bytes = 0

    def add_comm(self, nbytes):
        self.comm_bytes += nbytes


# --------------------------------------------------------------------------------------
# Collective communication primitives (with ring-algorithm cost accounting)
# --------------------------------------------------------------------------------------
# For a message of M elements across N ranks, the ring algorithm moves, PER rank:
#   all_reduce      : 2 * (N-1)/N * M   (reduce-scatter phase + all-gather phase)
#   reduce_scatter  :     (N-1)/N * M
#   all_gather      :     (N-1)/N * M
# We accumulate the reduction in fp32 for determinism (identical across all stages),
# but the *stored* gradient tensors keep their fp16 footprint.

def _ring_factor(n):
    return (n - 1) / n

def all_reduce_sum(gpus, name, elem_bytes):
    """Every GPU ends with the fp32 sum of `name` across all GPUs. Returns the sum."""
    n = len(gpus)
    total = np.zeros_like(gpus[0].store[name], dtype=FP32)
    for g in gpus:                       # fixed rank order -> deterministic
        total = total + gpus[g.rank].store[name].astype(FP32)
    m = total.size
    for g in gpus:
        g.add_comm(2 * _ring_factor(n) * m * elem_bytes)
    return total

def reduce_scatter_sum(gpus, name, shards, elem_bytes):
    """GPU s ends with the fp32 sum over GPUs of `name` restricted to its shard slice."""
    n = len(gpus)
    total = np.zeros_like(gpus[0].store[name], dtype=FP32)
    for g in gpus:
        total = total + gpus[g.rank].store[name].astype(FP32)
    m = total.size
    for g in gpus:
        g.add_comm(_ring_factor(n) * m * elem_bytes)
    return [total[s] for s in shards]    # one slice per GPU

def all_gather(gpus, shard_arrays, elem_bytes):
    """Concatenate per-GPU shards into the full vector; every GPU 'receives' it."""
    n = len(gpus)
    full = np.concatenate(shard_arrays)
    m = full.size
    for g in gpus:
        g.add_comm(_ring_factor(n) * m * elem_bytes)
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
# Demo model: a flat-parameter MLP for regression (real forward/backward)
# --------------------------------------------------------------------------------------
class FlatMLP:
    """MLP whose parameters live in ONE flat vector, so ZeRO can shard them trivially."""

    def __init__(self, sizes, seed=0):
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.shapes, self.P = [], 0
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

    def forward_backward(self, flat_params, X, Y):
        """Return (sum_of_squared_error, flat_gradient) for one micro-batch.
        Compute is done in fp32 on the fp16->fp32 upcast params (as real AMP does).
        Gradient is the SUM over samples (not mean) so per-GPU grads add up to the
        full-batch gradient exactly."""
        p = self._unpack(flat_params.astype(FP32))
        Ws = p[0::2]; bs = p[1::2]
        acts = [X.astype(FP32)]
        pre = []
        h = acts[0]
        L = len(Ws)
        for l in range(L):
            z = h @ Ws[l] + bs[l]
            pre.append(z)
            h = np.maximum(z, 0.0) if l < L - 1 else z   # ReLU except last layer
            acts.append(h)
        pred = acts[-1]
        diff = pred - Y.astype(FP32)
        sse = float(np.sum(diff ** 2))
        # backward (gradient of sum-of-squared-error)
        grads_W = [None] * L; grads_b = [None] * L
        delta = 2.0 * diff
        for l in reversed(range(L)):
            grads_W[l] = acts[l].T @ delta
            grads_b[l] = delta.sum(axis=0)
            if l > 0:
                dh = delta @ Ws[l].T
                delta = dh * (pre[l - 1] > 0)
        packed = []
        for l in range(L):
            packed += [grads_W[l], grads_b[l]]
        return sse, self._pack(packed)


# --------------------------------------------------------------------------------------
# Adam (operates elementwise on fp32 master params) -- identical math for every stage
# --------------------------------------------------------------------------------------
class AdamShard:
    """Adam state for ONE shard slice. Elementwise, so per-shard == global-on-slice."""
    def __init__(self, size, lr=1e-2, betas=(0.9, 0.999), eps=1e-8):
        self.lr, (self.b1, self.b2), self.eps = lr, betas, eps
        self.t = 0
        self.m = np.zeros(size, FP32)
        self.v = np.zeros(size, FP32)

    def step(self, master, grad_fp32):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad_fp32
        self.v = self.b2 * self.v + (1 - self.b2) * (grad_fp32 ** 2)
        mhat = self.m / (1 - self.b1 ** self.t)
        vhat = self.v / (1 - self.b2 ** self.t)
        master -= self.lr * mhat / (np.sqrt(vhat) + self.eps)
        return master
