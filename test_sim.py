import numpy as np
from zero_sim import FlatMLP, make_shards
from zero_stages import train, build_layout, init_state, STAGE_NAMES, VirtualGPU

np.random.seed(0)
N_GPUS = 32
STEPS = 40

# synthetic regression task
rng = np.random.default_rng(1)
D_IN, D_OUT, M = 16, 1, 32 * 8      # 256 samples -> 8 per GPU
X = rng.standard_normal((M, D_IN)).astype(np.float32)
true_w = rng.standard_normal((D_IN, D_OUT)).astype(np.float32)
Y = (np.tanh(X @ true_w) + 0.05 * rng.standard_normal((M, D_OUT))).astype(np.float32)

model = FlatMLP([D_IN, 64, 64, D_OUT], seed=0)
print(f"Model parameters P = {model.P:,}")
print(f"Virtual GPUs N = {N_GPUS}\n")

results = {}
for stage in [0, 1, 2, 3]:
    loss, master, gpus = train(stage, model, X, Y, N_GPUS, STEPS)
    results[stage] = (loss, master, gpus)

# ---- correctness: every stage must match the ZeRO-0 baseline trajectory ----
base_loss, base_master, _ = results[0]
print("=== CORRECTNESS (vs ZeRO-0 baseline) ===")
for stage in [1, 2, 3]:
    loss, master, _ = results[stage]
    dparam = np.max(np.abs(master - base_master))
    dloss = np.max(np.abs(loss - base_loss))
    ok = np.allclose(master, base_master, atol=1e-5) and np.allclose(loss, base_loss, atol=1e-5)
    print(f"  {STAGE_NAMES[stage]:24s}  max|dparam|={dparam:.2e}  max|dloss|={dloss:.2e}  {'OK' if ok else 'FAIL'}")
print(f"  final loss (all stages) = {base_loss[-1]:.6f}  (started {base_loss[0]:.6f})\n")

# ---- memory: per-GPU resident + peak, in BYTES PER PARAMETER (the ZeRO table unit) ----
P = model.P
print("=== MEMORY per GPU (bytes / parameter) ===")
print(f"{'stage':26s}{'resident':>10s}{'peak':>10s}{'theory':>10s}")
theory = {0: 16, 1: 4 + 12 / N_GPUS, 2: 2 + 14 / N_GPUS, 3: 16 / N_GPUS}
for stage in [0, 1, 2, 3]:
    _, _, gpus = results[stage]
    g = gpus[0]
    print(f"{STAGE_NAMES[stage]:26s}{g.resident_bytes / P:>10.3f}{g.peak_bytes / P:>10.3f}{theory[stage]:>10.3f}")

# ---- communication per step, per GPU, as multiple of the fp16 model size (Psi) ----
model_bytes = P * 2       # fp16 model = Psi
print("\n=== COMMUNICATION per step, per GPU (x fp16 model size) ===")
print(f"{'stage':26s}{'volume':>10s}{'theory':>10s}")
comm_theory = {0: 2, 1: 2, 2: 2, 3: 3}
for stage in [0, 1, 2, 3]:
    _, _, gpus = results[stage]
    g = gpus[0]
    print(f"{STAGE_NAMES[stage]:26s}{g.comm_bytes / model_bytes:>10.3f}{comm_theory[stage]:>10.3f}")
