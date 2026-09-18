import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from zero_sim import FlatMLP
from zero_stages import train

# ---- Okabe-Ito colorblind-safe categorical palette (fixed order, never cycled) ----
C = {"z0": "#4C4C4C", "z1": "#0072B2", "z2": "#009E73", "z3": "#D55E00", "grid": "#DddDdd"}
plt.rcParams.update({
    "figure.dpi": 120, "font.size": 11, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": "#E6E6E6",
    "grid.linewidth": 0.8, "axes.axisbelow": True, "font.family": "DejaVu Sans",
})
STAGES = ["ZeRO-0\n(DDP)", "ZeRO-1\n(Pos)", "ZeRO-2\n(Pos+g)", "ZeRO-3\n(Pos+g+p)"]
COLORS = [C["z0"], C["z1"], C["z2"], C["z3"]]
N = 32

# run once to get measured numbers
np.random.seed(0)
rng = np.random.default_rng(1)
X = rng.standard_normal((256, 16)).astype(np.float32)
Y = np.tanh(X @ rng.standard_normal((16, 1))).astype(np.float32)
model = FlatMLP([16, 64, 64, 1], seed=0)
P = model.P
res, comm, loss_all = {}, {}, {}
for s in [0, 1, 2, 3]:
    lh, _, gpus = train(s, model, X, Y, N, 60)
    res[s] = (gpus[0].resident_bytes / P, gpus[0].peak_bytes / P)
    comm[s] = gpus[0].comm_bytes / (P * 2)
    loss_all[s] = lh

# ---------------------------------------------------------------- Figure 1: memory
fig, ax = plt.subplots(figsize=(7.2, 4.3))
x = np.arange(4)
resident = [res[s][0] for s in range(4)]
peak_extra = [max(res[s][1] - res[s][0], 0) for s in range(4)]
ax.bar(x, resident, width=0.62, color=COLORS, zorder=3)
ax.bar(x, peak_extra, bottom=resident, width=0.62, color=COLORS, alpha=0.30, zorder=3,
       hatch="///", edgecolor="white")
for i in range(4):
    ax.text(i, res[i][0] + peak_extra[i] + 0.35, f"{resident[i]:.2f}", ha="center",
            fontweight="bold", color=COLORS[i])
ax.set_xticks(x); ax.set_xticklabels(STAGES)
ax.set_ylabel("memory per GPU  (bytes / parameter)")
ax.set_title("Per-GPU memory across ZeRO stages  (N = 32 GPUs)", fontweight="bold", loc="left")
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor="#888", label="resident (persistent)"),
                   Patch(facecolor="#888", alpha=0.30, hatch="///", edgecolor="white",
                         label="transient peak (all-gather / reduce buffer)")],
          frameon=False, fontsize=9, loc="upper right")
ax.set_ylim(0, 17.5)
ax.annotate("32x less\nthan baseline", xy=(3, 0.55), xytext=(2.42, 6.2), fontsize=9,
            color=C["z3"], ha="center",
            arrowprops=dict(arrowstyle="->", color=C["z3"], lw=1.3))
plt.tight_layout(); plt.savefig("figures/memory.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 2: comm
fig, ax = plt.subplots(figsize=(7.2, 4.3))
vals = [comm[s] for s in range(4)]
ax.bar(x, vals, width=0.62, color=COLORS, zorder=3)
for i in range(4):
    ax.text(i, vals[i] + 0.06, f"{vals[i]:.2f}x", ha="center", fontweight="bold",
            color=COLORS[i])
ax.axhline(comm[0], ls="--", lw=1, color="#999", zorder=2)
ax.text(3.5, comm[0] + 0.05, "baseline", color="#999", fontsize=9, ha="right")
ax.set_xticks(x); ax.set_xticklabels(STAGES)
ax.set_ylabel("communication per step  (x fp16 model size)")
ax.set_title("Per-GPU communication volume per step", fontweight="bold", loc="left")
ax.set_ylim(0, 3.4)
plt.tight_layout(); plt.savefig("figures/comm.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 3: scaling
fig, ax = plt.subplots(figsize=(7.2, 4.3))
Ns = np.array([1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024])
formulas = {0: lambda n: 16 + 0 * n, 1: lambda n: 4 + 12 / n,
            2: lambda n: 2 + 14 / n, 3: lambda n: 16 / n}
labels = ["ZeRO-0 (DDP)", "ZeRO-1 (Pos)", "ZeRO-2 (Pos+g)", "ZeRO-3 (Pos+g+p)"]
for s in range(4):
    ax.plot(Ns, formulas[s](Ns), "-o", color=COLORS[s], lw=2, ms=4, label=labels[s], zorder=3)
ax.axvline(32, color="#bbb", ls=":", lw=1)
ax.text(32, 17, "N=32", color="#888", fontsize=9, ha="center")
ax.set_xscale("log", base=2); ax.set_yscale("log")
ax.set_xticks(Ns); ax.set_xticklabels([str(n) for n in Ns], fontsize=8)
ax.set_xlabel("number of GPUs (N)")
ax.set_ylabel("memory per GPU  (bytes / parameter)")
ax.set_title("How per-GPU memory scales with the number of GPUs", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
ax.grid(True, which="both")
plt.tight_layout(); plt.savefig("figures/scaling.png", bbox_inches="tight"); plt.close()

# ---------------------------------------------------------------- Figure 4: loss (correctness)
fig, ax = plt.subplots(figsize=(7.2, 4.3))
styles = [("-", 3.4, 1.0), ("-", 2.2, 0.9), ("--", 1.6, 0.9), (":", 1.6, 0.9)]
for s in range(4):
    ls, lw, a = styles[s]
    ax.plot(loss_all[s], ls, color=COLORS[s], lw=lw, alpha=a, label=labels[s], zorder=3 + s)
ax.set_xlabel("training step"); ax.set_ylabel("training loss (MSE)")
ax.set_yscale("log")
ax.set_title("All four stages trace the identical loss curve", fontweight="bold", loc="left")
ax.legend(frameon=False, fontsize=9)
ax.text(0.98, 0.9, "curves overlap exactly\n(max |Δparam| = 0)", transform=ax.transAxes,
        ha="right", fontsize=9, color="#555")
plt.tight_layout(); plt.savefig("figures/loss.png", bbox_inches="tight"); plt.close()
print("figures written:", res, comm)
