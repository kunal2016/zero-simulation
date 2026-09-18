"""
Export the from-scratch simulation's MEASURED numbers to widget/data/measurements.json,
so the interactive widget can display what the simulation actually produced (real
ring-factor communication, real transient peaks, real shard rounding) instead of only
the idealized formula. The same export is reproduced by a cell in the notebook.
"""
import json, os
import numpy as np
from zero_sim import FlatMLP
from zero_stages import train

def export(path="widget/data/measurements.json"):
    rng = np.random.default_rng(1)
    M = 1024                                   # >= max N so every GPU gets a sample
    X = rng.standard_normal((M, 16)).astype(np.float32)
    Y = np.tanh(X @ rng.standard_normal((16, 1))).astype(np.float32)
    model = FlatMLP([16, 64, 64, 1], seed=0)   # P = 5313, mixed-precision Adam (K=12, 2B)
    P = model.P
    Ns = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]

    stages = {k: {} for k in ["z0", "z1", "z2", "z3"]}
    for N in Ns:
        for si, key in enumerate(["z0", "z1", "z2", "z3"]):
            _, _, gpus = train(si, model, X, Y, N, 3)
            g = gpus[0]
            param_b = g.store["param_fp16"].nbytes / P
            grad_b = g.store["grad_fp16"].nbytes / P
            opt_b = (g.store["master_fp32"].nbytes + g.store["m_fp32"].nbytes
                     + g.store["v_fp32"].nbytes) / P
            stages[key][str(N)] = {
                "param_bpp": round(param_b, 5),                   # bytes / parameter
                "grad_bpp":  round(grad_b, 5),
                "opt_bpp":   round(opt_b, 5),
                "resident_bpp": round(g.resident_bytes / P, 5),
                "peak_bpp":     round(g.peak_bytes / P, 5),
                "comm":         round(g.comm_bytes / (P * 2), 5),  # x fp16 model size
            }

    data = {
        "source": "from-scratch NumPy simulation (zero_from_scratch.ipynb / test_sim.py)",
        "model_P": P,
        "config": {"optimizer": "Adam", "optimizer_K": 12, "precision_bytes": 2,
                   "gpus_measured": Ns},
        "stages": stages,
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    return data

if __name__ == "__main__":
    d = export()
    print("wrote widget/data/measurements.json  (model P =", d["model_P"], ")")
    print("N=32 measured resident bytes/param:",
          {k: d["stages"][k]["32"]["resident_bpp"] for k in d["stages"]})
    print("N=32 measured comm (xΨ):",
          {k: d["stages"][k]["32"]["comm"] for k in d["stages"]})
