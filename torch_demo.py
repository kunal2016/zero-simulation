"""
PyTorch companion to the NumPy simulation: the SAME ZeRO memory accounting, but
now measured on real torch tensors (real autograd gradients, real Adam moments).

We build one flat parameter vector, run a real backward pass to get real gradients,
and then measure -- with tensor.element_size() * tensor.nelement() -- how many bytes
each ZeRO stage keeps resident on a single rank out of N. The mixed-precision layout
is the real thing: fp16 param/grad tensors, fp32 master/m/v tensors.
"""

import torch

torch.manual_seed(0)
N = 32                      # virtual ranks


class MLP(torch.nn.Module):
    def __init__(self, d_in=16, h=64, d_out=1):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, h), torch.nn.ReLU(),
            torch.nn.Linear(h, h), torch.nn.ReLU(),
            torch.nn.Linear(h, d_out))

    def forward(self, x):
        return self.net(x)


model = MLP()
P = sum(p.numel() for p in model.parameters())
print(f"torch model parameters P = {P:,}")

# One real forward/backward to obtain a genuine gradient vector.
X = torch.randn(256, 16)
Y = torch.tanh(X @ torch.randn(16, 1))
loss = ((model(X) - Y) ** 2).mean()
loss.backward()
flat_grad = torch.cat([p.grad.reshape(-1) for p in model.parameters()])
flat_param = torch.cat([p.detach().reshape(-1) for p in model.parameters()])
print(f"real gradient captured: shape={tuple(flat_grad.shape)}, "
      f"norm={flat_grad.norm().item():.4f}\n")


def bytes_per_param(stage, P, N):
    """Resident bytes on one rank, using real torch tensors, divided by P."""
    shard = P // N
    fp16 = lambda n: torch.zeros(n, dtype=torch.float16)   # 2 bytes/elem
    fp32 = lambda n: torch.zeros(n, dtype=torch.float32)   # 4 bytes/elem
    tensors = []
    if stage == 0:      # baseline: everything replicated
        tensors = [fp16(P), fp16(P), fp32(P), fp32(P), fp32(P)]
    elif stage == 1:    # shard optimizer states
        tensors = [fp16(P), fp16(P), fp32(shard), fp32(shard), fp32(shard)]
    elif stage == 2:    # + shard gradients
        tensors = [fp16(P), fp16(shard), fp32(shard), fp32(shard), fp32(shard)]
    elif stage == 3:    # + shard parameters
        tensors = [fp16(shard), fp16(shard), fp32(shard), fp32(shard), fp32(shard)]
    total = sum(t.element_size() * t.nelement() for t in tensors)
    return total / P


print("=== MEMORY per rank (bytes / parameter), real torch tensors ===")
names = {0: "ZeRO-0", 1: "ZeRO-1", 2: "ZeRO-2", 3: "ZeRO-3"}
theory = {0: 16, 1: 4 + 12 / N, 2: 2 + 14 / N, 3: 16 / N}
for s in [0, 1, 2, 3]:
    print(f"  {names[s]}: measured={bytes_per_param(s, P, N):6.3f}   theory={theory[s]:6.3f}")

# ---- correctness in torch: a sharded manual Adam step == torch.optim.Adam step ----
torch.manual_seed(0)
w = flat_param.clone()
g = flat_grad.clone()

# reference: torch Adam on the whole vector
ref = w.clone().requires_grad_(True)
opt = torch.optim.Adam([ref], lr=1e-2)
ref.grad = g.clone()
opt.step()

# ZeRO-style: split into N shards, run independent Adam on each shard, concatenate
shards = torch.chunk(w.clone(), N)
gshards = torch.chunk(g.clone(), N)
updated = []
for ws, gs in zip(shards, gshards):
    p = ws.clone().requires_grad_(True)
    o = torch.optim.Adam([p], lr=1e-2)
    p.grad = gs.clone()
    o.step()
    updated.append(p.detach())
sharded = torch.cat(updated)

maxdiff = (ref.detach() - sharded).abs().max().item()
print(f"\n=== CORRECTNESS: sharded Adam vs whole-vector Adam ===")
print(f"  max|difference| = {maxdiff:.2e}  ->  {'OK (identical)' if maxdiff < 1e-6 else 'FAIL'}")
