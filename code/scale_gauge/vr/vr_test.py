"""Test 1: is van Rossum's 'twirling degrades
expressivity' (Fig. 2a) a readout-scale effect?  CPU, float64.
Arm (i)  raw: f = <Z> (i-a, centred target) and f = <Z> + b (i-b, trainable offset == free target
         offset b of their Eq. 7); binary search for the largest target range converging to full-data
         MSE < 1e-5 within 150 Adam steps.
Arm (ii) rescaled: f = s<Z> + b (s, b trainable) at fixed target ranges; convergence within 150 / 1000.
Targets are batched: Adam is elementwise and each target's loss depends only on its own parameters,
so batching is exactly equivalent to separate runs."""
import sys, time, pickle, numpy as np, torch
from vr_model import channel, forward
torch.set_num_threads(4); torch.set_default_dtype(torch.float64)
L, LR, BETAS, BATCH, EPS = 2, 0.20, (0.45, 0.95), 25, 1e-5
K = int(sys.argv[1]) if len(sys.argv) > 1 else 200
X = torch.linspace(-2 * np.pi, 2 * np.pi, 250)
rng = np.random.default_rng(20260924)
coef = rng.uniform(-1, 1, (K, 4))                                  # a1, b1, a2, b2 (degree-2 truncated Fourier series)
xs = X.numpy()
G0 = coef[:, [0]] * np.cos(xs) + coef[:, [1]] * np.sin(xs) + coef[:, [2]] * np.cos(2 * xs) + coef[:, [3]] * np.sin(2 * xs)
G0 = (G0 - 0.5 * (G0.max(1, keepdims=True) + G0.min(1, keepdims=True))) / (G0.max(1, keepdims=True) - G0.min(1, keepdims=True))
G0 = torch.tensor(G0)                                              # unit range, zero mid-range
TH0 = torch.tensor(np.random.default_rng(777).uniform(0, 2 * np.pi, (K, 3 * (L + 1))))
PERMS = [torch.tensor(np.random.default_rng(1000 + e).permutation(250)) for e in range(200)]   # shared batch order

def train(R, T, t, mode, n_steps):
    """R (K,) target ranges. Returns first-convergence step (inf if none), full MSE at 150 and at end, s at end."""
    th = TH0.clone().requires_grad_(True)
    b = torch.zeros(K, requires_grad=True); s = torch.ones(K, requires_grad=True)
    params = [th] + ([b] if mode in ('raw_off', 'scaled') else []) + ([s] if mode == 'scaled' else [])
    opt = torch.optim.Adam(params, lr=LR, betas=BETAS)
    G = R[:, None] * G0
    f = lambda xx: (s[:, None] if mode == 'scaled' else 1.0) * forward(th, xx, T, t, L) + (b[:, None] if mode != 'raw' else 0.0)
    conv = torch.full((K,), float('inf')); mse150 = None
    for step in range(n_steps):
        idx = PERMS[step // 10][(step % 10) * BATCH:(step % 10 + 1) * BATCH]
        opt.zero_grad()
        ((f(X[idx]) - G[:, idx]) ** 2).mean(1).sum().backward()
        opt.step()
        with torch.no_grad():
            mse = ((f(X) - G) ** 2).mean(1)
            conv = torch.where((mse < EPS) & torch.isinf(conv), torch.tensor(float(step + 1)), conv)
            if step + 1 == 150: mse150 = mse.clone()
    return conv.numpy(), mse150.numpy(), mse.numpy(), s.detach().abs().numpy(), b.detach().numpy()

def max_range(T, t, mode, iters=10):
    lo, hi = torch.zeros(K), torch.full((K,), 2.0)
    for _ in range(iters):
        mid = (lo + hi) / 2
        ok = torch.tensor(train(mid, T, t, mode, 150)[0] <= 150)
        lo, hi = torch.where(ok, mid, lo), torch.where(ok, hi, mid)
    return lo.numpy()

if __name__ == '__main__':
    t0 = time.time(); out = {'K': K, 'coef': coef, 'theta0': TH0.numpy(), 'x': xs}
    CH, PS = ('None', 'AD', 'PauliAD', 'CliffAD'), (0.05, 0.1, 0.2)
    cfgs = [('None', 0.0)] + [(c, p) for p in PS for c in CH[1:]]
    for mode in ('raw', 'raw_off'):
        for c, p in cfgs:
            out[('range', mode, c, p)] = r = max_range(*channel(c, p), mode)
            print(f"arm(i) {mode:7s} {c:7s} p={p:.2f}: max range {r.mean():.4f} ± {r.std(ddof=1):.4f}  median {np.median(r):.4f}  ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open('vr_results.pkl', 'wb'))
    R_med = float(np.median(out[('range', 'raw_off', 'None', 0.0)]))
    out['R_fixed'] = R_fix = (1.0, R_med)
    for R in R_fix:
        for c, p in cfgs:
            conv, m150, mend, s_end, b_end = train(torch.full((K,), R), *channel(c, p), 'scaled', 1000)
            out[('scaled', R, c, p)] = dict(conv=conv, mse150=m150, mse1000=mend, s=s_end, b=b_end)
            print(f"arm(ii) R={R:.3f} {c:7s} p={p:.2f}: conv<=150 {np.mean(conv <= 150):.3f}  conv<=1000 {np.mean(conv <= 1000):.3f}  "
                  f"median MSE1000 {np.median(mend):.2e}  median |s| {np.median(s_end):.3f}  ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open('vr_results.pkl', 'wb'))
    print("VR_DONE", flush=True)
