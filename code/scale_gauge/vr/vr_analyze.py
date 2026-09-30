import os, pickle, numpy as np
from scipy import stats
D = pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'vr_results.pkl'), 'rb'))
K = D['K']; PS = (0.05, 0.1, 0.2); CH = ('AD', 'PauliAD', 'CliffAD')
def mcnemar(a, b):              # a, b boolean per target (paired); exact two-sided binomial on discordant pairs
    n10, n01 = int((a & ~b).sum()), int((~a & b).sum())
    return n10, n01, (stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0)
def boot(a, b, B=10000, seed=0):
    r = np.random.default_rng(seed); i = r.integers(0, len(a), (B, len(a)))
    d = a[i].mean(1) - b[i].mean(1); return np.percentile(d, [2.5, 97.5])
print(f"K = {K} targets (same set, same inits for every channel)\n")
print("ARM (i) raw, max converging range (mean ± sd over targets)")
for mode, lab in (('raw', 'i-a centred target'), ('raw_off', 'i-b free offset (Eq. 7)')):
    r0 = D[('range', mode, 'None', 0.0)]
    print(f"  [{lab}] None: {r0.mean():.3f} ± {r0.std(ddof=1):.3f}")
    for p in PS:
        print(f"    p={p:.2f}  " + "  ".join(f"{c}: {D[('range', mode, c, p)].mean():.3f} ± {D[('range', mode, c, p)].std(ddof=1):.3f}" for c in CH)
              + f"   AD-Pauli paired diff {np.mean(D[('range', mode, 'AD', p)] - D[('range', mode, 'PauliAD', p)]):+.3f} (Wilcoxon p={stats.wilcoxon(D[('range', mode, 'AD', p)], D[('range', mode, 'PauliAD', p)]).pvalue:.1e})")
print("\nARM (ii) rescaled f = s<Z> + b, fixed target range")
for R in D['R_fixed']:
    n0 = D[('scaled', R, 'None', 0.0)]
    s0 = np.median(n0['s'])
    print(f"  R = {R:.3f}.  None: conv<=150 {np.mean(n0['conv'] <= 150):.3f}  conv<=1000 {np.mean(n0['conv'] <= 1000):.3f}  median MSE1000 {np.median(n0['mse1000']):.1e}  median |s| {s0:.3f}")
    for p in PS:
        for c in CH:
            d = D[('scaled', R, c, p)]
            print(f"    p={p:.2f} {c:7s} conv<=150 {np.mean(d['conv'] <= 150):.3f}  conv<=1000 {np.mean(d['conv'] <= 1000):.3f}  median MSE1000 {np.median(d['mse1000']):.1e}  "
                  f"median |s| {np.median(d['s']):.3f}  shot factor vs None (s/s0)^2 {np.median(d['s'])**2 / s0**2:.2f}")
        for budget in (150, 1000):
            a = D[('scaled', R, 'AD', p)]['conv'] <= budget; b = D[('scaled', R, 'PauliAD', p)]['conv'] <= budget
            n10, n01, pv = mcnemar(a, b); lo, hi = boot(a.astype(float), b.astype(float))
            print(f"      AD - PauliAD conv<={budget}: {a.mean() - b.mean():+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  McNemar AD-only {n10} / Pauli-only {n01}, p={pv:.2g}")
