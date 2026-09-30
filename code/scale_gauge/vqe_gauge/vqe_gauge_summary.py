"""Summary table for vqe_gauge_results.pkl: min (landscape-level) and median (optimization-level)
relative energy error per point; AD vs reversed AD compared by Mann-Whitney U and a bootstrap
95% CI of the median difference."""
import pickle
import sys

import numpy as np
from scipy import stats

d = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else 'vqe_gauge_results.pkl', 'rb'))
R = d['results']
rng = np.random.default_rng(0)
print(f"E0 = {d['E0']:.6f}; settings {d['settings']}")
print("identity checks:", {k: f"{v:.1e}" for k, v in d['identity_checks'].items()})
print("paired post trajectories:", {k: f"{v:.1e}" for k, v in d['paired_post'].items()})
print(f"\n{'init':7s} {'place':5s} {'gamma':>5s} | {'AD min':>8s} {'rev min':>8s} {'twl min':>8s} | "
      f"{'AD med':>8s} {'rev med':>8s} {'twl med':>8s} | rev-AD median [95% CI]   MW p")
for scheme in ('uniform', 'small'):
    for placement in ('post', 'mid'):
        for g in d['settings']['GAMMAS']:
            a, r, t = (R[(scheme, placement, g, ch)]['rel_err'] for ch in ('AD', 'revAD', 'twirlAD'))
            boot = [np.median(rng.choice(r, len(r))) - np.median(rng.choice(a, len(a))) for _ in range(4000)]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            p = stats.mannwhitneyu(r, a).pvalue
            print(f"{scheme:7s} {placement:5s} {g:5.2f} | {a.min():8.4f} {r.min():8.4f} {t.min():8.4f} | "
                  f"{np.median(a):8.4f} {np.median(r):8.4f} {np.median(t):8.4f} | "
                  f"{np.median(r) - np.median(a):+.4f} [{lo:+.4f}, {hi:+.4f}]  {p:.2g}")
