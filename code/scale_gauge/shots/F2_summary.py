"""Summary of F2: feature-level input sensitivity (mean over test images of |d z~/d a|_F) per stage,
channel and L; Welch tests against None and against the Pauli twirl (S4: against None and AD_flip vs AD).
The one diverged run (S1 Pauli, L=1, seed 512: test accuracy 0.382) is reported separately and excluded from the tests; every other run is kept."""
import os, pickle, numpy as np
from scipy import stats
HERE = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(os.path.join(HERE, 'F2_results.pkl'), 'rb'))
W = lambda a, b: stats.ttest_ind(a, b, equal_var=False).pvalue
out = []
for stage, Ls, chs, refs in (('S1', (1, 2, 3, 4), ('None', 'AD', 'Pauli', 'Depol'), ('None', 'Pauli')),
                             ('S2', (4,), ('None', 'AD', 'Pauli', 'Depol'), ('None', 'Pauli')),
                             ('S4', (4,), ('None', 'AD', 'AD_flip'), ('None', 'AD'))):
    print(f"===== {stage} =====")
    for L in Ls:
        cell = {ch: [r for r in R if r['stage'] == stage and r['L'] == L and r['channel'] == ch] for ch in chs}
        bad = [r for ch in chs for r in cell[ch] if r['test_acc'] < 0.5]
        for r in bad: print(f"   excluded: {r['channel']} seed {r['seed']} (min z_std {r['min_z_std']:.1e}, test acc {r['test_acc']:.3f}, sens {r['sens_feat']:.3g})")
        v = {ch: np.array([r['sens_feat'] for r in cell[ch] if r['test_acc'] >= 0.5]) for ch in chs}
        vl = {ch: np.array([r['sens_logp_record'] for r in cell[ch] if r['test_acc'] >= 0.5]) for ch in chs}
        for ch in chs:
            line = f"   L={L} {ch:7s} n={len(v[ch])} feature-level {v[ch].mean():.4f} ± {v[ch].std(ddof=1):.4f}   (log-p level {vl[ch].mean():.3f})"
            row = dict(stage=stage, L=L, channel=ch, n=len(v[ch]), mean=v[ch].mean(), sd=v[ch].std(ddof=1), logp_mean=vl[ch].mean())
            for ref in refs:
                if ch != ref:
                    d, p = v[ch].mean() - v[ref].mean(), W(v[ch], v[ref])
                    line += f"   vs {ref}: {d:+.4f} (P {p:.2g})"; row[f'd_vs_{ref}'] = d; row[f'P_vs_{ref}'] = p
            print(line); out.append(row)
pickle.dump(out, open(os.path.join(HERE, 'F2_summary.pkl'), 'wb'))
