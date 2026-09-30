"""Operational shot cost under the 'calib' convention (F1_calib_results.pkl) next to the 'stored'
convention (F1_summary.pkl) and kappa. Same definition as F1_shots.op_cost (first N_s within 0.01 of the
exact accuracy, log-interpolated; geometric mean over seeds; the diverged run and censored runs excluded)."""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from F1_shots import op_cost
C = pickle.load(open(os.path.join(HERE, 'F1_calib_results.pkl'), 'rb'))
S = pickle.load(open(os.path.join(HERE, 'F1_summary.pkl'), 'rb'))
rows = []
for ds, Ls in (('MNIST', (1, 2, 3, 4)), ('FashionMNIST', (4,))):
    for L in Ls:
        gm, cens = {}, {}
        for ch in ('None', 'AD', 'Pauli', 'Depol'):
            cs = []
            for r in [x for x in C if x['dataset'] == ds and x['L'] == L and x['channel'] == ch]:
                (c, f), _ = op_cost(np.array(r['Ns'], float), r['acc'].mean(1), r['acc_exact'])
                if r['acc_exact'] > 0.5: cs.append(c)
            ok = [c for c in cs if np.isfinite(c)]; cens[ch] = len(cs) - len(ok)
            gm[ch] = np.exp(np.mean(np.log(ok))) if ok else np.nan
        for ch in ('None', 'AD', 'Pauli', 'Depol'):
            s = [x for x in S if x['dataset'] == ds and x['L'] == L and x['channel'] == ch][0]
            sN = [x for x in S if x['dataset'] == ds and x['L'] == L and x['channel'] == 'None'][0]
            rows.append(dict(dataset=ds, L=L, channel=ch, calib_cost=gm[ch], calib_ratio=gm[ch] / gm['None'], calib_censored=cens[ch],
                             stored_cost=s['cost_geomean'], stored_ratio=s['cost_geomean'] / sN['cost_geomean'], kappa=s['kappa']))
            print(f"{ds[:5]} L={L} {ch:5s}  calib: cost {gm[ch]:9.3g} ratio {gm[ch] / gm['None']:9.3g} (censored {cens[ch]})"
                  f"   stored: cost {s['cost_geomean']:9.3g} ratio {s['cost_geomean'] / sN['cost_geomean']:9.3g}   kappa {s['kappa']:9.3g}")
pickle.dump(rows, open(os.path.join(HERE, 'F1_calib_summary.pkl'), 'wb'))
