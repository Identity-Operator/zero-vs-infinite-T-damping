"""Summary of F1: accuracy against N_s, operational shot cost and its ratio to None, and kappa.
Combines the main grid (F1_results.pkl) with the supplementary extension (F1_ext_results.pkl, if present):
the cost is taken from the main grid when it is uncensored there, otherwise from the merged grid."""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from F1_shots import op_cost
B = pickle.load(open(os.path.join(HERE, 'F1_results.pkl'), 'rb'))
E = {(r['dataset'], r['L'], r['channel'], r['seed']): r for r in pickle.load(open(os.path.join(HERE, 'F1_ext_results.pkl'), 'rb'))} \
    if os.path.exists(os.path.join(HERE, 'F1_ext_results.pkl')) else {}
CH = ('None', 'AD', 'Pauli', 'Depol')
def merged_cost(r):
    k = (r['dataset'], r['L'], r['channel'], r['seed'])
    if r['op_flag'] != 'gt_max' or k not in E: return r['op_cost'], r['op_flag'], 'plan'
    e = E[k]; ns = np.array(list(r['Ns']) + list(e['Ns'][1:]), float)       # e['Ns'][0] == 1e5 repeats the main grid top
    am = np.concatenate([r['acc'].mean(1), e['acc'][1:].mean(1)])
    (c, f), _ = op_cost(ns, am, r['acc_exact']); return c, f, 'ext'
rows = []
for ds, Ls in (('MNIST', (1, 2, 3, 4)), ('FashionMNIST', (4,))):
    print(f"\n===== {ds} =====")
    for L in Ls:
        cell = {ch: [r for r in B if r['dataset'] == ds and r['L'] == L and r['channel'] == ch] for ch in CH}
        sig = {ch: np.mean([r['sigma'] for r in cell[ch]]) for ch in CH}
        print(f"-- L={L}: mean accuracy vs N_s (over seeds; exact in last column)   N_s = {B[0]['Ns']}")
        for ch in CH:
            acc = np.mean([r['acc'].mean(1) for r in cell[ch]], 0)
            print(f"   {ch:5s} " + " ".join(f"{a:.3f}" for a in acc) + f" | exact {np.mean([r['acc_exact'] for r in cell[ch]]):.4f}")
        gm = {}
        for ch in CH:
            cs = [(merged_cost(r), r) for r in cell[ch]]
            ok = [c for (c, f, src), r in cs if np.isfinite(c) and r['acc_exact'] > 0.5]    # excludes the diverged run (0.382)
            cens = sum(1 for (c, f, src), r in cs if not np.isfinite(c))
            le = sum(1 for (c, f, src), r in cs if f == 'le_min')
            gm[ch] = np.exp(np.mean(np.log(ok))) if ok else np.nan
            src = sorted({src for (c, f, src), r in cs})
            rows.append(dict(dataset=ds, L=L, channel=ch, cost_geomean=gm[ch], n_ok=len(ok), n_censored=cens, n_le_min=le, source=src,
                             costs=[c for (c, f, s), r in cs], kappa=(sig['None'] / sig[ch]) ** 2))
        print(f"   operational shot cost (geometric mean over seeds; censored runs excluded and counted):")
        for ch in CH:
            rr = [x for x in rows if x['dataset'] == ds and x['L'] == L and x['channel'] == ch][0]
            print(f"   {ch:5s} {rr['cost_geomean']:10.4g}  n={rr['n_ok']} censored={rr['n_censored']} at-min={rr['n_le_min']} src={rr['source']}"
                  f"   ratio to None {rr['cost_geomean'] / gm['None']:9.4g}   kappa {rr['kappa']:9.4g}")
pickle.dump(rows, open(os.path.join(HERE, 'F1_summary.pkl'), 'wb'))
print("\nmax mismatch against stored test_pred over all runs:", max(r['pred_mismatch'] for r in B), " runs:", len(B))
