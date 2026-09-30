"""Summary of F1_s5 (shot-trained S5 models): operational shot cost (geometric mean over seeds)
under 'calib' (primary) and 'stored', ratios to None and to AD, kappa from the exact training spreads,
comparison with the exact-trained S1 models (F1_calib_summary.pkl, F1_summary.pkl), and the cross-check of
'calib' at the training N_s against each record's test_acc_shot_mean. Output: F1_s5_summary.pkl."""
import os, sys, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from F1_shots import op_cost
R = pickle.load(open(os.path.join(HERE, 'F1_s5_results.pkl'), 'rb'))
S1c = pickle.load(open(os.path.join(HERE, 'F1_calib_summary.pkl'), 'rb'))
CH = ('None', 'AD', 'Pauli', 'Depol')
print("max exact-prediction mismatch:", max(r['pred_mismatch'] for r in R), " runs:", len(R))
d = np.array([r['calib_at_train_shots'] - r['record_shot_acc'] for r in R])
print(f"cross-check calib(train N_s) - record test_acc_shot_mean: mean {d.mean():+.4f}, sd {d.std(ddof=1):.4f}, max |d| {np.abs(d).max():.4f}")
rows = []
for ts, L in ((1000, 1), (1000, 2), (1000, 3), (1000, 4), (100, 4), (10000, 4)):
    cell = {ch: [r for r in R if r['train_shots'] == ts and r['L'] == L and r['channel'] == ch] for ch in CH}
    sig = {ch: np.mean([r['sigma'] for r in cell[ch]]) for ch in CH}
    res = {}
    for conv in ('calib', 'stored'):
        for ch in CH:
            cs, flags = [], []
            for r in cell[ch]:
                (c, f), _ = op_cost(np.array(r['Ns'], float), r['acc_' + conv].mean(1), r['acc_exact'])
                if r['acc_exact'] > 0.5: cs.append(c); flags.append(f)
            ok = [c for c in cs if np.isfinite(c)]
            res[(conv, ch)] = (np.exp(np.mean(np.log(ok))) if ok else np.nan, len(cs) - len(ok), sum(f == 'le_min' for f in flags))
    print(f"\n== trained at N_s={ts}, L={L}:  exact acc " + "  ".join(f"{ch} {np.mean([r['acc_exact'] for r in cell[ch]]):.4f}" for ch in CH)
          + "\n   acc at training N_s (calib) " + "  ".join(f"{ch} {np.mean([r['calib_at_train_shots'] for r in cell[ch]]):.4f}" for ch in CH))
    for ch in CH:
        cc, cen, lem = res[('calib', ch)]; cs_, cen2, _ = res[('stored', ch)]
        s1 = [x for x in S1c if x['dataset'] == 'MNIST' and x['L'] == L and x['channel'] == ch][0]
        row = dict(train_shots=ts, L=L, channel=ch, calib_cost=cc, calib_ratio_None=cc / res[('calib', 'None')][0],
                   calib_ratio_AD=cc / res[('calib', 'AD')][0], stored_cost=cs_, stored_ratio_None=cs_ / res[('stored', 'None')][0],
                   stored_ratio_AD=cs_ / res[('stored', 'AD')][0], kappa=(sig['None'] / sig[ch]) ** 2, censored=cen, at_min=lem,
                   s1_calib_cost=s1['calib_cost'], s1_calib_ratio=s1['calib_ratio'], exact_acc=np.mean([r['acc_exact'] for r in cell[ch]]))
        rows.append(row)
        print(f"   {ch:5s} calib {cc:9.3g} (xNone {row['calib_ratio_None']:8.3g}, xAD {row['calib_ratio_AD']:7.3g})  stored {cs_:9.3g} "
              f"(xNone {row['stored_ratio_None']:8.3g}, xAD {row['stored_ratio_AD']:7.3g})  kappa {row['kappa']:8.3g}  censored {cen} at-min {lem}"
              f"  | S1 exact-trained calib {s1['calib_cost']:9.3g} (xNone {s1['calib_ratio']:8.3g})  S5/S1 {cc / s1['calib_cost']:6.3g}")
pickle.dump(rows, open(os.path.join(HERE, 'F1_s5_summary.pkl'), 'wb'))
