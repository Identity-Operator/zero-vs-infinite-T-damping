"""Image-level bootstrap for the standardized-readout comparisons at L=4, from
the e1 re-run with per-image predictions (scale_e1e2/e1_save_indep_*.pkl).
Per-image correctness is averaged over the 8 initializations; images are resampled."""
import pathlib, pickle, numpy as np
D = pathlib.Path(__file__).resolve().parent.parent / 'scale_e1e2'
Yte = np.load(D / 'v5_angles.npz')['Yte']
C = {}
for ch in ('None', 'AD', 'Pauli', 'Depol'):
    rows = sorted(pickle.load(open(D / f'e1_save_indep_{ch}.pkl', 'rb')), key=lambda r: r['seed'])
    C[ch] = (np.stack([np.asarray(r['test_pred']) for r in rows]) == Yte[None, :]).mean(0)
idx = np.random.default_rng(0).integers(0, len(Yte), size=(20000, len(Yte)))
for ch in ('AD', 'Pauli', 'Depol'):
    d = C[ch] - C['None']; lo, hi = np.percentile(d[idx].mean(1), [2.5, 97.5])
    print(f'{ch} - None: {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]')
