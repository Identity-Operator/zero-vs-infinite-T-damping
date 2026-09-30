"""Two-level bootstrap for the standardized-readout comparisons at L=4, from the
e1 re-run with per-image predictions (scale_e1e2/e1_save_indep_*.pkl).

bootstrap_e1.py resamples test images with the 8 initializations held fixed, so its interval is
conditional on those initializations. Here each replicate draws 8 seeds with replacement and
3037 images with replacement, and uses the same draws for both channels of a comparison. Seeds
are paired across channels because every channel starts from the same initial parameters for a
given seed (scale_e1e2/init_draws_L4_8seeds.pkl), and images are paired because the test set is
shared. The statistic is the mean over the drawn (seed, image) cells of correct(channel) minus
correct(None). The image-only interval is recomputed alongside for reference.

Output (B = 20000, RNG seed 0 for image-only, 1 for two-level; about 13 s on CPU). The image-only
column reproduces bootstrap_e1.py exactly (same RNG seed and draw order).
AD - None: +0.0077   image-only 95% CI [+0.0037, +0.0116]   two-level 95% CI [+0.0022, +0.0132]
Pauli - None: +0.0009   image-only 95% CI [-0.0026, +0.0042]   two-level 95% CI [-0.0073, +0.0088]
Depol - None: +0.0009   image-only 95% CI [-0.0028, +0.0047]   two-level 95% CI [-0.0061, +0.0078]
"""
import pathlib, pickle, numpy as np
D = pathlib.Path(__file__).resolve().parent.parent / 'scale_e1e2'
Yte = np.load(D / 'v5_angles.npz')['Yte']
B, n_img = 20000, len(Yte)
H = {}
for ch in ('None', 'AD', 'Pauli', 'Depol'):
    rows = sorted(pickle.load(open(D / f'e1_save_indep_{ch}.pkl', 'rb')), key=lambda r: r['seed'])
    H[ch] = (np.stack([np.asarray(r['test_pred']) for r in rows]) == Yte[None, :]).astype(np.float64)
n_seed = H['None'].shape[0]
for ch in ('AD', 'Pauli', 'Depol'):
    d = H[ch] - H['None']                                   # (seeds, images)
    r_img, r_two = np.random.default_rng(0), np.random.default_rng(1)
    d_img = d.mean(0)
    img = np.array([d_img[r_img.integers(0, n_img, n_img)].mean() for _ in range(B)])
    two = np.array([d[np.ix_(r_two.integers(0, n_seed, n_seed), r_two.integers(0, n_img, n_img))].mean()
                    for _ in range(B)])
    (li, hi_), (lt, ht) = np.percentile(img, [2.5, 97.5]), np.percentile(two, [2.5, 97.5])
    print(f'{ch} - None: {d.mean():+.4f}   image-only 95% CI [{li:+.4f}, {hi_:+.4f}]   '
          f'two-level 95% CI [{lt:+.4f}, {ht:+.4f}]')
