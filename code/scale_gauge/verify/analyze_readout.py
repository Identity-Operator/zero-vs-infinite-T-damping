"""Verification and tables for stages S1 (m5_readout), S2
(m5_fashion_readout) and S4 (m5_inside), with the raw-readout Phase 1 records for comparison.

Per cell: test accuracy (mean, SD over seeds), train accuracy, input sensitivity, accuracy drop
under a common encoding offset delta = -1 and +1 (reported per sign), and the shot factor
kappa = (sigma_None / sigma)^2 from the stored training-feature std at the final parameters.
Comparisons use the primary CI: a two-level bootstrap that resamples seeds
(paired: conditions share initial parameters per seed) and test images together; B = 20000,
RNG seed 1, as in bootstrap_two_level.py.

Reads copies of the result files, so it is safe to run while a queue appends.
Usage: python analyze_readout.py [s1|s2|s4|all]
"""
import os, sys, shutil, tempfile, pickle, pathlib, collections
import numpy as np
from scipy import stats

HERE = pathlib.Path(__file__).resolve().parent
CODE = HERE.parent.parent / 'noise_structure_mnist'
B = 20000


def load(stage_dir):
    p = CODE / stage_dir / 'results.pkl'
    if not p.exists():
        return []
    tmp = pathlib.Path(tempfile.mkdtemp()) / 'r.pkl'
    shutil.copy(p, tmp)
    return pickle.load(open(tmp, 'rb'))


def labels(dataset):
    sys.path.insert(0, str(CODE))
    cwd = os.getcwd(); os.chdir(CODE)
    try:
        from mnist_hybrid_multi import load_mnist_binary
        cls = (1, 3, 5) if dataset == 'MNIST' else (0, 2, 4)
        return load_mnist_binary(classes=cls, n_train=1000, n_test=None, n_components=4,
                                 seed=0, dataset=dataset, pca_seed=0)[3]
    finally:
        os.chdir(cwd)


def two_level(Ha, Hb, seed=1):
    """Ha, Hb: (seeds, images) correctness, rows aligned by seed. CI of mean(Ha - Hb)."""
    d = Ha - Hb; ns, ni = d.shape; r = np.random.default_rng(seed)
    bs = np.array([d[np.ix_(r.integers(0, ns, ns), r.integers(0, ni, ni))].mean() for _ in range(B)])
    return d.mean(), np.percentile(bs, [2.5, 97.5])


def cells(R, **filt):
    out = collections.defaultdict(list)
    for r in R:
        if all(r.get(k) == v for k, v in filt.items()):
            out[(r['channel'], r['L'])].append(r)
    return {k: sorted(v, key=lambda r: r['seed']) for k, v in out.items()}


def correct(rows, Y):
    return (np.stack([np.asarray(r['test_pred']) for r in rows]) == Y[None, :]).astype(float)


def summarize(C, Y, chans, Ls, ref='None', title=''):
    print(f'\n== {title}')
    for L in Ls:
        if (ref, L) not in C:
            continue
        sref = np.mean([np.mean(r['params']['z_std']) for r in C[(ref, L)]]) if 'z_std' in C[(ref, L)][0]['params'] else None
        for ch in chans:
            rows = C.get((ch, L), [])
            if not rows:
                continue
            acc = np.array([r['test_acc'] for r in rows])
            line = (f'L={L} {ch:7s} n={len(rows)} test {acc.mean():.4f}±{acc.std(ddof=1):.4f} '
                    f'train {np.mean([r["train_acc"] for r in rows]):.4f} '
                    f'sens {np.mean([r["sensitivity"] for r in rows]):.3f} '
                    f'drop(-1) {np.mean([r["test_acc"] - r["offset_acc"][-1.0] for r in rows]):.3f} '
                    f'drop(+1) {np.mean([r["test_acc"] - r["offset_acc"][1.0] for r in rows]):.3f}')
            if sref is not None and 'z_std' in rows[0]['params']:
                s = np.mean([np.mean(r['params']['z_std']) for r in rows])
                line += f' sigma {s:.5f} kappa {(sref / s) ** 2:.3g}'
            if ch != ref and len(rows) == len(C[(ref, L)]):
                assert [r['seed'] for r in rows] == [r['seed'] for r in C[(ref, L)]]
                m, (lo, hi) = two_level(correct(rows, Y), correct(C[(ref, L)], Y))
                p = stats.ttest_ind(acc, [r['test_acc'] for r in C[(ref, L)]], equal_var=False).pvalue
                sa = [r['sensitivity'] for r in rows]; sb = [r['sensitivity'] for r in C[(ref, L)]]
                line += (f'\n        {ch}-{ref}: {m:+.4f} two-level CI [{lo:+.4f}, {hi:+.4f}] Welch p={p:.2g}; '
                         f'sens diff {np.mean(sa) - np.mean(sb):+.3f} (Welch p={stats.ttest_ind(sa, sb, equal_var=False).pvalue:.2g})')
            print(line)


def check(R, name):
    c = collections.Counter((r['channel'], r['L'], r['seed']) for r in R)
    print(f'{name}: {len(R)} records; duplicates {[k for k, v in c.items() if v > 1]}; '
          f'code_sha {sorted({r["code_sha"] for r in R})}; git_dirty {sorted({r["git_dirty"] for r in R})}; '
          f'readout {sorted({str(r.get("readout")) for r in R})}; placement {sorted({str(r.get("noise_placement")) for r in R})}')


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    CH = ('None', 'AD', 'Pauli', 'Depol')
    if what in ('s1', 'all'):
        Y = labels('MNIST')
        R = load('m5_readout'); check(R, 'm5_readout')
        summarize(cells(R), Y, CH, (1, 2, 3, 4), title='S1: MNIST, standardized readout, 100 steps')
        summarize(cells(load('m5_depth')), Y, CH, (1, 2, 3, 4), title='Phase 1: MNIST, raw readout, 100 steps')
    if what in ('s2', 'all'):
        Y = labels('FashionMNIST')
        R = load('m5_fashion_readout'); check(R, 'm5_fashion_readout')
        summarize(cells(R), Y, CH, (4,), title='S2: Fashion-MNIST, standardized readout, L=4')
        summarize(cells(load('m5_fashion')), Y, CH, (4,), title='Q6: Fashion-MNIST, raw readout, L=4')
    if what in ('s4', 'all'):
        Y = labels('MNIST')
        R = load('m5_inside'); check(R, 'm5_inside')
        summarize(cells(R), Y, ('None', 'AD', 'AD_flip'), (4,), ref='AD',
                  title='S4: MNIST, noise inside the entangler, standardized readout, L=4 (reference AD)')
