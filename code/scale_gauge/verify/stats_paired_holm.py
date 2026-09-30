"""The classifier comparisons of App. C with paired t tests
(conditions share initial parameters seed by seed) and a Holm correction over the tests of the
statistics table (Delta acc and Delta chi against the noiseless circuit, S1, L=1-4). The two-level
bootstrap CIs are unchanged (stats_appendix.py). Also prints the paired P values quoted in the text
(S2 Fashion-MNIST, S4, fixed-point control, S7/S7b)."""
import pathlib, sys
import numpy as np
from scipy import stats
HERE = pathlib.Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
from stats_appendix import load, cells, corr, f3, pfmt, CODE   # noqa: E402  (runs stats_appendix's prints)
from analyze_readout import labels, two_level

def paired(a, b, key):
    d = np.array([key(x) - key(y) for x, y in zip(a, b)])
    assert [x['seed'] for x in a] == [y['seed'] for y in b]
    return (np.nan if np.allclose(d, 0) else stats.ttest_1samp(d, 0).pvalue), d.mean()

def holm(ps):
    ps = np.asarray(ps, float); ok = np.isfinite(ps); idx = np.argsort(np.where(ok, ps, np.inf)); m = ok.sum()
    adj = np.full_like(ps, np.nan); run = 0.0
    for r, i in enumerate(idx[:m]):
        run = max(run, min(1.0, (m - r) * ps[i])); adj[i] = run
    return adj

Y = labels('MNIST')
S = cells(load(CODE / 'm5_readout' / 'results.pkl'))
rows, P = [], []
for L in (1, 2, 3, 4):
    for ch in ('AD', 'Pauli', 'Depol'):
        a, b = S[(ch, L)], S[('None', L)]
        m, (lo, hi) = two_level(corr(a, Y), corr(b, Y))
        pa, _ = paired(a, b, lambda r: r['test_acc']); ps, ds = paired(a, b, lambda r: r['sensitivity'])
        rows.append((L, ch, m, lo, hi, ds)); P += [pa, ps]
H = holm(P)
print(f'\n% TABLE (paired t, Holm-adjusted over {int(np.isfinite(P).sum())} tests): L & ch & dacc & CI & P_holm & dchi (P_holm)')
for k, (L, ch, m, lo, hi, ds) in enumerate(rows):
    pa, ps = H[2 * k], H[2 * k + 1]
    fa = 'exact' if np.isnan(pa) else pfmt(pa); fs = '--' if np.isnan(ps) else pfmt(ps)
    print(f'{L} & {ch} & ${f3(m)}$ & $[{f3(lo)},{f3(hi)}]$ & {fa} & ${f3(ds)}$ ({fs})\\\\   % raw P {P[2*k]:.2g} / {P[2*k+1]:.2g}')
print('surviving at 0.05 after Holm:', [(rows[k // 2][0], rows[k // 2][1], 'acc' if k % 2 == 0 else 'chi') for k in range(len(H)) if np.isfinite(H[k]) and H[k] < 0.05])

def show(tag, a, b):
    pa, da = paired(a, b, lambda r: r['test_acc']); ps, ds = paired(a, b, lambda r: r['sensitivity'])
    print(f'{tag}: dacc {da:+.4f} paired P {pa:.2g}; dchi {ds:+.3f} paired P {ps:.2g}')
F2 = cells(load(CODE / 'm5_fashion_readout' / 'results.pkl'))
for ch in ('AD', 'Pauli', 'Depol'):
    show(f'S2 Fashion {ch}-None', F2[(ch, 4)], F2[('None', 4)])
show('S2 Fashion AD-Pauli', F2[('AD', 4)], F2[('Pauli', 4)])
for L in (2, 3, 4):
    show(f'S1 L={L} AD-Pauli', S[('AD', L)], S[('Pauli', L)])
I = cells(load(CODE / 'm5_inside' / 'results.pkl')); show('S4 AD_flip-AD', I[('AD_flip', 4)], I[('AD', 4)])
Fx = cells(load(CODE / 'm5_fixedpoint' / 'results.pkl')); D = cells(load(CODE / 'm5_depth' / 'results.pkl'))
pa, da = paired(Fx[('AD_flip', 4)], D[('AD', 4)], lambda r: r['test_acc']); print(f'fixedpoint (raw readout) AD_flip-AD dacc {da:+.4f} paired P {pa:.2g}')
