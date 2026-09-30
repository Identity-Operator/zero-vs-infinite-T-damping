"""Statistics for the manuscript's Appendix: every hypothesis test quoted in
v5_pra, recomputed from the committed result files. Prints LaTeX table rows.
Accuracy CIs: two-level bootstrap (seeds paired, images), B=20000, RNG seed 1."""
import os, sys, pickle, pathlib, shutil, tempfile, collections
import numpy as np
from scipy import stats
HERE = pathlib.Path(__file__).resolve().parent; CODE = HERE.parent.parent / 'noise_structure_mnist'
sys.path.insert(0, str(HERE)); from analyze_readout import labels, two_level
def load(p):
    t = pathlib.Path(tempfile.mkdtemp()) / 'r.pkl'; shutil.copy(p, t); return pickle.load(open(t, 'rb'))
def cells(R, L=None):
    c = collections.defaultdict(list)
    for r in R:
        c[(r['channel'], r['L'])].append(r)
    return {k: sorted(v, key=lambda r: r['seed']) for k, v in c.items()}
def corr(rows, Y): return (np.stack([np.asarray(r['test_pred']) for r in rows]) == Y[None, :]).astype(float)
f3 = lambda x: f'{x:+.3f}'
def pfmt(p): return f'{p:.2g}' if p >= 1e-3 else f'{p:.0e}'.replace('e-0', 'e-')
Y = labels('MNIST')
print('% classifier, standardized readout (S1): X - None')
S = cells(load(CODE / 'm5_readout' / 'results.pkl'))
for L in (1, 2, 3, 4):
    for ch in ('AD', 'Pauli', 'Depol'):
        a, b = S[(ch, L)], S[('None', L)]
        m, (lo, hi) = two_level(corr(a, Y), corr(b, Y))
        pa = stats.ttest_ind([r['test_acc'] for r in a], [r['test_acc'] for r in b], equal_var=False).pvalue
        sa = [r['sensitivity'] for r in a]; sb = [r['sensitivity'] for r in b]
        ps = stats.ttest_ind(sa, sb, equal_var=False).pvalue
        print(f'{L} & {ch} & ${f3(m)}$ & $[{f3(lo)},{f3(hi)}]$ & {pfmt(pa)} & ${f3(np.mean(sa)-np.mean(sb))}$ & {pfmt(ps)}\\\\')
print('% S4 inside: AD_flip - AD; and None - AD')
I = cells(load(CODE / 'm5_inside' / 'results.pkl'))
a, b = I[('AD_flip', 4)], I[('AD', 4)]
m, (lo, hi) = two_level(corr(a, Y), corr(b, Y))
print('S4', f3(m), f3(lo), f3(hi), pfmt(stats.ttest_ind([r['test_acc'] for r in a], [r['test_acc'] for r in b], equal_var=False).pvalue),
      f3(np.mean([r['sensitivity'] for r in a]) - np.mean([r['sensitivity'] for r in b])),
      pfmt(stats.ttest_ind([r['sensitivity'] for r in a], [r['sensitivity'] for r in b], equal_var=False).pvalue))
print('% Phase 1 raw readout L=4: AD_flip - AD (m5_fixedpoint vs m5_depth)')
F = cells(load(CODE / 'm5_fixedpoint' / 'results.pkl')); D = cells(load(CODE / 'm5_depth' / 'results.pkl'))
a, b = F[('AD_flip', 4)], D[('AD', 4)]
assert [r['seed'] for r in a] == [r['seed'] for r in b]
m, (lo, hi) = two_level(corr(a, Y), corr(b, Y))
print('fixedpoint', f3(m), f3(lo), f3(hi), pfmt(stats.ttest_ind([r['test_acc'] for r in a], [r['test_acc'] for r in b], equal_var=False).pvalue))
print('% single-qubit fits: McNemar (conv<=1000, conv<=150) AD vs PauliAD, R=1 and R=median')
V = pickle.load(open(HERE.parent / 'vr' / 'vr_results.pkl', 'rb'))
for R in V['R_fixed']:
    for p in (0.05, 0.1, 0.2):
        out = []
        for B in (1000, 150):
            ca = V[('scaled', R, 'AD', p)]['conv'] <= B; cb = V[('scaled', R, 'PauliAD', p)]['conv'] <= B
            n10, n01 = int(np.sum(ca & ~cb)), int(np.sum(~ca & cb))
            pv = stats.binomtest(n10, n10 + n01, 0.5).pvalue if n10 + n01 else 1.0
            out.append(f'${100*(ca.mean()-cb.mean()):+.1f}$ & {pfmt(pv)}')
        print(f'{R:.3f} & {p} & ' + ' & '.join(out) + '\\\\')
print('% raw range AD - PauliAD, Wilcoxon')
for var in ('raw', 'raw_off'):
    for p in (0.05, 0.1, 0.2):
        a, b = V[('range', var, 'AD', p)], V[('range', var, 'PauliAD', p)]
        print(var, p, f3(a.mean() - b.mean()), pfmt(stats.wilcoxon(a, b).pvalue))
print('% VQE inside: revAD - AD relative error, Mann-Whitney (init uniform)')
Q = pickle.load(open(HERE.parent / 'vqe_gauge' / 'vqe_gauge_results.pkl', 'rb'))['results']
for g in (0.01, 0.05, 0.1, 0.2):
    for pl in ('post', 'mid'):
        a = np.asarray(Q[('uniform', pl, g, 'revAD')]['rel_err']); b = np.asarray(Q[('uniform', pl, g, 'AD')]['rel_err'])
        print(pl, g, f'{np.median(a)-np.median(b):+.4f}', pfmt(stats.mannwhitneyu(a, b).pvalue))
