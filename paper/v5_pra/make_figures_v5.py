# -*- coding: utf-8 -*-
"""
Figures for the v5_pra manuscript. Reads the committed result files directly;
nothing is transcribed.

  fig_vr.pdf   scale_gauge/vr/vr_results.pkl          single-qubit fits (test 1)
  fig_vqe.pdf  scale_gauge/vqe_gauge/vqe_gauge_results.pkl   eigensolver gauge (test 2)
  fig_clf.pdf  noise_structure_mnist/m5_depth (raw readout), m5_readout (standardized; S1)
  fig_width.pdf noise_structure_mnist/m5_width_diag (S6a) and scale_gauge/theory/t1_results.json (T1)
  fig_shots.pdf noise_structure_mnist/m5_shots (S5); scale_gauge/shots/F1_calib_summary.pkl, F1_s5_summary.pkl
  fig_sep.pdf  scale_gauge/theory/t1_p_dependence_n4.json, _n8.json (T1b)   separation against p

House style: one red family for every noisy channel, with
the noiseless reference in neutral near-black. Channel identity is carried by
lightness, marker shape and a direct label, so the figures read in grayscale.
Each channel keeps its shade in every figure. The three reds pass the dataviz
CVD (min dE 14.8) and normal-vision (16.6) separation checks, and reversed AD
(pale red, dashed) passes against AD and the Pauli twirl, the only series
shown with it. All figures are full text width, with the same row height and
panel titles.
"""
import pathlib
import pickle

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
EVID = HERE.parent.parent / 'code' / 'scale_gauge'
OUT = HERE / 'figures'
OUT.mkdir(exist_ok=True)

COL = {'None': '#252525', 'AD': '#99000d', 'Pauli': '#e0301e',
       'Cliff': '#fb8f67', 'Depol': '#fb8f67', 'revAD': '#f7a386'}
EDGE = {'None': 'white', 'AD': 'white', 'Pauli': 'white',
        'Cliff': '#b8431f', 'Depol': '#b8431f', 'revAD': '#b8431f'}
LSTY = {'revAD': (0, (4, 2))}
MFC = {'revAD': 'white'}          # hollow markers: reversed AD sits on top of AD
LW = {'revAD': 1.8}
MRK = {'None': 'o', 'AD': 's', 'Pauli': '^', 'Cliff': 'D', 'Depol': 'D', 'revAD': 'v'}
NAME = {'None': 'noiseless', 'AD': 'AD', 'Pauli': 'Pauli twirl',
        'Cliff': 'Clifford twirl', 'Depol': 'depolarizing', 'revAD': 'reversed AD'}
CODE = HERE.parent.parent / 'code' / 'noise_structure_mnist'
CH4 = ('None', 'AD', 'Pauli', 'Depol')
LS = (1, 2, 3, 4)
INK, MUTED = '#0b0b0b', '#8a8985'

plt.rcParams.update({
    'font.family': 'serif', 'font.serif': ['DejaVu Serif'],
    'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 7,
    'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': MUTED, 'axes.linewidth': 0.6,
    'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.labelcolor': INK, 'text.color': INK,
    'axes.spines.top': False, 'axes.spines.right': False,
    'axes.grid': True, 'axes.grid.axis': 'y', 'grid.color': '#e6e5e1',
    'grid.linewidth': 0.5, 'savefig.bbox': 'tight', 'savefig.dpi': 300,
    'lines.linewidth': 1.4,
})
WIDE, NARROW = 7.0, 3.4
ROW = 2.4                      # height of one row of panels, in inches
PS = (0.05, 0.1, 0.2)


def panel(ax, tag):
    ax.text(-0.16, 1.02, tag, transform=ax.transAxes, fontsize=8,
            fontweight='bold', va='bottom')


def line(ax, x, y, ch, yerr=None, dx=0.0):
    x = np.asarray(x, float) + dx
    ax.errorbar(x, y, yerr=yerr, color=COL[ch], marker=MRK[ch],
                ms=5.2 if ch in MFC else 4.5, mfc=MFC.get(ch, COL[ch]),
                mec=EDGE[ch], mew=0.9 if ch in MFC else 0.6, capsize=2,
                elinewidth=0.8, zorder=4 if ch in MFC else 3,
                ls=LSTY.get(ch, '-'), lw=LW.get(ch, plt.rcParams['lines.linewidth']))
    return x[-1], y[-1]


def title(ax, text):
    ax.set_title(text, fontsize=7.5, color=INK)


def label_ends(ax, ends, x, min_sep, logy=False):
    """Direct labels at a common x, spread vertically to avoid overlap.
    With logy, min_sep is in decades."""
    f, finv = (np.log10, lambda v: 10 ** v) if logy else ((lambda v: v), (lambda v: v))
    ends = sorted(ends, key=lambda e: e[2])
    ys = [f(e[2]) for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + min_sep)
    ys = [finv(y) for y in ys]
    for (ch, xe, ye), y in zip(ends, ys):
        ax.annotate(NAME[ch], xy=(xe, ye), xytext=(x, y), textcoords='data',
                    va='center', fontsize=7, color=INK,
                    arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4,
                                    shrinkA=1, shrinkB=3)
                    if abs(f(y) - f(ye)) > min_sep / 2 else None)


def wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def fig_vr():
    r = pickle.load(open(EVID / 'vr' / 'vr_results.pkl', 'rb'))
    chans = (('AD', 'AD'), ('Pauli', 'PauliAD'), ('Cliff', 'CliffAD'))
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}

    # (a) raw model, free target offset: largest converging range
    ends = []
    for k, (ch, key) in enumerate(chans):
        v = [r[('range', 'raw_off', key, p)] for p in PS]
        m = [x.mean() for x in v]
        e = [1.96 * x.std(ddof=1) / np.sqrt(len(x)) for x in v]
        ends.append((ch, *line(a, PS, m, ch, e, dx=(k - 1) * 0.004)))
        out[('a', ch)] = m
    n0 = r[('range', 'raw_off', 'None', 0.0)].mean()
    a.hlines(n0, 0.03, 0.205, color=COL['None'], lw=0.8, ls='--', zorder=1)
    ends.append(('None', PS[-1], n0))
    a.set_xlabel('noise strength $p$')
    a.set_ylabel('largest converging range')
    a.set_xticks(PS)
    a.set_xlim(0.03, 0.26)
    a.set_ylim(0.3, 1.0)
    label_ends(a, ends, 0.215, 0.05)
    title(a, 'raw output')
    panel(a, '(a)')

    # (b) rescaled model s<Z>+b, R=1: fraction converged within 1000 steps
    R = 1.0
    ends = []
    for k, (ch, key) in enumerate(chans):
        f, lo, hi = [], [], []
        for p in PS:
            conv = r[('scaled', R, key, p)]['conv']
            kk, nn = int(np.sum(conv <= 1000)), len(conv)
            f.append(kk / nn)
            wl, wh = wilson(kk, nn)
            lo.append(kk / nn - wl)
            hi.append(wh - kk / nn)
        ends.append((ch, *line(b, PS, f, ch, [lo, hi], dx=(k - 1) * 0.004)))
        out[('b', ch)] = f
    f0 = np.mean(r[('scaled', R, 'None', 0.0)]['conv'] <= 1000)
    b.hlines(f0, 0.03, 0.205, color=COL['None'], lw=0.8, ls='--', zorder=1)
    ends.append(('None', PS[-1], f0))
    b.set_xlabel('noise strength $p$')
    b.set_ylabel('fraction converged (1000 steps)')
    b.set_xticks(PS)
    b.set_xlim(0.03, 0.26)
    b.set_ylim(0.90, 1.02)
    b.set_yticks([0.90, 0.92, 0.94, 0.96, 0.98, 1.00])
    label_ends(b, ends, 0.215, 0.013)
    title(b, 'trained scale and offset')
    panel(b, '(b)')

    # (c) shot factor (s/s_None)^2, with the exact depolarizing value
    s0 = np.median(np.abs(r[('scaled', R, 'None', 0.0)]['s']))
    ends = []
    for k, (ch, key) in enumerate(chans):
        kap = [(np.median(np.abs(r[('scaled', R, key, p)]['s'])) / s0) ** 2
               for p in PS]
        ends.append((ch, *line(c, PS, kap, ch, dx=(k - 1) * 0.004)))
        out[('c', ch)] = kap
    pp = np.linspace(0.03, 0.21, 60)
    cc = np.sqrt(1 - pp)
    c.plot(pp, ((2 * cc + cc ** 2) / 3) ** -10, color=MUTED, lw=0.8, ls=':',
           zorder=1)
    c.text(0.10, 1.25, r'$\lambda_d^{-10}$', fontsize=6.5, color=MUTED)
    c.set_xlabel('noise strength $p$')
    c.set_ylabel(r'shot factor $\kappa$')
    c.set_xticks(PS)
    c.set_xlim(0.03, 0.26)
    c.set_ylim(1, 6.5)
    label_ends(c, ends, 0.215, 0.45)
    title(c, 'trained scale and offset')
    panel(c, '(c)')

    fig.tight_layout(w_pad=2.2)
    fig.savefig(OUT / 'fig_vr.pdf')
    fig.savefig(OUT / 'fig_vr.png')
    return out


def fig_vqe():
    r = pickle.load(open(EVID / 'vqe_gauge' / 'vqe_gauge_results.pkl', 'rb'))
    res, G = r['results'], r['settings']['GAMMAS']
    chans = (('AD', 'AD'), ('revAD', 'revAD'), ('Pauli', 'twirlAD'))
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, ROW), sharey=True)
    out = {}
    for ax, place, tag, ttl in ((axes[0], 'post', '(a)', 'noise after $R_{ZZ}$'),
                                (axes[1], 'mid', '(b)', 'noise inside $R_{ZZ}$')):
        ends = []
        for k, (ch, key) in enumerate(chans):
            med, lo, hi = [], [], []
            for g in G:
                e = np.asarray(res[('uniform', place, g, key)]['rel_err'])
                q1, q2, q3 = np.percentile(e, [25, 50, 75])
                med.append(q2)
                lo.append(q2 - q1)
                hi.append(q3 - q2)
            ends.append((ch, *line(ax, G, med, ch, [lo, hi],
                                   dx=(k - 1) * 0.007)))
            out[(place, ch)] = med
        ax.set_xlabel(r'noise strength $\gamma$')
        ax.set_xticks(G)
        ax.set_xticklabels([f'{g:g}' for g in G])
        ax.set_xlim(0.0, 0.27)
        ax.set_yscale('log')
        ax.set_ylim(0.008, 1.5)
        ax.set_yticks([0.01, 0.1, 1])
        ax.set_yticklabels(['0.01', '0.1', '1'])
        globals()['title'](ax, ttl)
        label_ends(ax, ends, 0.218, 0.16, logy=True)
        panel(ax, tag)
    axes[0].set_ylabel('relative energy error')
    fig.tight_layout(w_pad=1.5)
    fig.savefig(OUT / 'fig_vqe.pdf')
    fig.savefig(OUT / 'fig_vqe.png')
    return out


def med_iqr(cols):
    """Median and interquartile range as asymmetric error bars (robust to a diverged run)."""
    q = np.array([np.percentile(v, [25, 50, 75]) for v in cols])
    return q[:, 1], [q[:, 1] - q[:, 0], q[:, 2] - q[:, 1]]


def fig_clf():
    import collections
    raw = pickle.load(open(CODE / 'm5_depth' / 'results.pkl', 'rb'))
    std = pickle.load(open(CODE / 'm5_readout' / 'results.pkl', 'rb'))
    def cells(R):
        c = collections.defaultdict(list)
        for r in R:
            c[(r['channel'], r['L'])].append(r)
        return c
    cr, cs = cells(raw), cells(std)
    fig, axs = plt.subplots(2, 2, figsize=(WIDE, 2 * ROW))
    (a, b), (c, d) = axs
    out = {}
    for ax, C, tag, ttl in ((a, cr, '(a)', 'raw readout'), (b, cs, '(b)', 'standardized readout')):
        ends = []
        for k, ch in enumerate(CH4):
            acc = [[r['test_acc'] for r in C[(ch, L)]] for L in LS]
            m, e = med_iqr(acc)
            x = np.array(LS, float) + (k - 1.5) * 0.06
            ends.append((ch, *line(ax, LS, m, ch, e, dx=(k - 1.5) * 0.06)))
            for xi, v in zip(x, acc):
                vv = [u for u in v if u >= 0.66]
                ax.scatter([xi + 0.12] * len(vv), vv, s=8, marker=MRK[ch], facecolors='none',
                           edgecolors=COL[ch] if EDGE[ch] == 'white' else EDGE[ch],
                           linewidths=0.5, alpha=0.8, zorder=2)
                if len(vv) < len(v):
                    ax.annotate(f'one run at {min(v):.2f}', xy=(xi + 0.12, 0.662), xytext=(xi + 0.35, 0.69),
                                fontsize=6.3, color=MUTED, arrowprops=dict(arrowstyle='->', color=MUTED, lw=0.5))
            out[(tag, ch)] = m
        ax.set_ylim(0.66, 0.90)
        ax.set_ylabel('test accuracy')
        globals()['title'](ax, ttl)
        if tag == '(a)':
            label_ends(ax, ends, 4.35, 0.016)
        else:
            ax.annotate('all four channels', xy=(4.05, 0.873), xytext=(4.15, 0.835), fontsize=7,
                        color=INK, arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4))
        panel(ax, tag)
    ends = []
    for k, ch in enumerate(CH4):
        sv = [[r['sensitivity'] for r in cs[(ch, L)]] for L in LS]
        m, e = med_iqr(sv)
        ends.append((ch, *line(c, LS, m, ch, e, dx=(k - 1.5) * 0.06)))
        out[('(c)', ch)] = m
    c.set_ylabel(r'input sensitivity $\chi$')
    c.set_ylim(0.5, 1.1)
    title(c, 'standardized readout')
    label_ends(c, ends, 4.35, 0.04)
    panel(c, '(c)')
    ends = []
    for k, ch in enumerate(CH4[1:]):
        kap = []
        for L in LS:
            s0 = np.mean([np.mean(r['params']['z_std']) for r in cs[('None', L)]])
            s1 = np.mean([np.mean(r['params']['z_std']) for r in cs[(ch, L)]])
            kap.append((s0 / s1) ** 2)
        ends.append((ch, *line(d, LS, kap, ch, dx=(k - 1) * 0.06)))
        out[('(d)', ch)] = kap
    d.set_yscale('log')
    d.set_ylabel(r'shot factor $\kappa$')
    d.set_ylim(1, 5e4)
    title(d, 'standardized readout')
    for ch, x, y in ends:
        d.annotate(NAME[ch], xy=(x, y), xytext=(4.3, y * (1.6 if ch == 'Pauli' else 0.62 if ch == 'Depol' else 1.0)),
                   va='center', fontsize=7, color=INK,
                   arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4, shrinkA=1, shrinkB=3))
    panel(d, '(d)')
    for ax in (a, b, c, d):
        ax.set_xticks(LS)
        ax.set_xticklabels([f'$L={L}$' for L in LS])
        ax.set_xlim(0.7, 5.1)
    for ax in (c, d):
        ax.set_xlabel('number of re-uploading layers')
    fig.tight_layout(w_pad=2.0, h_pad=1.2)
    fig.savefig(OUT / 'fig_clf.pdf')
    fig.savefig(OUT / 'fig_clf.png')
    return out


def fig_width():
    """Feature scale before training against depth at n = 4, 6, 8 (S6a), with the
    exact theta-averaged prediction of the T1 pair chain. Both are the rms scale
    sqrt(E_theta mean_j Var_a z_j); the measured points use the five theta draws."""
    import json
    W = pickle.load(open(CODE / 'm5_width_diag' / 'results.pkl', 'rb'))
    T = json.load(open(EVID / 'theory' / 't1_results.json'))['chain']
    key = {'None': 'None', 'AD': 'AD', 'Pauli': 'twirl'}
    Ls = np.arange(1, 7)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, ROW), sharey=True)
    out = {}
    for ax, n, tag in zip(axes, (4, 6, 8), ('(a)', '(b)', '(c)')):
        ends = []
        for k, ch in enumerate(CH4):
            rms, se = [], []
            for L in Ls:
                sd = np.array([np.asarray(r['sigma_per_qubit']) for r in W
                               if r['n'] == n and r['L'] == L and r['channel'] == ch])
                v = (sd ** 2).mean(1)
                m = np.sqrt(v.mean())
                rms.append(m)
                se.append(v.std(ddof=1) / np.sqrt(len(v)) / (2 * m))
            rms, se = np.array(rms), np.array(se)
            out[(n, ch)] = rms
            if ch in key:
                th = np.array(T[str(n)][key[ch]]['rms'])
                ax.plot(Ls, th, color=COL[ch], lw=1.1, zorder=2)
                out[(n, ch, 'chain')] = th
            dx = {'None': -0.15, 'AD': -0.05, 'Pauli': 0.05, 'Depol': 0.15}[ch]
            ax.plot(Ls + dx, rms, color=COL[ch], marker=MRK[ch], ms=3.6 if ch == 'Depol' else 4.2,
                    mfc=COL[ch], mec=EDGE[ch], mew=0.6, ls='none', zorder=5 if ch == 'Pauli' else 3)
            ends.append((ch, 6 + dx, rms[-1]))
        ax.set_yscale('log')
        ax.set_ylim(1e-5, 1)
        ax.set_xlim(0.6, 8.9)
        ax.set_xticks(Ls)
        ax.set_xlabel('number of re-uploading layers $L$')
        globals()['title'](ax, f'$n={n}$ qubits')
        label_ends(ax, ends, 6.55, 0.42, logy=True)
        panel(ax, tag)
    axes[0].set_ylabel(r'feature scale $\sigma$ before training')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_width.pdf')
    fig.savefig(OUT / 'fig_width.png')
    return out


def op_cost(ns, acc_mean, acc_exact, tol=0.01):
    """Operational shot cost, as scale_gauge/shots/F1_shots.op_cost: the smallest N_s at
    which the mean accuracy is within tol of the exact accuracy, interpolated on log N_s."""
    ns = np.asarray(ns, float)
    within = np.abs(acc_exact - acc_mean) <= tol
    k = next((i for i in range(len(ns)) if within[i]), None)
    if k is None:
        return np.inf
    if k == 0:
        return ns[0]
    d0, d1 = acc_exact - acc_mean[k - 1] - tol, acc_exact - acc_mean[k] - tol
    f = d0 / (d0 - d1) if d0 != d1 else 1.0
    x0, x1 = np.log10(ns[k - 1]), np.log10(ns[k])
    return 10 ** (x0 + np.clip(f, 0, 1) * (x1 - x0))


def cost_ratio(recs, key, ch, L, B=4000, seed=1):
    """Geometric-mean cost of channel ch over that of None at depth L, with a 95% bootstrap interval
    that resamples initializations jointly (paired by seed). Runs with exact accuracy <= 0.5 or no
    finite cost are dropped, as in the committed summaries."""
    def per_seed(c):
        out = {}
        for r in recs:
            if r['L'] == L and r['channel'] == c:
                v = op_cost(r['Ns'], np.asarray(r[key]).mean(1), r['acc_exact'])
                out[r['seed']] = np.log(v) if (r['acc_exact'] > 0.5 and np.isfinite(v)) else np.nan
        return out
    a, n = per_seed(ch), per_seed('None')
    seeds = sorted(a)
    A = np.array([a[s] for s in seeds]); N = np.array([n[s] for s in seeds])
    est = np.exp(np.nanmean(A) - np.nanmean(N))
    rng = np.random.default_rng(seed); bs = []
    for _ in range(B):
        i = rng.integers(0, len(seeds), len(seeds))
        if np.isfinite(A[i]).any() and np.isfinite(N[i]).any():
            bs.append(np.exp(np.nanmean(A[i]) - np.nanmean(N[i])))
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def fig_shots():
    """Finite shots on MNIST. (a) Shot-sampled test accuracy at L=4 of models trained
    and evaluated with N_s shots per image ('calib' standardization; S5). (b), (c)
    Operational shot cost relative to the noiseless circuit (geometric mean over
    initializations) for exactly trained models (S1, F1_calib) and for models
    trained with N_s = 1000 shots (S5, F1_s5)."""
    import collections
    R = pickle.load(open(CODE / 'm5_shots' / 'results.pkl', 'rb'))
    S1 = pickle.load(open(EVID / 'shots' / 'F1_calib_summary.pkl', 'rb'))
    S5 = pickle.load(open(EVID / 'shots' / 'F1_s5_summary.pkl', 'rb'))
    C = collections.defaultdict(list)
    for r in R:
        C[(r['L'], r['shots'], r['channel'])].append(r['test_acc_shot_mean'])
    NS = (100, 1000, 10000)
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}
    ax = axes[0]
    ends = []
    for k, ch in enumerate(CH4):
        m, e = med_iqr([C[(4, N, ch)] for N in NS])
        xs = np.array(NS, float) * 10 ** ((k - 1.5) * 0.025)
        ax.errorbar(xs, m, yerr=e, color=COL[ch], marker=MRK[ch], ms=4.5, mfc=COL[ch],
                    mec=EDGE[ch], mew=0.6, capsize=2, elinewidth=0.8, lw=1.4, zorder=3)
        ends.append((ch, xs[-1], m[-1]))
        out[('acc', ch)] = m
    ax.set_xscale('log')
    ax.set_xlim(60, 1.5e5)
    ax.set_xticks(NS)
    ax.set_ylim(0.3, 0.92)
    ax.set_xlabel(r'shots per image $N_s$')
    ax.set_ylabel('test accuracy')
    globals()['title'](ax, 'fixed shot budget, $L=4$')
    label_ends(ax, ends, 1.5e4, 0.05)
    panel(ax, '(a)')
    R1 = [r for r in pickle.load(open(EVID / 'shots' / 'F1_calib_results.pkl', 'rb')) if r['dataset'] == 'MNIST']
    R5 = [r for r in pickle.load(open(EVID / 'shots' / 'F1_s5_results.pkl', 'rb')) if r['train_shots'] == 1000]
    for ax, recs, key, rows, skey, tag, ttl in (
            (axes[1], R1, 'acc', [r for r in S1 if r['dataset'] == 'MNIST'], 'calib_ratio', '(b)', 'exact training'),
            (axes[2], R5, 'acc_calib', [r for r in S5 if r['train_shots'] == 1000], 'calib_ratio_None', '(c)',
             'trained with $N_s=10^3$ shots')):
        ends = []
        for ch in ('AD', 'Pauli', 'Depol'):
            y, lo, hi = np.array([cost_ratio(recs, key, ch, L) for L in LS]).T
            ref = np.array([next(r[skey] for r in rows if r['L'] == L and r['channel'] == ch) for L in LS])
            assert np.allclose(y, ref, rtol=1e-6), (tag, ch, y, ref)   # same numbers as the committed summaries
            ends.append((ch, *line(ax, LS, y, ch, [y - lo, hi - y])))
            out[(tag, ch)] = np.array([y, lo, hi])
        ax.axhline(1, color=COL['None'], lw=0.8, ls=(0, (2, 2)), zorder=1)
        ax.text(4.25, 1.15, 'noiseless', va='bottom', fontsize=7, color=INK)
        ax.set_yscale('log')
        ax.set_ylim(0.5, 1e5)
        ax.set_xlim(0.7, 5.3)
        ax.set_xticks(LS)
        ax.set_xlabel('number of re-uploading layers $L$')
        globals()['title'](ax, ttl)
        label_ends(ax, ends, 4.25, 0.35, logy=True)
        panel(ax, tag)
    axes[1].set_ylabel('shot cost relative to noiseless')
    axes[2].sharey(axes[1])
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_shots.pdf')
    fig.savefig(OUT / 'fig_shots.png')
    return out


def fig_sep():
    """Dependence of the AD-twirl separation on p before training (T1b, t1_p_dependence.py): the
    exact theta-averaged feature scales against depth, the AD floor and its last-layer part, and
    the separation depth L* with Eq. (Lstar). The JSON 'br' entry at the last computed depth is
    sigma_br,inf of Table IV; sigma_None in Eq. (Lstar) is the noiseless scale at L = 30."""
    import json
    D = {n: json.load(open(EVID / 'theory' / f't1_p_dependence_n{n}.json')) for n in (4, 8)}
    fig, axes = plt.subplots(1, 3, figsize=(WIDE, ROW))
    out = {}

    # (a) feature scales against depth at n = 4
    ax = axes[0]
    d = D[4]
    ax.plot(np.arange(1, len(d['None']) + 1), d['None'], color=COL['None'], lw=1.1, zorder=2)
    ax.annotate(NAME['None'], xy=(30, d['None'][-1]), xytext=(40, 0.32), fontsize=7, va='center',
                arrowprops=dict(arrowstyle='-', color=MUTED, lw=0.4, shrinkA=1, shrinkB=2))
    for p, lab in (('0.1', '$p=0.1$'), ('0.01', '$p=10^{-2}$'), ('0.001', '$p=10^{-3}$')):
        v = d['p'][p]
        L = np.arange(1, len(v['AD']) + 1)
        n_p = 4 * float(p)
        mk = [i for i in range(len(L)) if L[i] in 2 ** np.arange(12) and L[i] >= 0.25 / n_p]
        for ch, key in (('AD', 'AD'), ('Pauli', 'twirl')):
            ax.plot(L, v[key], color=COL[ch], marker=MRK[ch], ms=3.8, mfc=COL[ch], mec=EDGE[ch],
                    mew=0.5, markevery=mk, lw=1.1, zorder=3)
        Ls = v['L_star']
        ax.plot([Ls, Ls], [v['twirl'][Ls - 1], v['AD'][Ls - 1]], color=INK, lw=0.9, zorder=4)
        ax.text(L[-1], v['twirl'][-1] * 0.6, lab, fontsize=7, ha='center', va='top')
        out[('a', p)] = (Ls, v['AD'][Ls - 1] / v['twirl'][Ls - 1])
    ax.text(1508 * 0.8, 1.0e-3, '$L^*$', fontsize=7, ha='right', va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(0.8, 6e3)
    ax.set_ylim(1e-5, 0.6)
    ax.set_xlabel('number of re-uploading layers $L$')
    ax.set_ylabel(r'feature scale $\sigma$ before training')
    for ch in ('AD', 'Pauli'):
        ax.plot([], [], color=COL[ch], marker=MRK[ch], ms=3.8, mec=EDGE[ch], mew=0.5, lw=1.1,
                label=NAME[ch])
    ax.legend(loc='lower left', frameon=False, handlelength=1.8)
    title(ax, 'feature scales, $n=4$ qubits')
    panel(ax, '(a)')

    # (b) floor of the AD scale and its last-layer part
    ax = axes[1]
    for n, mfc, ls in ((4, COL['AD'], '-'), (8, 'white', (0, (4, 2)))):
        ps = np.array(sorted(map(float, D[n]['p'])))
        br = np.array([D[n]['p'][k]['br'][-1] for k in map(str, ps)])
        fl = np.array([D[n]['p'][k]['floor_closed'] for k in map(str, ps)])
        ax.plot(ps, fl, color=COL['AD'], lw=1.0, ls=ls, zorder=2)
        ax.plot(ps, br, color=COL['AD'], marker=MRK['AD'], ms=4.2, mfc=mfc, mec=COL['AD'],
                mew=0.9, ls='none', zorder=3, label=f'$n={n}$')
        out[('b', n)] = (ps, br, fl)
    ax.axhline(1 / np.sqrt(1e3), color=MUTED, lw=0.8, ls=':', zorder=1)
    ax.text(6e-4, 1 / np.sqrt(1e3) * 1.25, r'$1/\sqrt{N_s}$, $N_s=10^3$', fontsize=6.5, color=INK)
    ax.text(1.3e-3, 1.75e-3, r'$\sigma_{\rm br,\infty}$', fontsize=7.5, color=INK, va='center')
    ax.text(3e-3, 7e-4, r'$\sigma_\infty$', fontsize=7.5, color=INK, va='center')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(6e-4, 0.5)
    ax.set_ylim(3e-4, 0.2)
    ax.set_xlabel('damping strength $p$')
    ax.set_ylabel(r'floor of the AD feature scale')
    ax.legend(loc='lower right', frameon=False, handlelength=1.2)
    title(ax, 'floor of the AD scale at large $L$')
    panel(ax, '(b)')

    # (c) separation depth against p, with Eq. (Lstar); its number is read from the LaTeX .aux
    ax = axes[2]
    import re
    aux = HERE / 'scale_gauge_v5.aux'
    if aux.exists():
        eq = re.search(r'\\newlabel\{eq:Lstar\}\{\{([^}]*)\}', aux.read_text()).group(1)
    else:  # fresh checkout, paper not built yet: the number in the committed PDF
        eq = '45'
        print('scale_gauge_v5.aux not found; labelling Eq. (45) as in the committed PDF')
    for n, mfc, ls in ((4, COL['AD'], '-'), (8, 'white', (0, (4, 2)))):
        ps = np.array(sorted(map(float, D[n]['p'])))
        Lx = np.array([D[n]['p'][k]['L_star'] for k in map(str, ps)])
        br = np.array([D[n]['p'][k]['br'][-1] for k in map(str, ps)])
        Leq = np.log(3 * D[n]['None'][-1] / br) / (n * ps)
        ax.plot(ps, Leq, color=MUTED, lw=1.0, ls=ls, zorder=2,
                label=f'Eq. ({eq}), $n={n}$')
        ax.plot(ps, Lx, color=COL['AD'], marker=MRK['AD'], ms=4.2, mfc=mfc, mec=COL['AD'],
                mew=0.9, ls='none', zorder=3, label=f'$n={n}$')
        out[('c', n)] = (ps, Lx, Leq)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(6e-4, 0.5)
    ax.set_ylim(1, 4e3)
    ax.set_xlabel('damping strength $p$')
    ax.set_ylabel(r'separation depth $L^*$')
    h, l = ax.get_legend_handles_labels()
    ax.legend(h[1::2] + h[0::2], l[1::2] + l[0::2], loc='lower left', frameon=False,
              handlelength=1.8, fontsize=6.5)
    title(ax, r'$(\sigma_{\rm AD}/\sigma_{\rm Pauli})^2=10$ at $L^*$')
    panel(ax, '(c)')

    for ax in axes:
        ax.grid(True, which='major', axis='both')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(OUT / 'fig_sep.pdf')
    fig.savefig(OUT / 'fig_sep.png')
    return out


if __name__ == '__main__':
    for k, v in fig_vr().items():
        print('vr', k, np.round(v, 4))
    for k, v in fig_vqe().items():
        print('vqe', k, np.round(v, 4))
    for k, v in fig_clf().items():
        print('clf', k, np.round(v, 4))
    for k, v in fig_shots().items():
        print('shots', k, np.array2string(np.asarray(v), precision=3))
    for k, v in fig_width().items():
        print('width', k, np.array2string(np.asarray(v), precision=3))
    for k, v in fig_sep().items():
        print('sep', k, [np.array2string(np.asarray(x), precision=4) for x in v])
    print('written to', OUT)
