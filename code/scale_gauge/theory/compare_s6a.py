# -*- coding: utf-8 -*-
"""
T1 against S6a. Compares the exact pair-chain prediction of
t1_scale_ratio.py with the before-training feature scales measured by run_phase1.py width_diag
(m5_width_diag/results.pkl), or, for n = 4 only, by P1.3 (m5_diagnostics/results.pkl, v5 variant).

The chain gives sqrt(E_theta mean_j Var_a z_j). The measured counterpart is formed from the stored
per-qubit standard deviations as sqrt(mean over draws and qubits of sd_j^2), so that the two are the
same quantity; the standard error is over the theta draws. The recorded sigma (mean over qubits of
sd_j, then over draws) is printed alongside.

    python compare_s6a.py                 # m5_width_diag
    python compare_s6a.py diagnostics     # m5_diagnostics (n = 4, L = 1 and 4)
    python compare_s6a.py table           # the chain prediction alone, as a markdown table
    python compare_s6a.py summary         # rates, prefactors and the AD floor against its closed form
    python compare_s6a.py jensen          # E_theta sigma / rms from the density-matrix numerics
"""
import json, os, pickle, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'noise_structure_mnist')
CH = {'AD': 'AD', 'twirl': 'Pauli'}


def measured(recs, n, L, ch):
    sd = np.array([np.asarray(r['sigma_per_qubit']) for r in recs
                   if r.get('n', r.get('n_qubits')) == n and r['L'] == L and r['channel'] == ch])
    if not len(sd):
        return None
    v = (sd ** 2).mean(1)                                    # per draw, mean over qubits
    rms = np.sqrt(v.mean()); se = v.std(ddof=1) / np.sqrt(len(v)) / (2 * rms)
    return rms, se, sd.mean(), len(v)


def prediction_table(T):
    """Markdown table of the chain prediction: sigma_AD, sigma_twirl, their ratio, sigma_None."""
    print('| n | L | σ_AD | σ_twirl | σ_AD/σ_twirl | σ_None |')
    print('|---|---|---|---|---|---|')
    for n in sorted(T, key=int):
        for L in range(1, 7):
            a, t = T[n]['AD']['rms'][L - 1], T[n]['twirl']['rms'][L - 1]
            no = T[n].get('None', {}).get('rms', [float('nan')] * 6)[L - 1]
            print(f'| {n} | {L} | {a:.4f} | {t:.2e} | {a / t:.1f} | {no:.3f} |')


def summary(R):
    """Markdown table: decay rates, prefactor, branching floor against the closed form, sigma_None."""
    T = R['chain']
    print('| n | Λ_n Markov | Λ_n exact | growth 1/Λ_n | C_n | σ_br(L=1) | σ_br(L=6) | closed form | σ_None(L=6) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for n in sorted(T, key=int):
        a, t = np.array(T[n]['AD']['rms']), np.array(T[n]['twirl']['rms'])
        lam = (t[5] / t[2]) ** (1 / 3)                        # exact rate, L = 3 -> 6
        C = np.exp(np.mean(np.log(t[2:]) - np.arange(3, 7) * np.log(lam)))
        br = np.sqrt(a ** 2 - t ** 2)
        no = T[n].get('None', {}).get('rms', [np.nan] * 6)[5]
        fl = f"{R['floor'][n]:.4f}" if n in R.get('floor', {}) else '—'   # from t1_scale_ratio.py lead
        print(f"| {n} | {R['decay'][n]['Lambda_data']:.3f} | {lam:.3f} | {1 / lam:.2f} | {C:.3f} | "
              f"{br[0]:.4f} | {br[5]:.4f} | {fl} | {no:.3f} |")


def jensen(R):
    """Markdown table from the density-matrix part: E_theta sigma / rms, and the chain's rms
    against the numerics' rms (a check of the chain with independent theta draws)."""
    E, T = R['exact'], R['chain']
    print('| n | L | E[σ]/rms AD | E[σ]/rms twirl | E[σ]/rms None | rms numerics / chain, AD | twirl |')
    print('|---|---|---|---|---|---|---|')
    for n in sorted(E, key=int):
        for L in sorted(E[n], key=int):
            row = E[n][L]
            f = {c: row[c][0] / row[c][2] for c in ('AD', 'twirl', 'None')}
            q = {c: row[c][2] / T[n][c]['rms'][int(L) - 1] for c in ('AD', 'twirl')}
            print(f"| {n} | {L} | {f['AD']:.2f} | {f['twirl']:.2f} | {f['None']:.2f} | {q['AD']:.2f} | {q['twirl']:.2f} |")


def main():
    which = (sys.argv[1:] or ['width_diag'])[0]
    R = json.load(open(os.path.join(HERE, 't1_results.json'))); T = R['chain']
    if which == 'table':
        return prediction_table(T)
    if which == 'summary':
        return summary(R)
    if which == 'jensen':
        return jensen(R)
    if which == 'diagnostics':
        recs = [r for r in pickle.load(open(os.path.join(SRC, 'm5_diagnostics', 'results.pkl'), 'rb'))
                if r['variant'] == 'v5']
        for r in recs:
            r['n'] = 4
    else:
        recs = pickle.load(open(os.path.join(SRC, 'm5_width_diag', 'results.pkl'), 'rb'))
    print(f"{'n':>2} {'L':>2} | {'AD meas':>17} {'AD chain':>9} | {'twirl meas':>19} {'twirl chain':>11} | "
          f"{'ratio meas':>10} {'ratio chain':>11} | draws")
    for n in sorted({r['n'] for r in recs}):
        for L in sorted({r['L'] for r in recs if r['n'] == n}):
            m = {c: measured(recs, n, L, CH[c]) for c in CH}
            if None in m.values() or str(n) not in T:
                continue
            ch = {c: T[str(n)][c]['rms'][L - 1] for c in CH}
            rm = m['AD'][0] / m['twirl'][0]
            rse = rm * np.hypot(m['AD'][1] / m['AD'][0], m['twirl'][1] / m['twirl'][0])
            print(f"{n:>2} {L:>2} | {m['AD'][0]:.4f}+-{m['AD'][1]:.4f}  {ch['AD']:.4f}   | "
                  f"{m['twirl'][0]:.3e}+-{m['twirl'][1]:.1e}  {ch['twirl']:.3e}   | "
                  f"{rm:8.1f}+-{rse:<6.1f} {ch['AD'] / ch['twirl']:9.1f} | {m['AD'][3]}")


if __name__ == '__main__':
    main()
