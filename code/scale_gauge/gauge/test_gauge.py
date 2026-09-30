# -*- coding: utf-8 -*-
"""
G1 test suite: when is reversing amplitude damping (damping toward |1>
instead of |0>) a reparametrisation of the model?  numpy, CPU, float64.

For every model the frame search is run over ALL Pauli frames, unitary ('U') and
antiunitary ('A', Pauli x complex conjugation; time reversal is the case P = Y...Y).
Where a gauge exists, the explicit identity y_rev(x; theta) = s * y_AD(x; theta') is
checked on random parameters and inputs. Where none exists, the failure is backed by a
class invariant when one is available (output range; optimised VQE energy).

    python test_gauge.py            # writes results.json next to this file
"""
import json, os, sys, time
import numpy as np
from scipy.optimize import minimize
from gauge_lib import (PX, PY, PZ, I2, PAULI, embed, cnot, kraus, same_channel, rot, conj_t,
                       random_params, simulate, outputs, search, reparametrise, readout_signs)

P_NOISE = 0.3
HERE = os.path.dirname(os.path.abspath(__file__))
AX = {'rx': (1, 0, 0), 'ry': (0, 1, 0), 'rz': (0, 0, 1)}


# ---------------------------------------------------------------- models
def ring(n):
    E = np.eye(2 ** n, dtype=complex)
    for j in range(n):
        E = cnot(j, (j + 1) % n, n) @ E
    return E


def classifier(n=3, L=2, enc='rx', placement='after', order='train_first'):
    """Our layout: per layer su2 -> entangler -> noise -> encoding -> noise; trailing su2 -> noise.
    placement='inside': noise after each CNOT of the ring instead of after the whole ring.
    order='enc_first': the encoding comes first in every layer (no trainable gate before the
    first noise), which exposes the input state |0> to the frame."""
    axis = AX[enc] if isinstance(enc, str) else enc
    allq = list(range(n)); c = []
    def ent():
        if placement == 'after':
            return [{'kind': 'fixed', 'U': ring(n), 'name': 'ring'}, {'kind': 'noise', 'qubits': allq}]
        out = []
        for j in range(n):
            out += [{'kind': 'fixed', 'U': cnot(j, (j + 1) % n, n), 'name': f'cx{j}'},
                    {'kind': 'noise', 'qubits': [j, (j + 1) % n]}]
        return out
    for _ in range(L):
        enc_el = [{'kind': 'enc', 'axes': {q: axis for q in allq}}, {'kind': 'noise', 'qubits': allq}]
        tr = [{'kind': 'su2', 'qubits': allq}] + ent()
        c += (enc_el + tr) if order == 'enc_first' else (tr + enc_el)
    c += [{'kind': 'su2', 'qubits': allq}, {'kind': 'noise', 'qubits': allq}]
    obs = [embed({q: PZ}, n) for q in allq]
    return dict(name='classifier', n=n, circuit=c, obs=obs, readout='affine', has_x=True)


def vanrossum(L=2, readout='fixed'):
    """van Rossum et al. 2510.24050 App. B1: |0>, W = RZ RY RZ (a general SU(2)), S(x) = RX(x),
    noise after every S and every W (their Eq. 17), f = <Z> used directly (no affine map)."""
    c = []
    for _ in range(L):
        c += [{'kind': 'su2', 'qubits': [0]}, {'kind': 'noise', 'qubits': [0]},
              {'kind': 'enc', 'axes': {0: AX['rx']}}, {'kind': 'noise', 'qubits': [0]}]
    c += [{'kind': 'su2', 'qubits': [0]}, {'kind': 'noise', 'qubits': [0]}]
    return dict(name='vanrossum_1q', n=1, circuit=c, obs=[PZ], readout=readout, has_x=True)


def tfim(n=3, J=1.0, h=0.5):
    H = np.zeros((2 ** n, 2 ** n), complex)
    for i in range(n):
        H += -J * embed({i: PZ, (i + 1) % n: PZ}, n) - h * embed({i: PX}, n)
    return H


def vqe(n=3, T=4, placement='post'):
    """van Rossum Sec. IV: RY on each qubit, then T blocks of RZZ on the PBC ring and RX on each
    qubit; single-qubit gates noiseless. 'post': noise after each whole RZZ; 'mid': RZZ =
    CNOT . RZ_t . CNOT with noise after each CNOT (their Fig. 4b)."""
    c = [{'kind': 'rot', 'gen': 'Y', 'qubits': [q]} for q in range(n)]
    for _ in range(T):
        for i in range(n):
            j = (i + 1) % n
            if placement == 'post':
                c += [{'kind': 'rot', 'gen': 'ZZ', 'qubits': [i, j]}, {'kind': 'noise', 'qubits': [i, j]}]
            else:
                c += [{'kind': 'fixed', 'U': cnot(i, j, n), 'name': f'cx{i}{j}'},
                      {'kind': 'noise', 'qubits': [i, j]},
                      {'kind': 'rot', 'gen': 'Z', 'qubits': [j]},
                      {'kind': 'fixed', 'U': cnot(i, j, n), 'name': f'cx{i}{j}'},
                      {'kind': 'noise', 'qubits': [i, j]}]
        c += [{'kind': 'rot', 'gen': 'X', 'qubits': [q]} for q in range(n)]
    return dict(name=f'vqe_{placement}', n=n, circuit=c, obs=[tfim(n)], readout='fixed', has_x=False)


# ---------------------------------------------------------------- checks
def gauge_case(model, rng, n_trials=12):
    n, c, obs = model['n'], model['circuit'], model['obs']
    res = {}
    for t in ('U', 'A'):
        ok, frames = search(c, n, obs, model['readout'], t, p=P_NOISE)
        entry = {'exists': ok}
        if ok:
            err = 0.0
            for _ in range(n_trials):
                th = random_params(c, rng)
                x = rng.uniform(0, 2 * np.pi, size=n)
                thp = reparametrise(c, th, frames, t, n)
                s = readout_signs(frames, obs, t)
                yr = outputs(simulate(c, th, x, 'ADrev', P_NOISE, n), obs)
                ya = outputs(simulate(c, thp, x, 'AD', P_NOISE, n), obs)
                err = max(err, np.abs(yr - s * ya).max())
            entry.update(identity_err=float(err), signs=[int(v) for v in s],
                         frame_after_first_block=''.join('IXYZ'[i] for i in frames[1]))
        res[t] = entry
    return res


def channel_checks(rng):
    K, Kr = kraus('AD', P_NOISE), kraus('ADrev', P_NOISE)
    out = {
        'ADrev == X.AD.X': same_channel(Kr, [PX @ k @ PX for k in K]),
        'ADrev == Y.AD.Y': same_channel(Kr, [PY @ k @ PY for k in K]),
        'AD == Z.AD.Z (phase covariant)': same_channel(K, [PZ @ k @ PZ for k in K]),
        'time reversal maps AD to ADrev: Y conj(K) Y': same_channel(Kr, [PY @ k.conj() @ PY for k in K]),
        'twirl == frame average (1/4) sum_P P.AD.P':
            bool(np.allclose(sum(_choi_of([P @ k @ P for k in K]) for P in PAULI) / 4,
                             _choi_of(kraus('twirl', P_NOISE)), atol=1e-12)),
    }
    worst = 0.0
    for _ in range(50):
        axis = rng.normal(size=3); axis /= np.linalg.norm(axis); a = rng.uniform(-6, 6)
        R = rot(axis, a); M = PY @ R.conj() @ PY
        worst = max(worst, np.abs(M - R).max())
    out['time reversal leaves every single-qubit rotation invariant (max err)'] = float(worst)
    return out


def _choi_of(K):
    from gauge_lib import choi
    return choi(K)


def vanrossum_range(rng, L=2):
    """Output range invariant of the single-qubit model: the last channel sits right before
    <Z>, so AD gives (1-p) z + p in [2p-1, 1], ADrev gives [-1, 1-2p], the twirl [-(1-p), 1-p]."""
    m = vanrossum(L); c = m['circuit']; res = {}
    for ch in ('AD', 'ADrev', 'twirl'):
        best_hi, best_lo = -9, 9
        for _ in range(400):
            th = random_params(c, rng); x = rng.uniform(0, 2 * np.pi, size=1)
            f = outputs(simulate(c, th, x, ch, P_NOISE, 1), m['obs'])[0]
            best_hi, best_lo = max(best_hi, f), min(best_lo, f)
        res[ch] = {'sampled_max': float(best_hi), 'sampled_min': float(best_lo)}
    p = P_NOISE
    res['analytic_bounds'] = {'AD': [2 * p - 1, 1.0], 'ADrev': [-1.0, 1 - 2 * p], 'twirl': [-(1 - p), 1 - p]}
    # Constructive witnesses that the upper bounds are attained (random sampling rarely hits
    # them). AD: identity blocks and x = 0 keep |0>, the fixed point, so f = 1. ADrev: the
    # first block X parks the state in |1>, the reversed channel's fixed point; x = 0 keeps
    # it there; the last block X maps it to |0>, and the final channel gives (1-p) - p.
    def blocks(first, last):
        th = random_params(c, rng); su = [i for i, el in enumerate(c) if el['kind'] == 'su2']
        for i in su:
            th[i] = {0: I2.copy()}
        th[su[0]] = {0: first}; th[su[-1]] = {0: last}
        return th
    x0 = np.zeros(1)
    res['witness_AD_max'] = float(outputs(simulate(c, blocks(I2, I2), x0, 'AD', p, 1), m['obs'])[0])
    res['witness_ADrev_max'] = float(outputs(simulate(c, blocks(PX, PX), x0, 'ADrev', p, 1), m['obs'])[0])
    return res


def vqe_minima(placement, rng, restarts=12, T=4):
    """Class invariant for the VQE: the lowest energy reachable. Equal under a gauge."""
    m = vqe(T=T, placement=placement); c, n, H = m['circuit'], m['n'], m['obs'][0]
    from gauge_lib import gen_full, expm_pauli
    out = {}
    for ch in ('AD', 'ADrev'):
        K = kraus(ch, P_NOISE)
        prog = []                                   # precompiled: ('rot', G) / ('U', U) / ('noise', [Ks])
        for el in c:
            if el['kind'] == 'rot':
                prog.append(('rot', gen_full(el['gen'], el['qubits'], n)))
            elif el['kind'] == 'fixed':
                prog.append(('U', el['U']))
            elif el['kind'] == 'noise':
                prog.append(('noise', [[embed({q: k}, n) for k in K] for q in el['qubits']]))
        nrot = sum(1 for op in prog if op[0] == 'rot')
        def E(v):
            rho = np.zeros((2 ** n, 2 ** n), complex); rho[0, 0] = 1; k = 0
            for kind, obj in prog:
                if kind == 'rot':
                    U = expm_pauli(obj, v[k]); k += 1; rho = U @ rho @ U.conj().T
                elif kind == 'U':
                    rho = obj @ rho @ obj.conj().T
                else:
                    for Ks in obj:
                        rho = sum(Kk @ rho @ Kk.conj().T for Kk in Ks)
            return np.trace(H @ rho).real
        idx = range(nrot)
        best = np.inf
        for r in range(restarts):
            v0 = np.random.default_rng(1000 + r).uniform(-np.pi, np.pi, size=len(idx))
            best = min(best, minimize(E, v0, method='L-BFGS-B').fun)
        out[ch] = float(best)
    return out


def main():
    rng = np.random.default_rng(20260924)
    t0 = time.time(); R = {}
    print('channel identities'); R['channels'] = channel_checks(rng)
    for k, v in R['channels'].items(): print(f'   {k}: {v}')
    cases = {
        'classifier rx, noise after ring':      classifier(enc='rx'),
        'classifier ry, noise after ring':      classifier(enc='ry'),
        'classifier rz, noise after ring':      classifier(enc='rz'),
        'classifier generic axis, after ring':  classifier(enc=tuple(np.array([.3, -.5, .8]) / np.linalg.norm([.3, -.5, .8]))),
        'classifier rx, noise INSIDE ring':     classifier(enc='rx', placement='inside'),
        'classifier rz, noise INSIDE ring':     classifier(enc='rz', placement='inside'),
        'classifier rx, encoding first':        classifier(enc='rx', order='enc_first'),
        'vanRossum 1q, f=<Z> (no sign freedom)': vanrossum(readout='fixed'),
        'vanRossum 1q, learnable sign':         vanrossum(readout='affine'),
        'VQE, noise after whole RZZ':           vqe(placement='post'),
        'VQE, noise after each CNOT':           vqe(placement='mid'),
    }
    R['cases'] = {}
    print(f"\n{'case':40s} {'unitary Pauli frame':>26s} {'antiunitary (time reversal)':>30s}")
    for name, m in cases.items():
        r = gauge_case(m, rng); R['cases'][name] = r
        f = lambda e: (f"gauge, err {e['identity_err']:.1e}" + (' (sign -)' if -1 in e['signs'] else '')
                       if e['exists'] else 'NO GAUGE')
        print(f'{name:40s} {f(r["U"]):>26s} {f(r["A"]):>30s}')
    print('\nvan Rossum single-qubit output range (class invariant):')
    R['vanrossum_range'] = vanrossum_range(rng)
    for k, v in R['vanrossum_range'].items(): print(f'   {k}: {v}')
    print('\nVQE lowest energy reached (class invariant; ground state -3.232 noiseless):')
    R['vqe_minima'] = {pl: vqe_minima(pl, rng) for pl in ('post', 'mid')}
    for k, v in R['vqe_minima'].items(): print(f'   {k}: AD {v["AD"]:.5f}  ADrev {v["ADrev"]:.5f}  diff {v["ADrev"]-v["AD"]:+.2e}')

    # expectations (the assertions that make this a test)
    C = R['cases']; tol = 1e-12
    expect = [
        ('rx after: unitary gauge',        C['classifier rx, noise after ring']['U']['exists'] and C['classifier rx, noise after ring']['U']['identity_err'] < tol),
        ('rz after: no unitary gauge',     not C['classifier rz, noise after ring']['U']['exists']),
        ('rx/ry/rz/generic after: time-reversal gauge',
         all(C[k]['A']['exists'] and C[k]['A']['identity_err'] < tol for k in
             ['classifier rx, noise after ring', 'classifier ry, noise after ring',
              'classifier rz, noise after ring', 'classifier generic axis, after ring'])),
        ('noise inside ring: no gauge of either type',
         not any(C[k][t]['exists'] for k in ['classifier rx, noise INSIDE ring', 'classifier rz, noise INSIDE ring'] for t in 'UA')),
        ('encoding first: no gauge of either type', not any(C['classifier rx, encoding first'][t]['exists'] for t in 'UA')),
        ('vanRossum f=<Z>: no gauge', not any(C['vanRossum 1q, f=<Z> (no sign freedom)'][t]['exists'] for t in 'UA')),
        ('vanRossum with sign: gauge up to f -> -f',
         C['vanRossum 1q, learnable sign']['U']['exists'] and C['vanRossum 1q, learnable sign']['U']['signs'] == [-1]
         and C['vanRossum 1q, learnable sign']['U']['identity_err'] < tol),
        ('VQE post: gauge', any(C['VQE, noise after whole RZZ'][t]['exists'] and C['VQE, noise after whole RZZ'][t]['identity_err'] < tol for t in 'UA')),
        ('VQE mid: no gauge', not any(C['VQE, noise after each CNOT'][t]['exists'] for t in 'UA')),
        ('vanRossum range: AD attains 1; ADrev attains but never exceeds 1-2p',
         abs(R['vanrossum_range']['witness_AD_max'] - 1) < 1e-12
         and abs(R['vanrossum_range']['witness_ADrev_max'] - (1 - 2 * P_NOISE)) < 1e-12
         and R['vanrossum_range']['ADrev']['sampled_max'] <= 1 - 2 * P_NOISE + 1e-9),
        ('ry after: unitary gauge (Y frame)', C['classifier ry, noise after ring']['U']['exists']
         and C['classifier ry, noise after ring']['U']['identity_err'] < tol),
        ('VQE post minima equal', abs(R['vqe_minima']['post']['AD'] - R['vqe_minima']['post']['ADrev']) < 1e-4),
        ('channel identities', all(v is True for k, v in R['channels'].items() if isinstance(v, bool))
                               and R['channels']['time reversal leaves every single-qubit rotation invariant (max err)'] < 1e-14),
    ]
    R['expectations'] = {k: bool(v) for k, v in expect}
    print('\nEXPECTATIONS')
    for k, v in expect: print(f"   {'PASS' if v else 'FAIL'}  {k}")
    R['p'] = P_NOISE; R['runtime_s'] = time.time() - t0
    json.dump(R, open(os.path.join(HERE, 'results.json'), 'w'), indent=1, default=str)
    print(f"\n{sum(bool(v) for _, v in expect)}/{len(expect)} expectations hold; results.json written; {R['runtime_s']:.0f}s")
    sys.exit(0 if all(v for _, v in expect) else 1)


if __name__ == '__main__':
    main()
