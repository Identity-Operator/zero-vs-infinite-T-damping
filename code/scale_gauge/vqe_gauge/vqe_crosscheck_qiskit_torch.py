"""V1: cross-check the torch VQE energy of vqe_gauge_test.py
(placement 'mid') against the reference Qiskit implementation
code/old_code/TFIM_VQE_Noise.py, noise_model "2" (channel on both qubits after every
CNOT), for None, AD, AD_reverse, Pauli (Pauli-twirled AD) and Depol (Clifford-twirled AD).

Parameter maps. Reference: par = [zs (T*n), xs (T*n), ys (n)], RZ(2 zs/T), RX(2 xs/T), RY(ys).
vqe_gauge_test: theta = [RY_q (n)] then, per block s, [RZ per pair (n), RX per qubit (n)].
vqe_gauge_test has no Depol channel; it is added here by wrapping `kraus`, leaving the
archived script unchanged. Its depolarizing parameter follows the reference:
lam = (g + 2 - 2 sqrt(1-g)) / 3 for rho -> (1-lam) rho + lam I/2.
"""
import os
import sys
import types

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'old_code'))
_m = types.ModuleType('qiskit_algorithms'); _o = types.ModuleType('qiskit_algorithms.optimizers')
_o.SPSA = None      # only the optimizer needs qiskit_algorithms; energies do not
sys.modules['qiskit_algorithms'] = _m; sys.modules['qiskit_algorithms.optimizers'] = _o
from TFIM_VQE_Noise import TFIM_VQE_NOISE  # noqa: E402

import vqe_gauge_test as v  # noqa: E402

_kraus = v.kraus


def kraus(channel, g):
    if channel == 'Depol':
        lam = (g + 2 - 2 * np.sqrt(1 - g)) / 3
        return [np.sqrt(1 - 3 * lam / 4) * v.I2] + [np.sqrt(lam / 4) * P for P in (v.X, v.Y, v.Z)]
    return _kraus(channel, g)


v.kraus = kraus
MAP = {'None': 'none', 'AD': 'AD', 'AD_reverse': 'revAD', 'Pauli': 'twirlAD', 'Depol': 'Depol'}
n, T = v.N, v.BLOCKS


def to_theta(par):
    zs, xs, ys = par[:T * n], par[T * n:2 * T * n], par[2 * T * n:]
    th = list(ys)
    for s in range(T):
        th += [2 * zs[s * n + k] / T for k in range(n)]
        th += [2 * xs[s * n + q] / T for q in range(n)]
    return th


rng = np.random.default_rng(2026)
worst = 0.0
rows = []
for g in (0.01, 0.05, 0.1, 0.2):
    ref = TFIM_VQE_NOISE(n, T, noise_str=g, use_density_matrix=True)
    pars = [rng.uniform(-np.pi, np.pi, 2 * T * n + n) for _ in range(5)]
    theta = torch.tensor([to_theta(p) for p in pars])
    for name, mine in MAP.items():
        with torch.no_grad():
            E_mine = v.energy(theta, mine, g, 'mid').numpy()
        E_ref = np.array([ref.compute_energy(p, name) for p in pars])
        d = float(np.max(np.abs(E_mine - E_ref)))
        worst = max(worst, d)
        rows.append((g, name, d))
        print(f"g={g:<5} {name:10s} max|E_torch - E_qiskit| over 5 random par = {d:.2e}", flush=True)
print(f"worst {worst:.2e}; {'PASS' if worst <= 1e-12 else 'FAIL'} at tolerance 1e-12")
np.save('vqe_crosscheck_qiskit_torch.npy', np.array(rows, dtype=object), allow_pickle=True)
