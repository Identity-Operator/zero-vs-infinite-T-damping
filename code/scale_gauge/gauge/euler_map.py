# -*- coding: utf-8 -*-
"""
Explicit parameter map of the AD <-> AD_flip gauge for the TorchCirc classifier (check of
Corollary 1). Layout: per layer, B = Rz(t2) Ry(t1) Rz(t0) on every qubit ->
CNOT ring E -> channel -> R_X(a) -> channel; trailing blocks -> channel; readout Z_j.

Two gauges, each giving z_flip(x; theta') = -z_AD(x; theta) for every input x:
  TR (time reversal Theta = Y^n K): Theta B Theta^-1 = B for every SU(2) block, Theta E Theta^-1 = E Q
     with Q = E^dag Y^n E Y^n (a Pauli string), Theta AD Theta^-1 = AD_flip, Theta R_X Theta^-1 = R_X,
     Theta |0> ~ |1>. So B'_l = Q B_l (l < L), B'_0 = Q B_0 Y, B'_L = B_L.
  X  (unitary frame F = X^n): F AD F = AD_flip, F R_X F = R_X, F E F = E Q_X with Q_X = E^dag X^n E X^n,
     F B F = Rz(-t2) Ry(-t1) Rz(-t0). So B'_l = Q_X (F B_l F) (l < L), B'_0 = Q_X X B_0, B'_L = F B_L F.
A Pauli multiplying a ZYZ triple from the left or right maps each angle to +-angle + k pi (rules
below, checked numerically). The map is therefore theta'_k = s_k theta_k + m_k pi, with s_k = +-1.

    python euler_map.py      # checks the rules, the maps against TorchCirc, prints the n=4, L=4 maps
"""
import itertools, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'noise_structure_mnist'))
from torch_circ import TorchCirc  # noqa: E402  (sets the default dtype to float64)

I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1.0, -1.0]).astype(complex); PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
rz = lambda t: np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])
ry = lambda t: np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], complex)
block = lambda a, b, c: rz(c) @ ry(b) @ rz(a)                     # Rz(t2) Ry(t1) Rz(t0), (a, b, c) = (t0, t1, t2)

# (sign, shift/pi) per angle (a, b, c) for P.B (left) and B.P (right), up to a global phase
LEFT = {'I': ((1, 0), (1, 0), (1, 0)), 'Z': ((1, 0), (1, 0), (1, 1)), 'X': ((1, 1), (-1, 1), (-1, 0)), 'Y': ((1, 0), (1, 1), (-1, 0))}
RIGHT = {'I': ((1, 0), (1, 0), (1, 0)), 'Z': ((1, 1), (1, 0), (1, 0)), 'X': ((-1, 1), (1, 1), (1, 0)), 'Y': ((-1, 0), (1, 1), (1, 0))}
CONJ_X = ((-1, 0), (-1, 0), (-1, 0))                                 # X B X


def compose(r2, r1):
    """Apply map r1, then r2, each ((s, m), ...): t -> s t + m pi."""
    return tuple((s2 * s1, s2 * m1 + m2) for (s1, m1), (s2, m2) in zip(r1, r2))


def apply(rule, abc):
    return tuple(s * t + m * np.pi for (s, m), t in zip(rule, abc))


def same_up_to_phase(U, V):
    return abs(abs(np.trace(U.conj().T @ V)) / 2 - 1) < 1e-12


def check_rules(rng):
    for P, M in PAULI.items():
        a, b, c = rng.uniform(0, 2 * np.pi, 3)
        assert same_up_to_phase(M @ block(a, b, c), block(*apply(LEFT[P], (a, b, c)))), ('left', P)
        assert same_up_to_phase(block(a, b, c) @ M, block(*apply(RIGHT[P], (a, b, c)))), ('right', P)
    a, b, c = rng.uniform(0, 2 * np.pi, 3)
    assert same_up_to_phase(X @ block(a, b, c) @ X, block(*apply(CONJ_X, (a, b, c))))
    Bq = block(a, b, c); assert np.allclose(Y @ Bq.conj() @ Y, Bq)  # time reversal leaves SU(2) blocks invariant


def ring_frame(n, F):
    """Pauli string Q = E^dag F^n E F^n for the TorchCirc ring, as letters per qubit."""
    def cnot(c, t):
        P0, P1 = np.diag([1, 0]).astype(complex), np.diag([0, 1]).astype(complex)
        k = lambda ops: __import__('functools').reduce(np.kron, ops)
        return k([P0 if q == c else I2 for q in range(n)]) + k([P1 if q == c else (X if q == t else I2) for q in range(n)])
    E = np.eye(2 ** n, dtype=complex)
    for j in range(n):
        E = cnot(j, (j + 1) % n) @ E
    Fn = __import__('functools').reduce(np.kron, [PAULI[F]] * n)
    Q = E.conj().T @ Fn @ E @ Fn
    for letters in itertools.product('IXYZ', repeat=n):
        Pm = __import__('functools').reduce(np.kron, [PAULI[l] for l in letters])
        if abs(abs(np.trace(Pm.conj().T @ Q)) / 2 ** n - 1) < 1e-10:
            return letters
    raise AssertionError


def theta_map(n, L, gauge):
    """Per block (l, j): rule ((s, m) for t0, t1, t2) with theta' = s theta + m pi."""
    Q = ring_frame(n, 'Y' if gauge == 'TR' else 'X'); rules = {}
    for l in range(L + 1):
        for j in range(n):
            r = ((1, 0), (1, 0), (1, 0)) if gauge == 'TR' else CONJ_X                # Theta B = B;  F B F
            if gauge == 'X' and l == 0:
                r = LEFT['X']                                                      # (F B_0 F) X = X B_0
            if gauge == 'TR' and l == 0:
                r = RIGHT['Y']                                                     # B_0 Y
            if l < L:
                r = compose(LEFT[Q[j]], r)                                         # Q B
            rules[(l, j)] = r
    return rules


def to_theta(theta, rules, n, L):
    th = theta.copy().reshape(L + 1, n, 3)
    out = np.empty_like(th)
    for (l, j), r in rules.items():
        out[l, j] = apply(r, th[l, j])
    return np.mod(out, 2 * np.pi).reshape(-1)


def main():
    rng = np.random.default_rng(0); check_rules(rng); print('Euler rules for P.B, B.P and X B X: checked (up to global phase)')
    worst = 0.0
    for n, L in ((2, 1), (3, 2), (4, 1), (4, 3), (4, 4)):
        ad = TorchCirc(n, L, noisetype='AD', p_noise=0.3, device='cpu')
        fl = TorchCirc(n, L, noisetype='AD_flip', p_noise=0.3, device='cpu')
        a = torch.as_tensor(rng.uniform(0, 2 * np.pi, (5, n)))
        for gauge in ('TR', 'X'):
            rules = theta_map(n, L, gauge)
            for _ in range(3):
                th = rng.uniform(0, 2 * np.pi, ad.n_theta)
                with torch.no_grad():
                    z = ad.features(a, torch.as_tensor(th)).numpy()
                    zf = fl.features(a, torch.as_tensor(to_theta(th, rules, n, L))).numpy()
                err = np.abs(zf + z).max(); worst = max(worst, err)
                assert err < 1e-10, (n, L, gauge, err)
    print(f'z_flip(theta\') = -z_AD(theta) for both gauges, n in 2-4, L in 1-4: max error {worst:.1e}')
    n, L = 4, 4
    for gauge in ('TR', 'X'):
        Q = ring_frame(n, 'Y' if gauge == 'TR' else 'X'); rules = theta_map(n, L, gauge)
        print(f'\n{gauge} gauge, n={n}, L={L}: ring frame Q = {"".join(Q)}; theta\'_k = s theta_k + m pi, listed (s, m) for (t0, t1, t2):')
        for l in range(L + 1):
            print(f'  layer {l}: ' + '  '.join(f'q{j} ' + ','.join(f'{"+" if s > 0 else "-"}{m % 2}' for s, m in rules[(l, j)]) for j in range(n)))
        s = np.array([[r[k][0] for k in range(3)] for r in rules.values()])
        print(f'  angles with sign flip: {int((s < 0).sum())} of {s.size}')


if __name__ == '__main__':
    main()
