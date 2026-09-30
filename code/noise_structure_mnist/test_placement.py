# -*- coding: utf-8 -*-
"""
Tests for TorchCirc's noise_placement option.
CPU only, small: about 0.3 GB of RAM and a minute.

    python test_placement.py

1. The default placement is unchanged: TorchCirc on the CPU, with a committed
   m5_depth model's parameters (L=4, AD, seed 256), reproduces its test
   predictions on the first 500 test images.
2. Time-reversal gauge, default placement: with Theta = (-iY)^n K, the map
   W -> P'' (Y conj(W) Y) on every trainable block, P'' = C^dag Y^n C Y^n for
   blocks followed by the entangler, the first block right-multiplied by X,
   gives z_AD_flip(a; theta) = -z_AD(a; theta~) to 1e-12, for Rx encoding.
3. The same explicit construction fails for noise_placement='inside_entangler'
   (difference > 1e-3), where noise sits between the CNOTs of the entangler.
4. Without an entangler (entangling='none') the two placements coincide.
Test 3 shows that this construction fails, not that no gauge exists; the
VQE test in code/scale_gauge/vqe_gauge shows a landscape-level
direction effect for the analogous placement.
"""
import itertools
import os
import pickle

import numpy as np
import torch

from torch_circ import TorchCirc

HERE = os.path.dirname(os.path.abspath(__file__))
N, L = 4, 2
I2 = np.eye(2)
X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1.0, -1.0]).astype(complex)
PAULI = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}


def kron(ms):
    out = np.eye(1)
    for m in ms:
        out = np.kron(out, m)
    return out


def ring_cnots(n):
    """The entangler as TorchCirc applies it: CNOT(j, j+1 mod n) for j = 0..n-1, in order."""
    C = np.eye(2 ** n)
    for c in range(n):
        t = (c + 1) % n
        P0 = kron([np.diag([1, 0]) if k == c else I2 for k in range(n)])
        P1 = kron([np.diag([0, 1]) if k == c else (X if k == t else I2) for k in range(n)])
        C = (P0 + P1) @ C
    return C


def local_factors(M, n):
    """Write a Pauli string M (up to phase) as a list of single-qubit Paulis."""
    for combo in itertools.product('IXYZ', repeat=n):
        P = kron([PAULI[c] for c in combo])
        if abs(abs(np.trace(P.conj().T @ M)) / 2 ** n - 1) < 1e-12:
            return [PAULI[c] for c in combo]
    raise ValueError("not a Pauli string")


def rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def ry(t):
    return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])


def block(t0, t1, t2):
    """TorchCirc applies Rz(t0), then Ry(t1), then Rz(t2)."""
    return rz(t2) @ ry(t1) @ rz(t0)


def zyz_angles(U):
    """(t0, t1, t2) with block(t0, t1, t2) = U up to a global phase."""
    V = U / np.sqrt(np.linalg.det(U))
    a, b = V[0, 0], V[1, 0]
    t1 = 2 * np.arctan2(abs(b), abs(a))
    s, d = -2 * np.angle(a), 2 * np.angle(b)          # s = t2 + t0, d = t2 - t0
    return (s - d) / 2, t1, (s + d) / 2


def time_reversed_theta(theta, n, L):
    """theta~ such that z_AD(a; theta~) = -z_AD_flip(a; theta) for the default placement."""
    C = ring_cnots(n)
    YY = kron([Y] * n)
    Pdd = local_factors(C.conj().T @ YY @ C @ YY, n)
    th = np.asarray(theta).reshape(L + 1, n, 3)
    out = np.empty_like(th)
    for ell in range(L + 1):
        for j in range(n):
            W = Y @ block(*th[ell, j]).conj() @ Y
            if ell < L:
                W = Pdd[j] @ W
            if ell == 0:
                W = W @ X
            out[ell, j] = zyz_angles(W)
    return out.reshape(-1)


def features(noisetype, theta, a, placement):
    sim = TorchCirc(N, L, noisetype=noisetype, p_noise=0.3, device='cpu',
                    noise_placement=placement)
    with torch.no_grad():
        return sim.features(torch.as_tensor(a), torch.as_tensor(theta)).numpy()


def gauge_error(placement, seed=0):
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, 3 * N * (L + 1))
    a = rng.uniform(0, 2 * np.pi, (5, N))
    z_flip = features('AD_flip', theta, a, placement)
    z_ad = features('AD', time_reversed_theta(theta, N, L), a, placement)
    return float(np.max(np.abs(z_flip + z_ad)))


def test_zyz_roundtrip():
    rng = np.random.default_rng(3)
    for _ in range(20):
        t = rng.uniform(0, 2 * np.pi, 3)
        U = block(*t)
        V = block(*zyz_angles(U))
        ph = np.trace(V.conj().T @ U) / 2
        assert abs(abs(ph) - 1) < 1e-12 and np.allclose(V * ph, U, atol=1e-12)


def test_default_placement_unchanged():
    from mnist_hybrid_multi import load_mnist_binary
    rec = [r for r in pickle.load(open(os.path.join(HERE, 'm5_depth', 'results.pkl'), 'rb'))
           if (r['L'], r['channel'], r['seed']) == (4, 'AD', 256)][0]
    Ate = load_mnist_binary(n_test=500, pca_seed=0)[2]
    sim = TorchCirc(4, 4, noisetype='AD', p_noise=0.3, device='cpu')
    P = rec['params']
    with torch.no_grad():
        z = sim.features(torch.as_tensor(Ate), torch.as_tensor(P['theta'])).numpy()
    pred = (z @ P['W2'].T + P['b2']).argmax(1)
    assert np.array_equal(pred, rec['test_pred'][:500])


def test_inside_placement_is_the_default_without_an_entangler():
    """With entangling='none' there are no CNOTs, so the two placements must agree
    (the inside placement must not drop that layer's noise)."""
    rng = np.random.default_rng(5)
    theta = rng.uniform(0, 2 * np.pi, 3 * N * (L + 1))
    a = rng.uniform(0, 2 * np.pi, (5, N))
    out = {}
    for placement in ('after_entangler', 'inside_entangler'):
        sim = TorchCirc(N, L, noisetype='AD', p_noise=0.3, device='cpu', entangling='none',
                        noise_placement=placement)
        with torch.no_grad():
            out[placement] = sim.features(torch.as_tensor(a), torch.as_tensor(theta)).numpy()
    assert np.array_equal(out['after_entangler'], out['inside_entangler'])


def test_time_reversal_gauge_holds_for_default_placement():
    for seed in range(3):
        err = gauge_error('after_entangler', seed)
        assert err < 1e-12, err


def test_construction_fails_inside_entangler():
    for seed in range(3):
        err = gauge_error('inside_entangler', seed)
        assert err > 1e-3, err


if __name__ == "__main__":
    import sys
    import traceback
    tests = [(k, v) for k, v in globals().items() if k.startswith('test_') and callable(v)]
    failed = []
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}", flush=True)
        except Exception:
            failed.append(name)
            print(f"FAIL  {name}\n{traceback.format_exc()}", flush=True)
    print(f"{len(tests) - len(failed)}/{len(tests)} passed")
    print(f"gauge error: default {gauge_error('after_entangler'):.2e}, "
          f"inside_entangler {gauge_error('inside_entangler'):.2e}")
    sys.exit(1 if failed else 0)
