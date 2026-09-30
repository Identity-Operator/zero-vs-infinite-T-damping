# -*- coding: utf-8 -*-
"""
Tests for the shot model. Tests 1-4 are CPU
only and small; test 5 needs the GPU for about two minutes.

    python test_shots.py

1. TorchCirc.probs is the Z-basis outcome distribution: its rows sum to 1 and
   sum_k p_k Z_j(k) = <Z_j> from features(), for n = 4 and 6.
2. The training noise z + sqrt((1 - z^2)/N_s) eps has mean z and variance
   (1 - z^2)/N_s.
3. Gradients flow through z: d z_hat / d z = 1 - z eps / sqrt(N_s (1 - z^2)).
4. The inference sampler draws exact multinomial shots: each qubit's sample
   mean has mean <Z_j> and the binomial variance (1 - <Z_j>^2)/N_s.
5. N_s = 1e12 reproduces the exact run: the same exact-expectation
   predictions, and every shot draw within one prediction of them.
"""
import numpy as np
import torch

from mnist_hybrid_multi import outcome_signs, sample_features, shot_noise, train_one
from torch_circ import TorchCirc


def test_probs_match_features():
    for n in (4, 6):
        sim = TorchCirc(n, 2, noisetype='AD', p_noise=0.3, device='cpu')
        rng = np.random.default_rng(n)
        th = torch.as_tensor(rng.uniform(0, 2 * np.pi, sim.n_theta))
        a = torch.as_tensor(rng.uniform(0, 2 * np.pi, (7, n)))
        with torch.no_grad():
            z, P = sim.features(a, th).numpy(), sim.probs(a, th).numpy()
        assert np.allclose(P.sum(1), 1, rtol=0, atol=1e-12) and P.min() > -1e-15
        assert np.allclose(P @ outcome_signs(n), z, rtol=0, atol=1e-12)


def test_training_noise_moments():
    z = torch.tensor([[0.9, -0.3, 0.0, 0.6]], dtype=torch.float64).repeat(200000, 1)
    g = torch.Generator().manual_seed(0)
    for shots in (10, 1000):
        zh = shot_noise(z, shots, torch.randn(z.shape, generator=g, dtype=torch.float64))
        want_var = (1 - z[0] ** 2) / shots
        assert torch.allclose(zh.mean(0), z[0], atol=4 * float(want_var.max().sqrt()) / 400)
        assert torch.allclose(zh.var(0), want_var, rtol=0.02)


def test_training_noise_gradient():
    z = torch.tensor([0.9, -0.3, 0.0, 0.6], dtype=torch.float64, requires_grad=True)
    eps = torch.tensor([0.5, -1.2, 2.0, 0.1], dtype=torch.float64)
    shots = 100
    shot_noise(z, shots, eps).sum().backward()
    zd = z.detach()
    want = 1 - zd * eps / torch.sqrt(shots * (1 - zd ** 2))
    assert torch.allclose(z.grad, want, rtol=0, atol=1e-12)


def test_sampler_moments():
    rng = np.random.default_rng(1)
    n, shots, draws = 4, 100, 20000
    sim = TorchCirc(n, 1, noisetype='Pauli', p_noise=0.3, device='cpu')
    th = torch.as_tensor(rng.uniform(0, 2 * np.pi, sim.n_theta))
    a = torch.as_tensor(rng.uniform(0, 2 * np.pi, (1, n)))
    with torch.no_grad():
        P, z = sim.probs(a, th).numpy(), sim.features(a, th).numpy()[0]
    zs = sample_features(np.repeat(P, draws, axis=0), shots, rng)
    assert np.allclose(zs.mean(0), z, atol=4 * np.sqrt(1.0 / shots / draws))
    assert np.allclose(zs.var(0), (1 - z ** 2) / shots, rtol=0.04)
    assert np.all(np.abs(zs * shots - np.round(zs * shots)) < 1e-9)     # integer counts


def test_huge_shot_count_reproduces_exact_run():
    from run_phase1 import load_data
    Xtr, Ytr, Xte, Yte, _ = load_data('MNIST')
    kw = dict(drop_w1=True, readout='standardized')
    exact = train_one(1, 'AD', 0.3, 256, Xtr, Ytr, Xte, Yte, 100, **kw)
    big = train_one(1, 'AD', 0.3, 256, Xtr, Ytr, Xte, Yte, 100, shots=10 ** 12, **kw)
    assert exact['shots'] is None and big['shots'] == 10 ** 12
    assert np.array_equal(big['test_pred'], exact['test_pred'])
    assert big['train_acc'] == exact['train_acc']
    for pred in big['shot_test_pred']:
        assert int((pred != exact['test_pred']).sum()) <= 1
    assert abs(big['test_acc_shot_mean'] - exact['test_acc']) <= 1 / len(Yte) + 1e-12


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
    sys.exit(1 if failed else 0)
