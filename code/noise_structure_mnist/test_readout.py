# -*- coding: utf-8 -*-
"""
Tests for the readout option of train_one.
Needs the GPU, for about two minutes.

    python test_readout.py

1. The default raw readout is unchanged: it reproduces a committed m5_depth
   record (L=1, None, seed 256) exactly in predictions and accuracies, and
   in loss, sensitivity and parameters to tolerances far below any real
   numerical change (see the test for why not bit for bit).
2. readout='standardized' is the e1 readout of scale_gauge/scale_e1e2: it matches a transcription
   of e1's training loop (code/scale_gauge/scale_e1e2, scale_test.py
   lines 28-48) to 1e-12.
3. Evaluation uses the training statistics: the stored z_mean and z_std are
   the training-feature statistics of the stored parameters, and applying
   them to the test features reproduces test_pred.
"""
import os
import pickle

import numpy as np
import torch

from mnist_hybrid_multi import train_one
from run_phase1 import HERE, OFFSETS, load_data
from torch_circ import TorchCirc

_data = {}


def data():
    if not _data:
        _data['d'] = load_data('MNIST')
    return _data['d']


def test_raw_default_reproduces_m5_depth():
    Xtr, Ytr, Xte, Yte, _ = data()
    ref = [r for r in pickle.load(open(os.path.join(HERE, 'm5_depth', 'results.pkl'), 'rb'))
           if (r['L'], r['channel'], r['seed']) == (1, 'None', 256)][0]
    r = train_one(1, 'None', 0.3, 256, Xtr, Ytr, Xte, Yte, 100, drop_w1=True,
                  entangling='ring', offsets=OFFSETS, sensitivity=True)
    # A fresh process need not reproduce a committed record bit for bit: the
    # committed values depend on what ran earlier in the same long-lived
    # process, and the drift is amplified in analytically flat directions of
    # theta (we measured 4e-14 relative in the loss and 4.7e-9
    # in theta for L=2 AD seed 768). This cell happens to be exact because it
    # was the first configuration of its process. Tolerances: about 20x the
    # observed drift; a real numerical change moves the loss far more.
    for k in ('train_acc', 'test_acc', 'test_acc_250', 'offset_acc', 'acc_hist'):
        assert r[k] == ref[k], f"{k}: {r[k]} != {ref[k]}"
    assert np.array_equal(r['test_pred'], ref['test_pred'])
    for k in ('loss', 'sensitivity'):
        assert abs(r[k] - ref[k]) <= 1e-12 * abs(ref[k]), f"{k}: {r[k]} vs {ref[k]}"
    assert np.allclose(r['params']['theta'], ref['params']['theta'], rtol=0, atol=1e-7)
    for k in ('W2', 'b2'):
        assert np.allclose(r['params'][k], ref['params'][k], rtol=0, atol=1e-12), k
    assert r['readout'] == 'raw' and 'z_mean' not in r['params']


def e1_reference(L, ch, seed, n_steps, Xtr, Ytr, Xte):
    """The e1 training loop, transcribed with TorchCirc on the GPU."""
    sim = TorchCirc(4, L, noisetype=ch, p_noise=0.3, device='cuda', entangling='ring')
    g = torch.Generator(device='cuda').manual_seed(seed)
    W2 = (0.5 * torch.randn(3, 4, generator=g, device='cuda')).requires_grad_(True)
    b2 = torch.zeros(3, device='cuda', requires_grad=True)
    th = (2 * np.pi * torch.rand(sim.n_theta, generator=g, device='cuda')).requires_grad_(True)
    opt = torch.optim.Adam([{'params': [W2, b2], 'weight_decay': 0.0},
                            {'params': [th], 'weight_decay': 0.0}], lr=0.05)

    def head(z, stats=None):
        mu, sdv = (z.mean(0), z.std(0, correction=0)) if stats is None else stats
        return ((z - mu) / sdv) @ W2.T + b2
    X, Xv = torch.as_tensor(Xtr, device='cuda'), torch.as_tensor(Xte, device='cuda')
    Yt = torch.as_tensor(Ytr, dtype=torch.long, device='cuda')
    for _ in range(n_steps):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(sim.features(X, th)), Yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        ztr, zte = sim.features(X, th), sim.features(Xv, th)
        st = (ztr.mean(0), ztr.std(0, correction=0))
        pred = head(zte, st).argmax(1).cpu().numpy()
    return {'theta': th.detach().cpu().numpy(), 'W2': W2.detach().cpu().numpy(),
            'b2': b2.detach().cpu().numpy(), 'pred': pred}


_std = {}


def standardized_run():
    if not _std:
        Xtr, Ytr, Xte, Yte, _ = data()
        _std['r'] = train_one(1, 'AD', 0.3, 512, Xtr, Ytr, Xte, Yte, 20, drop_w1=True,
                              entangling='ring', readout='standardized')
    return _std['r']


def test_standardized_matches_e1():
    Xtr, Ytr, Xte, Yte, _ = data()
    r = standardized_run()
    ref = e1_reference(1, 'AD', 512, 20, Xtr, Ytr, Xte)
    for k in ('theta', 'W2', 'b2'):
        assert np.allclose(r['params'][k], ref[k], rtol=0, atol=1e-12), k
    assert np.array_equal(r['test_pred'], ref['pred'])
    assert r['readout'] == 'standardized'


def test_evaluation_uses_training_statistics():
    Xtr, Ytr, Xte, Yte, _ = data()
    r, P = standardized_run(), standardized_run()['params']
    sim = TorchCirc(4, 1, noisetype='AD', p_noise=0.3, device='cuda', entangling='ring')
    th = torch.as_tensor(P['theta'], device='cuda')
    with torch.no_grad():
        ztr = sim.features(torch.as_tensor(Xtr, device='cuda'), th).cpu().numpy()
        zte = sim.features(torch.as_tensor(Xte, device='cuda'), th).cpu().numpy()
    assert np.allclose(P['z_mean'], ztr.mean(0), rtol=0, atol=1e-12)
    assert np.allclose(P['z_std'], ztr.std(0), rtol=0, atol=1e-12)
    pred = (((zte - P['z_mean']) / P['z_std']) @ P['W2'].T + P['b2']).argmax(1)
    assert np.array_equal(pred, r['test_pred'])


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
