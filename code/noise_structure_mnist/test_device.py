# -*- coding: utf-8 -*-
"""
Tests for the device and shard options. Needs the
GPU (for the initial draws and the committed twin) and about four CPU minutes.

    python test_device.py

1. A CPU run matches its GPU twin: train_one(device='cpu') for L=1, None,
   seed 256 reproduces the committed m5_depth record's per-image test
   predictions, train accuracy and loss, with the parameters equal to
   1e-6 (theta can drift in flat directions at the 1e-9 level).
2. The CPU run starts from the GPU twin's initial draws: after one step the
   two agree to 1e-9. Different draws would differ by O(0.1-1). The residual
   (up to about 6e-11) sits in analytically flat directions of theta, such
   as the first Rz acting on |0> and the last Rz before the Z measurement.
   Their true gradient is zero, so Adam scales float noise of about 1e-17 by
   lr/eps = 5e6.
3. Shards partition a stage's grid: for k = 1..5, the shards cover every
   configuration exactly once.
4. Five processes appending to one results file under the lock keep all
   100 records.
"""
import os
import pickle

import numpy as np

from mnist_hybrid_multi import train_one
from run_phase1 import (CHANNELS, HERE, OFFSETS, SEEDS, append_result, load_data,
                        shard_slice)


def test_cpu_run_matches_gpu_twin():
    Xtr, Ytr, Xte, Yte, _ = load_data('MNIST')
    ref = [r for r in pickle.load(open(os.path.join(HERE, 'm5_depth', 'results.pkl'), 'rb'))
           if (r['L'], r['channel'], r['seed']) == (1, 'None', 256)][0]
    r = train_one(1, 'None', 0.3, 256, Xtr, Ytr, Xte, Yte, 100, drop_w1=True,
                  entangling='ring', offsets=OFFSETS, sensitivity=True, device='cpu')
    assert r['device'] == 'cpu'
    assert np.array_equal(r['test_pred'], ref['test_pred'])
    assert r['train_acc'] == ref['train_acc'] and r['offset_acc'] == ref['offset_acc']
    assert abs(r['loss'] - ref['loss']) < 1e-10 and abs(r['sensitivity'] - ref['sensitivity']) < 1e-10
    for k in ('theta', 'W2', 'b2'):
        assert np.allclose(r['params'][k], ref['params'][k], rtol=0, atol=1e-6), k


def test_cpu_first_step_equals_gpu():
    """Equal to 1e-9 after one step is only possible from identical initial draws."""
    Xtr, Ytr, Xte, Yte, _ = load_data('MNIST')
    rc = train_one(2, 'AD', 0.3, 1024, Xtr[:50], Ytr[:50], Xte[:20], Yte[:20], 1, drop_w1=True,
                   device='cpu')
    rg = train_one(2, 'AD', 0.3, 1024, Xtr[:50], Ytr[:50], Xte[:20], Yte[:20], 1, drop_w1=True,
                   device='cuda')
    for k in ('theta', 'W2', 'b2'):
        assert np.allclose(rc['params'][k], rg['params'][k], rtol=0, atol=1e-9), k


def test_shards_partition_the_grid():
    grid = [(L, 'ring', 0.3, ch) for L in (1, 2, 3, 4) for ch in CHANNELS]
    full = [cfg + (sd,) for cfg in grid for sd in SEEDS]
    for k in range(1, 6):
        picked = [c for i in range(k) for c in shard_slice(full, i, k)]
        assert sorted(picked) == sorted(full) and len(picked) == len(set(picked))


def _append_many(path, worker, n):
    for m in range(n):
        append_result(path, {'worker': worker, 'm': m})


def test_parallel_appends_keep_every_record():
    import multiprocessing as mp
    import tempfile
    path = os.path.join(tempfile.mkdtemp(), 'results.pkl')
    procs = [mp.Process(target=_append_many, args=(path, w, 20)) for w in range(5)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    recs = pickle.load(open(path, 'rb'))
    assert sorted((r['worker'], r['m']) for r in recs) == [(w, m) for w in range(5) for m in range(20)]


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
