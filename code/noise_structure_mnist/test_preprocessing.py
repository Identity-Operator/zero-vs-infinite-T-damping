# -*- coding: utf-8 -*-
"""
Preprocessing tests for the m5_* runs. They must pass
before any run; run_queue.sh runs them first.

    python test_preprocessing.py            # or: python -m pytest test_preprocessing.py

CPU only, about two minutes (MNIST is re-read for every load).

Angles are compared with np.allclose, not bit for bit: the scaler and PCA
transforms use BLAS kernels whose blocking depends on the number of rows, so
the same image can come out 1e-15 rad apart in a 250-row and a 3037-row batch.
Indices, pixels and labels are compared exactly.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import mnist_hybrid_multi as m

CLASSES = (1, 3, 5)
N_TRAIN, N_OLD_TEST = 1000, 250
FULL_TEST_COUNTS = [1135, 1010, 892]      # MNIST test images of digits 1, 3, 5
ATOL = 1e-12

_cache = {}


def _load(n_train=N_TRAIN, n_test=None, dataset='MNIST', classes=CLASSES):
    key = (n_train, n_test, dataset, classes)
    if key not in _cache:
        info = {}
        out = m.load_mnist_binary(classes=classes, n_train=n_train, n_test=n_test,
                                  dataset=dataset, pca_seed=0, info=info)
        _cache[key] = out + (info,)
    return _cache[key]


def _pre_fix_selection(ds, n, classes=CLASSES, seed=0):
    """The image selection of load_mnist_binary as of d5ad69e, copied verbatim.

    Frozen reference for "the same images as the previous splits"; do not edit.
    """
    X, Y = [], []
    for x, y in ds:
        if y in classes:
            X.append(x.numpy().ravel())
            Y.append(classes.index(y))
    X, Y = np.array(X), np.array(Y)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))[:n]
    return X[idx], Y[idx], idx


def test_training_angles_do_not_depend_on_n_test():
    Atr, Ytr, _, _, info = _load(n_test=None)
    for n_test in (1, N_OLD_TEST):
        Atr2, Ytr2, _, _, info2 = _load(n_test=n_test)
        assert np.array_equal(Atr, Atr2), f"training angles change with n_test={n_test}"
        assert np.array_equal(Ytr, Ytr2)
        assert np.array_equal(info['train_idx'], info2['train_idx'])


def test_angles_are_the_training_extrema_transform():
    Atr, Ytr, Ate, Yte, info = _load(n_test=None)
    Xtr, _, _ = m.select_split(m.load_dataset('MNIST', True), CLASSES, N_TRAIN)
    Xte, _, _ = m.select_split(m.load_dataset('MNIST', False), CLASSES, None)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=4, random_state=0).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    want_tr = (Ztr - lo) / (hi - lo) * 2 * np.pi
    want_te_raw = (Zte - lo) / (hi - lo) * 2 * np.pi
    assert np.allclose(Atr, want_tr, rtol=0, atol=ATOL)
    assert np.allclose(Ate, np.clip(want_te_raw, 0, 2 * np.pi), rtol=0, atol=ATOL)
    assert np.allclose(Atr.min(0), 0, atol=ATOL) and np.allclose(Atr.max(0), 2 * np.pi, atol=ATOL)
    assert Ate.min() >= 0 and Ate.max() <= 2 * np.pi
    assert info['n_clip'] == int(((want_te_raw < 0) | (want_te_raw > 2 * np.pi)).sum())
    assert info['prep'] == m.PREP == 'v5_train_extrema' and info['pca_seed'] == 0
    # The bug: scaling the test split by its own extrema. Make sure it is gone.
    tlo, thi = Zte.min(0), Zte.max(0)
    per_split = (Zte - tlo) / (thi - tlo) * 2 * np.pi
    assert np.abs(Ate - per_split).mean(0).max() > 0.1


def test_first_test_images_are_the_previous_split():
    _, _, Ate_old, Yte_old, info_old = _load(n_test=N_OLD_TEST)
    _, _, Ate, Yte, info = _load(n_test=None)
    ref_X, ref_Y, ref_idx = _pre_fix_selection(m.load_dataset('MNIST', False), N_OLD_TEST)
    new_X, new_Y, new_idx = m.select_split(m.load_dataset('MNIST', False), CLASSES, None)
    assert np.array_equal(info['test_idx'][:N_OLD_TEST], ref_idx)
    assert np.array_equal(info_old['test_idx'], ref_idx)
    assert np.array_equal(new_idx[:N_OLD_TEST], ref_idx)
    assert np.array_equal(new_X[:N_OLD_TEST], ref_X)
    assert np.array_equal(new_Y[:N_OLD_TEST], ref_Y)
    assert np.array_equal(Yte[:N_OLD_TEST], ref_Y) and np.array_equal(Yte_old, ref_Y)
    assert np.allclose(Ate[:N_OLD_TEST], Ate_old, rtol=0, atol=ATOL)


def test_training_images_are_the_previous_ones():
    ref_X, ref_Y, ref_idx = _pre_fix_selection(m.load_dataset('MNIST', True), N_TRAIN)
    _, Ytr, _, _, info = _load(n_test=None)
    new_X, _, _ = m.select_split(m.load_dataset('MNIST', True), CLASSES, N_TRAIN)
    assert np.array_equal(info['train_idx'], ref_idx)
    assert np.array_equal(new_X, ref_X) and np.array_equal(Ytr, ref_Y)


def test_full_test_set():
    _, _, Ate, Yte, info = _load(n_test=None)
    assert len(Yte) == sum(FULL_TEST_COUNTS) == 3037
    assert np.bincount(Yte).tolist() == FULL_TEST_COUNTS
    assert Ate.shape == (3037, 4) and info['n_test_values'] == 4 * 3037


def test_per_split_option_is_gone():
    try:
        m.load_mnist_binary(n_test=1, scale_from_train=False)
    except TypeError:
        return
    raise AssertionError("load_mnist_binary still accepts scale_from_train")


def test_fashion_full_test_set():
    Atr, Ytr, Ate, Yte, info = _load(n_test=None, dataset='FashionMNIST', classes=(0, 2, 4))
    assert len(Yte) == 3000 and np.bincount(Yte).tolist() == [1000, 1000, 1000]
    assert len(Ytr) == N_TRAIN and Ate.min() >= 0 and Ate.max() <= 2 * np.pi
    assert np.allclose(Atr.min(0), 0, atol=ATOL) and np.allclose(Atr.max(0), 2 * np.pi, atol=ATOL)


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
