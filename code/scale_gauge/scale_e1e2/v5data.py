"""An independent implementation of the v5 preprocessing, written
from the spec, not from load_mnist_binary. Reads raw torchvision arrays."""
import os
import numpy as np
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data')
CLASSES = (1, 3, 5)

def _split(train, n, perm_seed=0):
    ds = datasets.MNIST(root=DATA, train=train, download=False)
    x = ds.data.numpy().reshape(len(ds), -1)
    y = ds.targets.numpy()
    keep = np.isin(y, CLASSES)                      # dataset order preserved
    X = x[keep].astype(np.float64) / 255.0
    Y = np.array([CLASSES.index(v) for v in y[keep]])
    idx = np.random.default_rng(perm_seed).permutation(len(X))
    idx = idx if n is None else idx[:n]
    return X[idx], Y[idx], np.flatnonzero(keep)[idx]

def load_v5(n_train=1000, n_test=None, pca_seed=0):
    Xtr, Ytr, itr = _split(True, n_train)
    Xte, Yte, ite = _split(False, n_test)
    sc = StandardScaler().fit(Xtr)
    pca = PCA(n_components=4, random_state=pca_seed).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: (Z - lo) / (hi - lo) * 2 * np.pi
    Atr, Ate_raw = f(Ztr), f(Zte)
    n_clip = int(((Ate_raw < 0) | (Ate_raw > 2 * np.pi)).sum())
    return dict(Atr=Atr, Ytr=Ytr, Ate=np.clip(Ate_raw, 0, 2 * np.pi), Yte=Yte,
                n_clip=n_clip, itr=itr, ite=ite, solver=pca._fit_svd_solver)

if __name__ == '__main__':
    d = load_v5()
    print("n_train", len(d['Ytr']), "n_test", len(d['Yte']), "test class counts", np.bincount(d['Yte']),
          "n_clip", d['n_clip'], "of", d['Ate'].size, "solver", d['solver'])
    here = os.path.dirname(os.path.abspath(__file__))
    old = np.load(os.path.join(here, 'pilot_angles.npz'))   # pilot loader output: Atr, Ytr, Yte, Bte (train-extrema test)
    print("train angles bit-identical to pilot loader:", np.array_equal(d['Atr'], old['Atr']), " labels:", np.array_equal(d['Ytr'], old['Ytr']))
    print("first 250 test: labels equal", np.array_equal(d['Yte'][:250], old['Yte']),
          " angles allclose", np.allclose(d['Ate'][:250], old['Bte'], atol=1e-12), " max|d|", np.abs(d['Ate'][:250]-old['Bte']).max())
    d250 = load_v5(n_test=250)
    print("n_test=250 run: train angles unchanged", np.array_equal(d250['Atr'], d['Atr']), " test idx = first 250 of full", np.array_equal(d250['ite'], d['ite'][:250]))
    np.savez(os.path.join(here, 'v5_angles.npz'),
             **{k: d[k] for k in ('Atr', 'Ytr', 'Ate', 'Yte', 'itr', 'ite')}, n_clip=d['n_clip'])
