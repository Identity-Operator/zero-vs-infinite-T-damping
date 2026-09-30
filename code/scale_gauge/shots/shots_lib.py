"""Shared code for F1 (finite-shot inference) and F2 (feature-level input
sensitivity). An independent density-matrix simulator (full 2^n x 2^n Kronecker
matrices, qubit 0 most significant) of the classifier of the v5 manuscript, with the channels
None/AD/AD_flip/Pauli/Depol and both noise placements, plus the v5 data for MNIST (1,3,5) and
Fashion-MNIST (0,2,4). Validated against TorchCirc in check_lib.py."""
import os, numpy as np, torch
from torchvision import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
torch.set_default_dtype(torch.float64)
CD = torch.complex128
HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, '..', '..', 'data')
I2 = torch.eye(2, dtype=CD); PX = torch.tensor([[0, 1], [1, 0]], dtype=CD)
PY = torch.tensor([[0, -1j], [1j, 0]], dtype=CD); PZ = torch.tensor([[1, 0], [0, -1]], dtype=CD)

def kron_list(ms):
    out = ms[0]
    for m in ms[1:]:
        out = torch.kron(out, m)
    return out

def on_qubit(op, j, n):
    return kron_list([op if k == j else I2 for k in range(n)])

def R(P, phi):
    return torch.cos(phi / 2).to(CD) * I2 - 1j * torch.sin(phi / 2).to(CD) * P

def kraus(ch, p):
    c = np.sqrt(1 - p)
    if ch == 'None':
        return []
    if ch == 'AD':
        return [torch.tensor([[1, 0], [0, c]], dtype=CD), torch.tensor([[0, np.sqrt(p)], [0, 0]], dtype=CD)]
    if ch == 'AD_flip':      # damping toward |1>: X.AD.X
        return [torch.tensor([[c, 0], [0, 1]], dtype=CD), torch.tensor([[0, 0], [np.sqrt(p), 0]], dtype=CD)]
    if ch == 'Pauli':
        px = py = p / 4; pz = (2 - p - 2 * c) / 4
    elif ch == 'Depol':
        Lam = 1 - (2 * c + c * c) / 3; px = py = pz = Lam / 4
    else:
        raise ValueError(ch)
    return [np.sqrt(w) * P for w, P in zip([1 - px - py - pz, px, py, pz], (I2, PX, PY, PZ))]

class Circ:
    """features(a, theta) -> (B, n) <Z_j>;  probs(a, theta) -> (B, 2^n) diag(rho)."""
    def __init__(self, n, L, channel, p, placement='after_entangler'):
        assert placement in ('after_entangler', 'inside_entangler')
        self.n, self.L, self.placement, D = n, L, placement, 2 ** n
        self.pairs = [(j, (j + 1) % n) for j in range(n)]
        self.CN = [self._cnot(c, t) for c, t in self.pairs]
        self.K = [[on_qubit(k, j, n) for k in kraus(channel, p)] for j in range(n)]
        bits = np.array([[(b >> (n - 1 - q)) & 1 for q in range(n)] for b in range(D)])
        self.sign = torch.as_tensor(1.0 - 2.0 * bits)            # (D, n): outcome b -> Z_j eigenvalue

    def _cnot(self, c, t):
        n, D = self.n, 2 ** self.n
        M = torch.zeros(D, D, dtype=CD)
        for b in range(D):
            bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[c]:
                bits[t] ^= 1
            M[sum(v << (n - 1 - k) for k, v in enumerate(bits)), b] = 1
        return M

    def _noise_q(self, rho, j):
        return sum(K @ rho @ K.conj().T for K in self.K[j]) if self.K[j] else rho

    def rho(self, a, theta):
        n, L, D = self.n, self.L, 2 ** self.n
        B = a.shape[0]
        rho = torch.zeros(B, D, D, dtype=CD); rho[:, 0, 0] = 1
        for ell in range(L + 1):
            b0 = 3 * n * ell
            Rl = kron_list([R(PZ, theta[b0 + 3 * j + 2]) @ R(PY, theta[b0 + 3 * j + 1]) @ R(PZ, theta[b0 + 3 * j]) for j in range(n)])
            rho = Rl @ rho @ Rl.conj().T
            if ell < L:
                for (c, t), CN in zip(self.pairs, self.CN):
                    rho = CN @ rho @ CN.T
                    if self.placement == 'inside_entangler':
                        rho = self._noise_q(self._noise_q(rho, c), t)
            if not (ell < L and self.placement == 'inside_entangler'):
                for j in range(n):
                    rho = self._noise_q(rho, j)
            if ell < L:
                cth, sth = torch.cos(a / 2).to(CD), torch.sin(a / 2).to(CD)
                rx = torch.stack([torch.stack([cth, -1j * sth], -1), torch.stack([-1j * sth, cth], -1)], -2)
                S = rx[:, 0]
                for j in range(1, n):
                    S = torch.einsum('bik,bjl->bijkl', S, rx[:, j]).reshape(B, S.shape[1] * 2, S.shape[2] * 2)
                rho = S @ rho @ S.conj().transpose(1, 2)
                for j in range(n):
                    rho = self._noise_q(rho, j)
        return rho

    def probs(self, a, theta):
        return torch.diagonal(self.rho(a, theta), dim1=1, dim2=2).real

    def features(self, a, theta):
        return self.probs(a, theta) @ self.sign

def load_v5(dataset='MNIST', n_train=1000):
    """v5 preprocessing: classes filtered in dataset order, seed-0 permutation,
    StandardScaler + PCA(4, random_state=0) on the first n_train training images, training-split
    min-max to [0, 2pi], test angles clipped. Returns train/test angles and labels."""
    src, classes = {'MNIST': (datasets.MNIST, (1, 3, 5)), 'FashionMNIST': (datasets.FashionMNIST, (0, 2, 4))}[dataset]
    def split(train, n):
        ds = src(root=DATA, train=train, download=False)
        x = ds.data.numpy().reshape(len(ds), -1); y = ds.targets.numpy()
        keep = np.isin(y, classes)
        X = x[keep].astype(np.float64) / 255.0; Y = np.array([classes.index(v) for v in y[keep]])
        idx = np.random.default_rng(0).permutation(len(X))
        idx = idx if n is None else idx[:n]
        return X[idx], Y[idx]
    Xtr, Ytr = split(True, n_train); Xte, Yte = split(False, None)
    sc = StandardScaler().fit(Xtr); pca = PCA(n_components=4, random_state=0).fit(sc.transform(Xtr))
    Ztr, Zte = pca.transform(sc.transform(Xtr)), pca.transform(sc.transform(Xte))
    lo, hi = Ztr.min(0), Ztr.max(0)
    f = lambda Z: (Z - lo) / (hi - lo) * 2 * np.pi
    return f(Ztr), Ytr, np.clip(f(Zte), 0, 2 * np.pi), Yte

def classify(z, prm):
    """Standardized readout with the stored training statistics: argmax of W2 (z - mu)/sigma + b2."""
    zt = (z - np.asarray(prm['z_mean'])) / np.asarray(prm['z_std'])
    return (zt @ np.asarray(prm['W2']).T + np.asarray(prm['b2'])).argmax(1)
