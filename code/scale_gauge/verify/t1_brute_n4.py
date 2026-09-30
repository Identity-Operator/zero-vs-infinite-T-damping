"""Independent check of T1: exact density-matrix simulation in numpy,
sharing no code with t1_scale_ratio.py or TorchCirc. n=4, p=0.3, circuit of Fig. 1:
per layer RZ RY RZ on every qubit, CNOT ring (1->2, 2->3, 3->4, 4->1 in that order),
channel, RX(a_j), channel; then a final RZ RY RZ block and the channel. theta ~ U[0, 2pi).
Output: sqrt(E_theta mean_j Var_a z_j) over the first 300 training images (v5 angles)."""
import sys, os, json, time
import numpy as np
n, p, NI, ND = 4, 0.3, 300, int(sys.argv[1]) if len(sys.argv) > 1 else 200
CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'noise_structure_mnist')
sys.path.insert(0, CODE); os.chdir(CODE)
from mnist_hybrid_multi import load_mnist_binary
A = np.asarray(load_mnist_binary(classes=(1, 3, 5), n_train=1000, n_test=None, n_components=n,
                                 seed=0, dataset='MNIST', pca_seed=0)[0], float)[:NI]
print('angles', A.shape, A.min().round(3), A.max().round(3), flush=True)
c = np.sqrt(1 - p)
KR = {'None': [np.eye(2)],
      'AD': [np.diag([1, c]), np.array([[0, np.sqrt(p)], [0, 0]])]}
X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1, -1])
pz = (2 - p - 2 * c) / 4; px = py = p / 4
KR['twirl'] = [np.sqrt(1 - px - py - pz) * np.eye(2), np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]
def rz(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
def ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
def rx_batch(a):  # (B,2,2)
    co, si = np.cos(a / 2), np.sin(a / 2)
    U = np.empty((len(a), 2, 2), complex); U[:, 0, 0] = U[:, 1, 1] = co; U[:, 0, 1] = U[:, 1, 0] = -1j * si
    return U
L_ = 'abcdefgh'
def op1(rho, U, q):  # rho (B, 2^n, 2^n); U (2,2) or (B,2,2)
    B = rho.shape[0]; r = rho.reshape((B,) + (2,) * (2 * n))
    ket = list(L_[:n]); bra = list(L_[n:2 * n])
    kin = ket.copy(); kin[q] = 'x'; bin_ = bra.copy(); bin_[q] = 'y'
    Us = 'z' + ket[q] + 'x' if U.ndim == 3 else ket[q] + 'x'
    Vs = 'z' + bra[q] + 'y' if U.ndim == 3 else bra[q] + 'y'
    out = np.einsum(f"z{''.join(kin)}{''.join(bin_)},{Us},{Vs}->z{''.join(ket)}{''.join(bra)}", r, U, U.conj())
    return out.reshape(B, 2 ** n, 2 ** n)
def chan(rho, ks, q):
    return sum(op1(rho, K, q) for K in ks) if len(ks) > 1 else op1(rho, ks[0], q)
def cnot_full(ctrl, tgt):
    D = 2 ** n; M = np.zeros((D, D))
    for i in range(D):
        bits = [(i >> (n - 1 - k)) & 1 for k in range(n)]
        if bits[ctrl]: bits[tgt] ^= 1
        M[sum(b << (n - 1 - k) for k, b in enumerate(bits)), i] = 1
    return M
RING = np.eye(2 ** n)
for j in range(n):
    RING = cnot_full(j, (j + 1) % n) @ RING
Zs = [np.diag([1 - 2 * ((i >> (n - 1 - j)) & 1) for i in range(2 ** n)]) for j in range(n)]
def run(L, ks, th):
    rho = np.zeros((NI, 2 ** n, 2 ** n), complex); rho[:, 0, 0] = 1
    k = 0
    for l in range(L):
        for q in range(n):
            for g in (rz, ry, rz):
                rho = op1(rho, g(th[k]), q); k += 1
        rho = RING @ rho @ RING.T
        for q in range(n): rho = chan(rho, ks, q)
        for q in range(n): rho = op1(rho, rx_batch(A[:, q]), q)
        for q in range(n): rho = chan(rho, ks, q)
    for q in range(n):
        for g in (rz, ry, rz):
            rho = op1(rho, g(th[k]), q); k += 1
    for q in range(n): rho = chan(rho, ks, q)
    return np.stack([np.einsum('bii,i->b', rho, np.diag(Zj)).real for Zj in Zs], 1)  # (NI, n)
rng = np.random.default_rng(12345)
res = {}
t0 = time.time()
for L in (1, 2, 3, 4):
    for name in ('None', 'AD', 'twirl'):
        v = []
        for d in range(ND):
            th = rng.uniform(0, 2 * np.pi, 3 * n * (L + 1))
            z = run(L, KR[name], th)
            v.append(z.var(0).mean())
        v = np.array(v); rms = np.sqrt(v.mean()); se = v.std(ddof=1) / np.sqrt(ND) / (2 * rms)
        res[(L, name)] = (rms, se)
        print(f'L={L} {name:5s} rms {rms:.4e} +- {se:.1e}  ({time.time() - t0:.0f}s)', flush=True)
T = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'theory', 't1_results.json')))['chain']['4']
print('\ncomparison with the T1 chain (n=4):')
for L in (1, 2, 3, 4):
    for name in ('None', 'AD', 'twirl'):
        rms, se = res[(L, name)]; ch = T[name]['rms'][L - 1]
        print(f'L={L} {name:5s} brute {rms:.4e} +- {se:.1e}  chain {ch:.4e}  z = {(rms - ch) / se:+.1f}')
