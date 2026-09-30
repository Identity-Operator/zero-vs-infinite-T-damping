"""Is the Adam result for reversed AD (noise inside RZZ) the landscape floor?
Independent numpy model (same layout as gauge_vqe_check.py), L-BFGS-B with restarts."""
import numpy as np, sys, time
from scipy.optimize import minimize
n, B = 3, 4
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Z = np.diag([1., -1.]).astype(complex)
def op(s, q):
    m = np.array([[1.]]);
    for k in range(n): m = np.kron(m, s if k == q else I2)
    return m
def cnot(c, t): return op(np.diag([1., 0.]), c) + op(np.diag([0., 1.]), c) @ op(X, t)
PAIRS = [(0, 1), (1, 2), (2, 0)]
H = -sum(op(Z, a) @ op(Z, b) for a, b in PAIRS) - 0.5*sum(op(X, q) for q in range(n))
E0 = np.linalg.eigvalsh(H)[0]
CN = {pr: cnot(*pr) for pr in PAIRS}
Zq = [np.diag(op(Z, q)).real for q in range(n)]; Yq = [op(np.array([[0, -1j], [1j, 0]]), q) for q in range(n)]; Xq = [op(X, q) for q in range(n)]
def kraus(g, rev):
    K0, K1 = np.array([[1, 0], [0, np.sqrt(1-g)]], complex), np.array([[0, np.sqrt(g)], [0, 0]], complex)
    Ks = [X @ K0 @ X, X @ K1 @ X] if rev else [K0, K1]
    return [[op(K, q) for K in Ks] for q in range(n)]
def rot_full(Pq, a):  # exp(-i a P/2) for single-qubit Pauli embedded (P^2=I)
    return np.cos(a/2)*np.eye(2**n) - 1j*np.sin(a/2)*Pq
def make_E(g, rev, place):
    KK = kraus(g, rev)
    def noise(rho, qs):
        for q in qs: rho = sum(K @ rho @ K.conj().T for K in KK[q])
        return rho
    def E(th):
        t0, rest = th[:n], th[n:].reshape(B, len(PAIRS) + n)
        rho = np.zeros((8, 8), complex); rho[0, 0] = 1
        for q in range(n): U = rot_full(Yq[q], t0[q]); rho = U @ rho @ U.conj().T
        for b in range(B):
            for k, (c, t) in enumerate(PAIRS):
                rho = CN[(c, t)] @ rho @ CN[(c, t)].T
                if place == 'mid': rho = noise(rho, (c, t))
                d = np.exp(-0.5j*rest[b, k]*Zq[t]); rho = (d[:, None]*rho)*d.conj()[None, :]
                rho = CN[(c, t)] @ rho @ CN[(c, t)].T
                rho = noise(rho, (c, t)) if place == 'mid' else noise(noise(rho, (c, t)), (c, t))
            for q in range(n): U = rot_full(Xq[q], rest[b, len(PAIRS) + q]); rho = U @ rho @ U.conj().T
        return np.real(np.trace(H @ rho))
    return E
place, restarts = sys.argv[1], int(sys.argv[2]); gammas = [float(x) for x in sys.argv[3].split(',')]
for g in gammas:
    for rev in (False, True):
        E = make_E(g, rev, place); t = time.time(); best = []
        for r in range(restarts):
            v0 = np.random.default_rng(500 + r).uniform(-np.pi, np.pi, n + B*(len(PAIRS) + n))
            best.append(minimize(E, v0, method='L-BFGS-B').fun)
        rel = (np.array(best) - E0)/abs(E0)
        print(f'{place} g={g} {"revAD" if rev else "AD   "}: best rel err {rel.min():.4f}  median {np.median(rel):.4f}  ({time.time()-t:.0f}s)', flush=True)
