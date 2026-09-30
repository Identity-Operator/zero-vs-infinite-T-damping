"""Cross-check: does an independent numpy density-matrix model reproduce the reference
Qiskit implementation code/old_code/TFIM_VQE_Noise.py (noise_model '2' = channel on both
qubits after every CNOT)? Channels: None, AD, AD_reverse, Pauli (Pauli twirl), Depol (Clifford twirl)."""
import os, sys, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'old_code'))
import types
# qiskit_algorithms is only needed for the SPSA optimizer, not for energies; stub it.
_m = types.ModuleType('qiskit_algorithms'); _o = types.ModuleType('qiskit_algorithms.optimizers'); _o.SPSA = None
sys.modules['qiskit_algorithms'] = _m; sys.modules['qiskit_algorithms.optimizers'] = _o
from TFIM_VQE_Noise import TFIM_VQE_NOISE
n, T = 3, 4
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])
def op(s, q):
    m = np.array([[1.]])
    for k in range(n): m = np.kron(m, s if k == q else I2)   # qubit 0 = leftmost factor
    return m
ry = lambda t: np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
rx = lambda t: np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])
rz = lambda t: np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def cnot(c, t): return op(np.diag([1., 0.]), c) + op(np.diag([0., 1.]), c) @ op(X, t)
def kraus(ch, g):
    K0, K1 = np.array([[1, 0], [0, np.sqrt(1-g)]]), np.array([[0, np.sqrt(g)], [0, 0]])
    if ch == 'AD': return [K0, K1]
    if ch == 'AD_reverse': return [X @ K0 @ X, X @ K1 @ X]
    if ch == 'Pauli':
        px = g/4; pz = (2 - g - 2*np.sqrt(1-g))/4; pI = 1 - 2*px - pz
        return [np.sqrt(pI)*I2, np.sqrt(px)*X, np.sqrt(px)*Y, np.sqrt(pz)*Z]
    if ch == 'Depol':
        lam = (g + 2 - 2*np.sqrt(1-g))/3   # qiskit depolarizing_error(lam,1): rho -> (1-lam) rho + lam I/2
        return [np.sqrt(1 - 3*lam/4)*I2] + [np.sqrt(lam/4)*P for P in (X, Y, Z)]
    return [I2]
def chan(rho, Ks, q): return sum(op(K, q) @ rho @ op(K, q).conj().T for K in Ks)
PAIRS = [(0, 1), (1, 2), (2, 0)]
H = -1.0*sum(op(Z, a) @ op(Z, b) for a, b in PAIRS) - 0.5*sum(op(X, q) for q in range(n))
def energy(par, ch, g):
    zs, xs, ys = par[:T*n], par[T*n:2*T*n], par[2*T*n:]
    Ks = kraus(ch, g); rho = np.zeros((8, 8), complex); rho[0, 0] = 1
    for q in range(n): rho = op(ry(ys[q]), q) @ rho @ op(ry(ys[q]), q).conj().T
    for s in range(T):
        for k, (c, t) in enumerate(PAIRS):
            for half in (0, 1):
                U = cnot(c, t); rho = U @ rho @ U.conj().T
                rho = chan(chan(rho, Ks, c), Ks, t)
                if half == 0:
                    R = op(rz(2*zs[s*n + k]/T), t); rho = R @ rho @ R.conj().T
        for q in range(n):
            R = op(rx(2*xs[s*n + q]/T), q); rho = R @ rho @ R.conj().T
    return np.real(np.trace(H @ rho))
rng = np.random.default_rng(7)
for g in (0.05, 0.2):
    vqe = TFIM_VQE_NOISE(n, T, noise_str=g, use_density_matrix=True)
    for ch in ('None', 'AD', 'AD_reverse', 'Pauli', 'Depol'):
        d = []
        for _ in range(4):
            par = rng.uniform(-np.pi, np.pi, 2*T*n + n)
            d.append(abs(vqe.compute_energy(par, ch) - energy(par, ch, g)))
        print(f'g={g} {ch:10s} max|E_qiskit - E_numpy| = {max(d):.2e}')
