"""Finite-temperature check: generalized amplitude damping
GAD(p, N) with excited-state population N has the contraction T = diag(c, c, c^2) of AD and the
non-unital vector t = (0, 0, p(1-2N)). Since the adjoint maps Z -> c^2 Z + t_z I and every path
through the identity stays there, each feature is affine in t_z:
    z_GAD(N) = z_twirl + (1 - 2N) (z_AD - z_twirl)   exactly, for all parameters and inputs,
where the Pauli twirl is GAD(p, 1/2). Checked by exact density-matrix simulation of the classifier
layout (RZ RY RZ blocks, CNOT ring, channel, RX(a), channel; final block and channel).
numpy only."""
import numpy as np
rng = np.random.default_rng(7)
X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])

def gad(p, N):
    c = np.sqrt(1 - p)
    return [np.sqrt(1 - N) * np.diag([1, c]), np.sqrt(1 - N) * np.sqrt(p) * np.array([[0, 1], [0, 0]]),
            np.sqrt(N) * np.diag([c, 1]), np.sqrt(N) * np.sqrt(p) * np.array([[0, 0], [1, 0]])]

def bloch(ks):
    P = [X, Y, Z]; ch = lambda r: sum(k @ r @ k.conj().T for k in ks)
    T = np.array([[np.trace(P[i] @ ch(P[j])).real / 2 for j in range(3)] for i in range(3)])
    t = np.array([np.trace(P[i] @ ch(np.eye(2))).real / 2 for i in range(3)])
    return T, t

def op(U, q, n):
    m = np.array([[1.]]);
    for k in range(n): m = np.kron(m, U if k == q else np.eye(2))
    return m

def rz(t): return np.diag([np.exp(-1j * t / 2), np.exp(1j * t / 2)])
def ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]])
def rx(t): return np.array([[np.cos(t / 2), -1j * np.sin(t / 2)], [-1j * np.sin(t / 2), np.cos(t / 2)]])

def ring(n):
    D = 2 ** n; M = np.eye(D)
    for j in range(n):
        C = np.zeros((D, D)); c_, t_ = j, (j + 1) % n
        for i in range(D):
            b = [(i >> (n - 1 - k)) & 1 for k in range(n)]
            if b[c_]: b[t_] ^= 1
            C[sum(v << (n - 1 - k) for k, v in enumerate(b)), i] = 1
        M = C @ M
    return M

def features(n, L, ks, th, a):
    rho = np.zeros((2 ** n, 2 ** n), complex); rho[0, 0] = 1; E = ring(n); k = 0
    noise = lambda r: _noise(r, ks, n)
    for l in range(L):
        for q in range(n):
            for g in (rz, ry, rz):
                U = op(g(th[k]), q, n); rho = U @ rho @ U.conj().T; k += 1
        rho = noise(E @ rho @ E.T)
        for q in range(n):
            U = op(rx(a[q]), q, n); rho = U @ rho @ U.conj().T
        rho = noise(rho)
    for q in range(n):
        for g in (rz, ry, rz):
            U = op(g(th[k]), q, n); rho = U @ rho @ U.conj().T; k += 1
    rho = noise(rho)
    return np.array([np.trace(op(Z, j, n) @ rho).real for j in range(n)])

def _noise(rho, ks, n):
    for q in range(n):
        Ks = [op(K, q, n) for K in ks]; rho = sum(K @ rho @ K.conj().T for K in Ks)
    return rho

p = 0.3
for N in (0.0, 0.25, 0.5, 1.0):
    T, t = bloch(gad(p, N)); print(f'N={N}: T diag {np.round(np.diag(T), 6)}, offdiag max {np.abs(T - np.diag(np.diag(T))).max():.1e}, t {np.round(t, 6)} (p(1-2N) = {p * (1 - 2 * N):.3f})')
worst = 0.0
for trial in range(20):
    n, L = 3, int(rng.integers(1, 4))
    th = rng.uniform(0, 2 * np.pi, 3 * n * (L + 1)); a = rng.uniform(0, 2 * np.pi, n)
    zAD, ztw = features(n, L, gad(p, 0.0), th, a), features(n, L, gad(p, 0.5), th, a)
    for N in (0.1, 0.25, 0.7, 1.0):
        zN = features(n, L, gad(p, N), th, a)
        worst = max(worst, np.abs(zN - (ztw + (1 - 2 * N) * (zAD - ztw))).max())
print(f'max |z_GAD(N) - [z_tw + (1-2N)(z_AD - z_tw)]| over 20 random circuits (n=3, L=1-3) and N in {{0.1,0.25,0.7,1}}: {worst:.1e}')

# The exact affine law fails on several qubits: a string such as Z(x)X branches once per qubit, so
# features carry t_z^2 and higher terms. The last-layer term of Eq. (floor) is a single-qubit branch
# and is exactly linear. Measure the feature scale over inputs (rms over random theta) against (1-2N).
print('\nfeature scale over inputs, p=0.3, n=3, 40 random theta, 60 random inputs:')
A = rng.uniform(0, 2 * np.pi, (60, 3))
for L in (1, 3):
    TH = [rng.uniform(0, 2 * np.pi, 9 * (L + 1)) for _ in range(40)]
    Z0 = {N: np.array([[features(3, L, gad(p, N), th, a) for a in A] for th in TH]) for N in (0.0, 0.25, 0.5)}
    var = lambda z: np.mean(z.var(1))                       # E_theta mean_j Var_a z_j
    br = lambda N: Z0[N] - Z0[0.5]                          # branching part: GAD(N) minus the twirl
    r = np.sqrt(var(br(0.25)) / var(br(0.0)))
    print(f'  L={L}: sigma_br(N=0.25)/sigma_br(N=0) = {r:.4f}  (1-2N = 0.5);  '
          f'sigma_AD {np.sqrt(var(Z0[0.0])):.4f}, sigma_GAD(0.25) {np.sqrt(var(Z0[0.25])):.4f}, sigma_tw {np.sqrt(var(Z0[0.5])):.4f}')
