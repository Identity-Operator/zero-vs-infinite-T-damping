"""Independent check of the VQE gauge identity: 3-qubit TFIM ansatz of van Rossum
Sec. IV. E_rev(theta) vs E_AD(theta') with theta' = theta except initial RY t -> pi - t.
Expect equality for noise after the full RZZ ('post'), a mismatch for noise after each CNOT ('mid')."""
import numpy as np
n, J, h, B = 3, 1.0, 0.5, 4
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Z = np.diag([1., -1.])
def op(single, q):
    m = np.array([[1.]])
    for k in range(n): m = np.kron(m, single if k == q else I2)
    return m
def ry(t): return np.array([[np.cos(t/2), -np.sin(t/2)], [np.sin(t/2), np.cos(t/2)]])
def rx(t): return np.array([[np.cos(t/2), -1j*np.sin(t/2)], [-1j*np.sin(t/2), np.cos(t/2)]])
def rz(t): return np.diag([np.exp(-1j*t/2), np.exp(1j*t/2)])
def cnot(c, t):
    P0, P1 = np.diag([1., 0.]), np.diag([0., 1.])
    return op(P0, c) + op(P1, c) @ op(X, t)
def kraus(g, rev):
    K0, K1 = np.array([[1, 0], [0, np.sqrt(1-g)]]), np.array([[0, np.sqrt(g)], [0, 0]])
    return [X @ K @ X for K in (K0, K1)] if rev else [K0, K1]
def chan(rho, Ks, q): return sum(op(K, q) @ rho @ op(K, q).conj().T for K in Ks)
def U(rho, u): return u @ rho @ u.conj().T
PAIRS = [(0, 1), (1, 2), (2, 0)]
H = -J*sum(op(Z, a) @ op(Z, b) for a, b in PAIRS) - h*sum(op(X, q) for q in range(n))
def energy(th, g, rev, place):
    Ks = kraus(g, rev); t0, rest = th[:n], th[n:].reshape(B, len(PAIRS) + n)
    rho = np.zeros((8, 8), complex); rho[0, 0] = 1
    for q in range(n): rho = U(rho, op(ry(t0[q]), q))
    for b in range(B):
        for k, (c, t) in enumerate(PAIRS):
            rho = U(rho, cnot(c, t))
            if place == 'mid': rho = chan(chan(rho, Ks, c), Ks, t)
            rho = U(rho, op(rz(rest[b, k]), t)); rho = U(rho, cnot(c, t))
            if place == 'mid': rho = chan(chan(rho, Ks, c), Ks, t)
            else:
                for _ in range(2): rho = chan(chan(rho, Ks, c), Ks, t)
        for q in range(n): rho = U(rho, op(rx(rest[b, len(PAIRS) + q]), q))
    return np.real(np.trace(H @ rho))
rng = np.random.default_rng(69)
print('E0 =', np.linalg.eigvalsh(H)[0])
for place in ('post', 'mid'):
    for g in (0.01, 0.05, 0.1, 0.2):
        d = []
        for _ in range(20):
            th = rng.uniform(0, 2*np.pi, n + B*(len(PAIRS) + n)); thp = th.copy(); thp[:n] = np.pi - th[:n]
            d.append(abs(energy(th, g, True, place) - energy(thp, g, False, place)))
        print(f'{place:4s} g={g}: max|E_rev(th) - E_AD(th\')| = {max(d):.2e}')
