"""Independent check of the time-reversal gauge. Our classifier layout:
R_l (general SU(2) per qubit) -> ring E -> N -> S(a) -> N, final R_L -> N, read <Z_j>. Claim: with
Theta = (x)(-iY)K, z_flip(a; R) = -z_AD(a; R'') for ANY encoding axis, where R''_l = P_l . Y R_l^* Y
(P_l the Pauli residue of Theta E Theta^-1 = E P_l, pushed before E) and R''_0 also absorbs Theta|0> ~ X|0>.
Also: noise inside the ring (after each CNOT) should break it."""
import numpy as np, itertools, functools
n, L, p = 3, 3, 0.3
I2 = np.eye(2); X = np.array([[0, 1], [1, 0]], complex); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.]).astype(complex)
kron = lambda ms: functools.reduce(np.kron, ms)
def op(s, q): return kron([s if k == q else I2 for k in range(n)])
def cnot(c, t): return op(np.diag([1., 0.]), c) + op(np.diag([0., 1.]), c) @ op(X, t)
def rot(axis, a): return np.cos(a/2)*I2 - 1j*np.sin(a/2)*axis
def su2(rng):
    q, r = np.linalg.qr(rng.normal(size=(2, 2)) + 1j*rng.normal(size=(2, 2))); return q @ np.diag(np.diag(r)/abs(np.diag(r)))
K_AD = [np.array([[1, 0], [0, np.sqrt(1-p)]], complex), np.array([[0, np.sqrt(p)], [0, 0]], complex)]
K_FL = [X @ K @ X for K in K_AD]
def noise(rho, Ks):
    for q in range(n): rho = sum(op(K, q) @ rho @ op(K, q).conj().T for K in Ks)
    return rho
PAIRS = [(j, (j+1) % n) for j in range(n)]
def run(R, a, axis, Ks, inside=False):
    rho = np.zeros((2**n, 2**n), complex); rho[0, 0] = 1
    for l in range(L):
        U = kron(R[l]); rho = U @ rho @ U.conj().T
        for c, t in PAIRS:
            G = cnot(c, t); rho = G @ rho @ G.conj().T
            if inside: rho = noise(rho, Ks)
        if not inside: rho = noise(rho, Ks)
        S = kron([rot(axis, a[j]) for j in range(n)]); rho = noise(S @ rho @ S.conj().T, Ks)
    U = kron(R[L]); rho = noise(U @ rho @ U.conj().T, Ks)
    return np.array([np.real(np.trace(op(Z, j) @ rho)) for j in range(n)])
# Pauli residue P with Theta E Theta^-1 = E P, where E = prod CNOT (applied in PAIRS order); Theta M Theta^-1 = Yn M^* Yn
Yn = kron([Y]*n); E = functools.reduce(lambda A, B: B @ A, [cnot(c, t) for c, t in PAIRS], np.eye(2**n))
P = np.linalg.inv(E) @ (Yn @ E.conj() @ Yn)
# factor P into single-qubit Paulis (up to phase)
paulis = {'I': I2, 'X': X, 'Y': Y, 'Z': Z}
fac = None
for combo in itertools.product('IXYZ', repeat=n):
    M = kron([paulis[s] for s in combo]); ph = np.trace(M.conj().T @ P)/2**n
    if abs(abs(ph) - 1) < 1e-9: fac = (combo, ph); break
print('ring residue P =', ''.join(fac[0]), 'phase', np.round(fac[1], 6))
rng = np.random.default_rng(69)
for axname, axis in (('rx', X), ('ry', Y), ('rz', Z), ('generic', (X + 2*Y - 0.5*Z)/np.linalg.norm([1, 2, -0.5]))):
    worst = 0; worst_in = 0
    for _ in range(10):
        R = [[su2(rng) for _ in range(n)] for _ in range(L + 1)]; a = rng.uniform(0, 2*np.pi, n)
        Rpp = [[Y @ R[l][j].conj() @ Y for j in range(n)] for l in range(L + 1)]
        for l in range(L): Rpp[l] = [paulis[fac[0][j]] @ Rpp[l][j] for j in range(n)]
        Rpp[0] = [Rpp[0][j] @ X for j in range(n)]
        worst = max(worst, np.max(abs(run(R, a, axis, K_FL) + run(Rpp, a, axis, K_AD))))
        worst_in = max(worst_in, np.max(abs(run(R, a, axis, K_FL, True) + run(Rpp, a, axis, K_AD, True))))
    print(f'{axname:8s} noise after ring: max|z_flip + z_AD(R\'\')| = {worst:.1e}   noise inside ring: {worst_in:.1e}')
