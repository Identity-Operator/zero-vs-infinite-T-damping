# -*- coding: utf-8 -*-
"""
T1 check: exact Heisenberg Pauli propagation of Z_j for the T1 layout, one theta
at a time, batched over images. Z_j is a vector over the 4^n Pauli strings; gates and channels act
as real 4x4 site maps (Tr(P_b U^dag P_a U)/2 and heis_channel), the CNOT ring as a signed
permutation of strings, and z_j = sum of the coefficients on {I, Z}^n. Agrees with the
density-matrix simulator of t1_scale_ratio.py to 4e-16. Used for the per-theta distribution of
the twirl's feature scale at n=8, L=4 (t1_note.md, comparison with S6a):

    python pauli_prop.py          # n=8, L=4, twirl, first 60 images, 120 draws (about 30 min)
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_scale_ratio as T

C2 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)

def signed_cnot(n, c, t):
    rule = {}
    for pc in range(4):
        for pt in range(4):
            Q = C2.conj().T @ np.kron(T.PAULI[pc], T.PAULI[pt]) @ C2
            for qc in range(4):
                for qt in range(4):
                    v = np.trace(np.kron(T.PAULI[qc], T.PAULI[qt]).conj().T @ Q) / 4
                    if abs(abs(v) - 1) < 1e-9:
                        assert abs(v.imag) < 1e-9
                        rule[(pc, pt)] = (qc, qt, v.real)
    idx = np.arange(4 ** n); digs = (idx[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4
    new = digs.copy(); sign = np.ones(4 ** n)
    for k in range(4 ** n):
        qc, qt, sg = rule[(digs[k, c], digs[k, t])]; new[k, c], new[k, t] = qc, qt; sign[k] = sg
    perm = (new * 4 ** np.arange(n - 1, -1, -1)).sum(1)
    return perm, sign

def site(V, M, q, n):
    b = V.shape[0]
    if M.ndim == 2:
        return np.matmul(M, V.reshape(b * 4 ** q, 4, -1)).reshape(b, -1)
    return np.matmul(M[:, None], V.reshape(b, 4 ** q, 4, -1)).reshape(b, -1)

class Prop:
    def __init__(self, n, ch):
        self.n = n; self.K = T.heis_channel(ch)
        ring = [signed_cnot(n, j, (j + 1) % n) for j in range(n)]
        self.ring = [(np.argsort(pm), sg[np.argsort(pm)]) for pm, sg in ring]   # gather form: V'[:, i] = sg[inv[i]] V[:, inv[i]]
        digs = (np.arange(4 ** n)[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4
        self.zmask = np.all((digs == 0) | (digs == 3), axis=1)

    def z(self, A, theta, L, j):
        n = self.n; B = A.shape[0]; th = theta.reshape(L + 1, n, 3)
        Kenc = np.array([[T.adj(T.rx(a)) for a in A[:, q]] for q in range(n)])     # (n, B, 4, 4)
        blk = lambda l, q: T.adj(T.rz(th[l, q, 2]) @ T.ry(th[l, q, 1]) @ T.rz(th[l, q, 0]))
        V = np.zeros((B, 4 ** n)); V[:, 3 * 4 ** (n - 1 - j)] = 1.0                 # Z_j
        for q in range(n): V = site(V, self.K, q, n)                                # trailing channel
        for q in range(n): V = site(V, blk(L, q), q, n)                             # trailing block
        for l in range(L - 1, -1, -1):
            for q in range(n): V = site(V, self.K, q, n)                            # post-encoding channel
            for q in range(n): V = site(V, Kenc[q], q, n)                           # R_X(a)
            for q in range(n): V = site(V, self.K, q, n)                            # post-entangler channel
            for inv, sg in reversed(self.ring): V = np.take(V, inv, axis=1) * sg      # CNOT ring, backwards
            for q in range(n): V = site(V, blk(l, q), q, n)                         # block l
        return V[:, self.zmask].sum(1)


if __name__ == '__main__':
    n, L, N, D = 8, 4, 60, 120
    A = T.pca_angles(n)[:N]
    chain, _ = T.chain_sigma(A, L, 'twirl', n)
    P = Prop(n, 'twirl'); rng = np.random.default_rng(88)
    vq = np.array([np.stack([P.z(A, th, L, j) for j in range(n)], 1).var(0).mean()
                   for th in (rng.uniform(0, 2 * np.pi, 3 * n * (L + 1)) for _ in range(D))])
    rms = np.sqrt(vq.mean()); se = vq.std(ddof=1) / np.sqrt(D) / (2 * rms)
    b5 = np.sqrt(vq.reshape(-1, 5).mean(1)) / rms
    print(f"n={n} L={L} twirl: chain {chain[L-1]:.4e} | {D}-draw rms {rms:.4e} +- {se:.1e}")
    print(f"  mean/median {vq.mean()/np.median(vq):.2f}, max/mean {vq.max()/vq.mean():.1f}, "
          f"top 5% of draws carry {np.sort(vq)[-D//20:].sum()/vq.sum():.0%}; "
          f"5-draw rms / full: median {np.median(b5):.2f}, P10 {np.percentile(b5, 10):.2f}, <= 0.70 in {np.mean(b5 <= 0.70):.0%}")
