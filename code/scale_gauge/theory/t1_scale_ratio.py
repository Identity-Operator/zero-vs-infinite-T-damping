# -*- coding: utf-8 -*-
"""
T1: the before-training feature scale of amplitude
damping (AD) against its Pauli twirl (same T, t = 0), as a function of n and L, for our
layout: per layer RZ-RY-RZ blocks -> ring of CNOTs -> channel -> RX(a_j) -> channel; a
trailing block -> channel; Z_j readout; theta uniform in [0, 2 pi). The scale of qubit j at fixed
theta is the std over inputs of z_j. See t1_note.md for the results and compare_s6a.py for S6a.

Four parts, all independent of TorchCirc:

lead   LEADING-ORDER AD (analytic). Propagating Z_j backwards, the last channel gives
       (1-p) Z_j + p I, the trailing block rotates Z_j to r.sigma, the post-encoding channel and
       R_X(a_j) give a Z coefficient (1-p)[(1-p) r_z cos a_j - c r_y sin a_j], and the
       post-entangler channel of the LAST layer branches it into the identity:
           delta z_j = p (1-p) [ (1-p) r_z cos a_j - c r_y sin a_j ],   c = sqrt(1-p).
       <0|I|0> = 1, so this data-dependent term reaches z_j whatever the earlier layers do; it is
       independent of L and of n. sigma_lead = E_theta E_j sqrt(Var_a[delta z_j]).
decay  TWIRL DECAY RATE (Markov transfer operator). Lambda_n = sqrt(nu_n), nu_n the leading
       eigenvalue of the layer map on Pauli-string second moments, with the encoding treated as
       independent in every layer. Power iteration on the 4^n vector (n <= 8).
chain  EXACT PAIR CHAIN (section 2b). E_theta[z_j(a) z_j(a')] propagated exactly on diagonal pair
       moments, with the same a re-uploaded in every layer; gives sqrt(E_theta mean_j Var_a z_j)
       for AD, the twirl and None, n = 2..8, L = 1..6. Also establishes sigma_AD^2 =
       sigma_twirl^2 + sigma_branch^2 (the cross moment is killed by the next uniform block).
exact  DENSITY-MATRIX NUMERICS (validation). Batched simulation on the first 300 training images
       (v5 preprocessing, n PCA components), n = 2..6, L = 1..6: E_theta sigma, its standard error,
       and the rms sqrt(E_theta mean_j Var).

    python t1_scale_ratio.py [lead] [decay] [exact] [chain]   # writes t1_results.json next to this file
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = 0.3
I2 = np.eye(2, dtype=complex); X = np.array([[0, 1], [1, 0]], complex)
Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.]).astype(complex); PAULI = [I2, X, Y, Z]


def rz(t): return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])
def ry(t): return np.array([[np.cos(t / 2), -np.sin(t / 2)], [np.sin(t / 2), np.cos(t / 2)]], complex)
def rx(t): return np.cos(t / 2) * I2 - 1j * np.sin(t / 2) * X


def kraus(ch, p=P):
    c = np.sqrt(1 - p)
    if ch == 'AD':
        return [np.diag([1, c]).astype(complex), np.sqrt(p) * np.array([[0, 1], [0, 0]], complex)]
    if ch == 'twirl':
        q, pz = p / 4, (2 - p - 2 * c) / 4
        return [np.sqrt(w) * M for w, M in zip([1 - 2 * q - pz, q, q, pz], PAULI)]
    if ch == 'None':
        return [I2]
    raise ValueError(ch)


# ------------------------------------------------------------------ data
def pca_angles(n, n_img=300):
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'noise_structure_mnist'))
    cwd = os.getcwd(); os.chdir(os.path.join(HERE, '..', '..', 'noise_structure_mnist'))
    try:
        from mnist_hybrid_multi import load_mnist_binary
        Xtr, _, _, _ = load_mnist_binary(n_train=1000, n_test=1, n_components=n, pca_seed=0)
    finally:
        os.chdir(cwd)
    return Xtr[:n_img]


# ------------------------------------------------------------------ 1. leading-order AD
def sigma_lead(A, p=P, n_theta=4000, seed=0):
    """E over theta and qubits of sqrt(Var_a[p(1-p)((1-p) r_z cos a - c r_y sin a)])."""
    rng = np.random.default_rng(seed); c = np.sqrt(1 - p); out = []
    for j in range(A.shape[1]):
        cs, sn = np.cos(A[:, j]), np.sin(A[:, j])
        vcc, vss = cs.var(), sn.var(); cov = np.mean((cs - cs.mean()) * (sn - sn.mean()))
        t1, t2 = rng.uniform(0, 2 * np.pi, n_theta), rng.uniform(0, 2 * np.pi, n_theta)
        a_, b_ = (1 - p) * np.cos(t2), -c * np.sin(t2) * np.sin(t1)
        out.append(np.mean(p * (1 - p) * np.sqrt(np.maximum(a_ ** 2 * vcc + b_ ** 2 * vss + 2 * a_ * b_ * cov, 0))))
    return float(np.mean(out))


def floor_formula(A, p=P):
    """Closed-form last-layer branching term, rms over theta and qubits (t1_note.md, statement 2):
    sigma_inf^2 = p^2 (1-p)^2 mean_j [ (1-p)^2/2 Var(cos a_j) + (1-p)/4 Var(sin a_j) ]."""
    v = [(1 - p) ** 2 / 2 * np.cos(A[:, j]).var() + (1 - p) / 4 * np.sin(A[:, j]).var() for j in range(A.shape[1])]
    return float(p * (1 - p) * np.sqrt(np.mean(v)))


# ------------------------------------------------------------------ 2. twirl transfer operator
def single_qubit_moment(Us):
    """4x4 matrix M[b, a] = mean over the unitaries Us of (Tr(P_b U^dag P_a U)/2)^2: the Heisenberg
    second-moment transfer of one site. For angles uniform on [0, 2 pi) an 8-point grid per angle is
    exact (the squared adjoint entries are trigonometric polynomials of degree 2 in each angle)."""
    M = np.zeros((4, 4))
    for U in Us:
        for a in range(4):
            Q = U.conj().T @ PAULI[a] @ U
            M[:, a] += np.array([(np.trace(PAULI[b] @ Q).real / 2) ** 2 for b in range(4)])
    return M / len(Us)


GRID = 2 * np.pi * np.arange(8) / 8
M_ROT = single_qubit_moment([rz(t2) @ ry(t1) @ rz(t0) for t0 in GRID for t1 in GRID for t2 in GRID])
M_ENC_UNIFORM = single_qubit_moment([rx(a) for a in GRID])


def cnot_perm(n, c, t):
    """Heisenberg image of each Pauli string under CNOT(c,t), as an index permutation."""
    # single-qubit symplectic rules for CNOT: (ctrl, tgt) Pauli pair -> image pair (signs dropped)
    rule = {}
    for pc in range(4):
        for pt in range(4):
            Pm = np.kron(PAULI[pc], PAULI[pt])
            C = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], complex)
            Q = C.conj().T @ Pm @ C
            for qc in range(4):
                for qt in range(4):
                    if abs(abs(np.trace(np.kron(PAULI[qc], PAULI[qt]).conj().T @ Q)) / 4 - 1) < 1e-9:
                        rule[(pc, pt)] = (qc, qt)
    perm = np.empty(4 ** n, dtype=np.int64)
    for idx in range(4 ** n):
        digs = [(idx // 4 ** (n - 1 - k)) % 4 for k in range(n)]
        qc, qt = rule[(digs[c], digs[t])]
        digs[c], digs[t] = qc, qt
        perm[idx] = sum(d * 4 ** (n - 1 - k) for k, d in enumerate(digs))
    return perm


def apply_site(v, M, q, n):
    T = v.reshape((4,) * n)
    return np.moveaxis(np.tensordot(M, T, axes=([1], [q])), 0, q).reshape(-1)


def twirl_decay(n, Menc=None, p=P, iters=400):
    """Menc: one 4x4 encoding matrix, or a list of n (one per qubit, from the data angles);
    default uniform angles."""
    c2 = 1 - p                                             # c^2
    Dn = np.diag([1.0, c2, c2, c2 ** 2])                   # squared twirl contractions (I,X,Y,Z)
    Menc = [M_ENC_UNIFORM] * n if Menc is None else Menc
    ring = [(j, (j + 1) % n) for j in range(n)]
    perms = [cnot_perm(n, cc, tt) for cc, tt in ring]
    v = np.ones(4 ** n); v[0] = 0; v /= v.sum(); nu = None
    for _ in range(iters):                                 # one layer, Heisenberg order
        w = v.copy()
        for q in range(n): w = apply_site(w, Dn, q, n)     # post-encoding channel
        for q in range(n): w = apply_site(w, Menc[q], q, n)  # R_X(a)
        for q in range(n): w = apply_site(w, Dn, q, n)     # post-entangler channel
        for pm in reversed(perms):                          # ring (last CNOT first, backwards)
            w2 = np.empty_like(w); w2[pm] = w; w = w2
        for q in range(n): w = apply_site(w, M_ROT, q, n)  # trainable block
        w[0] = 0.0                                          # identity carries no data dependence
        nu = w.sum() / v.sum(); v = w / w.sum()
    return float(nu), float(np.sqrt(nu))


# ------------------------------------------------------------------ 2b. exact pair chain
# E_theta[z_j(a) z_j(a')] for two inputs a, a' and a shared theta, propagated in the Heisenberg picture
# on the diagonal second moments s_P = E_theta[o_P(a) o_P(a')] of Pauli-string coefficients. This is
# exact: every uniform RZ-RY-RZ block kills the moments between different strings (first moments of
# its adjoint entries vanish and different adjoint columns are orthogonal on average), and between two
# blocks the circuit is the site-local map G(a) = K . A(a) . K (channel, R_X(a), channel) followed by the
# CNOT ring (a signed permutation of strings). So the pair transfer per site is the elementwise product
# G(a) o G(a'), exact also for AD, whose branching Z -> I creates (I, Z) cross moments inside G. The
# same a is re-uploaded in every layer (no Markov approximation), and
#     E_theta Var_a[z_j] = mean_a S(a, a) - mean_{a, a'} S(a, a').
def adj(U):
    """K[b, a] = Tr(P_b U^dag P_a U) / 2: Heisenberg coefficient map of a single-qubit unitary."""
    return np.array([[np.trace(PAULI[b] @ U.conj().T @ PAULI[a] @ U).real / 2 for a in range(4)] for b in range(4)])


def heis_channel(ch, p=P):
    c = np.sqrt(1 - p)
    if ch == 'AD':
        K = np.diag([1, c, c, 1 - p]); K[0, 3] = p; return K       # AD^dag(Z) = (1-p) Z + p I
    if ch == 'twirl':
        return np.diag([1, c, c, 1 - p])
    return np.eye(4)


def pair_chain(A, pairs, Lmax, ch, n, per_qubit=False, chunk=None):
    """S[L-1, j, k] = E_theta[z_j(A[i_k]) z_j(A[i'_k])] for L = 1..Lmax, pairs = (i, i').
    per_qubit=False returns the mean over j in one chain (index j of size 1): the chain is linear
    in its starting vector, so the qubit average costs one propagation instead of n."""
    K = heis_channel(ch); i1, i2 = pairs; nP = len(i1)
    G = np.array([[K @ adj(rx(a)) @ K for a in row] for row in A])        # (N, n, 4, 4)
    inv = [np.argsort(cnot_perm(n, j, (j + 1) % n)) for j in range(n)]    # gather form of each CNOT
    digits = (np.arange(4 ** n)[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4
    zmask = np.all((digits == 0) | (digits == 3), axis=1)                 # strings in {I, Z}^n
    site = M_ROT @ (K[:, 3] ** 2)                                         # trailing channel and block
    starts = []
    for j in range(n):
        full = np.ones(1)
        for q in range(n):
            full = np.kron(full, site if q == j else np.eye(4)[0])
        starts.append(full)
    starts = starts if per_qubit else [np.mean(starts, axis=0)]
    chunk = chunk or max(1, int(2e7 // 4 ** n))
    S = np.zeros((Lmax, len(starts), nP))
    for js, v0 in enumerate(starts):
        for s0 in range(0, nP, chunk):
            sl = slice(s0, min(nP, s0 + chunk)); b = sl.stop - sl.start
            Mmid = G[i1[sl]] * G[i2[sl]]                                    # (b, n, 4, 4)
            V = np.repeat(v0[None], b, axis=0)
            for L in range(1, Lmax + 1):
                for q in range(n):                                          # channel, R_X(a), channel
                    V = np.matmul(Mmid[:, q][:, None], V.reshape(b, 4 ** q, 4, -1)).reshape(b, -1)
                for iv in reversed(inv):                                    # CNOT ring, backwards
                    V = np.take(V, iv, axis=1)
                for q in range(n):                                          # RZ-RY-RZ block
                    V = np.matmul(M_ROT, V.reshape(b * 4 ** q, 4, -1)).reshape(b, -1)
                S[L - 1, js, sl] = V[:, zmask].sum(1)
    return S


def chain_sigma(A, Lmax, ch, n, n_off=None, seed=0):
    """sqrt of E_theta Var_a[z_j], averaged over j (rms), for L = 1..Lmax; exact up to pair sampling."""
    N = A.shape[0]; rng = np.random.default_rng(seed)
    if n_off is None:
        iu = np.triu_indices(N, 1); off = (iu[0], iu[1])
    else:
        a = rng.integers(0, N, 3 * n_off); b = rng.integers(0, N, 3 * n_off); keep = a != b
        off = (a[keep][:n_off], b[keep][:n_off])
    Sd = pair_chain(A, (np.arange(N), np.arange(N)), Lmax, ch, n)          # (Lmax, n, N)
    So = pair_chain(A, off, Lmax, ch, n)
    m_all = (Sd.sum(2) + (N * N - N) * So.mean(2)) / N ** 2
    var = Sd.mean(2) - m_all                                                # (Lmax, n)
    se = (N * N - N) / N ** 2 * So.std(2, ddof=1) / np.sqrt(So.shape[2]) if n_off else np.zeros_like(var)
    return np.sqrt(np.maximum(var.mean(1), 0)), np.sqrt(np.mean(se ** 2, 1))


# ------------------------------------------------------------------ 3. exact numerics
def apply_1q(rho, U, q, n):
    """rho: (B, 2^n, 2^n); U: (2,2) or (B,2,2)."""
    B = rho.shape[0]
    Ub = U if U.ndim == 3 else np.broadcast_to(U, (B, 2, 2))
    R = rho.reshape(B, 2 ** q, 2, 2 ** (n - q - 1), 2 ** n)
    R = np.einsum('bij,bxjyz->bxiyz', Ub, R).reshape(B, 2 ** n, 2 ** n)
    R = R.reshape(B, 2 ** n, 2 ** q, 2, 2 ** (n - q - 1))
    R = np.einsum('bxyjz,bij->bxyiz', R, Ub.conj()).reshape(B, 2 ** n, 2 ** n)
    return R


def apply_channel(rho, K, q, n):
    return sum(apply_1q(rho, k, q, n) for k in K)


def cnot_index(n, c, t):
    idx = np.arange(2 ** n); bc = (idx >> (n - 1 - c)) & 1
    return idx ^ (bc << (n - 1 - t))


def features(A, theta, L, ch, n):
    B = A.shape[0]; D = 2 ** n; K = kraus(ch)
    rho = np.zeros((B, D, D), complex); rho[:, 0, 0] = 1
    perms = [cnot_index(n, j, (j + 1) % n) for j in range(n)]
    th = theta.reshape(L + 1, n, 3)
    for l in range(L + 1):
        for q in range(n):
            rho = apply_1q(rho, rz(th[l, q, 2]) @ ry(th[l, q, 1]) @ rz(th[l, q, 0]), q, n)
        if l < L:
            for pm in perms:
                rho = rho[:, pm][:, :, pm]
        for q in range(n):
            rho = apply_channel(rho, K, q, n)
        if l < L:
            for q in range(n):
                Ub = np.stack([rx(a) for a in A[:, q]])
                rho = apply_1q(rho, Ub, q, n)
            for q in range(n):
                rho = apply_channel(rho, K, q, n)
    zs = []
    for q in range(n):
        d = np.einsum('bii->bi', rho).real.reshape(B, 2 ** q, 2, -1)
        zs.append(d[:, :, 0, :].sum((1, 2)) - d[:, :, 1, :].sum((1, 2)))
    return np.stack(zs, 1)                                  # (B, n)


def sigma(A, L, ch, n, n_theta, seed=0):
    rng = np.random.default_rng(seed); s = []
    for _ in range(n_theta):
        z = features(A, rng.uniform(0, 2 * np.pi, 3 * n * (L + 1)), L, ch, n)
        s.append((z.std(0).mean(), z.var(0).mean()))
    s = np.array(s)
    return float(s[:, 0].mean()), float(s[:, 0].std(ddof=1) / np.sqrt(n_theta)), float(np.sqrt(s[:, 1].mean()))


def main():
    """python t1_scale_ratio.py [lead] [decay] [exact] [chain]   (default: all four)"""
    t0 = time.time(); out = os.path.join(HERE, 't1_results.json')
    parts = sys.argv[1:] or ['lead', 'decay', 'exact', 'chain']
    R = json.load(open(out)) if os.path.exists(out) else {}
    R['p'] = P
    save = lambda: json.dump(R, open(out, 'w'), indent=1)
    if 'lead' in parts:
        print('1. leading-order AD sigma (L- and n-independent formula, per-n data angles):'); R['lead'] = {}
        for n in (2, 3, 4, 5, 6, 7, 8):
            A = pca_angles(n); R['lead'][n] = sigma_lead(A); R.setdefault('floor', {})[n] = floor_formula(A)
            u = np.random.default_rng(9).uniform(0, 2 * np.pi, (4000, n))
            print(f'   n={n}: sigma_lead = {R["lead"][n]:.4f}  (uniform angles: {sigma_lead(u):.4f})', flush=True)
        save()
    if 'decay' in parts:
        print('2. twirl per-layer decay Lambda_n (Markov transfer operator; uniform / data angles):'); R['decay'] = {}
        for n in (2, 3, 4, 5, 6, 7, 8):
            A = pca_angles(n)
            Md = [single_qubit_moment([rx(a) for a in A[:, q]]) for q in range(n)]
            nu_u, lam_u = twirl_decay(n); nu_d, lam_d = twirl_decay(n, Md)
            R['decay'][n] = {'nu_uniform': nu_u, 'Lambda_uniform': lam_u, 'nu_data': nu_d, 'Lambda_data': lam_d}
            print(f'   n={n}: Lambda_n = {lam_u:.4f} (uniform)  {lam_d:.4f} (data)', flush=True)
        save()
    if 'exact' in parts:
        print('3. exact density-matrix numerics (first 300 training images): mean_theta sigma +- se | rms'); R['exact'] = {}
        for n in (2, 3, 4, 5, 6):
            A = pca_angles(n); R['exact'][n] = {}
            n_theta = 24 if n <= 4 else 10
            for L in range(1, 7):
                row = {ch: sigma(A, L, ch, n, n_theta, seed=100 * n + L) for ch in ('AD', 'twirl', 'None')}
                R['exact'][n][L] = row
                print(f'   n={n} L={L}: ' + '  '.join(f'{ch} {row[ch][0]:.3e}+-{row[ch][1]:.1e} | {row[ch][2]:.3e}'
                                                    for ch in ('AD', 'twirl', 'None')) + f'   ({time.time()-t0:.0f}s)', flush=True)
            save()
    if 'chain' in parts:
        print('4. exact pair chain: sqrt(E_theta mean_j Var_a z_j), L = 1..6'); R['chain'] = {}
        for n in (2, 3, 4, 5, 6, 7, 8):
            A = pca_angles(n); n_off = None if n <= 5 else 5000; R['chain'][n] = {}
            for ch in ('AD', 'twirl', 'None'):
                s_, se = chain_sigma(A, 6, ch, n, n_off=n_off, seed=n)
                R['chain'][n][ch] = {'rms': s_.tolist(), 'se_var': se.tolist(), 'n_off': n_off}
                print(f'   n={n} {ch:5s}: ' + '  '.join(f'{x:.3e}' for x in s_) + f'   ({time.time()-t0:.0f}s)', flush=True)
            save()
    R['runtime_s_last'] = time.time() - t0; save()
    print(f'wrote t1_results.json ({time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()
