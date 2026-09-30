# -*- coding: utf-8 -*-
"""
T1b: dependence on the noise strength p of the
before-training feature scales of AD, its Pauli twirl and the noiseless circuit, from the exact pair
chain of t1_scale_ratio.py, stepped layer by layer until the AD and twirl scales have separated.

For each p it records sigma_AD(L), sigma_twirl(L) (rms scales, exact up to the pair set), the
branching part sigma_br = sqrt(sigma_AD^2 - sigma_twirl^2), the closed-form last-layer floor
sigma_inf(p) (Eq. floor_var), and two separation depths:
  L_star  = smallest L with (sigma_AD/sigma_twirl)^2 >= 10 (the twirl needs 10x the shots of AD);
  L_shot  = smallest L with sigma_twirl < 1/sqrt(N_s) <= sigma_AD at N_s = 1e3 (None if never).
The per-layer decay rate of the twirl is also taken from the Markov transfer operator at each p.

    python t1_p_dependence.py [n] [n_off] [p_min]   # default n=4, all image pairs; writes t1_p_dependence_n{n}.json
    python t1_p_dependence.py 4 5000                 # as run (n=4, 5000 random off-diagonal pairs)
    python t1_p_dependence.py 8 700 0.01             # as run (n=8, p >= 0.01; the chain is 4^8-dimensional)
    python t1_p_dependence.py summary [n]            # markdown table from the JSON (numpy only, seconds)

Memory: every pair's 4^n-vector stays resident for each live chain (AD and twirl together). Measured
with /usr/bin/time: n=4, 5000 pairs, 1.19 GB; n=8, 700 pairs, 3.57 GB as run, before the unused
None chain was freed (about 2.1 GB expected now; not re-measured).
"""
import json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import t1_scale_ratio as T

PS = (1e-3, 3e-3, 1e-2, 3e-2, 0.05, 0.1, 0.2, 0.3)
NS = 1e3


class Chain:
    """E_theta[z(a) z(a')] averaged over readout qubits, for a fixed pair set, one layer per step."""

    def __init__(self, A, pairs, ch, n, p):
        K = T.heis_channel(ch, p); i1, i2 = pairs; self.n = n
        G = np.array([[K @ T.adj(T.rx(a)) @ K for a in row] for row in A])
        self.M = G[i1] * G[i2]                                             # (b, n, 4, 4)
        self.inv = [np.argsort(T.cnot_perm(n, j, (j + 1) % n)) for j in range(n)]
        digits = (np.arange(4 ** n)[:, None] // 4 ** np.arange(n - 1, -1, -1)) % 4
        self.zmask = np.all((digits == 0) | (digits == 3), axis=1)
        site = T.M_ROT @ (K[:, 3] ** 2); starts = []
        for j in range(n):
            full = np.ones(1)
            for q in range(n):
                full = np.kron(full, site if q == j else np.eye(4)[0])
            starts.append(full)
        self.V = np.repeat(np.mean(starts, axis=0)[None], len(i1), axis=0)

    def step(self):
        b, n = self.V.shape[0], self.n; V = self.V
        for q in range(n):
            V = np.matmul(self.M[:, q][:, None], V.reshape(b, 4 ** q, 4, -1)).reshape(b, -1)
        for iv in reversed(self.inv):
            V = np.take(V, iv, axis=1)
        for q in range(n):
            V = np.matmul(T.M_ROT, V.reshape(b * 4 ** q, 4, -1)).reshape(b, -1)
        self.V = V
        return V[:, self.zmask].sum(1)


class Sigma:
    """rms scale sqrt(E_theta mean_j Var_a z_j), exact up to the off-diagonal pair set; the pairs are
    propagated in chunks of at most `chunk` to bound memory (4^n doubles per pair)."""

    def __init__(self, A, ch, n, p, n_off=None, seed=0, chunk=None):
        N = A.shape[0]; self.N = N
        if n_off is None:
            iu = np.triu_indices(N, 1); off = (iu[0], iu[1])
        else:
            r = np.random.default_rng(seed); a = r.integers(0, N, 3 * n_off); c = r.integers(0, N, 3 * n_off)
            k = a != c; off = (a[k][:n_off], c[k][:n_off])
        idx = np.arange(N)
        i1, i2 = np.concatenate([idx, off[0]]), np.concatenate([idx, off[1]])
        chunk = chunk or max(1, int(1e7 // 4 ** n))
        self.chains = [Chain(A, (i1[s:s + chunk], i2[s:s + chunk]), ch, n, p) for s in range(0, len(i1), chunk)]

    def step(self):
        S = np.concatenate([c.step() for c in self.chains]); N = self.N; Sd, So = S[:N], S[N:]
        var = Sd.mean() - (Sd.sum() + (N * N - N) * So.mean()) / N ** 2
        return float(np.sqrt(max(var, 0.0)))


def markov_rate(n, p, Md):
    """Per-layer decay of the twirl from the Markov transfer operator (data-angle encoding)."""
    c2 = 1 - p; Dn = np.diag([1.0, c2, c2, c2 ** 2])
    perms = [T.cnot_perm(n, j, (j + 1) % n) for j in range(n)]
    v = np.ones(4 ** n); v[0] = 0; v /= v.sum(); nu = 1.0
    for _ in range(3000):
        w = v.copy()
        for q in range(n): w = T.apply_site(w, Dn, q, n)
        for q in range(n): w = T.apply_site(w, Md[q], q, n)
        for q in range(n): w = T.apply_site(w, Dn, q, n)
        for pm in reversed(perms):
            w2 = np.empty_like(w); w2[pm] = w; w = w2
        for q in range(n): w = T.apply_site(w, T.M_ROT, q, n)
        w[0] = 0.0; nu_new = w.sum() / v.sum(); v = w / w.sum()
        if abs(nu_new - nu) < 1e-11 * nu_new:
            break
        nu = nu_new
    return float(np.sqrt(nu_new))


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4
    n_off = int(sys.argv[2]) if len(sys.argv) > 2 else None
    p_min = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0          # skip p below this (cost)
    r2_stop = 1e3 if n <= 5 else 1e2
    t0 = time.time(); A = T.pca_angles(n)
    out = os.path.join(HERE, f't1_p_dependence_n{n}.json')
    R = {'n': n, 'n_off': n_off, 'N_s': NS, 'p': {}}
    Md = [T.single_qubit_moment([T.rx(a) for a in A[:, q]]) for q in range(n)]
    sn = Sigma(A, 'None', n, 0.0, n_off); R['None'] = [sn.step() for _ in range(30)]; del sn      # free its chains
    print(f"n={n}: sigma_None L=1..30: {R['None'][0]:.4f} ... {R['None'][-1]:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    for p in sorted((q for q in PS if q >= p_min), reverse=True):
        sa, st = Sigma(A, 'AD', n, p, n_off), Sigma(A, 'twirl', n, p, n_off)
        ad, tw = [], []; L_star = L_shot = None; L = 0
        while True:
            L += 1; ad.append(sa.step()); tw.append(st.step())
            r2 = (ad[-1] / tw[-1]) ** 2 if tw[-1] > 0 else np.inf
            if L_star is None and r2 >= 10:
                L_star = L
            if L_shot is None and tw[-1] < NS ** -0.5 <= ad[-1]:
                L_shot = L
            if (r2 >= r2_stop and L >= 6) or L >= 20000:
                break
        ad, tw = np.array(ad), np.array(tw); br = np.sqrt(np.maximum(ad ** 2 - tw ** 2, 0))
        lam = markov_rate(n, p, Md)
        R['p'][repr(p)] = {'AD': ad.tolist(), 'twirl': tw.tolist(), 'br': br.tolist(), 'L_star': L_star, 'L_shot': L_shot,
                           'floor_closed': T.floor_formula(A, p), 'Lambda_markov': lam,
                           'Lambda_exact_tail': float((tw[-1] / tw[-6]) ** 0.2) if len(tw) > 6 else None}
        print(f"p={p:<6g} L*={L_star}  L_shot={L_shot}  br(L*)={br[L_star-1]:.4f}  br(end)={br[-1]:.4f}  floor_closed={T.floor_formula(A, p):.4f}  "
              f"Lambda markov={lam:.5f} tail={R['p'][repr(p)]['Lambda_exact_tail']}  twirl(L*)={tw[L_star-1]:.3e}  L_end={L}  ({time.time()-t0:.0f}s)", flush=True)
        json.dump(R, open(out, 'w'))
        del sa, st                                                          # free before the next p
    print(f"wrote {out} ({time.time()-t0:.0f}s)")


def summary(n):
    """Markdown table from t1_p_dependence_n{n}.json (numpy only): rates, floors, separation depths."""
    R = json.load(open(os.path.join(HERE, f't1_p_dependence_n{n}.json'))); sN = R['None'][-1]
    print(f"n={n}, sigma_None(L=30) = {sN:.4f}; pairs: {'all' if R['n_off'] is None else R['n_off']} off-diagonal\n")
    print('| p | −ln Λ/(np) | σ_br,∞ (chain) | closed form | κ = (σ_br,∞² − closed²)/p | N_s to resolve floor | (σ_AD/σ_tw)² at L=4 | L* (chain) | ln(3σ_None/σ_br,∞)/(−ln Λ) |')
    print('|---|---|---|---|---|---|---|---|---|')
    for key in sorted(R['p'], key=float):
        d = R['p'][key]; q = float(key); lam = d['Lambda_markov']; br = d['br'][-1]; cl = d['floor_closed']
        ad, tw = np.array(d['AD']), np.array(d['twirl']); r4 = (ad[3] / tw[3]) ** 2 if len(ad) > 3 else float('nan')
        pred = np.log(3 * sN / br) / (-np.log(lam))
        print(f"| {q:g} | {-np.log(lam) / (n * q):.3f} | {br:.4f} | {cl:.4f} | {(br ** 2 - cl ** 2) / q:.2e} | {1 / br ** 2:.1e} | {r4:.2f} | {d['L_star']} | {pred:.0f} |")


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'summary':
        summary(int(sys.argv[2]) if len(sys.argv) > 2 else 4)
    else:
        main()
