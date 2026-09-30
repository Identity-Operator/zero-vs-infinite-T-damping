# -*- coding: utf-8 -*-
"""
Pauli-frame / time-reversal gauge for reversed amplitude damping.

A circuit is a list of elements acting on n qubits, starting from |0...0><0...0|:
  {'kind': 'su2',   'qubits': [...]}          trainable general single-qubit unitaries
  {'kind': 'rot',   'gen': 'ZZ', 'qubits': [i, j]}  trainable Pauli rotation exp(-i t G/2)
  {'kind': 'fixed', 'U': <2^n x 2^n>, 'name': str}  fixed gate (e.g. a CNOT)
  {'kind': 'enc',   'axes': {q: (nx, ny, nz)}}  data encoding  (x) R_{n_q}(x_q)
  {'kind': 'noise', 'qubits': [...]}          the single-qubit channel on each listed qubit
and a readout: a list of observables, 'affine' (a learned sign/scale per observable is
available downstream) or 'fixed' (the expectation values are used as they are).

A GAUGE FRAME assigns to every cut k between elements a Pauli string P_k, all of one
type t: unitary (rho -> P rho P) or antiunitary (rho -> P conj(rho) P, conj in the
computational basis). Inserting g_k g_k = id at every cut rewrites the reversed-noise
circuit as g_m . [original elements, each replaced by g_k E_k g_{k-1}] . g_0. The
reversed model is a reparametrisation of the original one iff every replaced element is
back in its own family and the boundaries are absorbed (Proposition G1):

  noise    g_k N_rev g_{k-1} = N           (AD: frame X or Y on each noisy qubit)
  fixed    g_k U g_{k-1}   = U  (phase)    (frame propagates by conjugation; must stay Pauli)
  enc      g_k S(x) g_{k-1} = S(x) for all x (frame unchanged; unitary: P commutes with the
           generator; antiunitary: P conj(n.sigma) P = -n.sigma, always true for P = Y:
           time reversal leaves every single-qubit rotation invariant)
  rot      g_k R_G(t) g_{k-1} = R_G(+-t)   (frame unchanged; frame must map G to +-G)
  su2      always absorbed; the frame may change arbitrarily on the block's qubits
  input    g_0(|0><0|) must be absorbed by the first element on each qubit
  readout  g_m maps each observable to +-itself; 'fixed' readout needs +

search() propagates EVERY frame of the chosen type through the circuit and prunes on
these conditions, so an empty result means no gauge exists in the (anti)unitary
Pauli-frame group; a surviving frame is returned as an explicit assignment, from which
reparametrise() builds theta' and the readout signs so that the identity can be checked.
"""
import itertools
from functools import reduce
import numpy as np

I2 = np.eye(2, dtype=complex)
PX = np.array([[0, 1], [1, 0]], complex)
PY = np.array([[0, -1j], [1j, 0]])
PZ = np.array([[1, 0], [0, -1]], complex)
PAULI = [I2, PX, PY, PZ]
NAME = 'IXYZ'


def kron(ms):
    return reduce(np.kron, ms)


def embed(ops, n):
    """ops: {qubit: 2x2} -> 2^n x 2^n."""
    return kron([ops.get(q, I2) for q in range(n)])


def pauli_full(P):
    return kron([PAULI[i] for i in P])


def gen_full(gen, qubits, n):
    return embed({q: PAULI[NAME.index(c)] for c, q in zip(gen, qubits)}, n)


def conj_t(M, t):
    return M if t == 'U' else M.conj()


def haar2(rng):
    z = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    q, r = np.linalg.qr(z)
    return q * (np.diag(r) / abs(np.diag(r)))


def rot(axis, a):
    G = axis[0] * PX + axis[1] * PY + axis[2] * PZ
    return np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * G


def expm_pauli(G, t):
    """exp(-i t G/2) for a Pauli string matrix G (G^2 = 1)."""
    return np.cos(t / 2) * np.eye(len(G)) - 1j * np.sin(t / 2) * G


def cnot(c, tg, n):
    P0 = np.diag([1, 0]).astype(complex); P1 = np.diag([0, 1]).astype(complex)
    return embed({c: P0}, n) + embed({c: P1, tg: PX}, n)


# ---------------------------------------------------------------- channels
def kraus(kind, p):
    c = np.sqrt(1 - p)
    AD = [np.diag([1, c]).astype(complex), np.sqrt(p) * np.array([[0, 1], [0, 0]], complex)]
    if kind == 'AD':
        return AD
    if kind == 'ADrev':                                     # damping toward |1>
        return [PX @ k @ PX for k in AD]
    if kind == 'twirl':                                     # Pauli twirl of AD
        q, pz = p / 4, (2 - p - 2 * c) / 4
        return [np.sqrt(w) * P for w, P in zip([1 - 2 * q - pz, q, q, pz], PAULI)]
    raise ValueError(kind)


def choi(K):
    v = np.array([1, 0, 0, 1], complex) / np.sqrt(2)          # |00> + |11>, normalised
    out = np.zeros((4, 4), complex)
    for k in K:
        w = np.kron(k, I2) @ v
        out += np.outer(w, w.conj())
    return out


def same_channel(K1, K2):
    return np.allclose(choi(K1), choi(K2), atol=1e-12)


# ---------------------------------------------------------------- simulation
def random_params(circuit, rng):
    ps = []
    for el in circuit:
        if el['kind'] == 'su2':
            ps.append({q: haar2(rng) for q in el['qubits']})
        elif el['kind'] == 'rot':
            ps.append(rng.uniform(-np.pi, np.pi))
        else:
            ps.append(None)
    return ps


def simulate(circuit, params, x, noise, p, n):
    rho = np.zeros((2 ** n, 2 ** n), complex); rho[0, 0] = 1
    K = kraus(noise, p)
    for el, th in zip(circuit, params):
        k = el['kind']
        if k == 'noise':
            for q in el['qubits']:
                rho = sum(embed({q: kk}, n) @ rho @ embed({q: kk}, n).conj().T for kk in K)
            continue
        if k == 'su2':
            U = embed(th, n)
        elif k == 'rot':
            U = expm_pauli(gen_full(el['gen'], el['qubits'], n), th)
        elif k == 'fixed':
            U = el['U']
        elif k == 'enc':
            U = embed({q: rot(ax, x[q]) for q, ax in el['axes'].items()}, n)
        rho = U @ rho @ U.conj().T
    return rho


def outputs(rho, observables):
    return np.array([np.trace(O @ rho).real for O in observables])


# ---------------------------------------------------------------- frame search
_ALL = None


def all_frames(n):
    return list(itertools.product(range(4), repeat=n))


def identify(M, n):
    """Return (P, phase) if M is a phase times a Pauli string, else None."""
    for P in all_frames(n):
        ov = np.trace(pauli_full(P).conj().T @ M) / 2 ** n
        if np.isclose(abs(ov), 1, atol=1e-10):
            return P, ov
    return None


def touched(el, n):
    if el['kind'] == 'enc':
        return list(el['axes'].keys())
    if el['kind'] == 'fixed':
        return [q for q in range(n) if _acts_on(el['U'], q, n)]
    return list(el['qubits'])


def first_touch(circuit, n):
    """qubit -> index of the first element acting on it."""
    ft = {}
    for i, el in enumerate(circuit):
        for q in touched(el, n):
            ft.setdefault(q, i)
    return ft


def _acts_on(U, q, n):
    """Does U act nontrivially on qubit q? (U commutes with all Paulis on q iff not.)"""
    return any(not np.allclose(U @ embed({q: P}, n), embed({q: P}, n) @ U) for P in (PX, PZ))


def initial_frames(circuit, n):
    """Per qubit, the frames at cut 0 that the first element on that qubit can absorb."""
    ft = first_touch(circuit, n)
    allowed = []
    for q in range(n):
        el = circuit[ft[q]] if q in ft else None
        if el is not None and el['kind'] == 'su2':
            allowed.append((0, 1, 2, 3))
        elif el is not None and el['kind'] == 'rot' and len(el['qubits']) == 1 and el['gen'] in 'XY':
            allowed.append((0, 1, 2, 3))                     # X|0>, Y|0> ~ R_{X,Y}(pi)|0>
        else:
            allowed.append((0, 3))                           # only frames that fix |0>
    return [tuple(f) for f in itertools.product(*allowed)]


def _step(P, el, t, n, rev_K, K):
    """Frames reachable after element el from frame P, [] if a condition fails."""
    k = el['kind']
    Pm = pauli_full(P)
    if k == 'su2':                                          # the frame on these qubits is free
        qs = el['qubits']
        return [tuple(full) for full in
                itertools.product(*[(range(4) if q in qs else (P[q],)) for q in range(n)])]
    if k == 'noise':
        for q in el['qubits']:
            Pq = PAULI[P[q]]
            if not same_channel([Pq @ conj_t(kk, t) @ Pq for kk in rev_K], K):
                return []
        return [P]
    if k == 'rot':
        G = gen_full(el['gen'], el['qubits'], n)
        M = Pm @ conj_t(G, t) @ Pm
        return [P] if (np.allclose(M, G) or np.allclose(M, -G)) else []
    if k == 'enc':
        for q, ax in el['axes'].items():
            Pq = PAULI[P[q]]
            for a in (0.7, 1.9):
                R = rot(ax, a)
                M = Pq @ conj_t(R, t) @ Pq
                ph = np.trace(R.conj().T @ M) / 2
                if not (np.isclose(abs(ph), 1) and np.allclose(M, ph * R)):
                    return []
        return [P]
    if k == 'fixed':
        U = el['U']
        r = identify(U @ Pm @ conj_t(U, t).conj().T, n)
        return [r[0]] if r is not None else []
    raise ValueError(k)


def search(circuit, n, observables, readout, t, rev='ADrev', base='AD', p=0.3):
    """All-frames propagation. Returns (exists, assignment or None)."""
    rev_K, K = kraus(rev, p), kraus(base, p)
    layer = {P: None for P in initial_frames(circuit, n)}   # frame -> backpointer
    hist = [layer]
    for el in circuit:
        nxt = {}
        for P in layer:
            for Q in _step(P, el, t, n, rev_K, K):
                nxt.setdefault(Q, P)
        layer = nxt; hist.append(layer)
        if not layer:
            return False, None
    ok = []
    for P in layer:
        Pm = pauli_full(P); good = True
        for O in observables:
            M = Pm @ conj_t(O, t) @ Pm
            if np.allclose(M, O):
                continue
            if readout == 'affine' and np.allclose(M, -O):
                continue
            good = False; break
        if good:
            ok.append(P)
    if not ok:
        return False, None
    P = ok[0]; frames = [P]
    for h in reversed(hist[1:]):
        P = h[P]; frames.append(P)
    return True, frames[::-1]                                  # frames[k] = P at cut k


def reparametrise(circuit, params, frames, t, n):
    """theta' such that y_rev(x; theta) = s * y(x; theta'); returns (params', signs fn)."""
    ft = first_touch(circuit, n)
    new = []
    for i, (el, th) in enumerate(zip(circuit, params)):
        Pa, Pb = frames[i + 1], frames[i]                      # after, before
        if el['kind'] == 'su2':
            d = {}
            for q, W in th.items():
                Wn = PAULI[Pa[q]] @ conj_t(W, t)
                if ft.get(q) != i:                              # not the first element on q
                    Wn = Wn @ PAULI[Pb[q]]
                d[q] = Wn
            new.append(d)
        elif el['kind'] == 'rot':
            G = gen_full(el['gen'], el['qubits'], n)
            U = pauli_full(Pa) @ conj_t(expm_pauli(G, th), t) @ pauli_full(Pb)
            cands = [th, -th, th + np.pi, -th + np.pi, th - np.pi, -th - np.pi]
            first = all(ft.get(q) == i for q in el['qubits'])
            if first:                                           # absorb the input frame on |0>
                assert len(el['qubits']) == 1, "first-touch absorption needs a 1-qubit rotation"
                q = el['qubits'][0]; G1 = PAULI[NAME.index(el['gen'])]
                W = PAULI[Pa[q]] @ conj_t(expm_pauli(G1, th), t)
                tgt = W[:, 0]
                best = max(cands, key=lambda c: abs(np.vdot(expm_pauli(G1, c)[:, 0], tgt)))
                assert np.isclose(abs(np.vdot(expm_pauli(G1, best)[:, 0], tgt)), 1), "rot not absorbable"
            else:
                best = next(c for c in cands
                            if abs(np.trace(expm_pauli(G, c).conj().T @ U)) / 2 ** n > 1 - 1e-10)
            new.append(best)
        else:
            new.append(th)
    return new


def readout_signs(frames, observables, t):
    Pm = pauli_full(frames[-1])
    return np.array([1.0 if np.allclose(Pm @ conj_t(O, t) @ Pm, O) else -1.0 for O in observables])
