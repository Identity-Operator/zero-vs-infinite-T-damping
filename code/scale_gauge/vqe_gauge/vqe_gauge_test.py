"""Test 2: does the Pauli-frame gauge rule decide
when the direction of amplitude damping matters?

Setup of van Rossum et al. (2510.24050), Sec. IV and Fig. 4: 3-qubit periodic TFIM,
H = -J sum Z_i Z_{i+1} - h sum X_i, J=1, h=0.5. Ansatz: RY per qubit, then 4 Trotter blocks,
each RZZ on the 3 periodic pairs then RX per qubit; RZZ = CNOT . RZ_t . CNOT. Single-qubit gates
are noiseless. Noise placements:
  mid  (their setup): the channel on both qubits after each CNOT of the RZZ;
  post             : the channel on both qubits applied twice after the complete RZZ
                     (same number of insertions).
Channels: AD (fixed point |0>), reversed AD (fixed point |1>), Pauli-twirled AD (reference).
Gauge prediction: under `post`, E_rev(theta) = E_AD(theta') exactly, with theta' mapping each
initial RY angle t -> pi - t; under `mid` no X/Y frame exists and a gap is allowed.

CPU only, float64 / complex128. Every trial of a point is one batch element; Adam acts
elementwise, so the trials stay independent.
"""
import itertools
import pickle
import sys
import time

import numpy as np
import torch

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)
DEV = 'cpu'
CT = torch.complex128
N, BLOCKS, J, H_FIELD = 3, 4, 1.0, 0.5
PAIRS = [(0, 1), (1, 2), (2, 0)]
N_PARAMS = N + BLOCKS * (len(PAIRS) + N)
GAMMAS = (0.01, 0.05, 0.1, 0.2)
TRIALS, STEPS, LR = 50, 500, 0.05

I2 = np.eye(2); X = np.array([[0, 1], [1, 0]]); Y = np.array([[0, -1j], [1j, 0]]); Z = np.diag([1., -1.])


def kron(*ms):
    out = np.array([[1.0 + 0j]])
    for m in ms:
        out = np.kron(out, m)
    return out


def on(q, g):
    return kron(*[g if k == q else I2 for k in range(N)])


def cnot(c, t):
    P0, P1 = np.diag([1., 0.]), np.diag([0., 1.])
    return (kron(*[P0 if k == c else I2 for k in range(N)])
            + kron(*[P1 if k == c else (X if k == t else I2) for k in range(N)]))


HAM = sum(-J * on(i, Z) @ on(j, Z) for i, j in PAIRS) + sum(-H_FIELD * on(i, X) for i in range(N))
E0 = float(np.linalg.eigvalsh(HAM)[0])


def kraus(channel, g):
    c = np.sqrt(1 - g)
    ad = [np.array([[1, 0], [0, c]]), np.array([[0, np.sqrt(g)], [0, 0]])]
    if channel == 'AD':
        return ad
    if channel == 'revAD':
        return [X @ k @ X for k in ad]
    if channel == 'twirlAD':
        px = py = g / 4
        pz = (2 - g - 2 * c) / 4
        return [np.sqrt(1 - px - py - pz) * I2, np.sqrt(px) * X, np.sqrt(py) * Y, np.sqrt(pz) * Z]
    if channel == 'none':
        return [I2]
    raise ValueError(channel)


T = lambda m: torch.tensor(m, dtype=CT, device=DEV)
H_T = T(HAM)
CNOTS = {pr: T(cnot(*pr)) for pr in PAIRS}


def batched_1q(mats, q):
    """(B,2,2) single-qubit gates on qubit q -> (B,8,8)."""
    left, right = T(np.eye(2 ** q)), T(np.eye(2 ** (N - q - 1)))
    return torch.einsum('ab,xcd,ef->xacebdf', left, mats, right).reshape(mats.shape[0], 2 ** N, 2 ** N)


def rot(axis, th):
    c, s = torch.cos(th / 2).to(CT), torch.sin(th / 2).to(CT)
    m = torch.zeros(th.shape[0], 2, 2, dtype=CT)
    if axis == 'y':
        m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -s, s, c
    elif axis == 'x':
        m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -1j * s, -1j * s, c
    else:
        m[:, 0, 0], m[:, 1, 1] = torch.exp(-0.5j * th), torch.exp(0.5j * th)
    return m


def conj(U, rho):
    return U @ rho @ U.conj().transpose(-1, -2)


def make_noise(channel, g):
    """Return f(rho, qubits) applying the channel once to each listed qubit."""
    Ks = {q: [T(on(q, k)) for k in kraus(channel, g)] for q in range(N)}

    def apply(rho, qubits):
        for q in qubits:
            rho = sum(K @ rho @ K.conj().T for K in Ks[q])
        return rho
    return apply


def energy(theta, channel, g, placement):
    """theta (B, N_PARAMS) -> E (B,)."""
    B = theta.shape[0]
    noise = make_noise(channel, g)
    rho = torch.zeros(B, 2 ** N, 2 ** N, dtype=CT)
    rho[:, 0, 0] = 1
    k = 0
    for q in range(N):
        rho = conj(batched_1q(rot('y', theta[:, k]), q), rho); k += 1
    for _ in range(BLOCKS):
        for (c, t) in PAIRS:
            C = CNOTS[(c, t)]
            rho = C @ rho @ C.conj().T
            if placement == 'mid':
                rho = noise(rho, (c, t))
            rho = conj(batched_1q(rot('z', theta[:, k]), t), rho); k += 1
            rho = C @ rho @ C.conj().T
            rho = noise(rho, (c, t))
            if placement == 'post':
                rho = noise(rho, (c, t))
        for q in range(N):
            rho = conj(batched_1q(rot('x', theta[:, k]), q), rho); k += 1
    return torch.einsum('ij,bji->b', H_T, rho).real


def gauge_map(theta):
    """Initial RY angles t -> pi - t; everything else unchanged."""
    th = theta.clone()
    th[:, :N] = np.pi - th[:, :N]
    return th


def optimise(theta0, channel, g, placement):
    th = theta0.clone().requires_grad_(True)
    opt = torch.optim.Adam([th], lr=LR)
    for _ in range(STEPS):
        opt.zero_grad()
        E = energy(th, channel, g, placement)
        E.sum().backward()
        opt.step()
    with torch.no_grad():
        return energy(th, channel, g, placement).numpy(), th.detach().numpy()


def init(scheme, seed):
    gen = torch.Generator().manual_seed(seed)
    if scheme == 'uniform':
        return 2 * np.pi * torch.rand(TRIALS, N_PARAMS, generator=gen)
    return 0.1 * torch.randn(TRIALS, N_PARAMS, generator=gen)


def main():
    out = {'E0': E0, 'settings': dict(TRIALS=TRIALS, STEPS=STEPS, LR=LR, GAMMAS=GAMMAS,
                                      init_small='N(0, 0.1^2)', init_uniform='U[0, 2pi)')}
    print(f"E0 = {E0:.6f} (exact diagonalisation)", flush=True)
    # 1. Single-parameter-set checks of the gauge identity.
    th = init('uniform', 999)
    checks = {}
    for placement in ('post', 'mid'):
        for g in GAMMAS:
            with torch.no_grad():
                d = (energy(th, 'revAD', g, placement) - energy(gauge_map(th), 'AD', g, placement)).abs().max().item()
            checks[(placement, g)] = d
            print(f"identity check {placement:4s} gamma={g}: max|E_rev(th) - E_AD(th')| over {TRIALS} sets = {d:.2e}", flush=True)
    out['identity_checks'] = checks
    with torch.no_grad():
        out['noiseless_check'] = energy(th, 'none', 0.0, 'post').min().item()
    # 2. Optimisation.
    res = {}
    t0 = time.time()
    for scheme, placement, g, channel in itertools.product(('uniform', 'small'), ('mid', 'post'), GAMMAS,
                                                           ('AD', 'revAD', 'twirlAD')):
        th0 = init(scheme, 1)            # same initial parameters for every channel of a point
        E, thf = optimise(th0, channel, g, placement)
        res[(scheme, placement, g, channel)] = {'E': E, 'rel_err': (E - E0) / abs(E0), 'theta0': th0.numpy(), 'theta': thf}
        print(f"[{time.time() - t0:6.0f}s] {scheme:7s} {placement:4s} g={g:<5} {channel:7s} "
              f"min rel err {res[(scheme, placement, g, channel)]['rel_err'].min():.4f}  "
              f"median {np.median(res[(scheme, placement, g, channel)]['rel_err']):.4f}", flush=True)
    # 3. Paired exact check: rev with init th0 vs AD with init th0' (post placement, uniform init).
    th0 = init('uniform', 1)
    paired = {}
    for g in GAMMAS:
        Er, _ = optimise(th0, 'revAD', g, 'post')
        Ea, _ = optimise(gauge_map(th0), 'AD', g, 'post')
        paired[g] = float(np.max(np.abs(Er - Ea)))
        print(f"paired trajectories post g={g}: max|E_rev - E_AD(mapped init)| after {STEPS} steps = {paired[g]:.2e}", flush=True)
    out['paired_post'] = paired
    out['results'] = res
    with open(sys.argv[1] if len(sys.argv) > 1 else 'vqe_gauge_results.pkl', 'wb') as f:
        pickle.dump(out, f)


if __name__ == '__main__':
    main()
