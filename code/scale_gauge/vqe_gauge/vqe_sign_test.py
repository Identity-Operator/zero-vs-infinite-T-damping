"""Test: does the bond-parity mechanism of the
direction effect predict its sign? Reuses vqe_gauge_test.py unchanged (ansatz, channels, placements,
Adam 500 steps, lr 0.05, 50 uniform initializations) and swaps only the Hamiltonian and its bonds.

Mechanism (Sec. IV C): between the two CNOTs of R_ZZ the target holds the bond parity; damping it
drives the bond toward even parity under AD and toward odd parity under reversed AD. Prediction:
with the channel inside R_ZZ ('mid') reversed AD beats AD for an antiferromagnet (J<0) and loses
for a ferromagnet (J>0); with the channel after R_ZZ ('post') the two coincide (gauge).

Models: 'ferro_ring' (J=1, h=0.5, periodic; the paper's), 'ferro_open' (J=1, h=0.5, open chain),
'antiferro_open' (J=-1, h=0.5, open chain; the odd ring is frustrated), 'strong_field' (J=1, h=2,
periodic; the damping fixed point |000> is far from the ground state).
Reports the relative energy error and the mean bond correlation <Z_i Z_j> of the optimized states.
Output: vqe_sign_results.pkl, vqe_sign_log.txt (via tee)."""
import itertools, pickle, time
import numpy as np
import torch
import vqe_gauge_test as m

MODELS = {'ferro_ring': (1.0, 0.5, [(0, 1), (1, 2), (2, 0)]),
          'ferro_open': (1.0, 0.5, [(0, 1), (1, 2)]),
          'antiferro_open': (-1.0, 0.5, [(0, 1), (1, 2)]),
          'strong_field': (1.0, 2.0, [(0, 1), (1, 2), (2, 0)])}
GAMMAS = (0.1, 0.2)
CHANNELS = ('AD', 'revAD', 'twirlAD')


def set_model(J, h, pairs):
    m.PAIRS = pairs
    m.N_PARAMS = m.N + m.BLOCKS * (len(pairs) + m.N)
    m.HAM = sum(-J * m.on(i, m.Z) @ m.on(j, m.Z) for i, j in pairs) + sum(-h * m.on(i, m.X) for i in range(m.N))
    m.E0 = float(np.linalg.eigvalsh(m.HAM)[0])
    m.H_T = m.T(m.HAM)
    m.CNOTS = {pr: m.T(m.cnot(*pr)) for pr in pairs}
    return m.HAM, m.E0


def final_rho(theta, channel, g, placement):
    """The state of m.energy, returned instead of its energy (same gate sequence)."""
    B = theta.shape[0]; noise = m.make_noise(channel, g)
    rho = torch.zeros(B, 2 ** m.N, 2 ** m.N, dtype=m.CT); rho[:, 0, 0] = 1; k = 0
    for q in range(m.N):
        rho = m.conj(m.batched_1q(m.rot('y', theta[:, k]), q), rho); k += 1
    for _ in range(m.BLOCKS):
        for (c, t) in m.PAIRS:
            C = m.CNOTS[(c, t)]
            rho = C @ rho @ C.conj().T
            if placement == 'mid':
                rho = noise(rho, (c, t))
            rho = m.conj(m.batched_1q(m.rot('z', theta[:, k]), t), rho); k += 1
            rho = C @ rho @ C.conj().T
            rho = noise(rho, (c, t))
            if placement == 'post':
                rho = noise(rho, (c, t))
        for q in range(m.N):
            rho = m.conj(m.batched_1q(m.rot('x', theta[:, k]), q), rho); k += 1
    return rho


def bond_zz(rho):
    ops = [m.T(m.on(i, m.Z) @ m.on(j, m.Z)) for i, j in m.PAIRS]
    return np.mean([torch.einsum('ij,bji->b', O, rho).real.numpy() for O in ops], 0)


def main():
    out = {}; t0 = time.time()
    for name, (J, h, pairs) in MODELS.items():
        HAM, E0 = set_model(J, h, pairs)
        e000 = float(HAM[0, 0].real)
        print(f"== {name}: J={J} h={h} bonds={pairs} E0={E0:.4f}; |000> rel err {(e000 - E0) / abs(E0):.3f}; "
              f"maximally mixed rel err {(np.trace(HAM).real / 8 - E0) / abs(E0):.3f}", flush=True)
        th0 = m.init('uniform', 1)
        Emin, thf = m.optimise(th0, 'none', 0.0, 'post')
        rel = (Emin - E0) / abs(E0)
        zz = bond_zz(final_rho(torch.as_tensor(thf), 'none', 0.0, 'post'))
        out[(name, 'none')] = dict(rel_err=rel, zz=zz, E0=E0)
        print(f"   noiseless          median rel err {np.median(rel):.4f}  <ZZ> {np.median(zz):+.3f}  ({time.time() - t0:.0f}s)", flush=True)
        for g, placement, ch in itertools.product(GAMMAS, ('mid', 'post'), CHANNELS):
            E, thf = m.optimise(th0, ch, g, placement)
            rel = (E - E0) / abs(E0)
            zz = bond_zz(final_rho(torch.as_tensor(thf), ch, g, placement))
            out[(name, g, placement, ch)] = dict(rel_err=rel, zz=zz, E0=E0)
            print(f"   g={g} {placement:4s} {ch:7s} median rel err {np.median(rel):.4f} (min {rel.min():.4f})  "
                  f"<ZZ> {np.median(zz):+.3f}  ({time.time() - t0:.0f}s)", flush=True)
            pickle.dump(out, open('vqe_sign_results.pkl', 'wb'))
    print('VQE_SIGN_DONE', flush=True)


if __name__ == '__main__':
    main()
