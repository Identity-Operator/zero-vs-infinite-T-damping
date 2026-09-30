# -*- coding: utf-8 -*-
"""
T1 check: recompute S6a's own n=8 Pauli-twirl cells with the independent Pauli
propagation of pauli_prop.py and compare them with the per-qubit standard deviations that S6a
stored (m5_width_diag/results.pkl), at L = 4 and 6, for S6a's five theta draws, on the same 300
training images.

S6a's theta: run_phase1.py draws 2*pi*torch.rand(3n(L+1)) from a CUDA generator seeded 1-5. Because
it imports torch_circ, which sets the default dtype to float64, the draws are float64. The same
seeds in a process that has not imported torch_circ give float32 draws, a different set. For these
sizes the draws for depth L are the leading entries of those for L=6, so the depths share their
first L+1 blocks. s6a_theta_n8.npz holds the float64 draws, generated on the GPU as below.

    python check_s6a_n8.py              # uses s6a_theta_n8.npz (CPU only)
    python check_s6a_n8.py --gen-theta  # regenerates s6a_theta_n8.npz on the GPU first
"""
import os, pickle, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, '..', '..', 'noise_structure_mnist')
THETA = os.path.join(HERE, 's6a_theta_n8.npz')
N, LS, SEEDS = 8, (4, 6), (1, 2, 3, 4, 5)


def gen_theta():
    sys.path.insert(0, SRC)
    import torch
    import torch_circ  # noqa: F401  (sets the default dtype to float64, as in run_phase1)
    out = {f'L{L}_s{s}': (2 * np.pi * torch.rand(3 * N * (L + 1), generator=torch.Generator(device='cuda').manual_seed(s),
                                                 device='cuda')).cpu().numpy() for L in LS for s in SEEDS}
    assert all(v.dtype == np.float64 for v in out.values())
    np.savez(THETA, **out)


def main():
    if '--gen-theta' in sys.argv:
        gen_theta()
    sys.path.insert(0, HERE)
    import pauli_prop as PP
    import t1_scale_ratio as T
    t0 = time.time(); A = T.pca_angles(N); P = PP.Prop(N, 'twirl'); TH = np.load(THETA)
    R = pickle.load(open(os.path.join(SRC, 'm5_width_diag', 'results.pkl'), 'rb'))
    for L in LS:
        mine, s6a = [], []
        for s in SEEDS:
            mine.append(np.stack([P.z(A, TH[f'L{L}_s{s}'], L, j) for j in range(N)], 1).std(0))
            s6a.append([r for r in R if (r['n'], r['L'], r['channel'], r['theta_seed']) == (N, L, 'Pauli', s)][0]['sigma_per_qubit'])
        mine, s6a = np.array(mine), np.array(s6a)
        print(f"n={N} L={L} Pauli: per-qubit sd, Pauli propagation vs S6a record, max rel diff {np.abs(mine / s6a - 1).max():.1e}; "
              f"rms of the 5 draws {np.sqrt((mine ** 2).mean()):.4e} (S6a {np.sqrt((s6a ** 2).mean()):.4e})  ({time.time() - t0:.0f}s)",
              flush=True)


if __name__ == '__main__':
    main()
