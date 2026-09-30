# -*- coding: utf-8 -*-
"""
Finite temperature: generalized amplitude damping GAD(p, N) keeps AD's
contraction and has t_z = p(1-2N), so in the Heisenberg site map only the branching entry changes,
K[I, Z] = p(1-2N). The exact pair chain of t1_scale_ratio.py is unchanged otherwise, and the
decomposition sigma_GAD^2 = sigma_twirl^2 + sigma_br^2 still holds (the next uniform block kills the
cross moments), so sigma_br(N)/sigma_br(0) is computed exactly over the parameters (n=4, first 300
training images, 5000 random off-diagonal pairs), for L = 1..8.
Expected: exactly |1-2N| at L=1 (a single branching, the last-layer term of Eq. floor); multi-qubit
strings that branch on several qubits add t_z^2 and higher at L >= 2; exact symmetry N <-> 1-N (the X frame).
    python t1_gad_temperature.py | tee t1_gad_temperature.txt      (about 1.2 GB, 15 s; loads MNIST)
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import t1_scale_ratio as T
import t1_p_dependence as PD

_orig = T.heis_channel


def heis(ch, p=T.P):
    if ch.startswith('GAD'):
        N = float(ch[3:]); c = np.sqrt(1 - p); K = np.diag([1, c, c, 1 - p]); K[0, 3] = p * (1 - 2 * N)
        return K
    return _orig(ch, p)


def main(Ls=8):
    T.heis_channel = heis
    A = T.pca_angles(4)
    for p in (0.3, 0.1):
        tw = PD.Sigma(A, 'twirl', 4, p, 5000); stw = np.array([tw.step() for _ in range(Ls)]); del tw
        g0 = PD.Sigma(A, 'GAD0.0', 4, p, 5000); s0 = np.array([g0.step() for _ in range(Ls)]); del g0
        ad = PD.Sigma(A, 'AD', 4, p, 5000); sad = np.array([ad.step() for _ in range(3)]); del ad
        br0 = np.sqrt(s0 ** 2 - stw ** 2)
        print(f"p={p}: GAD(N=0) against AD, L=1-3: max rel diff {np.abs(s0[:3] / sad - 1).max():.1e}")
        for N in (0.1, 0.25, 0.4, 0.75, 1.0):
            g = PD.Sigma(A, f'GAD{N}', 4, p, 5000); s = np.array([g.step() for _ in range(Ls)]); del g
            r = np.sqrt(np.maximum(s ** 2 - stw ** 2, 0)) / br0
            print(f"   N={N:<4}: sigma_br(N)/sigma_br(0), L=1..{Ls}: " + ' '.join(f'{x:.4f}' for x in r) +
                  f"   |1-2N| = {abs(1 - 2 * N):.2f}; max rel deviation {np.abs(r / abs(1 - 2 * N) - 1).max():.2%}", flush=True)


if __name__ == '__main__':
    main()
