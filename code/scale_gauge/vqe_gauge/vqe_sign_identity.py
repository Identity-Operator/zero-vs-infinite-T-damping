"""Exact check behind the sign statement of Sec. IV. On a bipartite
coupling graph, flipping one sublattice, U = prod_{i in B} X_i, maps H(J) = -J sum ZZ - h sum X to H(-J)
and commutes with the transverse field. U acts after the last channel, where the final R_X layer absorbs
it (x_i -> x_i + pi for i in B), so E_{-J}(theta) = E_J(theta') for every channel and placement, and
x -> x + pi preserves the uniform initialization U[0, 2pi). On the odd (frustrated) ring there is no
such frame. Reuses vqe_gauge_test.energy unchanged.   python vqe_sign_identity.py | tee vqe_sign_identity_log.txt"""
import numpy as np
import vqe_sign_test as S, vqe_gauge_test as m


def worst(pairs, flip):
    w = 0.0
    for g in (0.1, 0.2):
        for placement in ('mid', 'post'):
            for ch in ('none', 'AD', 'revAD', 'twirlAD'):
                S.set_model(-1.0, 0.5, pairs); th = m.init('uniform', 7); E_af = m.energy(th, ch, g, placement)
                S.set_model(1.0, 0.5, pairs); thp = th.clone()
                for q in flip:
                    thp[:, m.N_PARAMS - m.N + q] += np.pi                         # final R_X on qubit q
                w = max(w, float((E_af - m.energy(thp, ch, g, placement)).abs().max()))
    return w


if __name__ == '__main__':
    print(f"open chain, B = {{1}}: max |E_(-J)(theta) - E_J(theta')| over {m.TRIALS} draws x gamma {{0.1, 0.2}} x "
          f"{{mid, post}} x {{none, AD, revAD, twirlAD}} = {worst([(0, 1), (1, 2)], [1]):.1e}")
    print(f"3-ring (frustrated), same shift: max difference {worst([(0, 1), (1, 2), (2, 0)], [1]):.2f} (not a gauge; on an odd cycle no product of X_i flips every bond, and Y or Z would flip the field)")
