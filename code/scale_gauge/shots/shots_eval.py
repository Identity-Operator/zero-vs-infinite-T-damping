"""Generic finite-shot inference evaluation of a standardized-readout model, in two
conventions for the standardization statistics:
  'stored' : the run's stored training z_mean/z_std (exact expectation values), as in F1;
  'calib'  : mu and sigma re-estimated in every draw from N_s-shot samples of the 1000 training images
             (population std), i.e. the readout calibrated with the same shot budget as the test.
             This is the convention of train_one's shot_evaluation (C4) and the one consistent with
             shot-noise training, whose readout sees noisy batch statistics.
Generators are keyed by (tag, L, channel, seed, train_shots, N_s, draw, convention)."""
import numpy as np, torch
from shots_lib import Circ

CH_ID = {'None': 0, 'AD': 1, 'Pauli': 2, 'Depol': 3, 'AD_flip': 4}

def run_probs(r, A, chunk=1000):
    prm = r['params']
    circ = Circ(4, r['L'], r['channel'], r['p_noise'], r.get('noise_placement', 'after_entangler'))
    th = torch.as_tensor(np.asarray(prm['theta']))
    with torch.no_grad():
        P = torch.cat([circ.probs(A[i:i + chunk], th) for i in range(0, len(A), chunk)]).numpy()
    P = np.clip(P, 0.0, None); P /= P.sum(1, keepdims=True)
    return P, circ.sign.numpy()

def evaluate(r, P_te, P_tr, sign, Yte, ns_grid, draws=5, tag=0, conventions=('stored', 'calib')):
    prm = r['params']; W, b = np.asarray(prm['W2']), np.asarray(prm['b2'])
    mu0, sd0 = np.asarray(prm['z_mean']), np.asarray(prm['z_std'])
    out = {c: np.zeros((len(ns_grid), draws)) for c in conventions}
    tsh = int(r.get('shots') or 0)
    for i, ns in enumerate(ns_grid):
        for k in range(draws):
            for ci, conv in enumerate(conventions):
                rng = np.random.default_rng([tag, r['L'], CH_ID[r['channel']], r['seed'], tsh, int(ns), k, ci])
                z = rng.multinomial(int(ns), P_te) @ sign / ns
                if conv == 'stored':
                    mu, sd = mu0, sd0
                else:
                    ztr = rng.multinomial(int(ns), P_tr) @ sign / ns
                    mu, sd = ztr.mean(0), ztr.std(0)
                out[conv][i, k] = (((z - mu) / sd) @ W.T + b).argmax(1).__eq__(Yte).mean()
    return out
