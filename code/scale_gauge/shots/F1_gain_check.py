"""Why does the operational shot cost of AD exceed kappa? Candidate predictor from the stored readout:
shot noise on z_j has variance (1 - z_j^2)/N_s, and the standardized readout maps it to class-score noise
with gain W2[:, j]/sigma_j. Noise gain G = mean over test images of sum_j |W2[:, j]|^2 (1 - z_j^2)/sigma_j^2
(exact z from the stored parameters). Predicted cost ratio = G / G_None (same accuracy tolerance assumed).
Also reports the smallest feature spread per run. Analysis of stored parameters only."""
import os, sys, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import Circ, load_v5
from F1_shots import SRC, load_records, CHUNK
from F1_summary import rows as SUMROWS
torch.set_num_threads(4)
out = []
for ds, Ls in (('MNIST', (1, 2, 3, 4)), ('FashionMNIST', (4,))):
    A = torch.as_tensor(load_v5(ds)[2]); recs = load_records(SRC[ds])
    for L in Ls:
        G = {}; smin = {}
        for ch in ('None', 'AD', 'Pauli', 'Depol'):
            g, sm = [], []
            for r in [x for x in recs if x['L'] == L and x['channel'] == ch and x['test_acc'] > 0.5]:
                prm = r['params']; c = Circ(4, L, ch, r['p_noise'])
                with torch.no_grad():
                    z = torch.cat([c.features(A[i:i + CHUNK], torch.as_tensor(np.asarray(prm['theta']))) for i in range(0, len(A), CHUNK)]).numpy()
                w2 = (np.asarray(prm['W2']) ** 2).sum(0); sd = np.asarray(prm['z_std'])
                g.append(np.mean(((1 - z ** 2) * w2 / sd ** 2).sum(1))); sm.append(sd.min())
            G[ch] = np.exp(np.mean(np.log(g))); smin[ch] = np.exp(np.mean(np.log(sm)))
        for ch in ('None', 'AD', 'Pauli', 'Depol'):
            srow = [x for x in SUMROWS if x['dataset'] == ds and x['L'] == L and x['channel'] == ch][0]
            opr = srow['cost_geomean'] / [x for x in SUMROWS if x['dataset'] == ds and x['L'] == L and x['channel'] == 'None'][0]['cost_geomean']
            out.append(dict(dataset=ds, L=L, channel=ch, gain_ratio=G[ch] / G['None'], op_ratio=opr, kappa=srow['kappa'], sigma_min_geo=smin[ch]))
            print(f"{ds[:5]} L={L} {ch:5s} op-cost ratio {opr:9.4g}   noise-gain ratio {G[ch]/G['None']:9.4g}   kappa {srow['kappa']:9.4g}   geo-mean min sigma {smin[ch]:.2e}")
pickle.dump(out, open(os.path.join(HERE, 'F1_gain_check.pkl'), 'wb'))
