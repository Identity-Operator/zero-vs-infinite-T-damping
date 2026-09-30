"""F1: inference-only finite-shot evaluation of the committed
standardized-readout models S1 (m5_readout, MNIST, L=1-4) and S2 (m5_fashion_readout, Fashion, L=4).
For every run: diag(rho) of each test image from the stored theta (independent simulator shots_lib.Circ),
exact multinomial sampling of N_s joint outcomes (16 at n=4), z_hat_j = mean of the +-1 outcomes of
qubit j, standardization with the stored training z_mean/z_std, readout with the stored W2, b2.
R=5 draws per (run, N_s), each with its own seeded generator. No training.
Output: F1_results.pkl (per run: exact accuracy, per-draw accuracies, operational shot cost)."""
import os, sys, time, shutil, pickle, tempfile, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import Circ, load_v5, classify
torch.set_num_threads(4)
NS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000); R_DRAWS = 5; CHUNK = 1000
SRC = {'MNIST': os.path.join(HERE, '..', '..', 'noise_structure_mnist', 'm5_readout', 'results.pkl'),
       'FashionMNIST': os.path.join(HERE, '..', '..', 'noise_structure_mnist', 'm5_fashion_readout', 'results.pkl')}
CH_ID = {'None': 0, 'AD': 1, 'Pauli': 2, 'Depol': 3, 'AD_flip': 4}

def load_records(path):
    dst = os.path.join(tempfile.gettempdir(), 'F1_' + os.path.basename(os.path.dirname(path)) + '.pkl')
    shutil.copy2(path, dst)                        # copy before loading: a writer may be live
    return pickle.load(open(dst, 'rb'))

def op_cost(ns, acc_mean, acc_exact, tol=0.01):
    """Smallest N_s at which the mean accuracy is within tol of the exact accuracy, interpolated on log N_s
   . Returns (cost, flag): flag 'ok', 'le_min' (already within at the smallest N_s) or
    'gt_max' (never within); also returns the robust variant (from which point on it stays within)."""
    within = np.abs(acc_exact - acc_mean) <= tol
    def interp(k):
        if k == 0: return float(ns[0]), 'le_min'
        d0, d1 = acc_exact - acc_mean[k - 1] - tol, acc_exact - acc_mean[k] - tol   # d0 > 0 >= d1 normally
        x0, x1 = np.log10(ns[k - 1]), np.log10(ns[k])
        f = d0 / (d0 - d1) if d0 != d1 else 1.0
        return float(10 ** (x0 + np.clip(f, 0, 1) * (x1 - x0))), 'ok'
    first = next((k for k in range(len(ns)) if within[k]), None)
    stay = next((k for k in range(len(ns)) if within[k:].all()), None)
    c1 = interp(first) if first is not None else (float('inf'), 'gt_max')
    c2 = interp(stay) if stay is not None else (float('inf'), 'gt_max')
    return c1, c2

if __name__ == '__main__':
    t0 = time.time(); out = []
    for ds in ('MNIST', 'FashionMNIST'):
        Atr, Ytr, Ate, Yte = load_v5(ds)
        recs = sorted(load_records(SRC[ds]), key=lambda r: (r['L'], CH_ID[r['channel']], r['seed']))
        print(f"{ds}: {len(recs)} runs, {len(Yte)} test images", flush=True)
        A = torch.as_tensor(Ate)
        for r in recs:
            prm = r['params']; L, ch = r['L'], r['channel']
            circ = Circ(4, L, ch, r['p_noise'], r.get('noise_placement', 'after_entangler'))
            th = torch.as_tensor(np.asarray(prm['theta']))
            with torch.no_grad():
                P = torch.cat([circ.probs(A[i:i + CHUNK], th) for i in range(0, len(A), CHUNK)]).numpy()
            P = np.clip(P, 0.0, None); P /= P.sum(1, keepdims=True)
            sign = circ.sign.numpy()
            pred = classify(P @ sign, prm)
            mism = int((pred != np.asarray(r['test_pred'])).sum())
            acc_exact = float(r['test_acc']); acc_recomp = float((pred == Yte).mean())
            acc = np.zeros((len(NS), R_DRAWS))
            for i, ns in enumerate(NS):
                for k in range(R_DRAWS):
                    rng = np.random.default_rng([0 if ds == 'MNIST' else 1, L, CH_ID[ch], r['seed'], ns, k])
                    zhat = rng.multinomial(ns, P) @ sign / ns
                    acc[i, k] = (classify(zhat, prm) == Yte).mean()
            (c1, f1), (c2, f2) = op_cost(np.array(NS, float), acc.mean(1), acc_exact)
            out.append(dict(dataset=ds, L=L, channel=ch, seed=r['seed'], placement=r.get('noise_placement'),
                            acc_exact=acc_exact, acc_exact_recomputed=acc_recomp, pred_mismatch=mism,
                            Ns=NS, acc=acc, op_cost=c1, op_flag=f1, op_cost_stay=c2, op_flag_stay=f2,
                            sigma=float(np.mean(prm['z_std']))))
            print(f"  {ds[:5]} L={L} {ch:5s} {r['seed']:4d}  mism {mism}  exact {acc_exact:.4f}  "
                  f"acc@10/100/1e3/1e4/1e5 {acc.mean(1)[[0,2,4,6,8]].round(3)}  cost {c1:.3g} ({f1})  ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open(os.path.join(HERE, 'F1_results.pkl'), 'wb'))
    print("F1_DONE", flush=True)
