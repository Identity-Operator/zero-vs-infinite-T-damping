"""F1 supplement: the main grid stops at N_s = 1e5, where the unital channels
at L >= 3 are still more than 0.01 below their exact accuracy, so their operational shot cost is censored.
This extends the same inference model (multinomial, R=5, same generators keyed by N_s) to
N_s in {1e5, 3e5, 1e6, 3e6, 1e7, 3e7, 1e8} for every S1/S2 run. numpy's multinomial sampler costs the same
for any N_s. Output: F1_ext_results.pkl. The main-grid result F1_results.pkl is unchanged."""
import os, sys, time, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import Circ, load_v5, classify
from F1_shots import SRC, CH_ID, CHUNK, R_DRAWS, load_records
torch.set_num_threads(4)
NS_EXT = (100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000)
if __name__ == '__main__':
    t0 = time.time(); out = []
    for ds in ('MNIST', 'FashionMNIST'):
        Ate, Yte = load_v5(ds)[2:]
        A = torch.as_tensor(Ate)
        for r in sorted(load_records(SRC[ds]), key=lambda r: (r['L'], CH_ID[r['channel']], r['seed'])):
            prm = r['params']; circ = Circ(4, r['L'], r['channel'], r['p_noise'], r.get('noise_placement', 'after_entangler'))
            th = torch.as_tensor(np.asarray(prm['theta']))
            with torch.no_grad():
                P = torch.cat([circ.probs(A[i:i + CHUNK], th) for i in range(0, len(A), CHUNK)]).numpy()
            P = np.clip(P, 0.0, None); P /= P.sum(1, keepdims=True); sign = circ.sign.numpy()
            acc = np.zeros((len(NS_EXT), R_DRAWS))
            for i, ns in enumerate(NS_EXT):
                for k in range(R_DRAWS):
                    rng = np.random.default_rng([0 if ds == 'MNIST' else 1, r['L'], CH_ID[r['channel']], r['seed'], ns, k])
                    acc[i, k] = (classify(rng.multinomial(ns, P) @ sign / ns, prm) == Yte).mean()
            out.append(dict(dataset=ds, L=r['L'], channel=r['channel'], seed=r['seed'], acc_exact=float(r['test_acc']), Ns=NS_EXT, acc=acc))
            print(f"  {ds[:5]} L={r['L']} {r['channel']:5s} {r['seed']:4d} exact {r['test_acc']:.4f} acc {acc.mean(1).round(4)} ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open(os.path.join(HERE, 'F1_ext_results.pkl'), 'wb'))
    print("F1_EXT_DONE", flush=True)
