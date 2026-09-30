"""F1 for the shot-trained models (S5, m5_shots): the same finite-shot inference
evaluation as F1, in both conventions of shots_eval.py ('calib' is the one consistent with shot-noise
training and with the record's own test_acc_shot; 'stored' matches F1_results.pkl), over the merged grid
10 ... 1e8, for every run. Cross-check: at the training N_s, 'calib' must agree with the record's
test_acc_shot_mean within sampling error. Output: F1_s5_results.pkl.
usage: F1_s5.py [path to results.pkl]  (default: ../../noise_structure_mnist/m5_shots/results.pkl)"""
import os, sys, time, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import load_v5
from shots_eval import run_probs, evaluate, CH_ID
from F1_shots import load_records
from F1_calib import NS
torch.set_num_threads(4)
if __name__ == '__main__':
    src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', '..', 'noise_structure_mnist', 'm5_shots', 'results.pkl')
    recs = [r for r in load_records(src) if r.get('shots')]
    print(f"{len(recs)} shot-trained records", flush=True)
    Atr, Ytr, Ate, Yte = load_v5('MNIST'); Atr, Ate = torch.as_tensor(Atr), torch.as_tensor(Ate)
    t0 = time.time(); out = []
    for r in sorted(recs, key=lambda r: (r['shots'], r['L'], CH_ID[r['channel']], r['seed'])):
        P_te, sign = run_probs(r, Ate); P_tr, _ = run_probs(r, Atr)
        exact_pred = (((P_te @ sign - np.asarray(r['params']['z_mean'])) / np.asarray(r['params']['z_std'])) @ np.asarray(r['params']['W2']).T
                      + np.asarray(r['params']['b2'])).argmax(1)
        acc = evaluate(r, P_te, P_tr, sign, Yte, NS, tag=20)
        at = list(NS).index(r['shots']) if r['shots'] in NS else None
        out.append(dict(L=r['L'], channel=r['channel'], seed=r['seed'], train_shots=r['shots'], acc_exact=float(r['test_acc']),
                        pred_mismatch=int((exact_pred != np.asarray(r['test_pred'])).sum()), Ns=NS, acc_stored=acc['stored'],
                        acc_calib=acc['calib'], record_shot_acc=r.get('test_acc_shot_mean'),
                        calib_at_train_shots=float(acc['calib'][at].mean()) if at is not None else None,
                        sigma=float(np.mean(r['params']['z_std']))))
        print(f"  shots={r['shots']:6d} L={r['L']} {r['channel']:5s} {r['seed']:4d} mism {out[-1]['pred_mismatch']} exact {r['test_acc']:.4f}"
              f"  record shot acc {r.get('test_acc_shot_mean', float('nan')):.4f} vs calib {out[-1]['calib_at_train_shots']}  ({time.time()-t0:.0f}s)", flush=True)
        pickle.dump(out, open(os.path.join(HERE, 'F1_s5_results.pkl'), 'wb'))
    print("F1_S5_DONE", flush=True)
