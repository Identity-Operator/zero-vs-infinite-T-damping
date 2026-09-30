"""F1 supplement: the S1/S2 models evaluated with the readout calibrated at the
same shot budget ('calib' convention of shots_eval.py) over the merged grid 10 ... 1e8, to compare like
for like with the shot-trained S5 models, whose evaluation uses that convention. F1_results.pkl
('stored' convention) is unchanged. Output: F1_calib_results.pkl."""
import os, sys, time, pickle, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import load_v5
from shots_eval import run_probs, evaluate, CH_ID
from F1_shots import SRC, load_records
torch.set_num_threads(4)
NS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000)
if __name__ == '__main__':
    t0 = time.time(); out = []
    for ti, ds in enumerate(('MNIST', 'FashionMNIST')):
        Atr, Ytr, Ate, Yte = load_v5(ds); Atr, Ate = torch.as_tensor(Atr), torch.as_tensor(Ate)
        for r in sorted(load_records(SRC[ds]), key=lambda r: (r['L'], CH_ID[r['channel']], r['seed'])):
            P_te, sign = run_probs(r, Ate); P_tr, _ = run_probs(r, Atr)
            acc = evaluate(r, P_te, P_tr, sign, Yte, NS, tag=10 + ti, conventions=('calib',))['calib']
            out.append(dict(dataset=ds, L=r['L'], channel=r['channel'], seed=r['seed'], acc_exact=float(r['test_acc']), Ns=NS, acc=acc))
            print(f"  {ds[:5]} L={r['L']} {r['channel']:5s} {r['seed']:4d} exact {r['test_acc']:.4f} calib acc@1e2/1e4/1e6/1e8 {acc.mean(1)[[2,6,10,14]].round(3)} ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(out, open(os.path.join(HERE, 'F1_calib_results.pkl'), 'wb'))
    print("F1_CALIB_DONE", flush=True)
