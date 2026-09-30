"""Shot evaluation of the p-dependence stages:
S7 (m5_pdep, L=4, p in {0.05, 0.1}) and S7b (m5_pdep_L8, L=8, p=0.1, with its own None reference).
'calib' convention over the grid 10 ... 1e8, as F1_calib.py; operational cost as F1_shots.op_cost (first N_s
within 0.01 of the exact accuracy, log-interpolated; geometric mean over seeds; censored runs excluded).
The L=4 None reference is S1's (m5_readout), whose calib costs are in F1_calib_results.pkl.
Usage: python F1_pdep.py m5_pdep | m5_pdep_L8        (evaluate; writes F1_pdep_<stage>.pkl)
       python F1_pdep.py summary                     (costs, ratios to None, kappa)"""
import os, sys, time, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
NS = (10, 30, 100, 300, 1_000, 3_000, 10_000, 30_000, 100_000, 300_000, 1_000_000, 3_000_000, 10_000_000, 30_000_000, 100_000_000)
CODE = os.path.join(HERE, '..', '..', 'noise_structure_mnist')


def evaluate_stage(stage):
    import torch
    from shots_lib import load_v5
    from shots_eval import run_probs, evaluate, CH_ID
    from F1_shots import load_records
    torch.set_num_threads(3)
    Atr, Ytr, Ate, Yte = load_v5('MNIST'); Atr, Ate = torch.as_tensor(Atr), torch.as_tensor(Ate)
    t0 = time.time(); out = []
    recs = sorted(load_records(os.path.join(CODE, stage, 'results.pkl')),
                  key=lambda r: (r['L'], r['p_noise'], CH_ID[r['channel']], r['seed']))
    for r in recs:
        P_te, sign = run_probs(r, Ate); P_tr, _ = run_probs(r, Atr)
        acc = evaluate(r, P_te, P_tr, sign, Yte, NS, tag=30, conventions=('calib',))['calib']
        out.append(dict(stage=stage, L=r['L'], p=r['p_noise'], channel=r['channel'], seed=r['seed'],
                        acc_exact=float(r['test_acc']), z_std=np.asarray(r['params']['z_std']), Ns=NS, acc=acc))
        print(f"  {stage} L={r['L']} p={r['p_noise']} {r['channel']:5s} {r['seed']:4d} exact {r['test_acc']:.4f} "
              f"calib acc@1e2/1e4/1e6/1e8 {acc.mean(1)[[2, 6, 10, 14]].round(3)} ({time.time() - t0:.0f}s)", flush=True)
        pickle.dump(out, open(os.path.join(HERE, f'F1_pdep_{stage}.pkl'), 'wb'))
    print('F1_PDEP_DONE', stage, flush=True)


def summary():
    from F1_shots import op_cost
    rows = []
    C1 = pickle.load(open(os.path.join(HERE, 'F1_calib_results.pkl'), 'rb'))
    S1 = pickle.load(open(os.path.join(CODE, 'm5_readout', 'results.pkl'), 'rb'))
    def gm_cost(recs):
        cs = []
        for r in recs:
            (c, f), _ = op_cost(np.array(r['Ns'], float), r['acc'].mean(1), r['acc_exact'])
            if r['acc_exact'] > 0.5: cs.append(c)
        ok = [c for c in cs if np.isfinite(c)]
        return (np.exp(np.mean(np.log(ok))) if ok else np.nan), len(cs) - len(ok)
    refs = {4: ([x for x in C1 if x['dataset'] == 'MNIST' and x['L'] == 4 and x['channel'] == 'None'],
                np.mean([np.mean(r['params']['z_std']) for r in S1 if r['L'] == 4 and r['channel'] == 'None']))}
    for stage in ('m5_pdep', 'm5_pdep_L8'):
        f = os.path.join(HERE, f'F1_pdep_{stage}.pkl')
        if not os.path.exists(f):
            continue
        E = pickle.load(open(f, 'rb'))
        for L in sorted({e['L'] for e in E}):
            if L not in refs:
                nn = [e for e in E if e['L'] == L and e['channel'] == 'None']
                refs[L] = (nn, np.mean([np.mean(e['z_std']) for e in nn]))
            cN, _ = gm_cost(refs[L][0]); sN = refs[L][1]
            for p in sorted({e['p'] for e in E if e['L'] == L}):
                for ch in ('None', 'AD', 'Pauli', 'Depol'):
                    rs = [e for e in E if e['L'] == L and e['p'] == p and e['channel'] == ch]
                    if not rs:
                        continue
                    c, cens = gm_cost(rs)
                    kap = (sN / np.mean([np.mean(e['z_std']) for e in rs])) ** 2
                    rows.append(dict(L=L, p=p, channel=ch, calib_cost=c, calib_ratio=c / cN, censored=cens, kappa=kap))
                    print(f'L={L} p={p} {ch:5s} cost {c:9.3g}  ratio to None {c / cN:8.3g}  kappa {kap:8.3g}  (censored {cens})')
    pickle.dump(rows, open(os.path.join(HERE, 'F1_pdep_summary.pkl'), 'wb'))


if __name__ == '__main__':
    if sys.argv[1] == 'summary':
        summary()
    else:
        evaluate_stage(sys.argv[1])
