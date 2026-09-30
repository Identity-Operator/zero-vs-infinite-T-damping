"""S1 regression: (0) e1 re-run reproduces original e1; (1) IndepCirc vs TorchCirc-CPU noise floor;
(2) m5_readout L=4 vs e1 image by image (if present); (3) per-cell S1 acceptance table with the
run's device (at most 1 differing prediction per cell, identical accuracies)."""
import os, sys, glob, shutil, pickle, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
CH, SEEDS = ('None', 'AD', 'Pauli', 'Depol'), (256, 512, 768, 1024, 1280, 1536, 1792, 2048)
ld = lambda s: {(r['channel'], r['seed']): r for c in CH if os.path.exists(os.path.join(HERE, f'e1_save_{s}_{c}.pkl')) for r in pickle.load(open(os.path.join(HERE, f'e1_save_{s}_{c}.pkl'), 'rb'))}
IND, TC = ld('indep'), ld('torchcirc')
orig = {(r['ch'], r['seed']): r['ck'][100] for r in pickle.load(open(os.path.join(HERE, 'scale_e1_None_AD_Pauli_Depol.pkl'), 'rb'))}
common = [k for k in orig if k in IND]
ok = all(IND[k]['test_acc'] == orig[k]['test'] and IND[k]['train_acc'] == orig[k]['train'] for k in common)
print(f"(0) re-run reproduces original e1 train/test accuracy in {len(common)}/{len(orig)} cells present: {ok}")
def cmp(A, B, la, lb):
    tot = ncell = 0
    for c in CH:
        seeds = [sd for sd in SEEDS if (c, sd) in A and (c, sd) in B]
        if not seeds: continue
        nd = [int((A[(c, s)]['test_pred'] != B[(c, s)]['test_pred']).sum()) for s in seeds]
        pdm = max(max(float(np.abs(np.asarray(A[(c, s)]['params'][k]) - np.asarray(B[(c, s)]['params'][k])).max()) for k in ('theta', 'W2', 'b2')) for s in seeds)
        dtr = max(abs(A[(c, s)]['train_acc'] - B[(c, s)]['train_acc']) for s in seeds)
        dte = max(abs(A[(c, s)]['test_acc'] - B[(c, s)]['test_acc']) for s in seeds)
        print(f"   {c:5s} seeds {seeds}: pred diffs {nd}  max|d train_acc| {dtr:.4f}  max|d test_acc| {dte:.4f}  max|d param| {pdm:.1e}")
        tot += sum(nd); ncell += len(seeds)
    print(f"   total differing test predictions {la} vs {lb}: {tot} over {ncell} cells x 3037 images")
print("(1) IndepCirc vs TorchCirc (CPU), standardized readout:"); cmp(IND, TC, 'indep', 'torchcirc')
src = os.path.join(HERE, '..', '..', 'noise_structure_mnist', 'm5_readout', 'results.pkl')
if os.path.exists(src):
    import tempfile; dst = os.path.join(tempfile.gettempdir(), 'm5_readout_cmp.pkl'); shutil.copy2(src, dst)   # copy before loading: the writer may be live
    R = [r for r in pickle.load(open(dst, 'rb')) if r.get('L') == 4 and r.get('dataset', 'MNIST') == 'MNIST']
    print(f"(2) m5_readout L=4 records: {len(R)}; readout fields: {sorted({str(r.get('readout')) for r in R})}")
    M = {(r['channel'], r['seed']): r for r in R}
    if all((c, s) in M for c in CH for s in SEEDS):
        print("  m5 (GPU) vs e1 IndepCirc:"); cmp(M, IND, 'm5', 'indep')
        print("  m5 (GPU) vs TorchCirc-CPU:"); cmp(M, TC, 'm5', 'torchcirc')
    else:
        print("  incomplete:", sorted(set((c, s) for c in CH for s in SEEDS) - set(M)))
else:
    print("(2) m5_readout/results.pkl not present yet")

# --- per-cell acceptance table for S1 ---
if os.path.exists(src):
    R = [r for r in pickle.load(open(dst, 'rb')) if r.get('L') == 4 and r.get('dataset', 'MNIST') == 'MNIST']
    M = {(r['channel'], r['seed']): r for r in R}
    devkeys = sorted({k for r in R for k in r if 'device' in k.lower() or 'shard' in k.lower()})
    print(f"\n(3) per-cell S1 acceptance vs e1_save_indep (device fields found: {devkeys})")
    bad = []
    for c in CH:
        for sd in SEEDS:
            if (c, sd) not in M: print(f"   {c:5s} {sd:4d}  MISSING"); bad.append((c, sd, 'missing')); continue
            r, e = M[(c, sd)], IND[(c, sd)]
            nd = int((np.asarray(r['test_pred']) != e['test_pred']).sum())
            same_tr, same_te = r['train_acc'] == e['train_acc'], r['test_acc'] == e['test_acc']
            dev = ' '.join(f"{k}={r[k]}" for k in devkeys)
            flag = '' if (nd <= 1 and same_tr and (same_te or nd == 1)) else '  <-- FAIL'
            print(f"   {c:5s} {sd:4d}  {dev:28s} pred diffs {nd}  train_acc equal {same_tr}  test_acc equal {same_te}{flag}")
            if flag: bad.append((c, sd, nd))
    print("S1 ACCEPTANCE:", "PASS (all 32 cells within <=1 prediction, accuracies identical)" if not bad else f"FAIL {bad}")
