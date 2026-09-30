# -*- coding: utf-8 -*-
"""
Acceptance checks for the Phase 1 stages.

Exits with status 1 on any failure, which stops run_queue.sh. Prints the
per-cell summary (mean and sample SD over seeds) that goes into the status
report.

    python check_stage.py baselines | diagnostics | fixedpoint | fashion
    python check_stage.py strength | topology | channels | fashion_readout | inside | shots
    python check_stage.py width_diag | pdep | pdep_L8
    python check_stage.py depth --L 4
    python check_stage.py readout --L 4
    python check_stage.py checkA
    python check_stage.py depth --L 1 --root /tmp/x --seeds 256 --allow-scratch   # test the checks

checkA bridges the pilot to the new runs: for None and AD at L=4, train_acc
must equal the pilot's final training accuracy (m4_rescale_train,
acc_hist[-1][1]) seed by seed, and test_acc_250 must be within one prediction
of the pilot's test accuracy. The pilot used the same training images and
preprocessing, and its test split is the first 250 images of the full one.

channels checks each noise channel's Bloch map (T, t) against Table I of the
manuscript at every strength p the runs use, and that it is trace preserving.
"""
import argparse
import math
import os
import pickle
import sys

import numpy as np

from mnist_hybrid_multi import PREP, load_mnist_binary
from run_phase1 import (CHANNELS, CLASSES, DIAG_CHANNELS, DIAG_L, DIAG_THETA_SEEDS,
                        HERE, N_STEPS, OFFSETS, P_NOISE, SEEDS, STAGES, TOPO)

N_TEST = {'MNIST': 3037, 'FashionMNIST': 3000}
PILOT = os.path.join(HERE, 'm4_rescale_train', 'mnist_results.pkl')

failures = []
opts = argparse.Namespace(root=HERE, seeds=SEEDS, allow_scratch=False)


def check(ok, msg):
    if not ok:
        failures.append(msg)
        print(f"  FAIL: {msg}", flush=True)
    return ok


def load(path):
    if not check(os.path.exists(path), f"{path} missing"):
        return []
    with open(path, 'rb') as f:
        return pickle.load(f)


def labels(dataset):
    return load_mnist_binary(classes=CLASSES[dataset], n_train=1000, n_test=None,
                             dataset=dataset, pca_seed=0)[3]


def check_stamp(r, dataset, tag):
    check(r.get('prep') == PREP, f"{tag}: prep={r.get('prep')}")
    check(r.get('n_test') == N_TEST[dataset], f"{tag}: n_test={r.get('n_test')}")
    check(r.get('pca_seed') == 0, f"{tag}: pca_seed={r.get('pca_seed')}")
    check(isinstance(r.get('n_clip'), int), f"{tag}: n_clip missing")
    check(isinstance(r.get('git_commit'), str) and len(r['git_commit']) == 40,
          f"{tag}: git_commit missing")
    if not opts.allow_scratch:
        check(r.get('git_dirty') is False and r.get('scratch') is False,
              f"{tag}: written from uncommitted code or as scratch")


def one_code_version(recs, what):
    shas = {r.get('code_sha') for r in recs}
    check(len(shas) == 1, f"{what}: records from {len(shas)} code versions {shas}")
    clips = {r.get('n_clip') for r in recs}
    check(len(clips) == 1, f"{what}: n_clip differs between records {clips}")
    return shas, clips


def sd(x):
    return float(np.std(x, ddof=1)) if len(x) > 1 else float('nan')


def check_circuits(path, dataset, cells, n_steps=N_STEPS, readout='raw',
                   noise_placement='after_entangler'):
    """cells: (L, channel), (L, channel, p, entangler) or (L, channel, p, entangler,
    shots); p and entangler default to Sec. 2, shots to None (exact)."""
    recs = load(path)
    if not recs:
        return
    Y = labels(dataset)
    keys = [(r.get('dataset'), r.get('L'), r.get('entangling'), r.get('p_noise'),
             r.get('channel'), r.get('shots'), r.get('seed')) for r in recs]
    check(len(keys) == len(set(keys)), f"{path}: duplicate configurations")
    for c in cells:
        L, ch, p, topo, shots = tuple(c) + (P_NOISE, TOPO, None)[len(c) - 2:]
        cell = sorted((r for r in recs if (r.get('dataset'), r.get('L'), r.get('channel'),
                                           r.get('entangling'), r.get('shots')) ==
                       (dataset, L, ch, topo, shots)
                       and abs(r.get('p_noise', -1) - p) < 1e-12), key=lambda r: r['seed'])
        tag = f"{dataset} L={L} {ch}" + (f" p={p}" if p != P_NOISE else '') + \
            (f" {topo}" if topo != TOPO else '') + (f" N_s={shots}" if shots else '')
        check(tuple(r['seed'] for r in cell) == opts.seeds,
              f"{tag}: seeds {[r['seed'] for r in cell]}, want {list(opts.seeds)}")
        for r in cell:
            t = f"{tag} seed={r['seed']}"
            check_stamp(r, dataset, t)
            pred = np.asarray(r.get('test_pred'))
            check(pred.shape == (N_TEST[dataset],), f"{t}: test_pred shape {pred.shape}")
            if pred.shape == Y.shape:
                check(abs((pred == Y).mean() - r['test_acc']) < 1e-12,
                      f"{t}: test_acc disagrees with test_pred")
                check(abs((pred[:250] == Y[:250]).mean() - r['test_acc_250']) < 1e-12,
                      f"{t}: test_acc_250 disagrees with test_pred")
            h = r.get('acc_hist') or [(None, None, None)]
            check(h[-1][0] == n_steps, f"{t}: last logged step {h[-1][0]}, want {n_steps}")
            check(r.get('readout', 'raw') == readout, f"{t}: readout {r.get('readout')}, want {readout}")
            check(r.get('noise_placement', 'after_entangler') == noise_placement,
                  f"{t}: noise_placement {r.get('noise_placement')}, want {noise_placement}")
            if readout == 'standardized':
                check({'z_mean', 'z_std'} <= set(r.get('params', {})), f"{t}: training statistics missing")
            check(r.get('train_acc') == h[-1][1], f"{t}: train_acc != acc_hist[-1][1]")
            check(math.isfinite(r.get('loss', float('nan'))), f"{t}: loss not finite")
            check(sorted(r.get('offset_acc', {})) == sorted(OFFSETS), f"{t}: offsets")
            s = r.get('sensitivity', float('nan'))
            check(math.isfinite(s) and s > 0, f"{t}: sensitivity={s}")
            check({'theta', 'W2', 'b2'} <= set(r.get('params', {})), f"{t}: params missing")
            if shots:
                sp = np.asarray(r.get('shot_test_pred'))
                acc = r.get('test_acc_shot') or []
                check(sp.shape == (r.get('shot_draws', -1), N_TEST[dataset]) and len(acc) == sp.shape[0],
                      f"{t}: shot draws malformed")
                if sp.shape[-1:] == Y.shape:
                    check(np.allclose([(q == Y).mean() for q in sp], acc, atol=1e-12)
                          and abs(np.mean(acc) - r['test_acc_shot_mean']) < 1e-12,
                          f"{t}: shot accuracies disagree with shot_test_pred")
        if cell:
            te = [r['test_acc'] for r in cell]
            tr = [r['train_acc'] for r in cell]
            se = [r['sensitivity'] for r in cell]
            off = '  '.join(f"{d:+g}:{np.mean([r['offset_acc'][d] for r in cell]):.3f}"
                            for d in OFFSETS)
            if shots:
                sm = [r['test_acc_shot_mean'] for r in cell]
                print(f"  {tag:24s} shot-sampled test {np.mean(sm):.4f} ± {sd(sm):.4f}", flush=True)
            print(f"  {tag:24s} n={len(cell)} test {np.mean(te):.4f} ± {sd(te):.4f}  "
                  f"train {np.mean(tr):.4f}  gap {np.mean(tr) - np.mean(te):+.4f}  "
                  f"first250 {np.mean([r['test_acc_250'] for r in cell]):.4f}  "
                  f"sens {np.mean(se):.3f} ± {sd(se):.3f}  offset acc {off}", flush=True)
    shas, clips = one_code_version(recs, path)
    print(f"  {path}: {len(recs)} records, code_sha {shas}, n_clip {clips}", flush=True)


def check_baselines(path, dataset):
    recs = load(path)
    Y = labels(dataset)
    check(sorted(r.get('model') for r in recs) == ['knn15', 'logreg', 'svc_rbf'],
          f"{path}: models {[r.get('model') for r in recs]}")
    for r in recs:
        check_stamp(r, dataset, f"{dataset} {r.get('model')}")
        pred = np.asarray(r.get('test_pred'))
        check(pred.shape == Y.shape and abs((pred == Y).mean() - r['test_acc']) < 1e-12,
              f"{r.get('model')}: test_acc disagrees with test_pred")
        print(f"  {dataset} {r['model']:8s} train {r['train_acc']:.4f}  test {r['test_acc']:.4f}  "
              f"first250 {r['test_acc_250']:.4f}  rest {r['test_acc_rest']:.4f}", flush=True)
    if recs:
        one_code_version(recs, path)


def check_diagnostics(path):
    recs = load(path)
    keys = [(r.get('variant'), r.get('L'), r.get('channel'), r.get('theta_seed')) for r in recs]
    want = {(v, L, ch, s) for v in ('v5', 'legacy300') for L in DIAG_L
            for ch in DIAG_CHANNELS for s in DIAG_THETA_SEEDS}
    check(len(keys) == len(set(keys)) and set(keys) == want,
          f"{path}: {len(set(keys))} distinct of {len(want)} expected, {len(keys)} records")
    for r in recs:
        tag = f"{r.get('variant')} L={r.get('L')} {r.get('channel')} seed={r.get('theta_seed')}"
        check(r.get('prep') == PREP and (opts.allow_scratch or
                                         (r.get('git_dirty') is False and r.get('scratch') is False)),
              f"{tag}: stamp")
        check(math.isfinite(r.get('J', float('nan'))) and math.isfinite(r.get('sigma', float('nan'))),
              f"{tag}: J or sigma not finite")
    for v in ('v5', 'legacy300'):
        for L in DIAG_L:
            base = np.mean([r['sigma'] for r in recs if (r['variant'], r['L'], r['channel']) == (v, L, 'None')] or [np.nan])
            for ch in DIAG_CHANNELS:
                c = [r for r in recs if (r['variant'], r['L'], r['channel']) == (v, L, ch)]
                if c:
                    J = [r['J'] for r in c]
                    print(f"  {v:9s} L={L} {ch:7s} sigma/sigma_None {np.mean([r['sigma'] for r in c]) / base:.4f}  "
                          f"J {np.mean(J):.3f} ± {np.std(J):.3f}", flush=True)
    if recs:
        shas = {r.get('code_sha') for r in recs}
        check(len(shas) == 1, f"{path}: records from {len(shas)} code versions")


def check_channels(ps=(0.1, 0.2, 0.3, 0.5, 0.7)):
    """Each channel's (T, t) against Table I, with c = sqrt(1 - p)."""
    from torch_circ import TorchCirc
    P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]),
         np.diag([1.0, -1.0])]
    for p in ps:
        c = np.sqrt(1 - p)
        aniso = np.diag([c, c, c ** 2])
        want = {'None': (np.eye(3), [0, 0, 0]), 'AD': (aniso, [0, 0, p]),
                'AD_flip': (aniso, [0, 0, -p]), 'Pauli': (aniso, [0, 0, 0]),
                'Depol': ((2 * c + c ** 2) / 3 * np.eye(3), [0, 0, 0])}
        for ch, (T_want, t_want) in want.items():
            K = [k.cpu().numpy() for k in TorchCirc(1, 1, noisetype=ch, p_noise=p,
                                                     device='cpu').kraus] or [np.eye(2)]

            def N(rho):
                return sum(k @ rho @ k.conj().T for k in K)
            T = np.array([[0.5 * np.trace(P[i] @ N(P[j])) for j in (1, 2, 3)] for i in (1, 2, 3)])
            t = np.array([0.5 * np.trace(P[i] @ N(P[0])) for i in (1, 2, 3)])
            tp = np.allclose(sum(k.conj().T @ k for k in K), np.eye(2), rtol=0, atol=1e-12)
            ok = (np.allclose(T, T_want, rtol=0, atol=1e-12) and
                  np.allclose(t, t_want, rtol=0, atol=1e-12))
            check(tp, f"channels: {ch} p={p} is not trace preserving")
            check(ok, f"channels: {ch} p={p} (T, t) differ from Table I")
            print(f"  p={p} {ch:7s} diag T {np.real(np.diag(T)).round(4)} t {np.real(t).round(4)} "
                  f"{'ok' if ok and tp else 'MISMATCH'}", flush=True)


def check_width_diag(path):
    from run_phase1 import WIDTH_L, WIDTH_N
    recs = load(path)
    keys = [(r.get('n'), r.get('L'), r.get('channel'), r.get('theta_seed')) for r in recs]
    want = {(n, L, ch, s) for n in WIDTH_N for L in WIDTH_L for ch in CHANNELS
            for s in DIAG_THETA_SEEDS}
    check(len(keys) == len(set(keys)) and set(keys) == want,
          f"{path}: {len(set(keys))} distinct of {len(want)} expected, {len(keys)} records")
    for r in recs:
        tag = f"n={r.get('n')} L={r.get('L')} {r.get('channel')} seed={r.get('theta_seed')}"
        check(r.get('prep') == PREP and (opts.allow_scratch or
                                         (r.get('git_dirty') is False and r.get('scratch') is False)),
              f"{tag}: stamp")
        check(math.isfinite(r.get('J', float('nan'))) and math.isfinite(r.get('sigma', float('nan'))),
              f"{tag}: J or sigma not finite")
        check(np.asarray(r.get('sigma_per_qubit')).shape == (r.get('n'),), f"{tag}: sigma_per_qubit shape")
    for n in WIDTH_N:
        for L in WIDTH_L:
            base = np.mean([r['sigma'] for r in recs if (r['n'], r['L'], r['channel']) == (n, L, 'None')] or [np.nan])
            row = []
            for ch in CHANNELS:
                c = [r for r in recs if (r['n'], r['L'], r['channel']) == (n, L, ch)]
                if c:
                    row.append(f"{ch} s/s0 {np.mean([r['sigma'] for r in c]) / base:.2e} J {np.mean([r['J'] for r in c]):.2f}")
            print(f"  n={n} L={L}  " + " | ".join(row), flush=True)
    if recs:
        shas = {r.get('code_sha') for r in recs}
        check(len(shas) == 1, f"{path}: records from {len(shas)} code versions")


def check_a():
    with open(PILOT, 'rb') as f:
        pilot = {(r['channel'], r['seed']): r for r in pickle.load(f)}
    new = {(r['channel'], r['seed']): r for r in load(os.path.join(opts.root, 'm5_depth', 'results.pkl'))
           if r.get('L') == 4 and r.get('dataset') == 'MNIST'}
    for ch in ('None', 'AD'):
        for s in opts.seeds:
            p, n = pilot.get((ch, s)), new.get((ch, s))
            if not check(p is not None and n is not None, f"CHECK A {ch} seed={s}: record missing"):
                continue
            p_train = p['acc_hist'][-1][1]
            d250 = round(abs(n['test_acc_250'] - p['test_acc']) * 250)
            check(n['train_acc'] == p_train,
                  f"CHECK A {ch} seed={s}: train_acc {n['train_acc']} != pilot {p_train}")
            check(d250 <= 1, f"CHECK A {ch} seed={s}: test_acc_250 {n['test_acc_250']} vs "
                             f"pilot {p['test_acc']} ({d250} predictions)")
            print(f"  CHECK A {ch:4s} seed={s}: train {n['train_acc']:.3f}/{p_train:.3f}  "
                  f"first250 {n['test_acc_250']:.3f}/{p['test_acc']:.3f} ({d250} pred)  "
                  f"loss identical: {n['loss'] == p['loss']}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['baselines', 'diagnostics', 'depth', 'fixedpoint',
                                      'fashion', 'strength', 'topology', 'channels', 'checkA',
                                      'readout', 'fashion_readout', 'inside', 'shots',
                                      'width_diag', 'pdep', 'pdep_L8'])
    ap.add_argument('--L', type=int, nargs='+')
    ap.add_argument('--root', default=HERE, help='directory holding the m5_* outputs')
    ap.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS))
    ap.add_argument('--allow-scratch', action='store_true',
                    help='accept scratch or uncommitted-code records (for testing the checks)')
    args = ap.parse_args()
    opts.root, opts.seeds, opts.allow_scratch = args.root, tuple(args.seeds), args.allow_scratch
    print(f"check_stage {args.stage} {args.L or ''}", flush=True)
    p = lambda d, f='results.pkl': os.path.join(opts.root, d, f)
    if args.stage == 'baselines':
        check_baselines(p('m5_baseline', 'baselines.pkl'), 'MNIST')
    elif args.stage == 'diagnostics':
        check_diagnostics(p('m5_diagnostics'))
    elif args.stage == 'depth':
        if not args.L:
            raise SystemExit("depth needs --L")
        check_circuits(p('m5_depth'), 'MNIST', [(L, ch) for L in args.L for ch in CHANNELS])
    elif args.stage == 'fixedpoint':
        check_circuits(p('m5_fixedpoint'), 'MNIST', [(4, 'AD_flip')])
    elif args.stage == 'fashion':
        check_baselines(p('m5_fashion', 'baselines.pkl'), 'FashionMNIST')
        check_circuits(p('m5_fashion'), 'FashionMNIST', [(4, ch) for ch in CHANNELS])
    elif args.stage in ('strength', 'topology'):
        cfg = STAGES[args.stage]
        check_circuits(p(cfg['outdir']), 'MNIST',
                       [(4, ch, q, topo) for topo in cfg.get('topo', (TOPO,))
                        for q in cfg.get('p', (P_NOISE,)) for ch in cfg['channels']])
    elif args.stage == 'channels':
        check_channels()
    elif args.stage in ('pdep', 'pdep_L8'):
        cfg = STAGES[args.stage]
        check_circuits(p(cfg['outdir']), 'MNIST',
                       [(L, ch, q, TOPO) for L in cfg['L'] for q in cfg['p'] for ch in cfg['channels']],
                       readout=cfg['readout'])
    elif args.stage == 'width_diag':
        check_width_diag(p('m5_width_diag'))
    elif args.stage == 'shots':
        cfg = STAGES['shots']
        check_circuits(p(cfg['outdir']), 'MNIST',
                       [(L, ch, P_NOISE, TOPO, sh) for L, sh in cfg['L_shots'] for ch in cfg['channels']],
                       readout=cfg['readout'])
    elif args.stage in ('readout', 'fashion_readout', 'inside'):
        cfg = STAGES[args.stage]
        Ls = args.L if args.stage == 'readout' else cfg['L']
        if not Ls:
            raise SystemExit(f"{args.stage} needs --L")
        check_circuits(p(cfg['outdir']), cfg['dataset'], [(L, ch) for L in Ls for ch in cfg['channels']],
                       n_steps=cfg.get('n_steps', N_STEPS), readout=cfg.get('readout', 'raw'),
                       noise_placement=cfg.get('noise_placement', 'after_entangler'))
    else:
        check_a()
    print(f"check_stage {args.stage}: {'FAILED, ' + str(len(failures)) + ' problems' if failures else 'PASSED'}",
          flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
