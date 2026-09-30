# -*- coding: utf-8 -*-
"""
Driver for every m5_* result.

One stage per call. Each stage checkpoints after every configuration and skips
the finished ones, so an interrupted stage resumes where it stopped.

  baselines    P1.0  logistic regression, RBF-SVC, kNN15 on the angles  -> m5_baseline/
  diagnostics  P1.3  feature scale sigma and Fisher ratio J at untrained
                     parameters: 5 channels x L in {1, 4} x 5 draws,
                     preprocessing variants v5 and legacy300              -> m5_diagnostics/
  depth        P1.1  {None, AD, Pauli, Depol} x --L x 8 seeds              -> m5_depth/
  fixedpoint   P1.2  AD_flip at L=4 x 8 seeds                              -> m5_fixedpoint/
  fashion      Q6    Fashion-MNIST (0, 2, 4): the baselines, then the four
                     channels at L=4 x 8 seeds                             -> m5_fashion/
  strength     P2.2  {AD, Pauli} x p in {0.1, 0.2} x 8 seeds at L=4, ring;
                     Pauli matched to AD at each p (Table I); None and
                     p=0.3 come from P1.1                                  -> m5_strength/
  topology     P2.1  {None, AD, Pauli} x entangler in {none, full} x 8 seeds
                     at L=4, p=0.3; ring comes from P1.1                   -> m5_topology/
  readout      S1    standardized readout, {None, AD, Pauli, Depol} x --L
                     x 8 seeds                                             -> m5_readout/
  fashion_readout S2 Fashion-MNIST (0, 2, 4), standardized readout, four
                     channels at L=4 x 8 seeds                             -> m5_fashion_readout/
  inside       S4    noise after every CNOT of the entangler, standardized
                     readout, {None, AD, AD_flip} at L=4 x 8 seeds         -> m5_inside/
  shots        S5    shot-noise training and evaluation,
                     standardized readout: L=1-4 at N_s=1e3, and L=4 at
                     N_s=100 and 1e4; four channels x 8 seeds              -> m5_shots/
  pdep         S7    noise strength p in {0.05, 0.1}: AD, Pauli, Depol at L=4,
                     standardized readout, exact expectation values, 8 seeds;
                     None and p=0.3 come from S1                           -> m5_pdep/
  pdep_L8      S7b   p=0.1 at L=8: None, AD, Pauli, Depol, otherwise as
                     pdep (L=8 is past the depth where AD's feature scale
                     separates from its twirl's at this p)                  -> m5_pdep_L8/
  width_diag   S6a   sigma and J at untrained parameters against width:
                     n in {4, 6, 8} (n PCA components), L=1-6, four
                     channels, 5 draws, 300 training images                -> m5_width_diag/

The strength and topology stages are not used in the paper. The
standardized readout is the one of scale_gauge/scale_e1e2 (e1; see train_one).

Settings: training-extrema preprocessing with
pca_seed=0, n_train=1000, the full test set, n=4, p=0.3, ring entangler, no
pre-processing layer, Adam lr 0.05, 100 full-batch steps. Every circuit run
also records the test accuracy with a coherent offset added to all four test
angles and the input sensitivity, its trained parameters, and its
per-image test predictions.

    python run_phase1.py depth --L 4
    python run_phase1.py depth --L 4 --channels None --seeds 256 --outdir /tmp/x
    python run_phase1.py readout --L 4 --shard 0/2                            # GPU worker
    python run_phase1.py readout --L 4 --device cpu --threads 3 --shard 1/2   # CPU worker

--shard i/k runs the configurations whose index in the stage's grid is i mod
k, so k workers split a stage without overlap. Every finished configuration
is appended under a file lock, so the workers can share one results file.
RAM, not cores, is the limit: a CPU worker at L=4
peaks at about 3.4 GB, because autograd keeps every intermediate density
matrix in host RAM, and a GPU worker needs about 1.7 GB of host RAM. Run at
most one CPU worker at L=4, next to at most one GPU worker, and keep 3 GB
free; check `free -m` before starting each worker.

Results go to the m5_* directories next to this file, and only when every
tracked .py file here is committed, so each record's git_commit and code_sha
identify its code. Any other output directory is scratch: allowed with
uncommitted code and with --channels, --seeds, --p, --topo and --n-steps, and
its records say so. m4_* outputs are refused.
"""
import argparse
import fcntl
import hashlib
import os
import pickle
import subprocess
import time

import numpy as np
import torch

from mnist_hybrid_multi import classical_baseline, load_mnist_binary, train_one
from noise_discrimination_check import fisher_ratio
from torch_circ import TorchCirc

HERE = os.path.dirname(os.path.abspath(__file__))

SEEDS = (256, 512, 768, 1024, 1280, 1536, 1792, 2048)
CHANNELS = ('None', 'AD', 'Pauli', 'Depol')
N_QUBITS, P_NOISE, TOPO = 4, 0.3, 'ring'
N_TRAIN, N_STEPS, PCA_SEED = 1000, 100, 0
# Sec. 4 asks for +0.25, +0.5 and +1.0 rad; the negative offsets cost nothing
# and show whether the robustness depends on the sign of the shift.
OFFSETS = (-1.0, -0.5, -0.25, 0.25, 0.5, 1.0)
CLASSES = {'MNIST': (1, 3, 5), 'FashionMNIST': (0, 2, 4)}

DIAG_CHANNELS = CHANNELS + ('AD_flip',)
DIAG_L = (1, 4)
DIAG_THETA_SEEDS = (1, 2, 3, 4, 5)     # as in noise_discrimination_check.py
DIAG_N = 300

# Circuit stages also take p (default P_NOISE), topo (default TOPO), readout
# (default 'raw'), n_steps (default N_STEPS) and noise_placement (default
# 'after_entangler').
STAGES = {
    'baselines':   dict(outdir='m5_baseline', dataset='MNIST'),
    'diagnostics': dict(outdir='m5_diagnostics', dataset='MNIST'),
    'depth':       dict(outdir='m5_depth', dataset='MNIST', id='P1.1', channels=CHANNELS),
    'fixedpoint':  dict(outdir='m5_fixedpoint', dataset='MNIST', id='P1.2',
                        channels=('AD_flip',), L=(4,)),
    'fashion':     dict(outdir='m5_fashion', dataset='FashionMNIST', id='Q6',
                        channels=CHANNELS, L=(4,)),
    'strength':    dict(outdir='m5_strength', dataset='MNIST', id='P2.2',
                        channels=('AD', 'Pauli'), L=(4,), p=(0.1, 0.2)),
    'topology':    dict(outdir='m5_topology', dataset='MNIST', id='P2.1',
                        channels=('None', 'AD', 'Pauli'), L=(4,), topo=('none', 'full')),
    'readout':     dict(outdir='m5_readout', dataset='MNIST', id='S1', channels=CHANNELS,
                        readout='standardized'),
    'fashion_readout': dict(outdir='m5_fashion_readout', dataset='FashionMNIST', id='S2',
                            channels=CHANNELS, L=(4,), readout='standardized'),
    'inside':      dict(outdir='m5_inside', dataset='MNIST', id='S4',
                        channels=('None', 'AD', 'AD_flip'), L=(4,), readout='standardized',
                        noise_placement='inside_entangler'),
    'shots':       dict(outdir='m5_shots', dataset='MNIST', id='S5', channels=CHANNELS,
                        readout='standardized',
                        L_shots=((1, 1000), (2, 1000), (3, 1000), (4, 1000), (4, 100), (4, 10000))),
    'width_diag':  dict(outdir='m5_width_diag', dataset='MNIST'),
    'pdep':        dict(outdir='m5_pdep', dataset='MNIST', id='S7', channels=('AD', 'Pauli', 'Depol'),
                        L=(4,), p=(0.05, 0.1), readout='standardized'),
    'pdep_L8':     dict(outdir='m5_pdep_L8', dataset='MNIST', id='S7b', channels=CHANNELS,
                        L=(8,), p=(0.1,), readout='standardized'),
}
WIDTH_N = (4, 6, 8)
WIDTH_L = (1, 2, 3, 4, 5, 6)


def code_state():
    """HEAD, a hash of the tracked .py blobs here, and any uncommitted .py changes.

    code_sha changes only when the code does, so records written before and
    after a results-only commit still share it.
    """
    def git(*a):
        return subprocess.run(['git', *a], cwd=HERE, capture_output=True, text=True,
                              check=True).stdout
    dirty = git('status', '--porcelain', '--', '*.py').strip()
    blobs = git('ls-files', '-s', '--', '*.py')
    return {'git_commit': git('rev-parse', 'HEAD').strip(),
            'code_sha': hashlib.sha1(blobs.encode()).hexdigest()[:12],
            'git_dirty': bool(dirty)}, dirty


def load_results(path):
    if os.path.exists(path):
        with open(path, 'rb') as f:
            return pickle.load(f)
    return []


def save_results(path, results):
    """Write to a temporary file and rename, so an interruption never truncates it."""
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(results, f)
    os.replace(tmp, path)


def append_result(path, r):
    """Re-read, append and save under an exclusive lock, so parallel workers never
    overwrite each other's records."""
    with open(path + '.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        results = load_results(path)
        results.append(r)
        save_results(path, results)


def load_data(dataset):
    info = {}
    Xtr, Ytr, Xte, Yte = load_mnist_binary(classes=CLASSES[dataset], n_train=N_TRAIN,
                                           n_test=None, n_components=N_QUBITS,
                                           dataset=dataset, pca_seed=PCA_SEED, info=info)
    return Xtr, Ytr, Xte, Yte, info


def stamp(r, info, dataset, stage, state, scratch, n_train=N_TRAIN, has_test=True):
    """Provenance fields. has_test=False (diagnostics) leaves n_clip and clip_frac None."""
    r.update(prep=info['prep'], pca_seed=info['pca_seed'],
             n_clip=info['n_clip'] if has_test else None,
             clip_frac=info['clip_frac'] if has_test else None, n_train=n_train, n_qubits=N_QUBITS,
             dataset=dataset, classes=CLASSES[dataset], stage=stage, scratch=scratch,
             **state)
    return r


def run_baselines(outdir, dataset, state, scratch, data):
    path = os.path.join(outdir, 'baselines.pkl')
    if len(load_results(path)) == 3:
        print(f"{path}: already complete", flush=True)
        return
    Xtr, Ytr, Xte, Yte, info = data
    recs = [stamp(dict(r, n_test=len(Yte)), info, dataset, 'P1.0', state, scratch)
            for r in classical_baseline(Xtr, Ytr, Xte, Yte)]
    save_results(path, recs)


def shard_slice(full, i, k):
    """The configurations of worker i out of k: every k-th one, starting at i."""
    return [c for j, c in enumerate(full) if j % k == i]


def run_circuits(outdir, stage, dataset, grid, seeds, n_steps, state, scratch, data,
                 readout='raw', device='cuda', shard=(0, 1), noise_placement='after_entangler'):
    """Train every (L, topo, p, channel, shots) in grid for every seed not yet in the
    file, or, with shard=(i, k), those whose index in the full list is i mod k."""
    path = os.path.join(outdir, 'results.pkl')
    done = {(r['dataset'], r['L'], r['entangling'], r['p_noise'], r['channel'], r.get('shots'),
             r['seed']) for r in load_results(path)}
    todo = [c for c in shard_slice([cfg + (sd,) for cfg in grid for sd in seeds], *shard)
            if (dataset,) + c not in done]
    print(f"{stage} {dataset}: {len(todo)} configurations to run (shard {shard[0]}/{shard[1]}, "
          f"{device}) -> {path}", flush=True)
    Xtr, Ytr, Xte, Yte, info = data
    for k, (L, topo, p, ch, shots, sd) in enumerate(todo, 1):
        t0 = time.time()
        r = train_one(L, ch, p, sd, Xtr, Ytr, Xte, Yte, n_steps, drop_w1=True,
                      entangling=topo, offsets=OFFSETS, sensitivity=True,
                      readout=readout, device=device, noise_placement=noise_placement,
                      shots=shots)
        r['runtime_s'] = time.time() - t0
        append_result(path, stamp(r, info, dataset, stage, state, scratch))
        print(f"[{k}/{len(todo)}] {stage} L={L} {topo:4s} p={p} {ch:7s} shots={shots} seed={sd} "
              f"train={r['train_acc']:.3f} test={r['test_acc']:.4f} "
              f"first250={r['test_acc_250']:.3f} sens={r['sensitivity']:.3f} "
              f"({r['runtime_s']:.0f}s)", flush=True)


def run_diagnostics(outdir, state, scratch, Ls=DIAG_L, theta_seeds=DIAG_THETA_SEEDS):
    """sigma and J at untrained theta, as in noise_discrimination_check.py.

    v5: the preprocessing of the training runs (fitted on 1000 training
    images), evaluated on the first 300 of them. legacy300: fitted on those
    300 images alone, as the v3 draft's Table III was. Both use the same 300
    images, since the selection permutation does not depend on n_train.
    """
    path = os.path.join(outdir, 'results.pkl')
    results = load_results(path)
    done = {(r['variant'], r['L'], r['channel'], r['theta_seed']) for r in results}
    for variant, n_fit in (('v5', N_TRAIN), ('legacy300', DIAG_N)):
        info = {}
        Xtr, Ytr, _, _ = load_mnist_binary(n_train=n_fit, n_test=1, pca_seed=PCA_SEED,
                                           info=info)
        X = torch.as_tensor(Xtr[:DIAG_N], device='cuda')
        Y = Ytr[:DIAG_N]
        for L in Ls:
            for ch in DIAG_CHANNELS:
                sim = TorchCirc(N_QUBITS, L, noisetype=ch, p_noise=P_NOISE,
                                device='cuda', entangling=TOPO)
                for s in theta_seeds:
                    if (variant, L, ch, s) in done:
                        continue
                    g = torch.Generator(device='cuda').manual_seed(s)
                    theta = 2 * np.pi * torch.rand(sim.n_theta, generator=g, device='cuda')
                    with torch.no_grad():
                        z = sim.features(X, theta).cpu().numpy()
                    sd = z.std(axis=0)
                    r = {'variant': variant, 'n_fit': n_fit, 'n_images': len(Y), 'L': L,
                         'channel': ch, 'p_noise': P_NOISE, 'entangling': TOPO,
                         'theta_seed': s, 'J': fisher_ratio(z, Y),
                         'sigma': float(sd.mean()), 'sigma_per_qubit': sd}
                    results.append(stamp(r, info, 'MNIST', 'P1.3', state, scratch,
                                         n_train=n_fit, has_test=False))
                save_results(path, results)
                print(f"P1.3 {variant:9s} L={L} {ch:7s} "
                      f"J={np.mean([r['J'] for r in results if (r['variant'], r['L'], r['channel']) == (variant, L, ch)]):.3f}",
                      flush=True)


def run_width_diagnostics(outdir, state, scratch, device='cuda'):
    """S6a: sigma and J at untrained theta against width, as in P1.3 (v5
    preprocessing, now with n PCA components; the first 300 training images)."""
    path = os.path.join(outdir, 'results.pkl')
    results = load_results(path)
    done = {(r['n'], r['L'], r['channel'], r['theta_seed']) for r in results}
    for n in WIDTH_N:
        info = {}
        Xtr, Ytr, _, _ = load_mnist_binary(n_train=N_TRAIN, n_test=1, n_components=n,
                                           pca_seed=PCA_SEED, info=info)
        X = torch.as_tensor(Xtr[:DIAG_N], device=device)
        Y = Ytr[:DIAG_N]
        for L in WIDTH_L:
            for ch in CHANNELS:
                sim = TorchCirc(n, L, noisetype=ch, p_noise=P_NOISE, device=device,
                                entangling=TOPO)
                for s in DIAG_THETA_SEEDS:
                    if (n, L, ch, s) in done:
                        continue
                    g = torch.Generator(device='cuda').manual_seed(s)
                    theta = (2 * np.pi * torch.rand(sim.n_theta, generator=g, device='cuda')).to(device)
                    with torch.no_grad():
                        z = sim.features(X, theta).cpu().numpy()
                    sd = z.std(axis=0)
                    r = {'n': n, 'n_fit': N_TRAIN, 'n_images': len(Y), 'L': L, 'channel': ch,
                         'p_noise': P_NOISE, 'entangling': TOPO, 'theta_seed': s,
                         'J': fisher_ratio(z, Y), 'sigma': float(sd.mean()), 'sigma_per_qubit': sd}
                    r = stamp(r, info, 'MNIST', 'S6a', state, scratch, has_test=False)
                    r['n_qubits'] = n
                    results.append(r)
                save_results(path, results)
                cell = [r for r in results if (r['n'], r['L'], r['channel']) == (n, L, ch)]
                print(f"S6a n={n} L={L} {ch:6s} sigma={np.mean([r['sigma'] for r in cell]):.3e} "
                      f"J={np.mean([r['J'] for r in cell]):.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=sorted(STAGES))
    ap.add_argument('--L', type=int, nargs='+', help='depths (depth stage only)')
    ap.add_argument('--channels', nargs='+', help='override the stage default')
    ap.add_argument('--seeds', type=int, nargs='+', help='override the default 8 seeds')
    ap.add_argument('--p', type=float, nargs='+', help='override the noise strengths')
    ap.add_argument('--topo', nargs='+', help='override the entanglers')
    ap.add_argument('--n-steps', type=int, help='override the stage\'s step count')
    ap.add_argument('--outdir', help='default: the stage\'s m5_* directory')
    ap.add_argument('--device', default='cuda', choices=['cuda', 'cpu'])
    ap.add_argument('--threads', type=int, help='torch CPU threads for this process')
    ap.add_argument('--shard', default='0/1', help='i/k: run every k-th configuration from i')
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    shard = tuple(int(x) for x in args.shard.split('/'))
    if not (len(shard) == 2 and 0 <= shard[0] < shard[1]):
        raise SystemExit(f"bad --shard {args.shard}")

    cfg = STAGES[args.stage]
    outdir = os.path.abspath(args.outdir or os.path.join(HERE, cfg['outdir']))
    base = os.path.basename(outdir)
    if base.startswith('m4_'):
        raise SystemExit(f"refusing to write to {outdir}: m4_* directories are archived")
    # Only an m5_* directory next to this file holds results; anything else is scratch.
    scratch = not (os.path.dirname(outdir) == HERE and base.startswith('m5_'))
    state, dirty = code_state()
    if dirty and not scratch:
        raise SystemExit(f"uncommitted .py changes; commit before writing to {outdir}:\n{dirty}")
    if not scratch and (args.channels or args.seeds or args.p or args.topo
                        or args.n_steps is not None):
        raise SystemExit("--channels, --seeds, --p, --topo and --n-steps are for scratch runs only")
    os.makedirs(outdir, exist_ok=True)
    print(f"stage={args.stage} outdir={outdir} scratch={scratch} {state}", flush=True)

    seeds = tuple(args.seeds or SEEDS)
    if args.stage == 'diagnostics':
        run_diagnostics(outdir, state, scratch)
        return
    if args.stage == 'width_diag':
        run_width_diagnostics(outdir, state, scratch, device=args.device)
        return
    data = load_data(cfg['dataset'])
    if args.stage in ('baselines', 'fashion'):
        run_baselines(outdir, cfg['dataset'], state, scratch, data)
    if args.stage == 'baselines':
        return
    if args.stage in ('depth', 'readout'):
        if not args.L:
            raise SystemExit(f"{args.stage} needs --L")
        L_shots = [(L, None) for L in args.L]
    elif 'L_shots' in cfg:
        L_shots = [(L, sh) for L, sh in cfg['L_shots'] if not args.L or L in args.L]
    else:
        L_shots = [(L, None) for L in cfg['L']]
    channels = tuple(args.channels or cfg['channels'])
    ps = tuple(args.p or cfg.get('p', (P_NOISE,)))
    topos = tuple(args.topo or cfg.get('topo', (TOPO,)))
    grid = [(L, topo, p, ch, sh) for L, sh in L_shots for topo in topos for p in ps
            for ch in channels]
    run_circuits(outdir, cfg['id'], cfg['dataset'], grid, seeds,
                 args.n_steps or cfg.get('n_steps', N_STEPS), state, scratch, data,
                 readout=cfg.get('readout', 'raw'), device=args.device, shard=shard,
                 noise_placement=cfg.get('noise_placement', 'after_entangler'))


if __name__ == "__main__":
    main()
