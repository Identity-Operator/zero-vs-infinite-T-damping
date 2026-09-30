"""Noiseless sanity check for vqe_gauge_test.py: the same ansatz and Adam settings (lr 0.05,
500 steps) with no noise must reach E0. The original run was an
inline pre-flight check whose output was not saved; this re-run saves it."""
import numpy as np
import vqe_gauge_test as v
for scheme, seed in (('uniform', 999), ('uniform', 1), ('small', 1)):
    E, th = v.optimise(v.init(scheme, seed), 'none', 0.0, 'post')
    rel = (E - v.E0) / abs(v.E0)
    np.savez(f'vqe_noiseless_{scheme}_{seed}.npz', E=E, rel_err=rel, theta=th, E0=v.E0)
    print(f"noiseless, init {scheme} seed {seed}: {len(E)} trials, rel err min {rel.min():.2e} "
          f"median {np.median(rel):.2e} max {rel.max():.2e}", flush=True)
