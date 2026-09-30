# -*- coding: utf-8 -*-
"""
Training-free cross-check: does the circuit's raw feature separability
(Fisher discriminant ratio on z = <Z_i>, at RANDOM untrained theta) predict
the same channel ordering we measured from actual training runs?

This sidesteps optimizer/training-dynamics confounds entirely — no Adam, no
gradient descent, just: propagate the real PCA-encoded MNIST angles through
the circuit at fixed random theta, and ask how separable the three classes
are in the resulting 4-d feature space, per noise channel.

Also reports each channel's raw feature attenuation (std of z_i relative to
the noiseless channel) as a cross-check against the theoretical PTM
contraction factors (lambda_x=lambda_y=sqrt(1-p), lambda_z=1-p for AD/Pauli;
isotropic lambda_d for Depol).
"""
import numpy as np
import torch

from mnist_hybrid_multi import load_mnist_binary
from torch_circ import TorchCirc

L = 4
P_NOISE = 0.30
N_SAMPLES = 300
THETA_SEEDS = [1, 2, 3, 4, 5]
CHANNELS = ['None', 'AD', 'Pauli', 'Depol']


def fisher_ratio(z, y):
    """Multivariate Fisher discriminant ratio J = tr(Sw^-1 Sb) for z (N,d), y (N,)."""
    classes = np.unique(y)
    d = z.shape[1]
    mu = z.mean(axis=0)
    Sb = np.zeros((d, d))
    Sw = np.zeros((d, d))
    for c in classes:
        zc = z[y == c]
        muc = zc.mean(axis=0)
        diff = (muc - mu).reshape(-1, 1)
        Sb += len(zc) * (diff @ diff.T)
        cdiff = zc - muc
        Sw += cdiff.T @ cdiff
    Sw += 1e-6 * np.eye(d)  # ridge: avoid singular Sw under heavy attenuation
    return float(np.trace(np.linalg.solve(Sw, Sb)))


def main():
    print(f"loading MNIST (n={N_SAMPLES})...")
    Xtr, Ytr, _, _ = load_mnist_binary(n_train=N_SAMPLES, n_test=1)
    Xt = torch.as_tensor(Xtr, device='cuda')

    print(f"\nL={L}  p={P_NOISE}  {len(THETA_SEEDS)} random-theta seeds "
          f"(untrained — no optimization)\n")

    results = {ch: {'fisher': [], 'std': []} for ch in CHANNELS}
    for ch in CHANNELS:
        sim = TorchCirc(Xtr.shape[1], L, noisetype=ch, p_noise=P_NOISE, device='cuda')
        for seed in THETA_SEEDS:
            g = torch.Generator(device='cuda').manual_seed(seed)
            theta = 2 * np.pi * torch.rand(sim.n_theta, generator=g, device='cuda')
            with torch.no_grad():
                z = sim.features(Xt, theta).cpu().numpy()
            results[ch]['fisher'].append(fisher_ratio(z, Ytr))
            results[ch]['std'].append(z.std(axis=0).mean())

    none_std = np.mean(results['None']['std'])
    print(f"{'channel':8s} {'fisher J (mean+-std)':24s} {'feature std':14s} "
          f"{'attenuation vs None':20s}")
    for ch in CHANNELS:
        fj = np.array(results[ch]['fisher'])
        sd = np.array(results[ch]['std'])
        print(f"{ch:8s} {fj.mean():8.3f} +- {fj.std():6.3f}       "
              f"{sd.mean():.4f}        {sd.mean() / none_std:.3f}")


if __name__ == "__main__":
    main()
