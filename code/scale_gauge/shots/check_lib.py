"""Validate shots_lib.Circ against the repo's TorchCirc (CPU) for every channel and placement."""
import os, sys, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, '..', '..', 'noise_structure_mnist'))
from shots_lib import Circ
from torch_circ import TorchCirc
torch.manual_seed(0); worst = 0.0
for place in ('after_entangler', 'inside_entangler'):
    for L in (1, 4):
        for ch in ('None', 'AD', 'AD_flip', 'Pauli', 'Depol'):
            p = 0.0 if ch == 'None' else 0.3
            tc = TorchCirc(4, L, noisetype=ch, p_noise=p, device='cpu', entangling='ring', noise_placement=place)
            ic = Circ(4, L, ch, p, place)
            th = 2 * np.pi * torch.rand(3 * 4 * (L + 1)); a = 2 * np.pi * torch.rand(64, 4)
            d = (tc.features(a, th) - ic.features(a, th)).abs().max().item(); worst = max(worst, d)
            pr = ic.probs(a, th); assert pr.min() > -1e-12 and (pr.sum(1) - 1).abs().max() < 1e-12
            print(f"{place:16s} L={L} {ch:7s} max|TorchCirc - Circ| = {d:.1e}")
print(f"WORST {worst:.2e}")
