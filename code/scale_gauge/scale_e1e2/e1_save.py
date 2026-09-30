"""e1 re-run with per-image outputs, for the S1 regression.
Identical training to scale_test.py mode e1: joint training 100 Adam steps (lr 0.05), features
standardized by the full-batch TRAINING mean and population std (correction=0) at every step,
gradients flowing through mean and std, no affine map; evaluation uses the training mean/std of
the final theta.  L=4, ring, p=0.3, same CUDA-generator init draws as train_one (W2 then theta).
usage: e1_save.py {indep|torchcirc} CHANNEL"""
import os, sys, time, pickle, numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from indep_sim import IndepCirc
d = np.load(os.path.join(HERE, 'v5_angles.npz'))
Atr, Ytr, Ate, Yte = d['Atr'], d['Ytr'], d['Ate'], d['Yte']
SEEDS, L, P = (256, 512, 768, 1024, 1280, 1536, 1792, 2048), 4, 0.3
draws = pickle.load(open(os.path.join(HERE, 'init_draws_L4_8seeds.pkl'), 'rb'))
simname, ch = sys.argv[1], sys.argv[2]
if simname == 'torchcirc':
    sys.path.insert(0, os.path.join(HERE, '..', '..', 'noise_structure_mnist'))
    from torch_circ import TorchCirc
    sim = TorchCirc(4, L, noisetype=ch, p_noise=0.0 if ch == 'None' else P, device='cpu', entangling='ring')
else:
    sim = IndepCirc(4, L, ch, 0.0 if ch == 'None' else P)
Xtr, Xte, Yt = torch.as_tensor(Atr), torch.as_tensor(Ate), torch.as_tensor(Ytr, dtype=torch.long)
out, path = [], os.path.join(HERE, f'e1_save_{simname}_{ch}.pkl')
for sd in SEEDS:
    t0 = time.time()
    W2 = draws[sd][0].clone().requires_grad_(True); b2 = torch.zeros(3, requires_grad=True)
    th = draws[sd][1].clone().requires_grad_(True)
    opt = torch.optim.Adam([{'params': [W2, b2], 'weight_decay': 0.0}, {'params': [th], 'weight_decay': 0.0}], lr=0.05)
    def head(z, stats=None):
        mu, sdv = (z.mean(0), z.std(0, correction=0)) if stats is None else stats
        return ((z - mu) / sdv) @ W2.T + b2
    for step in range(100):
        opt.zero_grad()
        loss = torch.nn.functional.cross_entropy(head(sim.features(Xtr, th)), Yt)
        loss.backward(); opt.step()
    with torch.no_grad():
        ztr, zte = sim.features(Xtr, th), sim.features(Xte, th)
        st = (ztr.mean(0), ztr.std(0, correction=0))
        ptr, pte = head(ztr, st).argmax(1).numpy(), head(zte, st).argmax(1).numpy()
    r = dict(sim=simname, channel=ch, seed=sd, L=L, loss=float(loss.item()),
             train_acc=float((ptr == Ytr).mean()), test_acc=float((pte == Yte).mean()),
             test_acc_250=float((pte[:250] == Yte[:250]).mean()), test_pred=pte.astype(np.int8),
             params={'theta': th.detach().numpy().copy(), 'W2': W2.detach().numpy().copy(), 'b2': b2.detach().numpy().copy()},
             feat_mean=st[0].numpy(), feat_std=st[1].numpy(), zstd_sample=float(ztr.std(0).mean()))
    out.append(r); pickle.dump(out, open(path, 'wb'))
    print(f"{simname} {ch:5s} {sd:4d} train {r['train_acc']:.4f} test {r['test_acc']:.4f} t250 {r['test_acc_250']:.3f} loss {r['loss']:.6f} ({time.time()-t0:.0f}s)", flush=True)
print("E1SAVE_DONE", simname, ch, flush=True)
