"""e1: joint training with a scale-free readout (features standardized by full-batch
mean/std at every step, no affine), 100 steps.  e2: original raw readout, 1000 steps,
checkpoints at 100/300/1000.  L=4, same init draws as the m5 runs.
usage: scale_test.py e1|e2 CHANNEL [CHANNEL...]"""
import os, sys, time, pickle, numpy as np, torch
torch.set_num_threads(2); torch.set_default_dtype(torch.float64)
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from indep_sim import IndepCirc
d = np.load(os.path.join(HERE, 'v5_angles.npz'))
Atr, Ytr, Ate, Yte = d['Atr'], d['Ytr'], d['Ate'], d['Yte']
SEEDS, L, P = (256, 512, 768, 1024, 1280, 1536, 1792, 2048), 4, 0.3
INIT = os.path.join(HERE, 'init_draws_L4_8seeds.pkl')
if not os.path.exists(INIT):
    dr = {}
    for sd in SEEDS:
        g = torch.Generator(device='cuda').manual_seed(sd)
        dr[sd] = ((0.5 * torch.randn(3, 4, generator=g, device='cuda')).cpu(),
                  (2 * np.pi * torch.rand(3 * 4 * (L + 1), generator=g, device='cuda')).cpu())
    pickle.dump(dr, open(INIT, 'wb')); print("init draws written"); sys.exit(0)
draws = pickle.load(open(INIT, 'rb'))
mode, chans = sys.argv[1], sys.argv[2:]
Xtr, Xte, Yt = torch.as_tensor(Atr), torch.as_tensor(Ate), torch.as_tensor(Ytr, dtype=torch.long)
out = []
for ch in chans:
    sim = IndepCirc(4, L, ch, 0.0 if ch == 'None' else P)
    for sd in SEEDS:
        t0 = time.time()
        W2 = draws[sd][0].clone().requires_grad_(True); b2 = torch.zeros(3, requires_grad=True)
        th = draws[sd][1].clone().requires_grad_(True)
        opt = torch.optim.Adam([{'params': [W2, b2], 'weight_decay': 0.0}, {'params': [th], 'weight_decay': 0.0}], lr=0.05)
        if mode == 'e1':
            def head(z, stats=None):
                mu, sdv = (z.mean(0), z.std(0, correction=0)) if stats is None else stats
                return ((z - mu) / sdv) @ W2.T + b2
        else:
            head = lambda z, stats=None: z @ W2.T + b2
        n_steps = 100 if mode == 'e1' else 1000
        ck = {}
        for step in range(n_steps):
            opt.zero_grad()
            loss = torch.nn.functional.cross_entropy(head(sim.features(Xtr, th)), Yt)
            loss.backward(); opt.step()
            if step + 1 in (100, 300, 1000):
                with torch.no_grad():
                    ztr, zte = sim.features(Xtr, th), sim.features(Xte, th)
                    st = (ztr.mean(0), ztr.std(0, correction=0))
                    ck[step + 1] = dict(train=float((head(ztr, st).argmax(1).numpy() == Ytr).mean()),
                                        test=float((head(zte, st).argmax(1).numpy() == Yte).mean()),
                                        zstd=float(ztr.std(0).mean()), W2n=float(np.linalg.norm((W2 - W2.mean(0)).detach().numpy())),
                                        loss=float(loss.item()))
        r = dict(mode=mode, ch=ch, seed=sd, ck=ck); out.append(r)
        pickle.dump(out, open(os.path.join(HERE, f'scale_{mode}_{"_".join(chans)}.pkl'), 'wb'))
        print(f"{mode} {ch:5s} {sd:4d} " + " | ".join(f"s{k}: tr {v['train']:.3f} te {v['test']:.4f} zstd {v['zstd']:.4f} W2 {v['W2n']:.1f}" for k, v in ck.items())
              + f" ({time.time()-t0:.0f}s)", flush=True)
print("SCALE_DONE", mode, chans, flush=True)
