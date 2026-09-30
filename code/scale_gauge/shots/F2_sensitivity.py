"""F2: feature-level input sensitivity without the readout gain.
For every committed standardized-readout run of S1 (m5_readout, MNIST, L=1-4), S2 (m5_fashion_readout,
Fashion, L=4) and S4 (m5_inside, MNIST, L=4, noise inside the entangler): the mean over test images of the
Frobenius norm of the Jacobian d z~ / d a, with z~ = (z - mu)/sigma and mu, sigma the stored training
statistics (z_mean, z_std). Exact expectation values; independent simulator shots_lib.Circ; test images in
chunks. Output: F2_results.pkl (per run: mean norm and the per-image norms)."""
import os, sys, time, shutil, pickle, tempfile, numpy as np, torch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from shots_lib import Circ, load_v5
torch.set_num_threads(4)
CHUNK = 500
NSM = os.path.join(HERE, '..', '..', 'noise_structure_mnist')
STAGES = (('S1', 'MNIST', 'm5_readout'), ('S2', 'FashionMNIST', 'm5_fashion_readout'), ('S4', 'MNIST', 'm5_inside'))
CH_ID = {'None': 0, 'AD': 1, 'Pauli': 2, 'Depol': 3, 'AD_flip': 4}

def jac_norms(circ, A, th, mu, sd):
    """Per-image Frobenius norm of d z~/d a (n x n), by n reverse passes per chunk."""
    out = []
    for i in range(0, len(A), CHUNK):
        a = A[i:i + CHUNK].clone().requires_grad_(True)
        zt = (circ.features(a, th) - mu) / sd
        J = torch.stack([torch.autograd.grad(zt[:, j].sum(), a, retain_graph=j < zt.shape[1] - 1)[0]
                         for j in range(zt.shape[1])], 1)          # (B, n_out, n_in)
        out.append(J.flatten(1).norm(dim=1).detach())
    return torch.cat(out).numpy()

if __name__ == '__main__':
    t0 = time.time(); res = []
    data = {ds: load_v5(ds) for ds in ('MNIST', 'FashionMNIST')}
    for stage, ds, d in STAGES:
        dst = os.path.join(tempfile.gettempdir(), f'F2_{d}.pkl'); shutil.copy2(os.path.join(NSM, d, 'results.pkl'), dst)
        recs = sorted(pickle.load(open(dst, 'rb')), key=lambda r: (r['L'], CH_ID[r['channel']], r['seed']))
        A = torch.as_tensor(data[ds][2])
        for r in recs:
            prm = r['params']
            circ = Circ(4, r['L'], r['channel'], r['p_noise'], r.get('noise_placement', 'after_entangler'))
            th = torch.as_tensor(np.asarray(prm['theta']))
            g = jac_norms(circ, A, th, torch.as_tensor(np.asarray(prm['z_mean'])), torch.as_tensor(np.asarray(prm['z_std'])))
            res.append(dict(stage=stage, dataset=ds, L=r['L'], channel=r['channel'], seed=r['seed'],
                            placement=r.get('noise_placement'), sens_feat=float(g.mean()), per_image=g.astype(np.float32),
                            sens_logp_record=float(r['sensitivity']), test_acc=float(r['test_acc']), min_z_std=float(np.min(prm['z_std']))))
            print(f"  {stage} L={r['L']} {r['channel']:7s} {r['seed']:4d}  |d z~/d a| {g.mean():.4f}  (record |grad log p| {r['sensitivity']:.3f})  ({time.time()-t0:.0f}s)", flush=True)
            pickle.dump(res, open(os.path.join(HERE, 'F2_results.pkl'), 'wb'))
    print("F2_DONE", flush=True)
