# -*- coding: utf-8 -*-
"""
Hybrid MNIST classification: data loading, training and classical baselines.

3-class MNIST (digits 1, 3, 5) -> PCA to n components -> angles in [0, 2pi]
-> data re-uploading circuit (torch_circ.TorchCirc) -> linear readout -> logits.

A library since 2026-09-23: run_phase1.py drives the m5_* runs, and the old
command-line entry point is retired.
"""
import os

# Must be set before CUDA initializes (i.e. before `import torch`) for
# use_deterministic_algorithms to work with CUBLAS ops.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms

torch.use_deterministic_algorithms(True)

from torch_circ import TorchCirc

CHANNELS = ['AD']


# Stamped on every result record produced with the loader below.
PREP = 'v5_train_extrema'


def load_dataset(dataset='MNIST', train=True):
    src = {'MNIST': datasets.MNIST, 'FashionMNIST': datasets.FashionMNIST}[dataset]
    return src(root='./data', train=train, download=True,
               transform=transforms.Compose([transforms.ToTensor()]))


def select_split(ds, classes, n, seed=0):
    """Images of `classes` in a seed-fixed random order, truncated to n (None: all).

    The permutation depends only on the number of matching images and the seed,
    never on n, so the first k images are the same for every n >= k. Returns the
    flattened pixels, the labels (index into `classes`) and the selected indices
    into the class-filtered set.
    """
    X, Y = [], []
    for x, y in ds:
        if y in classes:
            X.append(x.numpy().ravel())
            Y.append(classes.index(y))
    X, Y = np.array(X), np.array(Y)
    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    return X[idx], Y[idx], idx


def load_mnist_binary(classes=(1, 3, 5), n_train=1000, n_test=None, n_components=4,
                      seed=0, dataset='MNIST', pca_seed=0, info=None):
    """Load MNIST, keep `classes`, standardise, PCA to n_components, map to angles.

    Every fit uses the training split only: the StandardScaler, the PCA and the
    per-component minimum and maximum that map PCA scores onto [0, 2pi]. The test
    split goes through the same three transforms, so an image receives the same
    angles whichever split it is in. Test angles outside [0, 2pi] are clipped,
    and their number is reported through `info`.

    Until 2026-09-23 the test split was scaled by its own extrema. That shifted
    the test angles by 0.2-0.9 rad per component on average relative to
    training, and every archived m4_* result used it.
    That path has been removed; commit d5ad69e is the last version with it.

    n_test=None takes every test image of `classes` (3037 for MNIST {1, 3, 5}).
    dataset='FashionMNIST' swaps the source; everything downstream is the same.

    info, if given, receives prep, pca_seed, n_clip, n_test_values, clip_frac,
    and the selected train_idx and test_idx.
    """
    Xtr, Ytr, idx_tr = select_split(load_dataset(dataset, True), classes, n_train, seed)
    Xte, Yte, idx_te = select_split(load_dataset(dataset, False), classes, n_test, seed)
    sc = StandardScaler().fit(Xtr)
    # pca_seed pins the randomized SVD solver. sklearn picks svd_solver=
    # 'randomized' for this data shape and defaults random_state=None, so
    # without this every call returns a slightly different basis (angles move
    # by ~1e-3 rad). Results archived before 2026-09-23 were produced with the
    # unseeded solver; pass pca_seed=None to reproduce that behaviour.
    pca = PCA(n_components=n_components,
              random_state=pca_seed).fit(sc.transform(Xtr))
    Ztr = pca.transform(sc.transform(Xtr))
    Zte = pca.transform(sc.transform(Xte))

    # Training extrema for both splits. Test images beyond the training range
    # land outside [0, 2pi] and are clipped to its edges. Keep this expression
    # as it is: the operation order fixes the rounding, and the pilot runs in
    # m4_rescale_train are only reproduced bit for bit with the same order.
    lo, hi = Ztr.min(0), Ztr.max(0)

    def to_angles(Z):
        return (Z - lo) / np.maximum(hi - lo, 1e-9) * 2 * np.pi

    Atr = to_angles(Ztr)
    Ate_raw = to_angles(Zte)
    Ate = np.clip(Ate_raw, 0.0, 2 * np.pi)
    n_clip = int(((Ate_raw < 0) | (Ate_raw > 2 * np.pi)).sum())
    print(f"  [{PREP}] n_test={len(Yte)}: clipped {n_clip}/{Ate_raw.size} test "
          f"angle values ({100.0 * n_clip / max(Ate_raw.size, 1):.2f}%)")
    if info is not None:
        info.update(prep=PREP, pca_seed=pca_seed, n_clip=n_clip,
                    n_test_values=int(Ate_raw.size),
                    clip_frac=n_clip / max(Ate_raw.size, 1),
                    train_idx=idx_tr, test_idx=idx_te)

    return (Atr.astype(np.float64), Ytr, Ate.astype(np.float64), Yte)


def retire_legacy_driver(path):
    """Exit before a pre-2026-09-23 driver can run with the corrected loader.

    The m4_* drivers write to m4_* directories and skip configurations already
    present there, so running one now would mix preprocessing versions in an
    archived file or silently do nothing. Port a driver to m5_* before reuse.
    """
    raise SystemExit(f"{os.path.basename(path)} is a legacy m4 driver, retired on "
                     f"2026-09-23. Port it to an m5_* output "
                     f"before running it.")


def classical_baseline(Xtr, Ytr, Xte, Yte, n_first=250):
    """Circuit-free classifiers on the same angles the circuit receives.

    Logistic regression, an RBF-SVC and 15-nearest-neighbours, with sklearn
    defaults otherwise. Returns one record per model, including the accuracy on
    the first n_first test images (the pre-2026-09-23 test split) and on the rest.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import SVC
    models = {'logreg': LogisticRegression(max_iter=1000),
              'svc_rbf': SVC(kernel='rbf'),
              'knn15': KNeighborsClassifier(n_neighbors=15)}
    out = []
    for name, clf in models.items():
        clf.fit(Xtr, Ytr)
        pred = clf.predict(Xte)
        hit = pred == Yte
        r = {'model': name, 'train_acc': float(clf.score(Xtr, Ytr)),
             'test_acc': float(hit.mean()),
             'test_acc_250': float(hit[:n_first].mean()),
             'test_acc_rest': float(hit[n_first:].mean()) if len(hit) > n_first else None,
             'test_pred': pred.astype(np.int8)}
        print(f"  classical {name:8s} train_acc={r['train_acc']:.3f}  "
              f"test_acc={r['test_acc']:.3f}  first {n_first}={r['test_acc_250']:.3f}")
        out.append(r)
    return out


def train_one(L, channel, p_noise, seed, Xtr, Ytr, Xte, Yte, n_steps,
              freeze_theta=False, drop_w1=False, entangling='ring', log_every=10,
              inject_bias=False, theta_weight_decay=0.0,
              smoothness_penalty=0.0, fourier_penalty=0.0, fourier_rho=0.7,
              fourier_subsample=16, marginal_weights=None, cross_weights=None,
              fourier_cross=True, offsets=(), sensitivity=False, n_first=250,
              readout='raw', device='cuda', noise_placement='after_entangler',
              shots=None, shot_draws=5):
    """Train one configuration; return its record.

    readout='raw' feeds the features z to the linear readout W2 z + b2.
    readout='standardized' first standardizes each feature by the full-batch
    training mean and std (correction=0), recomputed at every step with
    gradients flowing through them, and with no learned scale or shift beyond
    W2 and b2. Evaluation uses the training statistics
    of the current parameters, which are stored with the parameters.

    After training, and without affecting it: offsets adds each delta (rad) to
    every test angle, unclipped since the circuit is 2pi-periodic in each angle,
    and records the accuracy; sensitivity=True records the test-set mean of
    ||grad_a log p(true class)||_2 over the input angles a.

    noise_placement is passed to TorchCirc ('inside_entangler' puts the channel
    after every CNOT of the entangler instead of once after it).

    device='cpu' runs the simulation on the CPU. The initial parameters are
    still drawn from the CUDA generator seeded with `seed` and then moved, so
    a CPU run starts exactly where its GPU twin does.

    shots=N_s trains and evaluates with a finite-shot model
    (default None: exact expectation values). Training uses
    z + sqrt((1 - z^2)/N_s) eps (shot_noise), with eps redrawn every step for
    every image and qubit from a CUDA generator seeded per run, and the
    standardized readout's statistics taken from the noisy batch. After
    training, shot_draws independent draws of N_s shots per image from the
    Z-basis outcome distribution (sample_features) give the test features and,
    for the standardized readout, the training statistics of that draw. The
    exact-expectation fields (test_acc, test_pred, offsets, sensitivity) are
    recorded as for an exact run.
    """
    assert readout in ('raw', 'standardized'), readout
    if smoothness_penalty > 0 or fourier_penalty > 0 or inject_bias:
        assert device == 'cuda', "the penalty and reconstruction paths are GPU-only"
    if smoothness_penalty > 0 or fourier_penalty > 0:
        assert drop_w1, "these penalties need pre==X (drop_w1) to be a free leaf"
        assert readout == 'raw', "the penalty paths build their own raw logits"
    if shots is not None:
        assert drop_w1 and not inject_bias and smoothness_penalty == 0 and fourier_penalty == 0, \
            "the shot model is implemented for the plain circuit only"
    n, D = Xtr.shape[1], Xtr.shape[1]
    if inject_bias:
        channel, p_noise = 'None', 0.0  # reconstruction runs on the noiseless circuit
    sim = TorchCirc(n, L, noisetype=channel, p_noise=p_noise, device=device,
                     entangling=entangling, noise_placement=noise_placement)
    g = torch.Generator(device='cuda').manual_seed(seed)
    C = len(np.unique(Ytr))  # number of classes
    if not drop_w1:
        W1 = (0.5 * torch.randn(n, D, generator=g, device='cuda')).to(device).requires_grad_(True)
        b1 = torch.zeros(n, device=device, requires_grad=True)
    W2 = (0.5 * torch.randn(C, n, generator=g, device='cuda')).to(device).requires_grad_(True)
    b2 = torch.zeros(C, device=device, requires_grad=True)
    theta = (2 * np.pi * torch.rand(sim.n_theta, generator=g,
                                     device='cuda')).to(device).requires_grad_(not freeze_theta)
    other_params = [W2, b2] if drop_w1 else [W1, b1, W2, b2]
    if inject_bias:
        # b_j(a_j) = beta0 + beta1*cos(a_j) + beta2*sin(a_j): the reconstructed
        # low-order-Fourier bias hypothesised to explain AD's advantage,
        # injected onto the noiseless circuit's own output instead of relying
        # on a physical noise channel.
        beta0 = torch.zeros(n, device='cuda', requires_grad=True)
        beta1 = torch.zeros(n, device='cuda', requires_grad=True)
        beta2 = torch.zeros(n, device='cuda', requires_grad=True)
        other_params += [beta0, beta1, beta2]
    # theta gets its own param group so weight decay (the regularization-toward
    # low-Fourier-degree hypothesis) applies only to the circuit, not to W1/W2.
    param_groups = [{'params': other_params, 'weight_decay': 0.0}]
    if not freeze_theta:
        param_groups.append({'params': [theta], 'weight_decay': theta_weight_decay})
    opt = torch.optim.Adam(param_groups, lr=0.05)
    Xt = torch.as_tensor(Xtr, device=device)
    Yt = torch.as_tensor(Ytr, dtype=torch.long, device=device)
    Xv = torch.as_tensor(Xte, device=device)
    if shots is not None:
        g_shot = torch.Generator(device='cuda').manual_seed(seed + SHOT_SEED)

    def features(X):
        pre = X if drop_w1 else X @ W1.T + b1
        z = sim.features(pre, theta)
        if inject_bias:
            z = z + beta0 + beta1 * torch.cos(pre) + beta2 * torch.sin(pre)
        return z

    def logit(X, stats=None, noisy=False):
        """stats=None standardizes by X's own statistics (the training batch).
        noisy=True adds the training shot noise before the readout."""
        z = features(X)
        if noisy:
            eps = torch.randn(z.shape, generator=g_shot, device='cuda').to(device)
            z = shot_noise(z, shots, eps)
        if readout == 'standardized':
            mu, sd = (z.mean(0), z.std(0, correction=0)) if stats is None else stats
            z = (z - mu) / sd
        return z @ W2.T + b2  # (B, C)

    def train_stats():
        """Training-set feature statistics for evaluation; None for the raw readout."""
        if readout == 'raw':
            return None
        with torch.no_grad():
            z = features(Xt)
            return z.mean(0), z.std(0, correction=0)

    def snapshot():
        out = {'theta': theta.detach().cpu().numpy(), 'W2': W2.detach().cpu().numpy(),
               'b2': b2.detach().cpu().numpy()}
        if not drop_w1:
            out.update(W1=W1.detach().cpu().numpy(), b1=b1.detach().cpu().numpy())
        if readout == 'standardized':
            mu, sd = train_stats()
            out.update(z_mean=mu.cpu().numpy(), z_std=sd.cpu().numpy())
        return out

    K = 2 * L + 1  # exact DFT grid size: z_i(a_i) is a degree-<=L trig polynomial
    grid = torch.linspace(0, 2 * np.pi, K + 1, device=device)[:-1]

    acc_hist = []
    for step in range(n_steps):
        opt.zero_grad()
        penalty_val = 0.0
        if smoothness_penalty > 0:
            # Exact Jacobian dz_i/da_j via n backward passes (n is small),
            # penalizing sum_{i,j} E_batch[(dz_i/da_j)^2] -- by Parseval,
            # exactly a degree^2-weighted penalty on z's Fourier coefficients
            # in the encoding angles, not a penalty on theta's raw magnitude.
            pre = Xt.detach().requires_grad_(True)
            z = sim.features(pre, theta)
            penalty = 0.0
            for i in range(n):
                grad_i = torch.autograd.grad(z[:, i].sum(), pre,
                                              create_graph=True, retain_graph=True)[0]
                penalty = penalty + (grad_i ** 2).sum()
            penalty = penalty / Xt.shape[0]
            loss = torch.nn.functional.cross_entropy(z @ W2.T + b2, Yt)
            total_loss = loss + smoothness_penalty * penalty
            penalty_val = penalty.item()
        elif fourier_penalty > 0:
            # Exact per-qubit Fourier coefficients via a (2L+1)-point DFT
            # (z_i(a_i) is provably bandlimited to degree L, so this grid
            # recovers gamma_m/delta_m exactly, no truncation error), then
            # penalize degree m>=1 by rho^(-2m) -- a geometric decay matching
            # AD's own per-insertion attenuation rate, not the quadratic
            # weighting an output-derivative penalty gives. Cross-qubit terms
            # from entangling (see below) are handled separately. Estimated
            # on a random subsample (fourier_subsample rows) since a
            # (2L+1)x-per-qubit sweep on the full batch is expensive.
            idx = torch.randperm(Xt.shape[0], device='cuda', generator=g)[:fourier_subsample]
            base = Xt[idx]
            S = base.shape[0]
            penalty = 0.0
            for qi in range(n):
                rep = base.unsqueeze(1).repeat(1, K, 1).reshape(S * K, n)
                rep[:, qi] = grid.repeat(S)
                z_grid = sim.features(rep, theta)[:, qi].reshape(S, K)
                coeffs = torch.fft.rfft(z_grid, dim=1) / K  # (S, L+1)
                mags2 = coeffs.real ** 2 + coeffs.imag ** 2
                if marginal_weights is not None:
                    weights = marginal_weights
                else:
                    m_idx = torch.arange(mags2.shape[1], device='cuda', dtype=torch.float64)
                    weights = fourier_rho ** (-2 * m_idx)
                    weights[0] = 0.0  # don't penalize the DC term (AD's own bias is ~pure DC)
                penalty = penalty + (mags2 * weights).sum()
            # Cross-qubit terms: under entangling!='none', a feature's bias can
            # depend on a neighbor's angle too (lightcone spreading, sec 11.3/
            # 11.6) -- the marginal loop above cannot see this. For each
            # entangled pair, sweep both angles jointly over a (2L+1)x(2L+1)
            # grid and penalize the genuine cross terms (m>=1 AND k>=1) via a
            # 2D DFT, geometric in total degree m+k.
            # fourier_cross=False isolates the marginal-only penalty, which is
            # the ablation Sec. VII uses to show the cross term is required.
            # The pair sweep is the dominant cost, so skipping it is also much
            # faster.
            for (qi, qj) in (sim._entangling_pairs() if fourier_cross else []):
                rep2 = base.unsqueeze(1).repeat(1, K * K, 1).reshape(S * K * K, n)
                rep2[:, qi] = grid.repeat_interleave(K).repeat(S)
                rep2[:, qj] = grid.repeat(K).repeat(S)
                z_grid2 = sim.features(rep2, theta)
                for feat in (qi, qj):
                    zf = z_grid2[:, feat].reshape(S, K, K)
                    c2 = torch.fft.rfft2(zf, dim=(1, 2)) / (K * K)  # (S, K, L+1)
                    mags2d = c2.real ** 2 + c2.imag ** 2
                    m_idx2 = torch.arange(K, device='cuda', dtype=torch.float64)
                    m_deg = torch.where(m_idx2 <= L, m_idx2, m_idx2 - K).abs()
                    k_idx2 = torch.arange(mags2d.shape[2], device='cuda', dtype=torch.float64)
                    if cross_weights is not None:
                        w2 = cross_weights[m_deg.long()]  # (K, L+1) gathered by degree
                    else:
                        w2 = fourier_rho ** (-2 * (m_deg.unsqueeze(1) + k_idx2.unsqueeze(0)))
                    cross_mask = (m_deg.unsqueeze(1) > 0) & (k_idx2.unsqueeze(0) > 0)
                    penalty = penalty + (mags2d * w2 * cross_mask).sum()
            penalty = penalty / S
            loss = torch.nn.functional.cross_entropy(logit(Xt), Yt)
            total_loss = loss + fourier_penalty * penalty
            penalty_val = penalty.item()
        else:
            loss = torch.nn.functional.cross_entropy(logit(Xt, noisy=shots is not None), Yt)
            total_loss = loss
        total_loss.backward()
        if (step + 1) % log_every == 0:
            theta_grad_norm = theta.grad.norm().item() if theta.grad is not None else 0.0
            classical_grads = [W2.grad.flatten()] if drop_w1 else \
                [W1.grad.flatten(), W2.grad.flatten()]
            classical_grad_norm = torch.cat(classical_grads).norm().item()
        opt.step()
        if (step + 1) % log_every == 0:
            with torch.no_grad():
                st = train_stats()
                train_acc = (logit(Xt).argmax(1).cpu().numpy() == Ytr).mean()
                test_acc = (logit(Xv, st).argmax(1).cpu().numpy() == Yte).mean()
            acc_hist.append((step + 1, float(train_acc), float(test_acc)))
            print(f"    step {step + 1:4d}/{n_steps}  train_loss={loss.item():.4f}  "
                  f"penalty={penalty_val:.4f}  "
                  f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
                  f"|grad theta|={theta_grad_norm:.2e}  "
                  f"|grad W1,W2|={classical_grad_norm:.2e}", flush=True)
    st = train_stats()
    with torch.no_grad():
        test_pred = logit(Xv, st).argmax(1).cpu().numpy()
        final_acc = (test_pred == Yte).mean()
        # Same expression as the train column of acc_hist, so the two agree.
        train_acc = (logit(Xt).argmax(1).cpu().numpy() == Ytr).mean()
        offset_acc = {float(d): float((logit(Xv + d, st).argmax(1).cpu().numpy() == Yte).mean())
                      for d in offsets}
    params = snapshot()
    r = {'L': L, 'channel': channel, 'p_noise': p_noise, 'seed': seed,
         'freeze_theta': freeze_theta, 'drop_w1': drop_w1,
         'entangling': entangling, 'theta_weight_decay': theta_weight_decay,
         'smoothness_penalty': smoothness_penalty,
         'fourier_penalty': fourier_penalty, 'fourier_rho': fourier_rho,
         'test_acc': float(final_acc), 'loss': float(loss.item()),
         'acc_hist': acc_hist, 'train_acc': float(train_acc),
         'n_test': len(Yte),
         'test_acc_250': float((test_pred[:n_first] == Yte[:n_first]).mean()),
         'test_pred': test_pred.astype(np.int8), 'params': params,
         'readout': readout, 'n_steps': n_steps, 'device': device,
         'noise_placement': noise_placement, 'shots': shots}
    if offsets:
        r['offset_acc'] = offset_acc
    if sensitivity:
        r['sensitivity'] = input_sensitivity(lambda X: logit(X, st), Xv, Yte)
    if shots is not None:
        r.update(shot_evaluation(sim, theta, W2, b2, Xt, Xv, Yte, shots, shot_draws,
                                 readout, seed))
    return r


# Offset that separates the shot-noise random streams from the initial draws.
SHOT_SEED = 7919


def shot_noise(z, shots, eps):
    """Training shot model: the mean of N_s outcomes of
    Z_j, approximated as Gaussian with its exact variance (1 - z_j^2)/N_s and
    qubit correlations neglected. Reparametrized, so gradients flow through z.
    The clamp keeps the gradient finite at |z| = 1."""
    return z + torch.sqrt(torch.clamp(1 - z ** 2, min=1e-12) / shots) * eps


def outcome_signs(n):
    """(2^n, n) matrix of Z_j = +-1 for each computational-basis outcome, qubit 0
    the most significant bit (the ordering of TorchCirc.probs)."""
    k = np.arange(2 ** n)
    return np.stack([1 - 2 * ((k >> (n - 1 - j)) & 1) for j in range(n)], axis=1).astype(float)


def sample_features(P, shots, rng):
    """Inference shot model: N_s shots per row of P (B, 2^n), exact multinomial,
    returning the sample means of the Z_j, shape (B, n)."""
    P = np.clip(P, 0.0, None)
    P = P / P.sum(1, keepdims=True)
    counts = rng.multinomial(int(shots), P)
    n = int(round(np.log2(P.shape[1])))
    return counts @ outcome_signs(n) / shots


def shot_evaluation(sim, theta, W2, b2, Xt, Xv, Yte, shots, draws, readout, seed):
    """Test accuracy from sampled features, one entry per independent draw. For the
    standardized readout each draw also samples the training images, and their
    statistics standardize that draw's test features."""
    with torch.no_grad():
        P_te = sim.probs(Xv, theta).cpu().numpy()
        P_tr = sim.probs(Xt, theta).cpu().numpy() if readout == 'standardized' else None
    W, b = W2.detach().cpu().numpy(), b2.detach().cpu().numpy()
    rng = np.random.default_rng([seed, SHOT_SEED, int(shots)])
    accs, preds = [], []
    for _ in range(draws):
        z = sample_features(P_te, shots, rng)
        if readout == 'standardized':
            z_tr = sample_features(P_tr, shots, rng)
            z = (z - z_tr.mean(0)) / z_tr.std(0)
        pred = (z @ W.T + b).argmax(1)
        accs.append(float((pred == Yte).mean()))
        preds.append(pred.astype(np.int8))
    return {'shot_draws': draws, 'test_acc_shot': accs,
            'test_acc_shot_mean': float(np.mean(accs)), 'shot_test_pred': np.stack(preds)}


def input_sensitivity(logit, X, Y, chunk=1000):
    """Mean over rows of ||grad_a log softmax(logit(a))[y]||_2, in chunks.

    Rows are independent, so the gradient of the chunk sum with respect to each
    row is that row's own gradient. chunk=1000 matches the training batch, whose
    backward pass is known to fit in GPU memory.
    """
    Yt = torch.as_tensor(Y, dtype=torch.long, device=X.device)
    norms = []
    for i in range(0, len(X), chunk):
        a = X[i:i + chunk].detach().clone().requires_grad_(True)
        logp = torch.log_softmax(logit(a), dim=1)
        total = logp[torch.arange(len(a), device=X.device), Yt[i:i + chunk]].sum()
        g, = torch.autograd.grad(total, a)
        norms.append(g.norm(dim=1).detach())
    return float(torch.cat(norms).mean().item())


if __name__ == "__main__":
    # The M4 sweep entry point that lived here is retired; the m5_* runs are
    # driven by run_phase1.py. This module is now a library.
    retire_legacy_driver(__file__)
