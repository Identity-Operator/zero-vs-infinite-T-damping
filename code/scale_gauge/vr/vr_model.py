"""Single-qubit data re-uploading model of van Rossum et al. 2510.24050 (App. B1),
in the Bloch (PTM) representation, batched over independent targets.
Time order per block l=0..L: RZ(th[3l]) -> RY(th[3l+1]) -> RZ(th[3l+2]) -> N ; if l<L: RX(x) -> N.
Output <Z> = r_z.  Channels: affine Bloch maps r -> T r + t (App. A, Eqs. A31-A33 and A30)."""
import numpy as np, torch
torch.set_default_dtype(torch.float64)

def channel(name, g):
    c = np.sqrt(1 - g)
    if name == 'None':  T, t = [1, 1, 1], [0, 0, 0]
    elif name == 'AD':  T, t = [c, c, 1 - g], [0, 0, g]
    elif name == 'PauliAD': T, t = [c, c, 1 - g], [0, 0, 0]
    elif name == 'CliffAD':
        lam = (2 * c + 1 - g) / 3; T, t = [lam, lam, lam], [0, 0, 0]
    else: raise ValueError(name)
    return torch.tensor(T), torch.tensor(t)

def rot(axis, phi):
    """Bloch rotation matrix of exp(-i phi P/2); phi any shape -> (..., 3, 3)."""
    c, s = torch.cos(phi), torch.sin(phi); o, z = torch.ones_like(phi), torch.zeros_like(phi)
    if axis == 'Z': m = [[c, -s, z], [s, c, z], [z, z, o]]
    elif axis == 'Y': m = [[c, z, s], [z, o, z], [-s, z, c]]
    else: m = [[o, z, z], [z, c, -s], [z, s, c]]
    return torch.stack([torch.stack(r, -1) for r in m], -2)

def forward(theta, x, T, t, L=2):
    """theta (K, 3(L+1)), x (B,) -> <Z> (K, B)."""
    K, B = theta.shape[0], x.shape[0]
    r = torch.zeros(K, B, 3); r[..., 2] = 1.0
    RX = rot('X', x)                                    # (B,3,3)
    for l in range(L + 1):
        for k, ax in enumerate('ZYZ'):
            r = torch.einsum('kij,kbj->kbi', rot(ax, theta[:, 3 * l + k]), r)
        r = r * T + t
        if l < L:
            r = torch.einsum('bij,kbj->kbi', RX, r)
            r = r * T + t
    return r[..., 2]

if __name__ == '__main__':
    # validate against explicit Kraus density-matrix evolution
    CD = torch.complex128
    I2 = torch.eye(2, dtype=CD); X = torch.tensor([[0, 1], [1, 0]], dtype=CD); Y = torch.tensor([[0, -1j], [1j, 0]], dtype=CD); Z = torch.tensor([[1, 0], [0, -1]], dtype=CD)
    R = lambda P, a: np.cos(a / 2) * I2 - 1j * np.sin(a / 2) * P
    def kraus(name, g):
        c = np.sqrt(1 - g)
        if name == 'None': return [I2]
        if name == 'AD': return [torch.tensor([[1, 0], [0, c]], dtype=CD), torch.tensor([[0, np.sqrt(g)], [0, 0]], dtype=CD)]
        if name == 'PauliAD': px = py = g / 4; pz = (2 - g - 2 * c) / 4
        if name == 'CliffAD': lam = (2 * c + 1 - g) / 3; px = py = pz = (1 - lam) / 4
        return [np.sqrt(w) * P for w, P in zip([1 - px - py - pz, px, py, pz], [I2, X, Y, Z])]
    rng = np.random.default_rng(0); worst = 0
    for name in ('None', 'AD', 'PauliAD', 'CliffAD'):
        for g in (0.05, 0.2):
            T, t = channel(name, g); Ks = kraus(name, g)
            th = rng.uniform(0, 2 * np.pi, (3, 9)); xs = rng.uniform(-2 * np.pi, 2 * np.pi, 7)
            fb = forward(torch.tensor(th), torch.tensor(xs), T, t)
            for k in range(3):
                for b, x in enumerate(xs):
                    rho = torch.tensor([[1, 0], [0, 0]], dtype=CD)
                    N = lambda r: sum(K @ r @ K.conj().T for K in Ks)
                    for l in range(3):
                        U = R(Z, th[k, 3*l+2]) @ R(Y, th[k, 3*l+1]) @ R(Z, th[k, 3*l])
                        rho = N(U @ rho @ U.conj().T)
                        if l < 2:
                            U = R(X, x); rho = N(U @ rho @ U.conj().T)
                    worst = max(worst, abs(torch.trace(rho @ Z).real.item() - fb[k, b].item()))
    print(f"Bloch model vs Kraus density matrix, all channels: max |diff| = {worst:.2e}")
    # depolarizing theorem: f_Cliff = lam^5 f_None
    th = torch.tensor(rng.uniform(0, 2 * np.pi, (5, 9))); xs = torch.linspace(-2 * np.pi, 2 * np.pi, 50)
    for g in (0.05, 0.1, 0.2):
        T, t = channel('CliffAD', g); lam = T[0].item()
        d = (forward(th, xs, T, t) - lam ** 5 * forward(th, xs, *channel('None', g))).abs().max().item()
        print(f"  g={g}: max |f_Cliff - lam^5 f_None| = {d:.1e}  (lam^5 = {lam**5:.4f})")
