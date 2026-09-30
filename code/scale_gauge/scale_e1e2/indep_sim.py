"""An independent density-matrix simulator, written from the
paper's Eqs. (rotblock)-(forwardmap) and Table I (channels), with full 2^n x 2^n
matrices built by Kronecker products. Qubit 0 is the most significant bit.
Same call signature as TorchCirc.features(a, theta) -> (B, n) <Z_j>."""
import numpy as np, torch
torch.set_default_dtype(torch.float64)
CD = torch.complex128
I2 = torch.eye(2, dtype=CD)
PX = torch.tensor([[0, 1], [1, 0]], dtype=CD)
PY = torch.tensor([[0, -1j], [1j, 0]], dtype=CD)
PZ = torch.tensor([[1, 0], [0, -1]], dtype=CD)

def kron_list(ms):
    out = ms[0]
    for m in ms[1:]:
        out = torch.kron(out, m)
    return out

def on_qubit(op, j, n):
    return kron_list([op if k == j else I2 for k in range(n)])

def R(P, phi):  # exp(-i phi P / 2) = cos(phi/2) I - i sin(phi/2) P
    return torch.cos(phi / 2).to(CD) * I2 - 1j * torch.sin(phi / 2).to(CD) * P

def kraus(ch, p):
    c = np.sqrt(1 - p)
    if ch == 'None':
        return []
    if ch == 'AD':
        return [torch.tensor([[1, 0], [0, c]], dtype=CD), torch.tensor([[0, np.sqrt(p)], [0, 0]], dtype=CD)]
    if ch == 'Pauli':
        px = py = p / 4; pz = (2 - p - 2 * c) / 4
    elif ch == 'Depol':
        Lam = 1 - (2 * c + c * c) / 3
        px = py = pz = Lam / 4
    else:
        raise ValueError(ch)
    ws = [1 - px - py - pz, px, py, pz]
    return [np.sqrt(w) * P for w, P in zip(ws, (I2, PX, PY, PZ))]

class IndepCirc:
    def __init__(self, n, L, channel, p, entangling='ring'):
        assert entangling == 'ring'
        self.n, self.L = n, L
        self.n_theta = 3 * n * (L + 1)
        D = 2 ** n
        # ring E = CNOT_{n-1,0} ... CNOT_{1,2} CNOT_{0,1}  (pairs applied in order j=0..n-1, as the code does)
        E = torch.eye(D, dtype=CD)
        for j in range(n):
            E = self._cnot(j, (j + 1) % n) @ E
        self.E = E
        self.K = [[on_qubit(k, j, n) for k in kraus(channel, p)] for j in range(n)]
        self.Z = [on_qubit(PZ, j, n).real.diagonal() for j in range(n)]

    def _cnot(self, c, t):
        n, D = self.n, 2 ** self.n
        M = torch.zeros(D, D, dtype=CD)
        for b in range(D):
            bits = [(b >> (n - 1 - k)) & 1 for k in range(n)]
            if bits[c]:
                bits[t] ^= 1
            b2 = sum(v << (n - 1 - k) for k, v in enumerate(bits))
            M[b2, b] = 1
        return M

    def _noise(self, rho):
        for Kj in self.K:
            if Kj:
                rho = sum(K @ rho @ K.conj().T for K in Kj)
        return rho

    def features(self, a, theta):
        n, L, D = self.n, self.L, 2 ** self.n
        B = a.shape[0]
        rho = torch.zeros(B, D, D, dtype=CD); rho[:, 0, 0] = 1
        for ell in range(L + 1):
            b = 3 * n * ell
            Rl = kron_list([R(PZ, theta[b + 3 * j + 2]) @ R(PY, theta[b + 3 * j + 1]) @ R(PZ, theta[b + 3 * j])
                            for j in range(n)])
            V = self.E @ Rl if ell < L else Rl
            rho = V @ rho @ V.conj().T
            rho = self._noise(rho)
            if ell < L:
                # S(a) = kron_j RX(a_j), batched
                c, s = torch.cos(a / 2).to(CD), torch.sin(a / 2).to(CD)
                rx = torch.stack([torch.stack([c, -1j * s], -1), torch.stack([-1j * s, c], -1)], -2)  # (B,n,2,2)
                S = rx[:, 0]
                for j in range(1, n):
                    S = torch.einsum('bik,bjl->bijkl', S, rx[:, j]).reshape(B, S.shape[1] * 2, S.shape[2] * 2)
                rho = S @ rho @ S.conj().transpose(1, 2)
                rho = self._noise(rho)
        diag = torch.diagonal(rho, dim1=1, dim2=2).real
        return torch.stack([diag @ z for z in self.Z], 1)

if __name__ == '__main__':
    import os, sys, time
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'noise_structure_mnist'))
    from torch_circ import TorchCirc
    torch.manual_seed(0)
    for L in (1, 4):
        for ch in ('None', 'AD', 'Pauli', 'Depol'):
            p = 0.0 if ch == 'None' else 0.3
            tc = TorchCirc(4, L, noisetype=ch, p_noise=p, device='cpu', entangling='ring')
            ic = IndepCirc(4, L, ch, p)
            worst = 0.0
            for trial in range(3):
                th = 2 * np.pi * torch.rand(tc.n_theta); a = 2 * np.pi * torch.rand(64, 4)
                worst = max(worst, (tc.features(a, th) - ic.features(a, th)).abs().max().item())
            print(f"L={L} {ch:5s}: max |TorchCirc - independent| over 3x64 random (a, theta) = {worst:.2e}")
    # gradient agreement
    th = (2 * np.pi * torch.rand(60)).requires_grad_(True); a = 2 * np.pi * torch.rand(32, 4)
    g1, = torch.autograd.grad(TorchCirc(4, 4, 'AD', 0.3, device='cpu').features(a, th).sum(), th)
    g2, = torch.autograd.grad(IndepCirc(4, 4, 'AD', 0.3).features(a, th).sum(), th)
    print(f"AD L=4 d/dtheta max diff {(g1 - g2).abs().max().item():.2e}")
    for nm, sim in (('TorchCirc', TorchCirc(4, 4, 'AD', 0.3, device='cpu')), ('indep', IndepCirc(4, 4, 'AD', 0.3))):
        t0 = time.time(); th2 = (2*np.pi*torch.rand(60)).requires_grad_(True); sim.features(2*np.pi*torch.rand(1000, 4), th2).sum().backward(); print(nm, f"fwd+bwd 1000: {time.time()-t0:.2f}s")
