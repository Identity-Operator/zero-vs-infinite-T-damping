# -*- coding: utf-8 -*-
"""
Torch-native density-matrix simulator for the hybrid ansatz (M3).

State rho is kept as a tensor of shape (B, 2,...,2 | 2,...,2) (2n qubit
indices: row indices then column indices). Gates act by einsum contractions
on the relevant qubit indices — no 2^n x 2^n matrices are ever built, and
everything is differentiable in torch (GPU).

Noise channels are applied as local Kraus operators with definitions matching
QDRCirc.py exactly (gate C8), so results are comparable to the n=1 campaign.

Validation gates:
  T1: noiseless n=1 outputs match QDRCirc.TrainableCirc (<=1e-10)
  T2: noisy n=1 AD/Pauli/Depol match QDRCirc (<=1e-9)
  T3: noisy n=2 matches PennyLane default.mixed (<=1e-9)
  T4: noisy n=4 training step in seconds (not minutes)
"""

import numpy as np
import torch

# Match hybrid_qnn.py: keep all simulation in float64 (torch defaults to
# float32, which silently degrades precision on gate angles near k*pi).
torch.set_default_dtype(torch.float64)


class TorchCirc:
    def __init__(self, n_qubits, num_layers, noisetype='None', p_noise=0.0,
                 device='cuda', x_scaling=1.0, dtype=torch.complex128,
                 entangling='ring', noise_placement='after_entangler'):
        """noise_placement='after_entangler' (default) applies the channel once to
        every qubit after the entangling layer. 'inside_entangler' instead applies
        it to both qubits after every CNOT of that layer, so a qubit is hit once
        per CNOT it takes part in. This breaks the Pauli-frame and time-reversal
        gauges that relate AD and AD_flip. Layers without
        an entangler (including every layer when entangling='none'), and the
        insertions after the encoding, are the same in both.
        """
        assert noise_placement in ('after_entangler', 'inside_entangler'), noise_placement
        self.n = n_qubits
        self.L = num_layers
        self.noisetype = noisetype
        self.p_noise = p_noise
        self.device = device
        self.x_scaling = x_scaling
        self.dtype = dtype
        self.entangling = entangling
        self.noise_placement = noise_placement
        self.n_theta = 3 * n_qubits * (num_layers + 1)
        self._prep_noise()

    # ------------------------------------------------------------------
    # gate constructors (complex128, differentiable)
    def _rz(self, t):
        z = torch.zeros(2, 2, dtype=self.dtype, device=self.device)
        z[0, 0] = torch.exp(-0.5j * t)
        z[1, 1] = torch.exp(0.5j * t)
        return z

    def _ry(self, t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        m = torch.zeros(2, 2, dtype=self.dtype, device=self.device)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -s, s, c
        return m

    def _rx(self, t):
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        m = torch.zeros(2, 2, dtype=self.dtype, device=self.device)
        m[0, 0], m[0, 1], m[1, 0], m[1, 1] = c, -1j * s, -1j * s, c
        return m

    def _rx_batch(self, t):
        """Batched RX: t (B,) -> (B,2,2)."""
        c, s = torch.cos(t / 2), torch.sin(t / 2)
        B = t.shape[0]
        m = torch.zeros(B, 2, 2, dtype=self.dtype, device=self.device)
        m[:, 0, 0], m[:, 0, 1], m[:, 1, 0], m[:, 1, 1] = c, -1j*s, -1j*s, c
        return m

    # ------------------------------------------------------------------
    # index-contraction gate application
    def _apply_1q_left(self, rho, G, j):
        n = self.n
        letters = [chr(ord('c') + k) for k in range(2 * n)]
        rho_str = ''.join(letters)
        left_letter = 'X'
        letters_out = letters.copy()
        letters_out[j] = left_letter
        eq = f"{left_letter}{letters[j]},b{rho_str}->b{''.join(letters_out)}"
        return torch.einsum(eq, G, rho)

    def _apply_1q_right(self, rho, G, j):
        n = self.n
        letters = [chr(ord('c') + k) for k in range(2 * n)]
        rho_str = ''.join(letters)
        left_letter = 'X'
        letters_out = letters.copy()
        letters_out[n + j] = left_letter
        eq = f"{left_letter}{letters[n + j]},b{rho_str}->b{''.join(letters_out)}"
        return torch.einsum(eq, G.conj(), rho)

    def apply_1q(self, rho, G, j):
        return self._apply_1q_right(self._apply_1q_left(rho, G, j), G, j)

    def apply_cnot(self, rho, control, target):
        """CNOT via 4-leg tensor contraction on (control, target)."""
        n = self.n
        # U[a,b,c,d] = <a,b|CNOT|c,d>: maps basis |c,d> -> |a,b>
        U = torch.zeros(2, 2, 2, 2, dtype=self.dtype, device=self.device)
        U[0, 0, 0, 0] = 1
        U[0, 1, 0, 1] = 1
        U[1, 0, 1, 1] = 1
        U[1, 1, 1, 0] = 1
        letters = [chr(ord('c') + k) for k in range(2 * n)]
        rho_str = ''.join(letters)
        lo = letters.copy(); lo[control] = 'P'; lo[target] = 'Q'
        eq = f"PQ{letters[control]}{letters[target]},b{rho_str}->b{''.join(lo)}"
        rho = torch.einsum(eq, U, rho)
        ro = letters.copy(); ro[n + control] = 'P'; ro[n + target] = 'Q'
        eq = f"PQ{letters[n + control]}{letters[n + target]},b{rho_str}->b{''.join(ro)}"
        rho = torch.einsum(eq, U.conj(), rho)
        return rho

    # ------------------------------------------------------------------
    def _prep_noise(self):
        """Local Kraus operators per qubit, matching QDRCirc definitions."""
        dev, dt = self.device, self.dtype
        p = self.p_noise
        self.kraus = []
        if self.noisetype == 'None':
            return
        if self.noisetype == 'AD':
            K0 = torch.tensor([[1, 0], [0, np.sqrt(1 - p)]], dtype=dt, device=dev)
            K1 = torch.tensor([[0, np.sqrt(p)], [0, 0]], dtype=dt, device=dev)
            self.kraus = [K0, K1]
        elif self.noisetype == 'AD_flip':
            # Amplitude damping toward |1> instead of |0>. Same contraction
            # T = diag(c, c, c^2) as 'AD' and its Pauli twirl; the bias is
            # exactly sign-flipped, t_z = -p. Isolates the SIGN of the
            # non-unital term relative to the |0> input and Z readout, with
            # every attenuation factor held fixed.
            K0 = torch.tensor([[np.sqrt(1 - p), 0], [0, 1]], dtype=dt, device=dev)
            K1 = torch.tensor([[0, 0], [np.sqrt(p), 0]], dtype=dt, device=dev)
            self.kraus = [K0, K1]
        elif self.noisetype == 'Depol':
            # Qiskit depolarizing_error(lam): rho -> (1-lam)rho + lam I/2,
            # contraction 1-lam. Kraus: I with 1-3*lam/4, X/Y/Z with lam/4.
            lam = (p + 2 - 2 * np.sqrt(1 - p)) / 3
            probs = [1 - 3 * lam / 4, lam / 4, lam / 4, lam / 4]
            mats = [torch.eye(2), torch.tensor([[0, 1], [1, 0]]),
                    torch.tensor([[0, -1j], [1j, 0]]),
                    torch.tensor([[1, 0], [0, -1]])]
            self.kraus = [np.sqrt(pr) * m.to(dtype=dt, device=dev)
                          for pr, m in zip(probs, mats)]
        elif self.noisetype == 'Pauli':
            q = p / 4
            pz = (2 - p - 2 * np.sqrt(1 - p)) / 4
            probs = [1 - 2 * q - pz, q, q, pz]
            mats = [torch.eye(2), torch.tensor([[0, 1], [1, 0]]),
                    torch.tensor([[0, -1j], [1j, 0]]),
                    torch.tensor([[1, 0], [0, -1]])]
            self.kraus = [np.sqrt(pr) * m.to(dtype=dt, device=dev)
                          for pr, m in zip(probs, mats)]
        else:
            raise ValueError(f"unknown noisetype {self.noisetype}")

    def _noise(self, rho, j):
        if not self.kraus:
            return rho
        return sum(self.apply_1q(rho, K, j) for K in self.kraus)

    def _entangling_pairs(self):
        n = self.n
        if self.entangling == 'none':
            return []
        if self.entangling == 'ring':
            return [(j, (j + 1) % n) for j in range(n)]
        if self.entangling == 'star':
            return [(0, j) for j in range(1, n)]
        if self.entangling == 'full':
            return [(c, t) for c in range(n) for t in range(c + 1, n)]
        raise ValueError(f"unknown entangling pattern {self.entangling}")

    # ------------------------------------------------------------------
    def features(self, a_batch, theta):
        """Batched forward pass. a_batch: (B, n) angles; theta: (n_theta,).

        Returns z: (B, n) real expectation values <Z_j>.
        """
        rho = self.evolve(a_batch, theta)
        z = torch.stack([self._measure_z(rho, j) for j in range(self.n)], dim=1)
        return z.real

    def probs(self, a_batch, theta):
        """Z-basis outcome distribution diag(rho): (B, 2^n), qubit 0 the most
        significant bit, so outcome k gives Z_j = 1 - 2 * ((k >> (n-1-j)) & 1)."""
        rho = self.evolve(a_batch, theta)
        d = 2 ** self.n
        return torch.diagonal(rho.reshape(-1, d, d), dim1=1, dim2=2).real

    def evolve(self, a_batch, theta):
        """The output density matrix, shape (B,) + (2,) * 2n."""
        B = a_batch.shape[0]
        n, L = self.n, self.L
        rho = torch.zeros((B,) + (2,) * (2 * n), dtype=self.dtype,
                          device=self.device)
        idx = (slice(None),) + (0,) * (2 * n)
        rho[idx] = 1.0

        for ell in range(L + 1):
            base = 3 * n * ell
            for j in range(n):
                rho = self.apply_1q(rho, self._rz(theta[base + 3 * j]), j)
                rho = self.apply_1q(rho, self._ry(theta[base + 3 * j + 1]), j)
                rho = self.apply_1q(rho, self._rz(theta[base + 3 * j + 2]), j)
            inside = (ell < L and n > 1 and self.noise_placement == 'inside_entangler'
                      and len(self._entangling_pairs()) > 0)
            if ell < L and n > 1:
                for c, t in self._entangling_pairs():
                    rho = self.apply_cnot(rho, c, t)
                    if inside:
                        rho = self._noise(rho, c)
                        rho = self._noise(rho, t)
            if not inside:
                for j in range(n):
                    rho = self._noise(rho, j)
            if ell < L:
                for j in range(n):
                    # encoding angle varies per batch element: (B,2,2) gate
                    t = self.x_scaling * a_batch[:, j]
                    G = self._rx_batch(t)
                    rho = self._apply_1q_batch_left(rho, G, j)
                    rho = self._apply_1q_batch_right(rho, G, j)
                for j in range(n):
                    rho = self._noise(rho, j)
        return rho

    def _apply_1q_batch_left(self, rho, G, j):
        n = self.n
        letters = [chr(ord('c') + k) for k in range(2 * n)]
        rho_str = ''.join(letters)
        letters_out = letters.copy(); letters_out[j] = 'X'
        eq = f"bX{letters[j]},b{rho_str}->b{''.join(letters_out)}"
        return torch.einsum(eq, G, rho)

    def _apply_1q_batch_right(self, rho, G, j):
        n = self.n
        letters = [chr(ord('c') + k) for k in range(2 * n)]
        rho_str = ''.join(letters)
        letters_out = letters.copy(); letters_out[n + j] = 'X'
        eq = f"bX{letters[n + j]},b{rho_str}->b{''.join(letters_out)}"
        return torch.einsum(eq, G.conj(), rho)

    def _measure_z(self, rho, j):
        """Tr(rho Z_j) — one-sided application (apply_1q would give Z rho Z)."""
        Z = torch.tensor([[1.0, 0], [0, -1]], dtype=self.dtype,
                         device=self.device)
        rho = self._apply_1q_left(rho, Z, j)
        n = self.n
        letters = [chr(ord('c') + k) for k in range(n)]
        eq = 'b' + ''.join(letters) + ''.join(letters)
        return torch.einsum(eq, rho)
