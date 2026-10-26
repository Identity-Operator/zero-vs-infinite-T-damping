# Equations in scale_gauge_v5.tex

Equation numbers are from the current build (Eq. L* = 32). The table includes the numbered equations and the inline relations that carry content; plain parameter values such as $p=0.3$ or $L=4$ are left out. Rows are ranked by how much the paper's claims rest on them.

| Rank | Eq. | Formula | What it means | Where | Figures / tables |
|---|---|---|---|---|---|
| **Core results** | | | | | |
| 1 | 32 | $L^*\approx\frac{1}{np}\ln\frac{3\sigma_{\rm None}}{\sigma_{\rm br,\infty}}$ | Depth at which AD and the Pauli twirl separate; grows as $(np)^{-1}\ln(1/p)$ | V C | Fig. 7(c), Table IV |
| 2 | 29 | $\sigma_\infty^2=\frac{p^2(1-p)^2}{n}\sum_j\Big[\frac{(1-p)^2}{2}\mathrm{Var}(\cos a_j)+\frac{1-p}{4}\mathrm{Var}(\sin a_j)\Big]$ | Closed form of the AD floor (last-layer part) | V A | Fig. 7(b), Table IV |
| 3 | 28 | $\delta z_j=p(1-p)\big[(1-p)\,v_z\cos a_j-c\,v_y\sin a_j\big]$ | Last-layer branching term that keeps AD features from vanishing | V A | Fig. 5 (AD floor) |
| 4 | 31 | $\sigma_{\rm AD}^2=\sigma_{\rm Pauli}^2+\sigma_{\rm br}^2$ | AD scale = twirl part + branching part | V C, App D | Fig. 5, Fig. 7 |
| 5 | 22 | $y^{\rm rev}(x;\theta)=\eta\,y(x;\theta')$ | Proposition 1: reversing the damping is a reparametrization | IV A | Table III |
| 6 | 30 | $t_z=p(1-2N)=p\tanh(\hbar\omega/2k_BT)$ | Temperature law of the non-unital term; the floor falls with it | V A | — (abstract claim) |
| 7 | 18 | $\kappa(\mathcal N)=\big(\sigma(\text{None})/\sigma(\mathcal N)\big)^2$ | Shot factor: extra shots needed to read the features at fixed precision | III A | Fig. 2(c), Fig. 3(c), Table V |
| 8 | inline | $\sigma_{\rm br,\infty}^2\approx\sigma_\infty^2+b_{\rm br}\,p$ | Floor at weak damping: last layer plus earlier layers | V C | Fig. 7(b), Table IV |
| 9 | inline | $\sigma_{\rm Pauli}\approx\sigma_{\rm None}\,e^{-npL}$ | Twirled scale decays by $e^{-np}$ per layer | V C | Fig. 5, Fig. 7(a) |
| 10 | 3 | $\mathrm{AD}_{\rm Pauli}=\tfrac12\big(\mathrm{AD}+\mathrm{AD}^{\rm flip}\big)$ | The Pauli twirl is the average of the two damping directions | II A | — |
| 11 | 4 | $T_{\rm GAD}=T_{\rm AD},\ \mathbf t_{\rm GAD}=\big(0,0,p(1-2N)\big)$ | Pauli twirl = damping at infinite temperature | II A | — (finite-T results in V A, App D) |
| 12 | inline | $\sigma_{\rm AD}^2\ge\sigma_\infty^2$ | The floor holds at every depth (counterpart of the bound of Mele et al.) | V A | Fig. 5 |
| 13 | 27 | $E_{\mathrm{AD}^{\rm flip}}(\theta)=E_{\rm AD}(\theta'),\ \theta'_q=\pi-\theta_q$ | Eigensolver gauge when the noise follows the whole $R_{ZZ}$ | IV C | Fig. 4(a) |
| 14 | 24 | $\Theta R_{\mathbf n}(a)\Theta^{-1}=R_{\mathbf n}(a),\ \Theta\,\mathrm{AD}\,\Theta^{-1}=\mathrm{AD}^{\rm flip}$ | Time reversal keeps every rotation and flips the damping | IV A | Table III |
| 15 | 26 | $\Theta\,\mathrm{CNOT}\,\Theta^{-1}=-\mathrm{CNOT}\,(Z\otimes X)$ | Why the classifier ring is a gauge: the residues are absorbed | IV B | Table III |
| 16 | inline | $G=\big\langle\sum_j\lVert\mathbf w_j\rVert^2(1-z_j^2)/\sigma_j^2\big\rangle$ | Readout noise gain; explains why AD costs more shots than $\kappa$ | V B | — (Fig. 6(b) ratios) |
| 17 | 25 | $f\in[2p-1,1],\ f^{\rm rev}\in[-1,1-2p]$ | The two single-qubit model classes differ | IV A | Table III (single-qubit rows) |
| **Key definitions** | | | | | |
| 18 | inline | $\mathbf r\mapsto T\mathbf r+\mathbf t$ | Bloch map of a channel; $\mathbf t$ is the non-unital term | II A | Table I |
| 19 | inline | $\mathcal N(I)=I\iff\mathbf t=\mathbf 0$ | Definition of unital | II A | Table I |
| 20 | 1 | $T_{ij}=\tfrac12\mathrm{Tr}[P_i\mathcal N(P_j)],\ t_i=\tfrac12\mathrm{Tr}[P_i\mathcal N(I)]$ | Entries of the Bloch map | II A | Table I |
| 21 | inline | $c=\sqrt{1-p}$ | Transverse shrink factor of AD | II A | Table I |
| 22 | inline | $\lambda_{\rm depol}=\tfrac13(2c+c^2)$ | Shrink factor of depolarizing noise (the Clifford twirl) | II A | Table I, Fig. 2(c) |
| 23 | 2 | $\mathrm{AD}^{\rm flip}=\mathcal X\circ\mathrm{AD}\circ\mathcal X$ | Damping toward $\lvert1\rangle$ | II A | Table I, Fig. 4 |
| 24 | inline | $X\mathrm{AD}X=Y\mathrm{AD}Y=\mathrm{AD}^{\rm flip},\ Z\mathrm{AD}Z=\mathrm{AD}$ | Which Pauli conjugations flip the damping | II A | — |
| 25 | inline | $\mathbf t_{\rm flip}=(0,0,-p)$ | Reversed AD has the opposite non-unital term | II A | Table I |
| 26 | 11 | $\ell(\mathbf a)=W\mathbf z+\mathbf b$ (raw); $W\,\mathrm{diag}(\boldsymbol\sigma)^{-1}(\mathbf z-\boldsymbol\mu)+\mathbf b$ (standardized) | The two readouts behind the scale story | II C | Fig. 3, Fig. 6, Tables II, V, VI |
| 27 | 15 | $\tilde z_j=s\,z_j+b_j$ | If noise only rescales the features, a trained readout undoes it | III A | — |
| 28 | 17 | $\mathrm{Var}(\hat z_j)=(1-z_j^2)/N_s\le1/N_s$ | Shot noise of one feature | III A | Fig. 7(b) dotted line; Fig. 6 |
| 29 | 13 | $R_{ZZ}(\phi)=\mathrm{CNOT}\,[I\otimes R_Z(\phi)]\,\mathrm{CNOT}$ | Decomposed gate; noise inside it breaks the gauge | II D | Fig. 4(b), Table III |
| 30 | D1 | $\sigma^2=\mathbb E_\theta\frac1n\sum_j\mathrm{Var}_{\mathbf a}z_j$ | Feature scale before training | App D | Fig. 5, Fig. 7(a), Table IV |
| 31 | inline | $\theta_k\mapsto\pm\theta_k+m_k\pi,\ W\mapsto-W,\ \boldsymbol\mu\mapsto-\boldsymbol\mu$ | Explicit parameter map of the classifier gauge | IV A | Table III |
| 32 | inline | $\partial(sf)/\partial\theta=s\,\partial f/\partial\theta$ | A trained scale restores gradient size but not its signal-to-noise | III B | — |
| 33 | 16 | $\tilde f=\lambda_{\rm depol}^{N_{\rm ins}}f$ | Depolarizing noise on one qubit is an exact rescaling | III A | Fig. 2(c) dotted line |
| 34 | inline | $z_j^{\rm Depol}=\lambda_{\rm depol}^3\,z_j^{\rm None}$ | Exact check: at $L=1$ the depolarizing and noiseless classifiers agree | III C | Table VI ($L=1$, Depol) |
| 35 | inline | $\kappa=(s/s_{\rm None})^2$ | Shot factor of the single-qubit fits | III A | Fig. 2(c) |
| 36 | 14 | $\epsilon=\big(E(\theta)-E_0\big)/\lvert E_0\rvert$ | Relative energy error of the eigensolver | II D | Fig. 4 |
| 37 | inline | $N=\big(e^{\hbar\omega/k_BT}+1\big)^{-1}$ | Thermal excited-state population | V A | — |
| 38 | inline | $N\leftrightarrow1-N$; $N=\tfrac12$ is the Pauli twirl | Symmetry of the branching part; the floor vanishes at infinite temperature | V A | — |
| 39 | inline | $\prod_{i\in B}X_i$: $J\to-J$ | Sublattice flip: the coupling sign is a gauge on an open chain | IV C | — |
| 40 | inline | $\sigma_{\rm br}=3\,\sigma_{\rm Pauli}$ | Condition for $(\sigma_{\rm AD}/\sigma_{\rm Pauli})^2=10$; source of the 3 in Eq. 32 | V C | — |
| 41 | inline | $(1-p)^{-72}\approx10^{11}$ | Shot overhead of error cancellation, for comparison | V B | — |
| 42 | 19 | $g_k(\rho)=P_k\rho P_k$ or $P_k\bar\rho P_k$ | Pauli frame, unitary or antiunitary | IV A | Table III |
| 43 | 20–21 | $\Phi^{\rm rev}=g_m\circ\tilde E_m\circ\dots\circ\tilde E_1\circ g_0,\ \tilde E_k=g_k\circ E^{\rm rev}_k\circ g_{k-1}$ | Reversed circuit rewritten with frames | IV A | — |
| 44 | 23 | $\Theta=\bigotimes_q(-iY_q)\,K$ | Spin time reversal | IV A | Table III |
| 45 | inline | $R_{\mathbf n}(a)=e^{-ia\,\mathbf n\cdot\boldsymbol\sigma_q/2}$ | Single-qubit rotation | IV A | — |
| 46 | inline | $\tfrac14\sum_{P}P\,\mathrm{AD}\,P$ | Pauli twirl of AD (Corollary 2) | IV A | — |
| 47 | 10 | $z_j=\mathrm{Tr}\big[Z_j\,\Phi(\lvert0\rangle\langle0\rvert^{\otimes n})\big]$ | The classifier features | II C | Fig. 1, Fig. 5 |
| 48 | inline | $\mathcal N(\rho)=\sum_kK_k\rho K_k^\dagger$ | Kraus form of a channel | II A | Table I |
| 49 | inline | $\mathcal X(\rho)=X\rho X$ | Bit flip | II A | — |
| 50 | inline | $\mathbb E_\theta[R_{ab}R_{a'c}]=0\ (b\ne c)$ | Random blocks remove cross moments; makes the exact recursion close | App D | Fig. 5 lines |
| **Model specification** | | | | | |
| 51 | 9 | $\Phi=\mathcal N^{\otimes n}\circ\mathcal U(\mathcal R^{(L)})\circ\mathcal L^{(L-1)}\circ\dots\circ\mathcal L^{(0)}$ | Classifier circuit | II C | Fig. 1 |
| 52 | 8 | $\mathcal L^{(\ell)}=\mathcal N^{\otimes n}\circ\mathcal U(S(\mathbf a))\circ\mathcal N^{\otimes n}\circ\mathcal U(E\mathcal R^{(\ell)})$ | One layer: noise after the ring and after the encoding | II C | Fig. 1 |
| 53 | 7 | $\mathcal R^{(\ell)}=\bigotimes R_ZR_YR_Z,\ S(\mathbf a)=\bigotimes R_X(a_j),\ E=\mathrm{CNOT}_{n,1}\cdots\mathrm{CNOT}_{1,2}$ | Building blocks of the classifier | II C | Fig. 1 |
| 54 | 5 | $\Phi_{x,\theta}=\mathcal N\circ\mathcal W_2\circ\mathcal N\circ\mathcal S_x\circ\mathcal N\circ\mathcal W_1\circ\mathcal N\circ\mathcal S_x\circ\mathcal N\circ\mathcal W_0$ | Single-qubit re-uploading model, five insertions | II B | Fig. 2 |
| 55 | 6 | $f(x)=\mathrm{Tr}\big[Z\,\Phi_{x,\theta}(\lvert0\rangle\langle0\rvert)\big]$ | Raw single-qubit output | II B | Fig. 2(a) |
| 56 | inline | $f_{s,b}(x)=s\,f(x)+b$ | Rescaled single-qubit output | II B | Fig. 2(b), (c) |
| 57 | 12 | $H=-J\sum_iZ_iZ_{i+1}-h\sum_iX_i$ | Three-site transverse-field Ising ring | II D | Fig. 4 |
| 58 | E1 | $\hat z_j=z_j+\sqrt{(1-z_j^2)/N_s}\,\varepsilon_j$ | Gaussian shot model used in training | App E | Fig. 6(a), (c) |
| 59 | inline | $\mathcal U(V)\colon\rho\mapsto V\rho V^\dagger$, $\mathcal S_x$, $\mathcal W_\ell$ | Notation for gates as channels | II B | — |
| 60 | inline | $\Phi^{\rm rev}=E^{\rm rev}_m\circ\dots\circ E^{\rm rev}_1,\ g_k\circ g_k=\mathrm{id}$ | Setup for inserting frames | IV A | — |
| 61 | inline | $z\mapsto-z$ | Shot noise is symmetric, so the gauge survives finite shots | IV A | — |
| 62 | inline | $p_0=1-\sum_{k\ne0}p_k$ | Identity weight of a Pauli channel | Table I caption | Table I |
| **Proof steps and protocol** | | | | | |
| 63 | B1 | $y_i=\mathrm{Tr}[O_i\,\Phi(\rho_0)]$ | General model for the proof | App B | — |
| 64 | B2 | $g\circ\mathcal E\circ g'\colon\{K_j\}\mapsto\{P\,K_j^{(t)}P'\}$ | How a frame acts on Kraus operators | App B | — |
| 65 | B3 | $g_k\circ\mathrm{AD}^{\rm flip}\circ g_{k-1}=\mathrm{AD}$ | Condition (i): an $X$ or $Y$ frame flips the damping | App B | — |
| 66 | B4 | $P_k\propto U\,P_{k-1}\,U^{(t)\dagger}$ | Condition (ii): the frame passes fixed gates | App B | — |
| 67 | B5 | $P_q\,\overline{\mathbf n_q\cdot\boldsymbol\sigma_q}\,P_q=-\mathbf n_q\cdot\boldsymbol\sigma_q$ | Condition (iii): the encoding is left invariant | App B | — |
| 68 | B6 | $P\,e^{\mp i\theta Q^{(t)}/2}P=e^{-i\theta'Q/2}$ | Condition (iv): trainable rotations stay in their family | App B | — |
| 69 | B7 | $\mathrm{Tr}[O\,g_m(\tau)]=\mathrm{Tr}[P_mO^{(t)}P_m\,\tau]$ | Condition (v): the output frame gives a sign | App B | — |
| 70 | B8 | $y^{\rm rev}_i(x;\theta)=\eta_i\,y_i(x;\theta')$ | End of the proof (unlabeled) | App B | — |
| 71 | B9–B10 | $\mathcal P\colon(T,\mathbf t)\mapsto(DTD,D\mathbf t),\ \mathcal K\colon(T,\mathbf t)\mapsto(FTF,F\mathbf t)$ | Frames keep $\lVert\mathbf t\rVert$, so the twirl is not frame-equivalent to AD | App B | — |
| 72 | inline | $\theta'=D_\theta\theta+\pi\mathbf m,\ W\mapsto W\,\mathrm{diag}(\boldsymbol\eta)$ | Parameter map in the proof of Corollary 1 | App B | — |
| 73 | inline | $\mathcal P(\rho)=P\rho P,\ \mathcal K(\rho)=\bar\rho,\ \mathcal K\circ\mathcal E\circ\mathcal K=\bar{\mathcal E}$ | Frame algebra used in the proof | App B | — |
| 74 | inline | e.g. $YXY=-X$, $P_{0,q}\in\{I,Z\}$, $P_mO^{(t)}P_m=\eta O$ | Remaining small steps of the proof | App B | — |
| 75 | inline | $\mathbb E\,v_z^2=\tfrac12,\ \mathbb E\,v_y^2=\tfrac14,\ \mathbb E\,v_zv_y=0$ | Averages over the final block used to get Eq. 29 | App D | — |
| 76 | A1 | $g'(x)=a_1\cos x+b_1\sin x+a_2\cos2x+b_2\sin2x$ | Target functions of the single-qubit fits | App A | Fig. 2 |
| 77 | inline | $g(x)=R\,\frac{g'(x)-\frac12(g'_{\max}+g'_{\min})}{g'_{\max}-g'_{\min}}$ | Target scaled to range $R$ | App A | Fig. 2 |
| 78 | inline | $\frac1{250}\sum_i\big[f(x_i)-g(x_i)\big]^2<10^{-5}$ | Convergence criterion | App A | Fig. 2 |
| 79 | inline | $W_\ell=R_Z(\theta_{3\ell+2})R_Y(\theta_{3\ell+1})R_Z(\theta_{3\ell})$ | Angle layout of the single-qubit blocks | App A | — |
| 80 | inline | $f(x)=r_z,\ f=\langle Z\rangle+b$ | Output in the Bloch picture; raw model with a free offset | App A | Fig. 2(a) |
| 81 | inline | $z_j=\langle Z_j\rangle$, $y^{\rm rev}=\eta\,y$, $f=\langle Z\rangle$ | Restatements of numbered equations in captions and text | Fig. 1, Table III, IV A | — |
