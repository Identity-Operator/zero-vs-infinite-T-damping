# Theory of the damping direction — v3: proofs (English)

*2026-10-25. English version of [ly_thuyet_gauge_v3.md](ly_thuyet_gauge_v3.md), with the same content and the same numbering. It keeps statements and proofs only. Each proof is a list of numbered steps; each step is one transformation, with its reason after the dash "—". Interpretation, numerical-check tables and the literature discussion are in the Vietnamese [v2](ly_thuyet_gauge_v2.md); the check scripts are in [checks/](checks/).*

| Changes from v2 | |
|---|---|
| Removed | §0 (summary, answers to the co-author's notes), the "meaning" paragraphs, Remark 12.3, §11 (numerical checks), the positioning part of §12 |
| Revised | Every proof rewritten as one-transformation steps |
| Added | Proofs of Proposition 6 (Lemmas 6.1–6.4) and Theorem 9 (Lemma 9.1), which v2 only justified in a table; proofs of Corollaries 1.3 and 2.5; the section "What is needed for novelty" below |

## What is needed for novelty

| Part | Possibly new | Overlap risk | To do |
|---|---|---|---|
| **A** (gauge, QAOA) | Parity selection rule (Theorem 1); exact last-layer decomposition (Theorem 2); $E(\mathbf y)=E_{\rm tw}+a^2\langle\mathbf y\rvert H\lvert\mathbf y\rangle$, which explains the empirical NDAR correlation (Corollary 2.3); Ising landscape (Theorem 3); reset coefficient (Theorem 4); symmetry-breaking duality (Theorem 5); architecture classification and noise boundary | Read in full, no overlap: NDAR (2404.01412), ND-AWS (2607.09368), Marshall et al. 2020 (see the reading notes below). Not yet read: NDAR for qudits (2606.28234), Tam et al. (2504.12628), Shaydulin et al. 2021 *Classical symmetries and the QAOA* | 1. Read the three unread papers. 2. **New, high value:** extend Theorem 2 to the warm-start mixer $aX+bZ$, which gives the exact difference between the ND and Standard ansätze of ND-AWS at depth 1 (under AD the Standard ansatz is the ND ansatz with direction pattern $\hat{\mathbf y}$, by Lemma 4). 3. Tighten the bound of Theorem 5 with a light-cone argument (currently loose by about $10^3$). 4. Check at optimized parameters, $n=8$–12. 5. Hardware: measure the reset coefficient (Theorem 4) with mid-circuit resets, and the direction effect through the gauge $H^{\mathbf y}$ (Lemma 4) |
| **B** (gauge averaging) | Gauge averaging equals the Pauli twirl exactly at depth 1, with an $O(p^2)$ remainder that is bounded (Theorem 12); criterion for fixing versus averaging the gauge (Corollary 12.1) | Spin-reversal transforms are standard practice in annealing (Boixo et al. 2013, Pelofske et al.). There may be work on twirling $R_{ZZ}$ gates with Paulis that commute with them. **Not yet searched.** ND-AWS (read) does not discuss gauge averaging | 1. Find and read the literature on spin-reversal transforms and on twirling non-Clifford gates with commuting Paulis. 2. Explicit formula for the remainder at $P=2$. 3. Shot-cost analysis of averaging versus fixing the gauge. 4. Experiment: compare the two on the same instance |
| **D** (feature scale, separation depth) | Decomposition $\sigma^2=\sum s^{2m}B_m$ with $B_m\ge0$ (Theorem 15); last-layer floor as a lower bound at every depth (Theorem 16); contraction rate $2n'p$ (Theorem 18); derivation of the $L^*$ law that the paper only fits numerically (Proposition 19) | Mele et al., Fefferman et al., Schumann et al. have results on moments under non-unital and unital noise. **The value $1/(2^n+1)$ in Theorem 17 is the known Haar value**, $\mathbb E\langle\psi\rvert P\lvert\psi\rangle^2=1/(2^n+1)$; what is new is only the proof that this circuit reaches it | 1. State in the paper that $1/(2^n+1)$ is the Haar value. 2. Prove Assumption G for general $n$, and prove A1, A2. 3. Closed form of $b_{\rm br}$. 4. Positioning against Mele et al.: the difference is the *comparative* law between AD and its twirl |

**Reading notes (2026-10-25):**
- *Marshall et al. 2020* (arXiv:2002.11682): only depolarizing noise (Pauli Kraus operators) after each complete QAOA round, and only an approximate binomial formula $F\approx1+\alpha[(1-p\frac{\kappa-1}{\kappa})^N-1]$ with $\alpha,\kappa$ fitted numerically. No amplitude damping and no exact formula, so **no overlap** with Theorem 2 or Corollary 2.1. Their expansion in the number of noise events (after Xue et al. 2019) is in the same spirit as Lemma 3 and should be cited as precedent.
- *ND-AWS* (arXiv:2607.09368v2): no theorem for the noisy case. Appendix A.4.b proves that **without noise** the ND and Standard ansätze give the same distribution; this is the noiseless case of Lemma 4 extended to warm starts. The noisy case is only a heuristic argument ("noise will sometimes act against the bias", Appendix A.4.c) plus simulations with AD after every single-qubit gate ($n=20$). Their $p=1$ simulator uses two-qubit reduced density matrices and is noiseless. The paper does not discuss gauge averaging.

**Priorities:**
1. Extend Theorem 2 to the warm-start mixer, to obtain the exact ND − Standard difference: the analytical contribution that ND-AWS lacks.
2. Read NDAR for qudits, Tam et al., Shaydulin et al. 2021, and the literature on spin-reversal transforms or twirling with commuting Paulis (for Part B).
3. Tighten the bound of Theorem 5, so that the weak-breaking part has usable numbers.
4. Check at optimized parameters, since all checks of A and B are currently at random parameters.
5. Decide the structure of the paper: A + B as the core (fixed gauge versus averaged gauge), D as support or as a separate paper. Following the co-author's notes, drop the two models of van Rossum et al.

---

## 0. Notation

- Channel insertion $i$ acts on qubit $q(i)$ with parameters $(p_i,s_i)$. $\mathcal I_q$ is the set of insertions on qubit $q$. $c=\sqrt{1-p}$.
- $\mathrm{GAD}_{p,s}$ has Bloch matrix $T=\mathrm{diag}(c,c,c^2)$ and vector $\mathbf t=(0,0,ps)$, with $s=1-2N$. Cases: $s=1$ is AD, $s=-1$ is AD$^{\rm flip}$, $s=0$ is the twirl $\mathcal T_p$. For a thermal bath, $s=\tanh(\hbar\omega/2k_BT)$.
- Reset: $\mathcal R_b(\rho)=\lvert b\rangle\langle b\rvert_q\otimes\mathrm{Tr}_q\rho$.
- QAOA: $H=\sum_{u<v}J_{uv}Z_uZ_v+\sum_uh_uZ_u$. Input $\lvert+\rangle^{\otimes n}$. Layer $l$: $e^{-i\gamma_lH}$, noise stage A, $e^{-i\beta_l\sum_qX_q}$, noise stage B.
- Gauge: $P_{\mathbf y}=\bigotimes_qX_q^{y_q}$, $H^{\mathbf y}=P_{\mathbf y}HP_{\mathbf y}$, $\hat y_q=(-1)^{y_q}$.
- $E(\mathbf s)$ is the expectation value of the observable $O$; $E_{\rm tw}=E(\mathbf 0)$.

---

## 1. Basic lemmas

**Lemma 1.** $\mathrm{GAD}_{p,s}=\mathcal T_p+s\,\mathcal D_p$, with $\mathcal D_p(\rho)=\tfrac p2\,Z_q\otimes\mathrm{Tr}_q\rho=\tfrac p2(\mathcal R_0-\mathcal R_1)(\rho)$.

*Proof.*

(1) One qubit: $\mathcal N\bigl(\tfrac12(\tau I+\mathbf r\cdot\boldsymbol\sigma)\bigr)=\tfrac12\bigl(\tau I+(T\mathbf r+\tau\mathbf t)\cdot\boldsymbol\sigma\bigr)$, with $\tau=\mathrm{Tr}\rho$ — affine form of a qubit channel.

(2) $\mathrm{GAD}_{p,s}(\rho)-\mathcal T_p(\rho)=\tfrac12\tau\,(0,0,ps)\cdot\boldsymbol\sigma$ — both channels have the same $T$; $\mathbf t_{\mathcal T}=0$.

(3) $=\tfrac{ps}{2}\,\mathrm{Tr}(\rho)\,Z$.

(4) For $n$ qubits and $\rho=A_q\otimes B$: the difference is $\tfrac{ps}2\mathrm{Tr}(A_q)\,Z_q\otimes B=\tfrac{ps}2Z_q\otimes\mathrm{Tr}_q\rho$ — (3) applied to the factor on $q$.

(5) It holds for every $\rho$ — linearity.

(6) $Z=\lvert0\rangle\langle0\rvert-\lvert1\rangle\langle1\rvert$ gives the reset form. □

**Lemma 2.**

(a) $\mathcal D_p^\dagger(O)=\tfrac p2\,I_q\otimes\mathrm{Tr}_q[Z_qO]$. On one qubit: $\mathcal D^\dagger(Z)=pI$ and $\mathcal D^\dagger(I)=\mathcal D^\dagger(X)=\mathcal D^\dagger(Y)=0$.

(b) $\lVert\mathcal D_p(A)\rVert_1\le p\lVert A\rVert_1$.

*Proof.*

(1) $\mathrm{Tr}[O\,\mathcal D(\rho)]=\tfrac p2\mathrm{Tr}\bigl[O\,(Z_q\otimes\mathrm{Tr}_q\rho)\bigr]$ — Lemma 1.

(2) $=\tfrac p2\mathrm{Tr}\bigl[(I_q\otimes\mathrm{Tr}_q[Z_qO])\,\rho\bigr]$ — property of the partial trace.

(3) (2) gives (a). On one qubit: $\mathrm{Tr}[ZZ]=2$ and $\mathrm{Tr}[Z\sigma]=0$ for $\sigma\in\{I,X,Y\}$.

(4) $\lVert Z_q\otimes B\rVert_1=\lVert Z\rVert_1\lVert B\rVert_1=2\lVert B\rVert_1$ — the trace norm is multiplicative over tensor products.

(5) $\lVert\mathrm{Tr}_qA\rVert_1\le\lVert A\rVert_1$ — the partial trace is CPTP.

(6) (4) and (5) times $\tfrac p2$ give (b). □

**Lemma 3.** For a circuit of fixed CPTP maps with $K$ GAD insertions,
$$
E(\mathbf s)=\sum_{S}s^Sc_S,\qquad s^S=\prod_{i\in S}s_i,\qquad c_S=\mathrm{Tr}[O\,\Phi_S(\rho_0)],\qquad |c_S|\le\lVert O\rVert_\infty\prod_{i\in S}p_i ,
$$
where $\Phi_S$ has $\mathcal D_i$ at $i\in S$ and $\mathcal T_i$ at every other insertion.

*Proof.*

(1) Replace each insertion by $\mathcal T_i+s_i\mathcal D_i$ — Lemma 1.

(2) Expand the product of maps into a sum over $S$ — distributivity.

(3) $\lVert\Phi_S(\rho_0)\rVert_1\le\prod_{i\in S}p_i$ — Lemma 2b; CPTP maps do not increase $\lVert\cdot\rVert_1$.

(4) $|\mathrm{Tr}[OA]|\le\lVert O\rVert_\infty\lVert A\rVert_1$ — Hölder. □

**Lemma 4.** For every depth and every $p_i$: $E\bigl(H^{\mathbf y};\{s_i\}\bigr)=E\bigl(H;\{\hat y_{q(i)}s_i\}\bigr)$.

*Proof.*

(1) $e^{-i\gamma H^{\mathbf y}}=P_{\mathbf y}e^{-i\gamma H}P_{\mathbf y}$ — since $P_{\mathbf y}^2=I$.

(2) $P_{\mathbf y}e^{-i\beta\sum X}P_{\mathbf y}=e^{-i\beta\sum X}$ and $P_{\mathbf y}\lvert+\rangle^{\otimes n}=\lvert+\rangle^{\otimes n}$ — $X$ commutes with $X$, and $X\lvert+\rangle=\lvert+\rangle$.

(3) $X\,\mathrm{GAD}_{p,s}\,X=\mathrm{GAD}_{p,-s}$ — $X\mathcal TX=\mathcal T$ because $\mathcal T$ is a Pauli channel; $X\mathcal DX=-\mathcal D$ because $XZX=-Z$ (Lemma 1).

(4) Insert $P_{\mathbf y}P_{\mathbf y}=I$ between all elements of the circuit for $H^{\mathbf y}$ and use (1)–(3): it becomes the circuit for $H$ with $s_i\to\hat y_{q(i)}s_i$.

(5) Observable: $\mathrm{Tr}[H^{\mathbf y}\,P_{\mathbf y}\sigma P_{\mathbf y}]=\mathrm{Tr}[H\sigma]$ — cyclicity of the trace. □

---

## 2. Part A — Gauge

**Theorem 1 (parity).** Let $G$ be a Pauli string, $\mathcal G(\rho)=G\rho G$, such that:
- (H0) $G_{q(i)}\in\{X,Y\}$ for every insertion $i$;
- (H1) $\mathcal G$ commutes with every map that is not a GAD insertion;
- (H2) $\mathcal G(\rho_0)=\rho_0$;
- (H3) $GOG=\eta O$ with $\eta=\pm1$.

Then $c_S=\eta(-1)^{|S|}c_S$.

*Proof.*

(1) $\mathcal G\mathcal T_i\mathcal G=\mathcal T_i$ — a Pauli channel is invariant under Pauli conjugation.

(2) $\mathcal G\mathcal D_i\mathcal G(\rho)=\tfrac p2\,G\bigl(Z_q\otimes\mathrm{Tr}_q[G\rho G]\bigr)G$ — Lemma 1.

(3) $\mathrm{Tr}_q[G\rho G]=G'(\mathrm{Tr}_q\rho)G'$ with $G=G_q\otimes G'$ — $G_q$ is unitary on $q$.

(4) $\mathcal G\mathcal D_i\mathcal G(\rho)=\tfrac p2\,(G_qZ_qG_q)\otimes\mathrm{Tr}_q\rho$ — since $G'^2=I$.

(5) $=-\mathcal D_i(\rho)$ — (H0): $G_qZG_q=-Z$ for $G_q\in\{X,Y\}$.

(6) $\mathcal G\Phi_S\mathcal G=(-1)^{|S|}\Phi_S$ — insert $\mathcal G\mathcal G=\mathrm{id}$ between all maps; use (H1), (1), (5).

(7) $c_S=\mathrm{Tr}\bigl[GOG\cdot\mathcal G\Phi_S\mathcal G(\mathcal G\rho_0)\bigr]$ — $\mathcal G^2=\mathrm{id}$; the trace is invariant under unitary conjugation.

(8) $=\eta(-1)^{|S|}c_S$ — (H2), (H3), (6). □

**Corollary 1.1.** QAOA with $h=0$: $E(\mathbf s)=E(-\mathbf s)$ for every $\mathbf s\in[-1,1]^K$.

*Proof.*

(1) $G=X^{\otimes n}$ satisfies (H0).

(2) $X^{\otimes n}$ commutes with $Z_uZ_v$ and with $X_q$, hence with $e^{-i\gamma H}$ and the mixer; (H1) holds.

(3) $X\lvert+\rangle=\lvert+\rangle$; (H2) holds.

(4) $X^{\otimes n}Z_uZ_vX^{\otimes n}=Z_uZ_v$; (H3) holds with $\eta=1$.

(5) Theorem 1: $c_S=0$ for odd $|S|$, so $E(\mathbf s)-E(-\mathbf s)=2\sum_{|S|\ \text{odd}}s^Sc_S=0$. □

**Corollary 1.2.** Under the assumptions of Theorem 1 with $\eta=1$:
$$
|E_{\rm AD}-E_{\rm tw}|\le\lVert O\rVert_\infty\Bigl[\tfrac12\prod_i(1+p_i)+\tfrac12\prod_i(1-p_i)-1\Bigr]=O(p^2).
$$

*Proof.*

(1) $E_{\rm AD}-E_{\rm tw}=\sum_{S\ne\emptyset}c_S$ — Lemma 3 with $s_i=1$; $c_\emptyset=E_{\rm tw}$.

(2) $=\sum_{|S|\ \text{even}\ \ge2}c_S$ — Theorem 1.

(3) $\le\lVert O\rVert\sum_{|S|\ \text{even}\ \ge2}\prod_Sp_i$ — Lemma 3.

(4) $\sum_{|S|\ \text{even}}\prod_Sp_i=\tfrac12\bigl[\prod(1+p_i)+\prod(1-p_i)\bigr]$ — expand both products; the odd orders cancel. □

**Corollary 1.3.** For uniform $s_i=s$: $E=\sum_kf_k\,s^{2k}$, and $\partial E/\partial N=0$ at $N=\tfrac12$.

*Proof.*

(1) $E(s)=\sum_Ss^{|S|}c_S$ keeps only even $|S|$ — Lemma 3, Theorem 1.

(2) $\partial_N=-2\,\partial_s$ — since $s=1-2N$.

(3) $\partial_sE|_{s=0}=\sum_{|S|=1}c_S=0$ — (1). □

**Corollary 1.4 (propagating frame).** Suppose the model satisfies Proposition 1 of the paper, with an involutive parameter map $\theta\to\theta'$. Then $E_{\mathbf s}(\theta)=E_{-\mathbf s}(\theta')$, and the odd part $\mathcal O(\theta)=\tfrac12[E_{\mathbf s}(\theta)-E_{-\mathbf s}(\theta)]$ satisfies $\mathcal O(\theta')=-\mathcal O(\theta)$.

*Proof.*

(1) At every noise location the frame is $X$ or $Y$, so $g\,\mathrm{GAD}_{p,s}\,g=\mathrm{GAD}_{p,-s}$ — as in step (5) of Theorem 1. The same holds for antiunitary frames, since $\overline{\mathcal T}=\mathcal T$ and $\overline{\mathcal D}=\mathcal D$ ($Z$ is real).

(2) The telescoping proof of Proposition 1, applied with $\mathrm{GAD}_{p,-s}$ in place of AD$^{\rm flip}$, gives $E_{-\mathbf s}(\theta)=E_{\mathbf s}(\theta')$.

(3) Replace $\theta$ by $\theta'$: $E_{-\mathbf s}(\theta')=E_{\mathbf s}(\theta'')=E_{\mathbf s}(\theta)$ — involution.

(4) $\mathcal O(\theta')=\tfrac12[E_{-\mathbf s}(\theta)-E_{\mathbf s}(\theta)]=-\mathcal O(\theta)$ — (2), (3). □

**Theorem 2 (last-layer decomposition).** Let $\sigma_C$ be the state right after the last phase layer (arbitrary). Let $E^{(0)}$ and $m^{(0)}_u=\langle Z_u\rangle$ be the values obtained when the last two noise stages have $s=0$. Then
$$
E=E^{(0)}+\sum_ua_u\Bigl(h_u+\sum_vJ_{uv}m^{(0)}_v\Bigr)+\sum_{u<v}J_{uv}a_ua_v,\qquad a_u=s_{B,u}p_{B,u}+(1-p_{B,u})\cos(2\beta_P)\,s_{A,u}p_{A,u}.
$$

*Proof.*

(1) $\mathrm{GAD}^\dagger$: $Z\mapsto c^2Z+ps\,I$, $X\mapsto cX$, $Y\mapsto cY$ — Bloch matrix $(T,\mathbf t)$.

(2) Stage B: $Z_u\mapsto(1-p_B)Z_u+s_Bp_B\,I$ — (1).

(3) Mixer: $e^{i\beta X}Ze^{-i\beta X}=\cos2\beta\,Z+\sin2\beta\,Y$.

(4) Stage A: $Z\mapsto(1-p_A)Z+s_Ap_A\,I$, $Y\mapsto\sqrt{1-p_A}\,Y$ — (1).

(5) Combining (2)–(4): $Z_u\mapsto\hat Z_u+a_uI$ with $\hat Z_u=(1-p_B)\bigl[\cos2\beta\,(1-p_A)Z_u+\sin2\beta\sqrt{1-p_A}\,Y_u\bigr]$, independent of $s$.

(6) $Z_uZ_v\mapsto(\hat Z_u+a_u)(\hat Z_v+a_v)$ and $I\mapsto I$ — the maps in (2)–(4) are tensor products over qubits.

(7) $E=\sum_{u<v}J_{uv}\langle(\hat Z_u+a_u)(\hat Z_v+a_v)\rangle_{\sigma_C}+\sum_uh_u\langle\hat Z_u+a_u\rangle_{\sigma_C}$ — (6).

(8) Setting $a=0$: $E^{(0)}=\sum J\langle\hat Z_u\hat Z_v\rangle+\sum h\langle\hat Z_u\rangle$ and $m^{(0)}_u=\langle\hat Z_u\rangle$.

(9) Expand (7) and use (8). □

**Corollary 2.1 ($P=1$).** With input $\lvert+\rangle^{\otimes n}$ and no noise before the phase layer: $E^{(0)}=E_{\rm tw}$ and
$$
m^{(0)}_u=(1-p_{B,u})\sqrt{1-p_{A,u}}\,\sin2\beta\,\sin(2\gamma h_u)\prod_{w\ne u}\cos(2\gamma J_{uw}).
$$

*Proof.*

(1) $e^{i\gamma H}Y_ue^{-i\gamma H}=Y_u\,e^{-2i\gamma H_u}$, with $H_u=h_uZ_u+\sum_wJ_{uw}Z_uZ_w$ — $Y_u$ anticommutes with the terms containing $Z_u$ and commutes with the others.

(2) $=Y_u(\cos2\gamma h_u-i\sin2\gamma h_u\,Z_u)\prod_w(\cos2\gamma J_{uw}-i\sin2\gamma J_{uw}\,Z_uZ_w)$ — $e^{-i\phi P}=\cos\phi-i\sin\phi\,P$; the factors commute.

(3) In $\lvert+\rangle^{\otimes n}$: $\langle X\rangle=1$, $\langle Y\rangle=\langle Z\rangle=0$, so only products of $X$ operators have a nonzero expectation.

(4) Every term that picks a factor $Z_uZ_w$ contains $Z_w$ an odd number of times, so its expectation vanishes — (3).

(5) The remaining term is $Y_u(-i\sin2\gamma h_u)Z_u\prod_w\cos2\gamma J_{uw}$, and $-iY_uZ_u=X_u$ — since $YZ=iX$.

(6) $\langle Y_u\rangle_C=\sin(2\gamma h_u)\prod_w\cos(2\gamma J_{uw})$ — (3)–(5).

(7) $\langle Z_u\rangle_C=0$ — $e^{-i\gamma H}$ is diagonal; $\lvert+\rangle^{\otimes n}$ is uniform on the computational basis.

(8) $m^{(0)}_u=\langle\hat Z_u\rangle_C$; insert (6), (7) into step (5) of Theorem 2. □

**Corollary 2.2.** $P=1$, $h=0$, uniform $p$, AD: $E_{\rm AD}-E_{\rm tw}=a^2\langle0^n\rvert H\lvert0^n\rangle$ at every $(\gamma,\beta)$. Hence $\sum_{u<v}J_{uv}\ge0\Rightarrow\min E_{\rm AD}\ge\min E_{\rm tw}$, and $\sum_{u<v}J_{uv}\le0\Rightarrow\min E_{\rm AD}\le\min E_{\rm tw}$.

*Proof.*

(1) $m^{(0)}_u=0$ — Corollary 2.1 with $h_u=0$.

(2) $E_{\rm AD}-E_{\rm tw}=a^2\sum_{u<v}J_{uv}$ — Theorem 2 with $a_u=a$.

(3) $\sum_{u<v}J_{uv}=\langle0^n\rvert H\lvert0^n\rangle$ — since $Z\lvert0\rangle=\lvert0\rangle$.

(4) The sign of $E_{\rm AD}-E_{\rm tw}$ is the sign of $\sum J$ at every point — (2), (3), $a^2\ge0$.

(5) An inequality that holds pointwise also holds for the minima. □

**Corollary 2.3 (gauge landscape, $P=1$, uniform AD).**
$$
E(\mathbf y)=E_{\rm tw}+a\sum_u\hat y_u\Bigl(h_u+\sum_vJ_{uv}m^{(0)}_v\Bigr)+a^2\sum_{u<v}J_{uv}\hat y_u\hat y_v,\qquad h=0:\ E(\mathbf y)=E_{\rm tw}+a^2\langle\mathbf y\rvert H\lvert\mathbf y\rangle .
$$

*Proof.*

(1) The gauge $\mathbf y$ is equivalent to direction $\hat y_u$ on qubit $u$ — Lemma 4.

(2) $a_u$ is linear in $s$, so $a_u\to\hat y_ua$.

(3) $E^{(0)}$ and $m^{(0)}$ do not depend on $\mathbf y$ — Lemma 4 with $s=0$.

(4) Insert (2), (3) into Theorem 2.

(5) For $h=0$: $m^{(0)}=0$ (Corollary 2.1), and $\sum J_{uv}\hat y_u\hat y_v=\langle\mathbf y\rvert H\lvert\mathbf y\rangle$ — since $Z_u\lvert\mathbf y\rangle=\hat y_u\lvert\mathbf y\rangle$. □

**Corollary 2.4 (direction effect, $P=1$).**
$$
E(\mathbf s)-E(-\mathbf s)=2\sum_ua_u\Bigl(h_u+\sum_vJ_{uv}m^{(0)}_v\Bigr),\qquad
|E(\mathbf s)-E(-\mathbf s)|\le2\sum_u|a_u|\Bigl(|h_u|+2|\gamma|\sum_v|J_{uv}||h_v|\Bigr).
$$

*Proof.*

(1) $\mathbf s\to-\mathbf s$ maps $a\to-a$; $E^{(0)}$ and $m^{(0)}$ are unchanged — Theorem 2.

(2) Subtract the two instances of Theorem 2: the term $\sum Ja_ua_v$ cancels, the linear term doubles.

(3) $|m^{(0)}_v|\le|\sin2\gamma h_v|\le2|\gamma||h_v|$ — Corollary 2.1; the other factors have modulus at most 1.

(4) Triangle inequality. □

**Corollary 2.5.** If $\cos2\beta_P=-s_Bp_B/[(1-p_B)s_Ap_A]$ on every qubit, then $E=E^{(0)}$.

*Proof.*

(1) The condition is equivalent to $a_u=0$ — definition of $a_u$.

(2) Set $a=0$ in Theorem 2. □

**Theorem 3 (gauge landscape, any depth, $h=0$).**
$$
E(\mathbf y)=C+\sum_{u<v}K_{uv}\hat y_u\hat y_v+R(\mathbf y),\qquad K_{uv}=\sum_{i\in\mathcal I_u,\,j\in\mathcal I_v}s_is_jc_{ij},
$$
$$
|R(\mathbf y)|\le\lVert H\rVert_\infty\Bigl[\tfrac12\prod(1+p_i)+\tfrac12\prod(1-p_i)-1-e_2(\mathbf p)\Bigr],\qquad e_2(\mathbf p)=\sum_{i<j}p_ip_j .
$$

*Proof.*

(1) $E(\mathbf y)=\sum_S\bigl(\prod_{i\in S}\hat y_{q(i)}\bigr)s^Sc_S$ — Lemma 4, Lemma 3.

(2) Odd $|S|$: $c_S=0$ — Corollary 1.1.

(3) $|S|=2$ with $q(i)=q(j)$: $\hat y^2=1$; absorbed into $C$ together with $c_\emptyset$.

(4) $|S|=2$ with $q(i)=u\ne v=q(j)$: gives $\hat y_u\hat y_v\,s_is_jc_{ij}$; summing gives $K_{uv}$.

(5) $|S|\ge4$ is $R$; bound it with Lemma 3 and the expansion of Corollary 1.2, minus the orders 0 and 2. □

**Theorem 4 (first-order coefficient).** For every circuit: $c_i=\tfrac{p_i}2\bigl(E_{\rm tw}[i\to\mathcal R_0]-E_{\rm tw}[i\to\mathcal R_1]\bigr)$, where $E_{\rm tw}[i\to\mathcal R_b]$ is the energy of the circuit twirled everywhere, except that insertion $i$ is replaced by the reset $\mathcal R_b$. Moreover, $E(\mathbf s)-E(-\mathbf s)=2\sum_is_ic_i+O(p^3)$.

*Proof.*

(1) $c_i=\mathrm{Tr}[O\,\Phi_{\{i\}}\rho_0]$, where $\Phi_{\{i\}}$ has $\mathcal D_i$ at $i$ and twirls everywhere else — Lemma 3.

(2) $\mathcal D_i=\tfrac{p_i}2(\mathcal R_0-\mathcal R_1)$ — Lemma 1.

(3) Insert (2) into (1); use linearity.

(4) $E(\mathbf s)-E(-\mathbf s)=2\sum_{|S|\ \text{odd}}s^Sc_S$ — Lemma 3.

(5) The terms with $|S|\ge3$ satisfy $|c_S|\le\lVert O\rVert\prod_Sp_i=O(p^3)$ — Lemma 3. □

**Theorem 5 (weak symmetry breaking).** Assume (H0) and
$$
\mathcal G\mathcal U_k(\varepsilon)\mathcal G=\mathcal U_k(-\varepsilon),\qquad\mathcal G\rho_0(\varepsilon)=\rho_0(-\varepsilon),\qquad GO(\varepsilon)G=O(-\varepsilon).
$$
Then:
- (a) $E_{\mathbf s}(\varepsilon)=E_{-\mathbf s}(-\varepsilon)$;
- (b) $\Delta E(\varepsilon):=E_{\mathbf s}(\varepsilon)-E_{-\mathbf s}(\varepsilon)$ is odd in $\varepsilon$;
- (c) $|\Delta E|\le\tfrac12\Lambda\bigl[\prod(1+|s_i|p_i)-\prod(1-|s_i|p_i)\bigr]$, with
$$
\Lambda=\lVert O(\varepsilon)-O(-\varepsilon)\rVert_\infty+\lVert O\rVert_\infty\Bigl(\sum_k\lVert\mathcal U_k(\varepsilon)-\mathcal U_k(-\varepsilon)\rVert_{1\to1}+\lVert\rho_0(\varepsilon)-\rho_0(-\varepsilon)\rVert_1\Bigr).
$$
For a field $h$ in QAOA: $\Lambda\le2\lVert h\rVert_1\bigl(1+2\lVert H\rVert_\infty\sum_l|\gamma_l|\bigr)$.

*Proof.* Write $X^\pm$ for $X$ at $\pm\varepsilon$.

(1) $c_S(-\varepsilon)=(-1)^{|S|}c_S(\varepsilon)$ — steps (6)–(7) of Theorem 1, with the new assumptions in place of (H1)–(H3).

(2) (a): $E_{-\mathbf s}(-\varepsilon)=\sum_S(-1)^{|S|}s^Sc_S(-\varepsilon)=\sum_Ss^Sc_S(\varepsilon)$ — Lemma 3, (1).

(3) (b): $\Delta E(\varepsilon)=E_{\mathbf s}(\varepsilon)-E_{\mathbf s}(-\varepsilon)$ — (a); the right-hand side changes sign under $\varepsilon\to-\varepsilon$.

(4) $\Delta E=\sum_{|S|\ \text{odd}}s^S\bigl[c_S^+-c_S^-\bigr]$ — Lemma 3, (1).

(5) $c_S^+-c_S^-=\mathrm{Tr}[(O^+-O^-)\Phi_S^+\rho_0^+]+\mathrm{Tr}[O^-(\Phi_S^+-\Phi_S^-)\rho_0^+]+\mathrm{Tr}[O^-\Phi_S^-(\rho_0^+-\rho_0^-)]$ — add and subtract.

(6) $\Phi_S^+-\Phi_S^-=\sum_k\mathcal M_m^-\cdots\mathcal M_{k+1}^-(\mathcal M_k^+-\mathcal M_k^-)\mathcal M_{k-1}^+\cdots\mathcal M_1^+$ — telescoping; only the unitaries depend on $\varepsilon$.

(7) Each term in (5), (6) is at most $\prod_Sp_i$ times the corresponding norm — Lemma 2b, Hölder, CPTP maps are contractive.

(8) $|c_S^+-c_S^-|\le\Lambda\prod_Sp_i$ — sum (7).

(9) $|\Delta E|\le\Lambda\sum_{|S|\ \text{odd}}\prod_S|s_i|p_i=\tfrac\Lambda2\bigl[\prod(1+|s|p)-\prod(1-|s|p)\bigr]$ — expand; the even orders cancel.

(10) Field $h$: $U^\pm=e^{-i\gamma H_J}e^{\mp i\gamma h\cdot Z}$, so $\lVert U^+-U^-\rVert_\infty=\lVert e^{-2i\gamma h\cdot Z}-I\rVert_\infty\le2|\gamma|\lVert h\rVert_1$ — since $|e^{-ix}-1|\le|x|$.

(11) $\lVert\mathcal U^+-\mathcal U^-\rVert_{1\to1}\le2\lVert U^+-U^-\rVert_\infty$ and $\lVert O^+-O^-\rVert_\infty=\lVert2h\cdot Z\rVert\le2\lVert h\rVert_1$ — add and subtract $U\rho V^\dagger$. □

### Proposition 6 — architecture catalogue (each case proved)

**Lemma 6.1.** QAOA in which every term $Z_{u_1}\cdots Z_{u_k}$ has even $k$ (any graph, any weights, any $P$, including multi-angle QAOA) satisfies (H0)–(H3) with $G=X^{\otimes n}$ and $\eta=1$.

*Proof.*

(1) $X^{\otimes n}Z_{u_1}\cdots Z_{u_k}X^{\otimes n}=(-1)^kZ_{u_1}\cdots Z_{u_k}$ — $XZX=-Z$ on each factor.

(2) For even $k$ the term is invariant, so every gate $e^{-i\gamma_tZ_{u_1}\cdots Z_{u_k}}$ commutes with $X^{\otimes n}$, also when each gate has its own angle.

(3) Mixer, input and observable: steps (2)–(4) of Corollary 1.1. □

**Lemma 6.2.** If $H$ has $h_u\ne0$ on a noisy qubit $u$, no fixed Pauli string $G$ satisfies (H0) together with (H1) for all $\gamma$. Moreover, $E_{\mathbf s}(h)=E_{-\mathbf s}(-h)$.

*Proof.*

(1) (H1) for all $\gamma$ implies $[G,H]=0$ — derivative at $\gamma=0$.

(2) $GHG=\sum_t(\pm)_t\,t$ over the Pauli terms $t$ of $H$; the $t$ are linearly independent.

(3) Hence $[G,H]=0$ if and only if $G$ commutes with every term; for $Z_u$ this requires $G_u\in\{I,Z\}$.

(4) This contradicts (H0) at $u$.

(5) Duality: $X^{\otimes n}$ maps $h\to-h$ in both the phase layer and the observable; apply Theorem 5 (a). □

**Lemma 6.3.** XY mixer $\sum_q(X_qX_{q+1}+Y_qY_{q+1})$, ZZ cost, Dicke input $\lvert D_k\rangle$: $E_{\mathbf s}(D_k)=E_{-\mathbf s}(D_{n-k})$. The gauge ($E_{\mathbf s}=E_{-\mathbf s}$) holds for $k=n/2$.

*Proof.*

(1) $X^{\otimes n}(XX+YY)X^{\otimes n}=XX+YY$ — $XXX=X$, $XYX=-Y$, and $(-1)^2=1$.

(2) $X^{\otimes n}\lvert x\rangle=\lvert\bar x\rangle$ maps Hamming weight $k\to n-k$, so $X^{\otimes n}\lvert D_k\rangle=\lvert D_{n-k}\rangle$.

(3) Insert $\mathcal G\mathcal G$ as in steps (6)–(7) of Theorem 1, with (H2) replaced by (2): $E_{\mathbf s}(D_k)=E_{-\mathbf s}(D_{n-k})$.

(4) For $k=n/2$, $D_{n-k}=D_k$. □

**Lemma 6.4.** HVA for the open XXZ chain with even $n$ (even-bond layer, then odd-bond layer, noise after each layer), Néel input $\lvert0101\cdots\rangle$: $E_{\mathbf s}=E_{-\mathbf s}$ if $p_q=p_{n-1-q}$.

*Proof.*

(1) $X^{\otimes n}$ preserves $XX$, $YY$, $ZZ$ — as in Lemma 6.3 (1).

(2) The reflection $\pi:q\mapsto n-1-q$ maps bond $(i,i+1)$ to $(n-2-i,n-1-i)$, and $n-2-i\equiv i\pmod2$; so the even and odd layers are preserved.

(3) Néel: the bit at $q$ is $q\bmod2$; after $X^{\otimes n}$ it is $1-(q\bmod2)$; after $\pi$ it is $1-((n-1-q)\bmod2)=q\bmod2$ — since $n-1$ is odd.

(4) $H_{\rm XXZ}$ is invariant under $X^{\otimes n}$ and $\pi$ — (1), (2).

(5) Apply Proposition 8 with $G=X^{\otimes n}$ and $\pi$. □

**Lemma 7 (CNOT traps the frame).** Suppose $R_{ZZ}$ is compiled as $\mathrm{CNOT}_{uv}R_Z^{(v)}\mathrm{CNOT}_{uv}$, with noise on both qubits after the first CNOT and a noise location on $v$ before the gate, and with only Pauli rotations and CNOTs on $v$ between these two locations. Then no Pauli frame (unitary or antiunitary) satisfies condition (i) of Proposition 1.

*Proof.*

(1) After the first CNOT, the frame has $x$-bit 1 on both $u$ and $v$ — condition (i).

(2) Before the CNOT: $x_v^{\rm before}=x_u^{\rm after}\oplus x_v^{\rm after}=0$ — CNOT propagation rule; the CNOT is real, so this also holds for antiunitary frames.

(3) Pauli rotations do not change the frame — condition (iv) of Proposition 1.

(4) At the noise location before the gate on $v$, the frame lies in $\{I,Z\}$, which violates (i). □

**Proposition 8 (Pauli × permutation).** Let $\pi$ be a qubit permutation with operator $\Pi$. Suppose $G\Pi$ preserves every gate (within its parameter family), $\rho_0$ and $O$, (H0) holds, and $p_i=p_{\pi(i)}$, $s_i=s_{\pi(i)}$. Then $E_{\mathbf s}=E_{-\mathbf s}$.

*Proof.*

(1) $\Pi\,\mathrm{GAD}^{(q)}_{p,s}\,\Pi^\dagger=\mathrm{GAD}^{(\pi(q))}_{p,s}$ — relabelling of qubits.

(2) Conjugation by $G\Pi$ maps insertion $i$ to $\mathrm{GAD}_{p_i,-s_i}$ at location $\pi(i)$ — (1), Theorem 1 step (5).

(3) By assumption $(p,s)$ is invariant under $\pi$, so the circuit becomes the circuit with $-\mathbf s$.

(4) $\rho_0$ and $O$ are invariant, and the trace is invariant under conjugation, so $E_{\mathbf s}=E_{-\mathbf s}$. □

### Theorem 9 — noise boundary

**Theorem 9.** Let $\mathcal M^{\rm rev}$ denote a noise component $\mathcal M$ with every damping direction reversed. If every noise component satisfies $\mathcal G\mathcal M\mathcal G=\mathcal M^{\rm rev}$ and the unitary part satisfies (H1)–(H3), then $E(\text{model})=E(\text{model}^{\rm rev})$.

*Proof.*

(1) Insert $\mathcal G\mathcal G=\mathrm{id}$ around every component.

(2) Each noise component becomes $\mathcal M^{\rm rev}$; the unitary part is unchanged.

(3) $\rho_0$ and $O$ are invariant — (H2), (H3). □

**Lemma 9.1 (component by component, with $G=X^{\otimes n}$).**
- (a) $\mathrm{GAD}_{p_q,s_q}$ with $p_q$, $s_q$ arbitrary per qubit and per time: $\mathcal G\mathcal M\mathcal G=\mathrm{GAD}_{p_q,-s_q}=\mathcal M^{\rm rev}$.
- (b) Pauli channels (dephasing, depolarizing, bit flip): $\mathcal G\mathcal M\mathcal G=\mathcal M$.
- (c) Unitary $e^{-i\epsilon A}$: becomes $e^{-i\epsilon\,GAG}$. Invariant if $GAG=A$ ($ZZ$ crosstalk, $X$ over-rotation); becomes $e^{+i\epsilon A}$ if $GAG=-A$ ($Z$ detuning, $Y$ errors).
- (d) Lindblad $\mathcal L=-i[H_g,\cdot]+\sum_k\mathcal D[L_k]$: $\mathcal G\mathcal L\mathcal G=-i[GH_gG,\cdot]+\sum_k\mathcal D[GL_kG]$. If $[G,H_g]=0$ and $L_k\in\{\sigma^-_q,\ \sum_q\sigma^-_q,\ Z_q\}$, the result is $\mathcal L^{\rm rev}$.
- (e) Readout with a confusion matrix $(r_0,r_1)$ fixed in the lab frame: global relabelling maps $(r_0,r_1)\to(r_1,r_0)$; invariant if and only if $r_0=r_1$.
- (f) CNOT with noise inside the gate: Lemma 7.

*Proof.*

(a) Lemma 4 step (3) on each qubit.

(b) $X^{\otimes n}QX^{\otimes n}=\pm Q$ for every Pauli string $Q$, and the sign cancels in $Q\rho Q$.

(c) $Ge^{-i\epsilon A}G=e^{-i\epsilon GAG}$; $X^{\otimes2}ZZX^{\otimes2}=ZZ$, $XXX=X$, $XZX=-Z$, $XYX=-Y$.

(d)

(1) $\mathcal Ge^{t\mathcal L}\mathcal G=e^{t\,\mathcal G\mathcal L\mathcal G}$ — since $\mathcal G^2=\mathrm{id}$.

(2) $\mathcal G\,\mathcal D[L]\,\mathcal G=\mathcal D[GLG]$ — direct substitution into $L\rho L^\dagger-\tfrac12\{L^\dagger L,\rho\}$.

(3) $X\sigma^-X=\sigma^+$; $X^{\otimes n}(\sum\sigma^-_q)X^{\otimes n}=\sum\sigma^+_q$; $\mathcal D[-Z]=\mathcal D[Z]$.

(e) Relabelling $b\to1-b$ swaps both columns and both rows of the confusion matrix.

(f) Lemma 7. □

---

## 3. Part B — Gauge averaging

Define $E_{\rm avg}=2^{-n}\sum_{\mathbf y}E(H^{\mathbf y};\{s_i\})$.

**Lemma 10.** $E_{\rm avg}=\sum_{S:\ |S\cap\mathcal I_q|\ \text{even}\ \forall q}s^Sc_S$.

*Proof.*

(1) $E_{\rm avg}=2^{-n}\sum_{\mathbf y}\sum_S\bigl(\prod_{i\in S}\hat y_{q(i)}\bigr)s^Sc_S$ — Lemma 4, Lemma 3.

(2) $\prod_{i\in S}\hat y_{q(i)}=\prod_q\hat y_q^{\,k_q}$ with $k_q=|S\cap\mathcal I_q|$ — grouping by qubit.

(3) $2^{-n}\sum_{\mathbf y}\prod_q\hat y_q^{\,k_q}=\prod_q\tfrac12\sum_{\hat y=\pm1}\hat y^{\,k_q}$ — the $\hat y_q$ are independent.

(4) $\tfrac12\sum_{\hat y=\pm1}\hat y^{\,k}=[k\ \text{even}]$.

(5) Insert (2)–(4) into (1). □

**Lemma 11.** Let $i<j$ in $\mathcal I_q$, with no other insertion on $q$ between them and only single-qubit gates on $q$ between them. Then $c_S=0$ for every $S\supseteq\{i,j\}$.

*Proof.*

(1) $\mathcal D_j^\dagger(O_j)=\tfrac{p_j}2I_q\otimes\mathrm{Tr}_q[Z_qO_j]$ — Lemma 2a.

(2) Going backwards from $j$ to $i$, the factor on $q$ stays $I_q$ — $U^\dagger IU=I$; maps on other qubits do not touch $q$.

(3) $\mathcal D_i^\dagger(I_q\otimes B)=\tfrac{p_i}2I_q\otimes\mathrm{Tr}[Z]\,B=0$ — Lemma 2a.

(4) $c_S=\mathrm{Tr}[\Phi_S^\dagger(O)\,\rho_0]=0$. □

**Theorem 12.**

(a) $P=1$, no noise before the phase layer, arbitrary $h$, $p_i$, $s_i$: $E_{\rm avg}=E_{\rm tw}$.

(b) Any $P$:
$$
E_{\rm avg}-E_{\rm tw}=\sum_{\substack{S\ne\emptyset\\ |S\cap\mathcal I_q|\ \text{even}}}s^Sc_S,\qquad
|E_{\rm avg}-E_{\rm tw}|\le\lVert O\rVert_\infty\Bigl[\prod_q\tfrac12\Bigl(\prod_{i\in\mathcal I_q}(1+|s_i|p_i)+\prod_{i\in\mathcal I_q}(1-|s_i|p_i)\Bigr)-1\Bigr].
$$

*Proof of (a).*

(1) $\mathcal I_q=\{A_q,B_q\}$, and between them on $q$ there is only $R_X(2\beta)$ — circuit structure.

(2) Only the sets $S$ with $|S\cap\mathcal I_q|\in\{0,2\}$ remain — Lemma 10.

(3) If $|S\cap\mathcal I_q|=2$, then $c_S=0$ — Lemma 11.

(4) Only $S=\emptyset$ remains: $E_{\rm avg}=c_\emptyset=E_{\rm tw}$. □

*Proof of (b).*

(1) The identity: Lemma 10 minus the term $S=\emptyset$.

(2) $|s^Sc_S|\le\lVert O\rVert\prod_S|s_i|p_i$ — Lemma 3.

(3) $\sum_{S\ \text{even on every qubit}}\prod_Sx_i=\prod_q\tfrac12\bigl[\prod_{\mathcal I_q}(1+x_i)+\prod_{\mathcal I_q}(1-x_i)\bigr]$ — expand qubit by qubit and keep the even orders.

(4) Set $x_i=|s_i|p_i$ and subtract 1. □

**Corollary 12.1.** $P=1$, uniform AD: $E(\mathbf y)-E_{\rm avg}=a\sum_u\hat y_u\bigl(h_u+\sum_vJ_{uv}m^{(0)}_v\bigr)+a^2\sum_{u<v}J_{uv}\hat y_u\hat y_v$. For $h=0$ the right-hand side is $a^2\langle\mathbf y\rvert H\lvert\mathbf y\rangle$, whose average over $\mathbf y$ is 0.

*Proof.*

(1) $E_{\rm avg}=E_{\rm tw}$ — Theorem 12 (a).

(2) Corollary 2.3.

(3) $\sum_{\mathbf y}\langle\mathbf y\rvert H\lvert\mathbf y\rangle=\mathrm{Tr}H=0$ — $\mathrm{Tr}\,Z_uZ_v=\mathrm{Tr}\,Z_u=0$. □

**Corollary 12.2.** For a propagating frame (Corollary 1.4): $\tfrac12[f_{\mathbf s}(\theta)+f_{\mathbf s}(\theta')]=f_{\rm tw}(\theta)+\sum_{|S|\ge2\ \text{even}}s^Sc_S(\theta)$.

*Proof.*

(1) $f_{\mathbf s}(\theta')=f_{-\mathbf s}(\theta)$ — Corollary 1.4.

(2) $\tfrac12[f_{\mathbf s}+f_{-\mathbf s}]$ keeps only even $|S|$ — Lemma 3. □

---

## 4. Part D — Feature scale of the classifier

**Setting.**
- Layer $l=0..L-1$: block $\bigotimes_qR_Z(t_2)R_Y(t_1)R_Z(t_0)$ with independent uniform angles; CNOT ring $E$; GAD stage; $\bigotimes_qR_X(a_q)$; GAD stage.
- End of the circuit: one block, one GAD stage, then measurement of $z_j=\langle Z_j\rangle$. Input $\mathbf a\sim\mu$.
- Feature scale: $\sigma^2=\mathbb E_\theta\tfrac1n\sum_j\mathrm{Var}_\mu z_j$.
- Heisenberg maps in the basis $(I,X,Y,Z)$ (column = input, row = output):
$$
K_s=\begin{pmatrix}1&0&0&sp\\0&c&0&0\\0&0&c&0\\0&0&0&c^2\end{pmatrix},\quad
A(a)=\begin{pmatrix}1&0&0&0\\0&1&0&0\\0&0&\cos a&-\sin a\\0&0&\sin a&\cos a\end{pmatrix},\quad
G(a)=K_sA(a)K_s .
$$
- Heisenberg operator $O=\sum_Po_PP$; $z=\sum_{P\in\{I,Z\}^n}o_P$.

**Lemma 13 (Euler block).** $\mathbb E[R_{ab}R_{cd}]=\delta_{ac}\delta_{bd}\mathsf Q_{ab}$, with
$$
\mathsf Q=\begin{pmatrix}1&0&0&0\\0&\tfrac38&\tfrac38&\tfrac14\\0&\tfrac38&\tfrac38&\tfrac14\\0&\tfrac14&\tfrac14&\tfrac12\end{pmatrix}\quad\text{(doubly stochastic)}.
$$

*Proof.* On $\{X,Y,Z\}$, $R$ has the same distribution as $\mathcal Z(\alpha)\mathcal Y(\beta)\mathcal Z(\gamma)$, with $\alpha,\beta,\gamma$ independent and uniform.

(1) $\mathcal Z(\alpha+\pi)=\mathrm{diag}(-1,-1,1)\mathcal Z(\alpha)$, so $\mathbb E[R_{ab}R_{cd}]=0$ when exactly one of $a,c$ equals $z$.

(2) Likewise with $\gamma+\pi$: $\mathbb E[R_{ab}R_{cd}]=0$ when exactly one of $b,d$ equals $z$.

(3) $F=\mathrm{diag}(1,-1,1)$ satisfies $\mathcal Z(-\alpha)=F\mathcal Z(\alpha)F$ and $F\mathcal YF=\mathcal Y$; the change $(\alpha,\gamma)\to(-\alpha,-\gamma)$ gives $R\to FRF$, so $\mathbb E[R_{ab}R_{cd}]=0$ when an odd number of the indices equal $y$.

(4) The remaining cases with $(a,b)\ne(c,d)$: $\mathbb E[R_{xx}R_{yy}]$, $\mathbb E[R_{xy}R_{yx}]$ and their permutations — enumeration using (1)–(3).

(5) $\mathbb E_\alpha[\mathcal Z_{xk}\mathcal Z_{yk'}]=\tfrac12\epsilon_{kk'}$ — $\mathbb E\cos\sin=0$, $\mathbb E\cos^2=\tfrac12$, $\mathbb E(-\sin^2)=-\tfrac12$.

(6) $\mathbb E_\gamma[\mathcal Z_{lx}\mathcal Z_{l'y}]=\tfrac12\epsilon_{ll'}$ — as (5).

(7) $\mathbb E[R_{xx}R_{yy}]=\tfrac14\sum\epsilon_{kk'}\epsilon_{ll'}\mathbb E[\mathcal Y_{kl}\mathcal Y_{k'l'}]=\tfrac12\mathbb E[\mathcal Y_{xx}\mathcal Y_{yy}-\mathcal Y_{xy}\mathcal Y_{yx}]$ — (5), (6).

(8) $=\tfrac12\mathbb E[\cos\beta\cdot1-0]=0$ — $\mathcal Y$ is a rotation about the $y$ axis.

(9) $\mathbb E[R_{xy}R_{yx}]=-(7)=0$ — $\epsilon_{l'l}=-\epsilon_{ll'}$.

(10) Diagonal entries: $R_{zz}=\cos\beta\Rightarrow\tfrac12$; $R_{zx}=\pm\sin\beta\cos\gamma\Rightarrow\tfrac14$; $R_{xx}=\cos\alpha\cos\beta\cos\gamma-\sin\alpha\sin\gamma\Rightarrow\tfrac18+\tfrac14=\tfrac38$; the others by symmetry.

(11) $\sum_aR_{ab}^2=\sum_bR_{ab}^2=1$ — $R$ is orthogonal. □

**Lemma 14 (pair chain).** Let $m^{(l)}_P=\mathbb E_\theta[o_P(\mathbf a)o_P(\mathbf a')]$ right after block $l$ (in the Heisenberg direction). Then:
- (i) the cross moments vanish at every block boundary;
- (ii) $m^{(l)}=\mathsf Q^{\otimes n}\Pi_E\mathcal W\,m^{(l+1)}$, with $\mathcal W=\bigotimes_q[G(a_q)\odot G(a'_q)]$;
- (iii) $m^{(L)}$ has $(sp)^2$ on $I^{\otimes n}$ and $c^4\mathsf Q_{bZ}/n$ on the string with $b\ne I$ at qubit $j$;
- (iv) $S(\mathbf a,\mathbf a')=\mathbb E_\theta\tfrac1n\sum_jz_j(\mathbf a)z_j(\mathbf a')=\sum_{P\in\{I,Z\}^n}m^{(0)}_P$.

*Proof.*

(1) Through a block: $o'_Q=\sum_P\mathcal R_{QP}o_P$, with the same $\mathcal R$ for both inputs.

(2) $\mathbb E[o'_Qo'_{Q'}]=\sum_{P,P'}\mathbb E[\mathcal R_{QP}\mathcal R_{Q'P'}]\,\mathbb E[o_Po_{P'}]$ — the block is independent of the blocks after it.

(3) $\mathbb E[\mathcal R_{QP}\mathcal R_{Q'P'}]=\prod_q\mathbb E[R_{Q_qP_q}R_{Q'_qP'_q}]=\delta_{QQ'}\delta_{PP'}\prod_q\mathsf Q_{Q_qP_q}$ — independent qubits; Lemma 13.

(4) (2) and (3) give (i) and $m'=\mathsf Q^{\otimes n}m$.

(5) Between two blocks: $o_{\pi_E(Q)}=\pm\sum_P\mathcal G(\mathbf a)_{QP}o_P$, $\mathcal G=\bigotimes G(a_q)$ — local maps, then the CNOT ring (Clifford).

(6) $\mathbb E[o_{\pi_E(Q)}(\mathbf a)o_{\pi_E(Q)}(\mathbf a')]=\sum_P\mathcal G(\mathbf a)_{QP}\mathcal G(\mathbf a')_{QP}m_P$ — the input is diagonal by (i); $(\pm)^2=1$.

(7) $\mathcal G(\mathbf a)_{QP}\mathcal G(\mathbf a')_{QP}=\prod_qG(a_q)_{Q_qP_q}G(a'_q)_{Q_qP_q}$ — tensor product. (4)–(7) give (ii).

(8) (iii): $Z_j\mapsto c^2Z_j+sp\,I$ (Theorem 2 step (1)), then block $L$ by (4).

(9) (iv): $\langle0^n\rvert P\lvert0^n\rangle=[P\in\{I,Z\}^n]$, together with (i). □

**Theorem 15.** $\sigma^2(s)=\sum_{m\ge0}s^{2m}B_m$, with $B_m\ge0$ independent of $s$, and $B_0=\sigma^2_{\rm tw}$.

*Proof.*

(1) $S(\mathbf a,\mathbf a')=\sum_\Gamma w_\Gamma u_\Gamma(\mathbf a)u_\Gamma(\mathbf a')$ over paths $\Gamma$ of Pauli strings. Here $w_\Gamma$ is a product of entries of $\mathsf Q$ (so $w_\Gamma\ge0$), and $u_\Gamma$ is the product of entries of $G$ along the path, including the initial factor — expansion of Lemma 14 (ii), (iii).

(2) $\sigma^2=\mathbb E_\mu S(\mathbf a,\mathbf a)-\mathbb E_{\mu\otimes\mu}S(\mathbf a,\mathbf a')$ — definition of the variance.

(3) $=\sum_\Gamma w_\Gamma\bigl[\mathbb Eu_\Gamma^2-(\mathbb Eu_\Gamma)^2\bigr]=\sum_\Gamma w_\Gamma\mathrm{Var}_\mu u_\Gamma$ — (1), (2).

(4) $G_{II}=1$, $G_{cI}=0$ ($c\ne I$), $G_{cb}=K_{cc}A_{cb}K_{bb}$ ($b,c\ne I$, free of $s$), $G_{Ib}=sp(\delta_{bZ}+A_{Zb}K_{bb})$ ($b\ne I$) — compute $K_sAK_s$ using $A_{cI}=A_{ZI}=0$.

(5) $u_\Gamma=s^{m(\Gamma)}\tilde u_\Gamma$, where $m(\Gamma)$ counts the uses of $G_{Ib}$ ($b\ne I$), plus 1 if the path starts with the factor $sp$ — (4).

(6) $\sigma^2=\sum_ms^{2m}B_m$ with $B_m=\sum_{m(\Gamma)=m}w_\Gamma\mathrm{Var}\,\tilde u_\Gamma\ge0$ — (3), (5).

(7) $s=0$ keeps only the paths with $m=0$, so $B_0=\sigma^2_{\rm tw}$. □

**Theorem 16 (last-layer floor).**
$$
B_1\ge\sigma^2_\infty:=p^2(1-p)^2\,\frac1n\sum_j\Bigl[\frac{(1-p)^2}{2}\mathrm{Var}_\mu\cos a_j+\frac{1-p}{4}\mathrm{Var}_\mu\sin a_j\Bigr],
\qquad\sigma^2(s)\ge\sigma^2_{\rm tw}+s^2\sigma^2_\infty\ \ \forall L .
$$

*Proof.*

(1) $B_1\ge$ the sum over the paths that branch into $I^{\otimes n}$ at the local map of the last layer on qubit $j$ — Theorem 15 (6); the other terms are nonnegative.

(2) After block $L$: the string has $b\in\{X,Y,Z\}$ at $j$, with $w=\mathsf Q_{bZ}/n$ and a factor $1-p$ in $u$ — Lemma 14 (iii).

(3) $G_{IZ}(a)=sp[1+(1-p)\cos a]$, $G_{IY}(a)=sp\,c\sin a$, $G_{IX}=0$ — Theorem 15 (4).

(4) $I^{\otimes n}$ keeps weight 1 to the end — $G_{II}=\mathsf Q_{II}=1$; $\Pi_E$ fixes $I^{\otimes n}$.

(5) Contribution $=\tfrac1n\sum_j\bigl[\mathsf Q_{ZZ}(1-p)^2s^2p^2(1-p)^2\mathrm{Var}\cos a_j+\mathsf Q_{YZ}(1-p)^2s^2p^2(1-p)\mathrm{Var}\sin a_j\bigr]$ — (2)–(4).

(6) $\mathsf Q_{ZZ}=\tfrac12$, $\mathsf Q_{YZ}=\tfrac14$ give exactly $s^2\sigma^2_\infty$ — Lemma 13.

(7) Combine with $B_0=\sigma^2_{\rm tw}$ and $B_{m\ge2}\ge0$ — Theorem 15. □

**Assumption G.** For generic $\mathbf a$, $T_{\mathbf a}=\mathsf Q^{\otimes n}\Pi_E\bigotimes_qA(a_q)^{\odot2}$, restricted to the $4^n-1$ non-identity strings, is primitive. *(Not proved for general $n$; checked numerically for $n\le5$.)*

**Theorem 17 (noiseless limit).** Assume G. Then $\lim_{L\to\infty}\sigma^2_{\rm None}=\dfrac1{2^n+1}$ if $\mu$ has no atoms, and $=\dfrac{1-1/M}{2^n+1}$ if $\mu$ is the empirical distribution of $M$ distinct generic inputs.

*Proof.*

(1) $p=0$: $G=A$, and $A$ is orthogonal, so $A^{\odot2}$ is doubly stochastic.

(2) $T_{\mathbf a}$ is doubly stochastic — (1), Lemma 13, $\Pi_E$ is a permutation.

(3) The initial mass on the non-identity strings is $\sum_b\mathsf Q_{bZ}=1$ — Lemma 14 (iii) with $p=0$.

(4) $T_{\mathbf a}^Lv\to(\mathbf 1^\top v/N)\mathbf 1$ with $N=4^n-1$ — Perron–Frobenius for a primitive doubly stochastic matrix.

(5) $S(\mathbf a,\mathbf a)\to(2^n-1)/(4^n-1)=1/(2^n+1)$ — Lemma 14 (iv), counting $\{I,Z\}^n\setminus\{I^{\otimes n}\}$.

(6) For $\mathbf a\ne\mathbf a'$: $|G(a)_{cb}G(a')_{cb}|\le|A(a)_{cb}||A(a')_{cb}|$, so $|m|\le(T^{\rm abs})^L|m^{(L)}|$ with $T^{\rm abs}=\mathsf Q^{\otimes n}\Pi_E\bigotimes|A(a_q)|\odot|A(a'_q)|$.

(7) The $Y$ column of $|A(a)|\odot|A(a')|$ sums to $|\cos a\cos a'|+|\sin a\sin a'|\le1$, with equality if and only if $a'\equiv\pm a\pmod\pi$ — Cauchy–Schwarz; the $Z$ column likewise; the $X$ column sums to 1.

(8) If $a'_q\not\equiv\pm a_q$ for some $q$, then $T^{\rm abs}$ is primitive (by G), has column sums $\le1$ and one column sum strictly below 1, so $\rho(T^{\rm abs})<1$ and $S(\mathbf a,\mathbf a')\to0$ — Perron–Frobenius.

(9) $\sigma^2=\mathbb ES(\mathbf a,\mathbf a)-\mathbb ES(\mathbf a,\mathbf a')$; with $M$ images, $\Pr[\mathbf a'=\mathbf a]=1/M$ — (5), (8). □

**Theorem 18 (contraction rate of the twirl).** Assume G. The leading eigenvalue of the single-input transfer matrix at $s=0$ is $\nu=1-2n'p+O(p^2)$, with $n'=n\,4^n/(4^n-1)$.

*Proof.*

(1) $G^{\odot2}_{cb}=\kappa_cA_{cb}^2\kappa_b$ with $\kappa=(1,1-p,1-p,(1-p)^2)$ — $K_0$ is diagonal.

(2) $\kappa=\mathbf 1-p\,\delta+O(p^2)$ with $\delta=(0,1,1,2)$.

(3) $T(p)=T(0)-pT_1+O(p^2)$, with $T_1=\mathsf Q^{\otimes n}\Pi_E\sum_q(\Delta_qA^{\odot2}+A^{\odot2}\Delta_q)\otimes\bigotimes_{r\ne q}A^{\odot2}$ — (1), (2).

(4) $T(0)$ has a simple eigenvalue 1, with left and right eigenvector $\mathbf 1$ — Theorem 17 (2), G.

(5) $\nu'(0)=-\mathbf 1^\top T_1\mathbf 1/\mathbf 1^\top\mathbf 1$ — first-order perturbation of a simple eigenvalue.

(6) $\mathbf 1^\top T_1\mathbf 1=\sum_q2\,\mathbf 1^\top\Delta_q\mathbf 1$ — double stochasticity: $\mathbf 1^\top\mathsf Q^{\otimes n}\Pi_E=\mathbf 1^\top$, $(\bigotimes A^{\odot2})\mathbf 1=\mathbf 1$.

(7) $\mathbf 1^\top\Delta_q\mathbf 1=4^{n-1}(1+1+2)=4^n$ — counting strings by their Pauli at $q$.

(8) $\nu'(0)=-2n\,4^n/(4^n-1)$ — (5)–(7). □

**Proposition 19 (separation depth).** Assume (A1) $\sigma^2_{\rm tw}(L)=\sigma^2_{\rm None}\nu^L(1+o(1))$ and (A2) $\sigma^2_{\rm br}(L)\to\sigma^2_{\rm br,\infty}$. Then
$$
L^*(R_0)\simeq\Bigl\lceil\frac{\ln[(R_0-1)\sigma^2_{\rm None}/\sigma^2_{\rm br,\infty}]}{-\ln\nu}\Bigr\rceil,\qquad
L^*(10)\lesssim\frac1{-\ln\nu}\ln\frac{9}{(2^n+1)s^2\sigma^2_\infty}\approx\frac1{n'p}\Bigl[\ln\frac1p+\ln\frac{3}{|s|(1-p)\kappa_\mu\sqrt{2^n+1}}\Bigr],
$$
with $\sigma_\infty=p(1-p)\kappa_\mu$.

*Proof.*

(1) $\sigma^2_{\rm AD}/\sigma^2_{\rm tw}\ge R_0\iff\sigma^2_{\rm br}\ge(R_0-1)\sigma^2_{\rm tw}$ — Theorem 15.

(2) Insert A1, A2: $(R_0-1)\sigma^2_{\rm None}\nu^L\le\sigma^2_{\rm br,\infty}$.

(3) Take logarithms, divide by $-\ln\nu>0$, round up.

(4) Upper bound: $\sigma^2_{\rm br,\infty}\ge s^2\sigma^2_\infty$ (Theorem 16) and $\sigma^2_{\rm None}=1/(2^n+1)$ (Theorem 17).

(5) $-\ln\nu=2n'p+O(p^2)$ — Theorem 18. □

---

## 5. Not yet proved

1. Assumption G for general $n$ (checked numerically for $n\le5$).
2. The asymptotic assumptions A1, A2 of Proposition 19 (checked numerically).
3. Tighter bounds for Theorem 5 and Corollary 1.2: both hold but are loose by about $10^2$–$10^3$.
4. Closed form of the $b_{\rm br}\,p$ part of $\sigma^2_{\rm br,\infty}$.
5. All statements in Parts A and B hold at fixed or arbitrary parameters; none is yet stated for optimized parameters, except Corollary 2.2.
