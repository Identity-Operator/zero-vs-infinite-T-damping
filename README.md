# Zero- versus infinite-temperature damping in variational quantum circuits

Code, result files and LaTeX source for

> V.-Q.-M. Nguyen, T.-V. Truong, H.-L. Nguyen, and T.-K. Le,
> *Zero- Versus Infinite-Temperature Damping in Variational Quantum Circuits:
> Feature Scale, Sampling Cost, and Frame Gauge* (2026).

Every number and figure in the paper is computed from the files here. The
figures are regenerated from the committed results in a few seconds; the
results themselves can be recomputed with the scripts below.

The paper compares amplitude damping (AD) with its Pauli twirl, which is
generalized amplitude damping at infinite temperature, in single-qubit
re-uploading fits, a four-qubit MNIST classifier, and a three-qubit
variational eigensolver. The first and third follow the setups of van Rossum
et al., [Quantum Sci. Technol. 11, 025047 (2026)](https://doi.org/10.1088/2058-9565/ae636a).

## Layout

```
paper/v5_pra/                  manuscript (scale_gauge_v5.tex, refs.bib), figures/, make_figures_v5.py
code/noise_structure_mnist/    the classifier
    torch_circ.py              density-matrix simulator (torch)
    mnist_hybrid_multi.py      data loading, preprocessing, training
    run_phase1.py              driver for every trained result, one stage per call
    check_stage.py, test_*.py  checks and tests
    m5_*/                      trained results (results.pkl, baselines.pkl)
code/scale_gauge/
    vr/                        single-qubit fits
    vqe_gauge/                 eigensolver
    gauge/                     frame gauge: proposition, proof, exhaustive frame search
    theory/                    exact average over parameters: feature scale, separation depth, finite temperature
    shots/                     finite-shot evaluation and operational shot cost
    scale_e1e2/                output-scale runs at L = 4 and an independent density-matrix simulator
    verify/                    independent re-implementations and the statistics of App. C
```

The folder names `noise_structure_mnist` and `v5_pra` are historical; the
scripts refer to each other by these paths, so they are kept.

## From results to the paper

| Paper | Produced by | Result files |
|---|---|---|
| Fig. 1 | drawn in LaTeX (quantikz) | — |
| Fig. 2 | `code/scale_gauge/vr/vr_test.py` | `vr/vr_results.pkl` |
| Fig. 3, Table II | `run_phase1.py depth` (raw readout, 100 steps), `run_phase1.py readout` (standardized readout); `scale_e1e2/scale_test.py` mode e2 (raw readout, 1000 steps, L = 4) | `m5_depth/`, `m5_readout/`, `scale_e1e2/scale_e2_*.pkl` |
| Fig. 4 | `vqe_gauge/vqe_gauge_test.py` | `vqe_gauge/vqe_gauge_results.pkl` |
| Sign of the coupling in the eigensolver | `vqe_gauge/vqe_sign_test.py`, `vqe_gauge/vqe_sign_identity.py` | `vqe_sign_results.pkl`, `vqe_sign_log.txt`, `vqe_sign_identity_log.txt` |
| Table III | `gauge/test_gauge.py` | `gauge/results.json` |
| Fig. 5 | `run_phase1.py width_diag`; `theory/t1_scale_ratio.py` (exact average) | `m5_width_diag/`, `theory/t1_results.json` |
| Fig. 6 | `run_phase1.py shots`; `shots/F1_calib.py` + `F1_calib_summary.py`; `shots/F1_s5.py` + `F1_s5_summary.py` | `m5_shots/`, `shots/F1_calib_summary.pkl`, `shots/F1_s5_summary.pkl` |
| Table IV, Fig. 7 | `theory/t1_p_dependence.py` | `theory/t1_p_dependence_n4.json`, `_n8.json` (logs in `.txt`) |
| Table V | `run_phase1.py pdep`, `run_phase1.py pdep_L8`; `shots/F1_pdep.py` | `m5_pdep/`, `m5_pdep_L8/`, `shots/F1_pdep_*.pkl` |
| Table VI, App. C | `verify/stats_paired_holm.py`, `verify/stats_appendix.py` | `verify/stats_paired_holm.txt` |
| Finite temperature | `theory/t1_gad_temperature.py`, `verify/gad_linearity_check.py` | the `.txt` files beside them |
| Reversed-damping null test | `run_phase1.py fixedpoint` | `m5_fixedpoint/` |
| Channel after every CNOT of the entangler | `run_phase1.py inside` | `m5_inside/` |
| Fashion-MNIST | `run_phase1.py fashion`, `run_phase1.py fashion_readout` | `m5_fashion/`, `m5_fashion_readout/` |
| Classical baselines | `run_phase1.py baselines` | `m5_baseline/` |
| Input sensitivity without the readout gain | `shots/F2_sensitivity.py` + `F2_summary.py` | `shots/F2_*.pkl` |

Independent checks, each sharing no simulation code with what it checks:
`verify/t1_brute_n4.py` and `t1_brute_n4_p.py` (density-matrix simulation
against the exact average), `theory/pauli_prop.py` (Pauli-path propagation),
`verify/timerev_gauge_check.py` and `gauge/trsym.py` (time-reversal gauge),
`gauge/euler_map.py` (the parameter map of Corollary 1),
`scale_e1e2/indep_sim.py` and `shots/check_lib.py` (second simulator of the
classifier), and `verify/gauge_vqe_check.py` and `vqe_lbfgs_floor.py`
(eigensolver).

## Reproducing

Linux or macOS (the training driver locks its results file with `fcntl`),
Python 3.12. From the repository root:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Figures and paper** (seconds; reads the committed results):

```bash
cd paper/v5_pra
python make_figures_v5.py
pdflatex scale_gauge_v5 && bibtex scale_gauge_v5 && pdflatex scale_gauge_v5 && pdflatex scale_gauge_v5
```

**Exact average over parameters** (CPU; times and peak memory from the logs):

```bash
cd code/scale_gauge/theory
python t1_p_dependence.py 4 5000        # about 6 min, 1.2 GB
python t1_p_dependence.py 8 700 0.01    # about 28 min, up to 3.6 GB
```

**Training** (the `m5_*` results). Run from `code/noise_structure_mnist`:
torchvision downloads MNIST and Fashion-MNIST into `./data` relative to the
working directory.

```bash
cd code/noise_structure_mnist
python test_preprocessing.py
python run_phase1.py depth --L 4 --outdir my_depth
```

- The initial parameters are drawn from the CUDA generator, so the committed
  runs are reproduced exactly only on a CUDA GPU (ours: NVIDIA RTX 3060,
  12 GB). `--device cpu` runs the simulation on the CPU but still draws the
  parameters on the GPU.
- A CPU worker at L = 4 peaks at about 3.4 GB of RAM, because autograd keeps
  every intermediate density matrix.
- The driver skips configurations already present in its output directory,
  and it writes into `m5_*` only from a working tree with committed `.py`
  files. Use `--outdir` to recompute into a new directory.
- The `git_commit` and `code_sha` fields of the committed records refer to the
  development repository. For this release, comments and docstrings were
  edited and hard-coded paths made relative; the simulation and training code
  is unchanged.

**Cross-check against the reference eigensolver code.** Two scripts
(`verify/vqe_crosscheck_qiskit.py`, `vqe_gauge/vqe_crosscheck_qiskit_torch.py`)
compare our eigensolver with the Qiskit implementation of van Rossum et al.,
which is not redistributed here. Download `TFIM_VQE_Noise.py` from their
deposit, [doi:10.6084/m9.figshare.30123904](https://doi.org/10.6084/m9.figshare.30123904),
into `code/old_code/`. These two scripts need `qiskit` and `qiskit-aer`
(listed in `requirements.txt`); the other scripts do not.

## License

- Code (`*.py`, `*.sh`): MIT, see [LICENSE](LICENSE).
- Result files and figures (`m5_*/`, `*.pkl`, `*.npz`, `*.npy`, `*.json`,
  the `.txt` and `.log` run logs, `paper/v5_pra/figures/`): CC BY 4.0, see
  [LICENSE-CC-BY-4.0.txt](LICENSE-CC-BY-4.0.txt).
- The manuscript (`paper/v5_pra/scale_gauge_v5.tex`, `.bbl`, `.pdf`,
  `refs.bib`) and `code/scale_gauge/gauge/gauge_proposition.tex` are not
  covered by these licenses; they remain with the authors. Cite the paper.

## Citation

Citation metadata, including the authors' ORCID iDs, is in
[CITATION.cff](CITATION.cff).

```bibtex
@article{nguyen2026damping,
  title  = {Zero- Versus Infinite-Temperature Damping in Variational Quantum Circuits:
            Feature Scale, Sampling Cost, and Frame Gauge},
  author = {Nguyen, Vu-Quoc-Minh and Truong, Tuan-Vu and Nguyen, Hoang-Long and Le, Trung-Khanh},
  year   = {2026}
}
```

## Contact

Trung-Khanh Le (corresponding author), ltkhanh@hcmus.edu.vn
