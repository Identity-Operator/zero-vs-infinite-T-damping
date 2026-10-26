# Remaining simulations for the gauge paper: instructions

*For Minh and the coding agent in his environment. Written 2026-10-26.*

## What this is

We are testing whether a result about noisy QAOA survives beyond depth one. Real qubits leak energy toward |0⟩ (amplitude damping, AD). In QAOA, relabeling the problem by flipping chosen qubits (a "gauge" y) is equivalent to reversing the damping on those qubits. At depth one we have proved that the energy of each gauge is the energy of the Pauli-twirled circuit plus a² times ⟨y|H|y⟩, so the gauge landscape *is* the problem Hamiltonian. These simulations check numerically whether this still holds at depths 2 and 3, and whether choosing the best gauge beats the twirled circuit as the qubit number grows. The theory is in `damping_gauge_theory_v3_en.md` (repo root); you do not need it to run anything.

Tiers 1 (n = 8) and 2 (n = 10) are **finished**; their results are in the repo. **Your job is tier 3 (n = 12)**, which was started but not finished, and then a summary.

## Branch

Everything below lives on the **`hoang`** branch, not on `main`:

```bash
git clone git@github.com:Identity-Operator/zero-vs-infinite-T-damping.git
cd zero-vs-infinite-T-damping
git checkout hoang
```

Commit and push your results to `hoang` as well. Do not merge into `main`.

## Files

All code and results are in `code/gauge_numerics/`:

| File | Role |
|---|---|
| `full.py` | main driver: `python3 full.py TIER NWORKERS` |
| `simi.py`, `simf.py`, `sim12.py` | exact density-matrix simulators (numpy) |
| `agg.py` | prints the summary table from all `out_tier*.json` present |
| `out_tier1.json`, `out_tier2.json` | finished results for n = 8 and n = 10 — **do not overwrite** |
| `tier1.log`, `tier2.log` | logs of those runs |

## Environment

- Python 3.9 or newer, with numpy (tested 1.26.4) and scipy (tested 1.13.1). Nothing else.
- Each worker is single-threaded (the script sets `OMP_NUM_THREADS=1` etc.). Use one worker per physical core you can spare.
- Memory: a 12-qubit density matrix in complex64 is about 130 MB; budget about 1 GB per worker.

## What to run

```bash
cd code/gauge_numerics
python3 full.py 3 NWORKERS 2>&1 | tee tier3.log
```

Set `NWORKERS` to the number of free cores (5 was used on a 12-core laptop). Tier 3 is 24 points: 6 graphs (3 Gaussian-weighted, 3 three-regular MaxCut) × depth P ∈ {2, 3} × damping p ∈ {0.05, 0.1}, about 81 gauges per point, with QAOA parameters optimized for the twirled circuit. The graphs come from a fixed seed (303), so the instances are the same every time.

**Expected time:** roughly 30–60 minutes per point per worker, so about 3–5 hours with 5 workers, less with more cores. Run it in `tmux`, `screen` or `nohup` so it survives a closed terminal.

**Saving:** after every finished point, `out_tier3.json` is rewritten with all points so far, and `"complete": true` is added at the end. Partial results are usable. **There is no resume:** if the run is killed, restarting starts tier 3 from scratch (same instances). If time is short, it is fine to stop after some points finish and report what you have.

## Sanity checks before trusting the output

1. **Quick smoke test.** Before the full run, make sure everything imports: `python3 -c "import simi, simf, sim12"` inside `code/gauge_numerics`.
2. **The tier-3 log line** for each point looks like `12 <graph> <family> <P> <p> opt pe=... gs=... gain=... pg=...`. `pe` should be a correlation between −1 and 1 (we expect about 0.9–1.0), and `gs` is 1 if the best gauge is a ground state.
3. **Running `python3 agg.py`** must still show the unchanged n = 8 and n = 10 rows. If they differ from the table below, stop and report.

## What to report back

When tier 3 finishes (or you stop it):

1. Commit `out_tier3.json` and `tier3.log` to `code/gauge_numerics/` and push.
2. Run `python3 agg.py > SUMMARY.txt`, commit it to `hoang`, and paste the n = 12 rows in your message.
3. In a short note (a few sentences), say: how many of the 24 points finished; whether the n = 12 rows follow the trend below; anything that looked wrong (negative `pe` at the optimized point, crashes, memory errors, very different run times).

Please do not change `full.py`'s configuration (graph seeds, gauge counts, restarts) without asking. The point is to stay comparable to tiers 1 and 2.

## Results so far (for comparison)

At parameters optimized for the twirled circuit. "Correlation" is the median Pearson correlation between E(y) − E_twirl and ⟨y|H|y⟩ over gauges. "GS rate" is how often the best gauge is a ground state. "GS-prob ratio" is the best gauge's ground-state probability over the twirled circuit's (median).

| n | P | p | correlation (min) | GS rate | GS-prob ratio |
|---|---|---|---|---|---|
| 8 | 2 | 0.05 | 0.989 (0.974) | 1.00 | 1.20 |
| 8 | 2 | 0.10 | 0.987 (0.975) | 1.00 | 1.83 |
| 8 | 3 | 0.05 | 0.946 (0.823) | 0.90 | 1.28 |
| 8 | 3 | 0.10 | 0.947 (0.696) | 0.80 | 2.16 |
| 10 | 2 | 0.05 | 0.991 (0.965) | 1.00 | 1.36 |
| 10 | 2 | 0.10 | 0.991 (0.971) | 1.00 | 2.69 |
| 10 | 3 | 0.05 | 0.971 (0.889) | 0.88 | 1.49 |
| 10 | 3 | 0.10 | 0.975 (0.941) | 0.88 | 3.15 |

The question tier 3 answers: does the correlation stay high and does the GS-prob ratio keep growing at n = 12?

Questions: ask Hoang-Long.
