"""Table of the readout-weighted noise gain G against the operational shot cost (appendix table tab:gain).
Reads F1_gain_check.pkl (G / G_None from the stored parameters) and F1_calib_summary.pkl (shot cost / None,
'calib' convention, as in Fig. fig_shots and Table tab:pdep_trained). No simulation.
Output: F1_gain_table.txt."""
import os, pickle
HERE = os.path.dirname(os.path.abspath(__file__))
G = {(r['dataset'], r['L'], r['channel']): r for r in pickle.load(open(os.path.join(HERE, 'F1_gain_check.pkl'), 'rb'))}
lines = []
for r in pickle.load(open(os.path.join(HERE, 'F1_calib_summary.pkl'), 'rb')):
    if r['channel'] == 'None': continue
    g = G[(r['dataset'], r['L'], r['channel'])]['gain_ratio']
    lines.append(f"{r['dataset']:12s} L={r['L']} {r['channel']:5s} cost/None {r['calib_ratio']:10.4g}  G/G_None {g:10.4g}  "
                 f"cost/G {r['calib_ratio'] / g:5.2f}  kappa {r['kappa']:10.4g}")
dev = max(max(x, 1 / x) for r in pickle.load(open(os.path.join(HERE, 'F1_calib_summary.pkl'), 'rb'))
          if r['channel'] != 'None' and r['L'] >= 2
          for x in [r['calib_ratio'] / G[(r['dataset'], r['L'], r['channel'])]['gain_ratio']])
lines.append(f"largest factor between cost and G for L >= 2: {dev:.2f}")
print('\n'.join(lines)); open(os.path.join(HERE, 'F1_gain_table.txt'), 'w').write('\n'.join(lines) + '\n')
