#!/usr/bin/env bash
# Overnight Phase 1 queue.
#
# Every step checkpoints per configuration, so rerunning this script resumes
# where it stopped. Every check fails fast: the first nonzero exit ends the
# queue (set -e), so Q6 runs only if the preprocessing tests, Q1-Q5 and all
# checks passed. Lines starting with "@@" mark step boundaries in the log.
#
# Launch, or resume after an interruption, from this directory:
#   systemd-run --user --unit=m5-phase1 --working-directory="$PWD" -E PYTHONUNBUFFERED=1 \
#       -p StandardOutput=append:"$PWD/phase1_queue.log" \
#       -p StandardError=append:"$PWD/phase1_queue.log" /bin/bash run_queue.sh
# A failed unit stays listed; `systemctl --user reset-failed m5-phase1` clears it.
set -euo pipefail
cd "$(dirname "$0")"
PY="$(git rev-parse --show-toplevel)/.venv/bin/python"

mark() { echo "@@ $(date '+%F %T') $*"; }
trap 'mark "FAILED at line $LINENO; queue stopped"' ERR

mark "START T0 preprocessing tests";  $PY test_preprocessing.py
mark "DONE T0"
mark "START Q1 baselines";    $PY run_phase1.py baselines;       $PY check_stage.py baselines
mark "DONE Q1"
mark "START Q2 diagnostics";  $PY run_phase1.py diagnostics;     $PY check_stage.py diagnostics
mark "DONE Q2"
mark "START Q3 depth L=4";    $PY run_phase1.py depth --L 4;     $PY check_stage.py depth --L 4
mark "DONE Q3"
mark "START CHECK_A";         $PY check_stage.py checkA
mark "DONE CHECK_A"
mark "START Q4 fixedpoint";   $PY run_phase1.py fixedpoint;      $PY check_stage.py fixedpoint
mark "DONE Q4"
mark "START Q5 depth L=1-3";  $PY run_phase1.py depth --L 1 2 3; $PY check_stage.py depth --L 1 2 3
mark "DONE Q5"
mark "START Q6 fashion";      $PY run_phase1.py fashion;         $PY check_stage.py fashion
mark "DONE Q6"
mark "QUEUE COMPLETE"
