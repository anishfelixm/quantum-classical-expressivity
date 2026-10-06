#!/usr/bin/env bash
# Amendment 17 - the last run on the cluster. Classical only, ~1 hour.
#   nohup bash scripts/run_capacity_followup.sh > logs_capacity_followup.txt 2>&1 &
set -u
cd "$(dirname "$0")/.."
if [[ -n "$(git status --porcelain -- src scripts docs)" ]]; then
    echo "Uncommitted changes in src/, scripts/ or docs/. Commit and push first:"
    git status --short -- src scripts docs; exit 1
fi
echo "H-S5 follow-up sha=$(git rev-parse --short HEAD) started $(date)" | tee -a logs_gap_status.txt
PY="python -u src/01_frozen_backbone_ablation.py"
CONF=$(python -c "import sys; sys.path.insert(0,'src'); import config; print(' '.join(map(str, config.CONFIRMATORY_SEEDS)))")
for R in 0 8; do
    echo "[$(date '+%m-%d %H:%M')] START H-S5 follow-up rank $R" | tee -a logs_gap_status.txt
    if $PY --arms low_rank --head-rank $R --dims 4 --seeds $CONF --lr-head 1e-2 \
           --experiment 27_capacity_tuned > logs_hs5f_r$R.txt 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] DONE  H-S5 follow-up rank $R" | tee -a logs_gap_status.txt
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL  H-S5 follow-up rank $R (see logs_hs5f_r$R.txt)" | tee -a logs_gap_status.txt
    fi
done
echo "H-S5 follow-up finished $(date)" | tee -a logs_gap_status.txt
