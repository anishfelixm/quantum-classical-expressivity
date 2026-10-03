#!/usr/bin/env bash
# The last experiments in the project, specified in docs/analysis_plan.md,
# Amendment 16, written BEFORE these runs.
#
#   nohup bash scripts/run_followup.sh > logs_followup.txt 2>&1 &
#
#   E7    fourier_rff_r2 into the confirmatory namespace, 40 seeds    ~1 h
#   H-S6  follow-up, frozen PCA bottleneck, primary protocol           ~10 h
#   H-S6  follow-up, frozen random bottleneck, primary protocol        ~10 h
#
# Estimates from the 1 October runs on this machine: ~42 s per quantum run
# (H-S1, 800 runs in 9 h 23 min) and ~2.5 s per classical run (H-S2, 800 runs
# in 34 min), PathMNIST included.
#
# DO NOT `git pull` while this is running - every shard records the git SHA.
set -u
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain -- src scripts docs)" ]]; then
    echo "Uncommitted changes in src/, scripts/ or docs/. Commit and push first:"
    git status --short -- src scripts docs
    exit 1
fi

# E7 adds an arm to the confirmatory namespace. Back it up first, once per day.
BK="backup_01_frozen_tuned_$(date +%F).tgz"
[[ -f "$BK" ]] || tar czf "$BK" artifacts/shards/01_frozen_tuned artifacts/predictions/01_frozen_tuned
echo "follow-up sha=$(git rev-parse --short HEAD) started $(date)" | tee -a logs_gap_status.txt

PY="python -u src/01_frozen_backbone_ablation.py"
CONF=$(python -c "import sys; sys.path.insert(0,'src'); import config; print(' '.join(map(str, config.CONFIRMATORY_SEEDS)))")
LR="--lr-head 1e-2 --lr-quantum 1e-2"

step() {   # step <name> <logfile> <command...>
    local name="$1" log="$2"; shift 2
    echo "[$(date '+%m-%d %H:%M')] START $name" | tee -a logs_gap_status.txt
    if "$@" > "$log" 2>&1; then
        echo "[$(date '+%m-%d %H:%M')] DONE  $name" | tee -a logs_gap_status.txt
    else
        echo "[$(date '+%m-%d %H:%M')] FAIL  $name  (see $log)" | tee -a logs_gap_status.txt
    fi
}

# E7: same key set as fourier_rff (lrh=1e-02 only), so it pairs with
# quantum_reupload on all 40 confirmatory seeds. Adds shards; touches none.
step "E7 fourier_rff_r2" logs_e7.txt \
    $PY --arms fourier_rff_r2 --dims 4 --seeds $CONF --lr-head 1e-2 \
        --experiment 01_frozen_tuned

# H-S6 follow-up: the primary's protocol exactly - same arms, 40 confirmatory
# seeds, tuned 1e-2 - with only the bottleneck policy changed. Its OWN
# namespace: a bn key added to 01_frozen_tuned would collide with the learned
# shards there, which carry none. The learned reference IS 01_frozen_tuned.
step "H-S6 follow-up pca" logs_hs6f_pca.txt \
    $PY --arms quantum_vqc matched_param_fullrank --dims 4 --bottleneck pca \
        --seeds $CONF $LR --experiment 26_bottleneck_tuned
step "H-S6 follow-up random" logs_hs6f_random.txt \
    $PY --arms quantum_vqc matched_param_fullrank --dims 4 --bottleneck random \
        --seeds $CONF $LR --experiment 26_bottleneck_tuned

echo "follow-up finished $(date)" | tee -a logs_gap_status.txt
