#!/usr/bin/env bash
# Closes every remaining gap, in priority order.
#
#   nohup bash scripts/run_gap_closure.sh > logs_gap_closure.txt 2>&1 &
#
# Specified in docs/analysis_plan.md, Amendments 13 and 14, written BEFORE any
# of these runs. Each step logs to its own file and records its outcome in
# logs_gap_status.txt, so one failure does not hide the others.
#
#   H-S2  fourier_rff into the confirmatory namespace       ~1.5 h
#   H-S3  noise v2: native resolution, declared rates        ~5 h
#   H-S1  quantum_reupload into the confirmatory namespace  ~10 h
#   H-S4  frozen + adaptive encoder, paired on seed          ~20 h
#   E1    ansatz check (exploratory)                         ~5 h
#
# DO NOT `git pull` while this is running - every shard records the git SHA,
# and pulling mid-run splits one experiment across two commits.
set -u
cd "$(dirname "$0")/.."

if [[ -n "$(git status --porcelain -- src scripts docs)" ]]; then
    echo "Uncommitted changes in src/, scripts/ or docs/. Commit and push first:"
    git status --short -- src scripts docs
    exit 1
fi
echo "gap closure sha=$(git rev-parse --short HEAD) started $(date)" | tee -a logs_gap_status.txt

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

# H-S2: the declared secondary arm, never run in the confirmatory namespace.
# Tuned rate 1e-2 gives the same key set as matched_param_fullrank, so it lands
# in the existing cells. Adds shards; touches none.
step "H-S2 fourier_rff" logs_hs2.txt \
    $PY --arms fourier_rff --dims 4 --seeds $CONF --use-tuned-lr --experiment 01_frozen_tuned

# H-S3: noise at native resolution; every arm at a tuned or inherited rate.
step "H-S3 noise v2" logs_noise_v2.txt \
    python -u src/03_robustness_evaluation.py --use-tuned-lr

# H-S1: re-uploading. Explicit 1e-2 reproduces quantum_vqc's tuned keys
# (lrh=1e-02, lrq=1e-02) exactly, so the two pair on all 40 seeds.
step "H-S1 reupload" logs_hs1.txt \
    $PY --arms quantum_reupload --dims 4 --seeds $CONF $LR --experiment 01_frozen_tuned

# H-S4: frozen and adaptive in ONE namespace so they pair on seed. No
# augmentation on either side, so freezing is the only difference.
step "H-S4 encoder" logs_hs4.txt \
    $PY --arms quantum_vqc matched_param_fullrank --dims 4 \
        --freeze-policies all layer3_only --use-tuned-lr --experiment 25_encoder

# E1: ansatz check, exploratory (Amendment 14).
step "E1 ansatz" logs_e1.txt \
    $PY --arms quantum_vqc quantum_basic --dims 4 $LR --experiment 21_ansatz

echo "gap closure finished $(date)" | tee -a logs_gap_status.txt
