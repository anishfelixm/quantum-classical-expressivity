#!/usr/bin/env bash
# Amendments 20, 21, 23 and 25 on the pod.
#
#   bash scripts/launch_v57.sh gate     J   # quick checks, J in parallel, in the foreground.
#                                           # Every re-execution must be bit-identical; prints PASSED/FAILED.
#   bash scripts/launch_v57.sh wave1    J   # R_ and B_ runs (GPU), in the background
#   bash scripts/launch_v57.sh hardware J   # 33_hardware_tuned (Amendment 23), 8 threads per job
#   bash scripts/launch_v57.sh cpu17    1   # C_17_fulldata (Amendment 25), on CPU; use J = 1,
#                                           # as the original ran: one process at a time
#   bash scripts/launch_v57.sh wave2    J   # after wave 1 has ended: unclipped re-executions
#
# Different modes may run at the same time (watch memory: about 3.2 GB per job,
# 16 GiB in all); the same mode is never launched twice.
# Refuses to start if tracked files are modified (every shard records the commit).
# Never `git pull` while anything is running. Progress: tail -f logs_v57/status.txt
# Re-running a mode resumes it: runs whose shard exists are skipped.
set -euo pipefail
MODE="${1:?usage: launch_v57.sh gate|wave1|hardware|cpu17|wave2 J}"
J="${2:?usage: launch_v57.sh gate|wave1|hardware|cpu17|wave2 J}"
cd "$(git rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files are modified. Commit or restore them first:"; git status --short; exit 1
fi
mkdir -p logs_v57
stamp=$(date +%Y%m%d_%H%M)
make_trees() {
    while read -r sha; do
        [[ -z "$sha" ]] && continue
        d="../replay_trees/${sha:0:12}"
        [[ -d "$d/src" ]] || git worktree add --detach "$d" "$sha"
    done < "$1"
}
case "$MODE" in
  gate)
    if pgrep -f "scripts/replay.py" > /dev/null; then
        echo "Replay jobs are still running; the gate runs alone:"; pgrep -af "scripts/replay.py"; exit 1
    fi
    python scripts/plan_v57_jobs.py --write logs_v57 | tee logs_v57/plan_wave1_$stamp.txt
    make_trees logs_v57/commits_wave1.txt
    echo "$(date '+%F %T') GATE START J=$J sha=$(git rev-parse --short=8 HEAD)" >> logs_v57/status.txt
    xargs -a logs_v57/jobs_gate.txt -d '\n' -P "$J" -n 1 bash
    echo; echo "=== gate results ==="
    fail=0
    for f in logs_v57/gate_*.log; do
        r=$(grep -h "QUICK CHECK" "$f" || echo "NO RESULT (see $f)")
        echo "$(basename "$f" .log): $r"
        [[ "$r" == *PASSED* ]] || fail=1
    done
    if [[ $fail == 0 ]]; then echo "GATE PASSED"
    else echo "GATE FAILED - do not launch; send logs_v57/gate_*.log"; exit 1; fi
    ;;
  wave1|hardware|cpu17|wave2)
    case "$MODE" in
      wave1)    args="--kinds R,B --tag wave1" ;;
      hardware) args="--kinds H --tag hardware" ;;
      cpu17)    args="--kinds C --tag cpu17" ;;
      wave2)    args="--wave 2 --tag wave2" ;;
    esac
    if pgrep -f "logs_v57/jobs_${MODE}/" > /dev/null; then
        echo "Mode $MODE is already running:"; pgrep -af "logs_v57/jobs_${MODE}/"; exit 1
    fi
    python scripts/plan_v57_jobs.py --write logs_v57 $args | tee logs_v57/plan_${MODE}_$stamp.txt
    make_trees logs_v57/commits_${MODE}.txt
    n=$(grep -c . logs_v57/jobs_${MODE}.txt || true)
    if [[ "$n" == "0" ]]; then echo "nothing left to run in $MODE"; exit 0; fi
    echo "$(date '+%F %T') ${MODE^^} LAUNCH J=$J sha=$(git rev-parse --short=8 HEAD) jobs=$n" \
        >> logs_v57/status.txt
    nohup bash -c "xargs -a logs_v57/jobs_${MODE}.txt -d '\n' -P $J -n 1 bash; \
        echo \"\$(date '+%F %T') ${MODE^^} ENDED\" >> logs_v57/status.txt" > logs_v57/launcher_${MODE}.log 2>&1 &
    echo "$MODE launched in the background (pid $!). Progress: tail -f logs_v57/status.txt"
    ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
