#!/usr/bin/env bash
# Amendments 20, 21 and 23 on the pod.
#
#   bash scripts/launch_v57.sh gate  J    # quick checks, J in parallel, in the foreground.
#                                         # Every re-execution must be bit-identical; prints PASSED/FAILED.
#   bash scripts/launch_v57.sh wave1 J    # only after the gate PASSED; runs in the background
#   bash scripts/launch_v57.sh wave2 J    # after wave 1 has ended: unclipped re-executions
#
# Refuses to start if tracked files are modified (every shard records the commit).
# Never `git pull` while a wave is running. Progress: tail -f logs_v57/status.txt
# Re-running a mode resumes it: runs whose shard exists are skipped.
set -euo pipefail
MODE="${1:?usage: launch_v57.sh gate|wave1|wave2 J}"
J="${2:?usage: launch_v57.sh gate|wave1|wave2 J}"
cd "$(git rev-parse --show-toplevel)"
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
    echo "Tracked files are modified. Commit or restore them first:"; git status --short; exit 1
fi
if pgrep -f "scripts/replay.py" > /dev/null; then
    echo "Replay jobs are still running:"; pgrep -af "scripts/replay.py"; exit 1
fi
mkdir -p logs_v57
stamp=$(date +%Y%m%d_%H%M)
case "$MODE" in
  gate)
    python scripts/plan_v57_jobs.py --write logs_v57 | tee logs_v57/plan_wave1_$stamp.txt
    # Worktrees are created one at a time here; parallel jobs only read them.
    while read -r sha; do
        [[ -z "$sha" ]] && continue
        d="../replay_trees/${sha:0:12}"
        [[ -d "$d/src" ]] || git worktree add --detach "$d" "$sha"
    done < logs_v57/commits_wave1.txt
    echo "$(date '+%F %T') GATE START J=$J sha=$(git rev-parse --short=8 HEAD)" >> logs_v57/status.txt
    xargs -a logs_v57/jobs_gate.txt -d '\n' -P "$J" -n 1 bash
    echo; echo "=== gate results ==="
    fail=0
    for f in logs_v57/gate_*.log; do
        r=$(grep -h "QUICK CHECK" "$f" || echo "NO RESULT (see $f)")
        echo "$(basename "$f" .log): $r"
        [[ "$r" == *PASSED* ]] || fail=1
    done
    if [[ $fail == 0 ]]; then echo "GATE PASSED - launch wave 1: bash scripts/launch_v57.sh wave1 $J"
    else echo "GATE FAILED - do not launch; send logs_v57/gate_*.log"; exit 1; fi
    ;;
  wave1|wave2)
    w=${MODE#wave}
    python scripts/plan_v57_jobs.py --write logs_v57 --wave "$w" | tee logs_v57/plan_wave${w}_$stamp.txt
    while read -r sha; do
        [[ -z "$sha" ]] && continue
        d="../replay_trees/${sha:0:12}"
        [[ -d "$d/src" ]] || git worktree add --detach "$d" "$sha"
    done < logs_v57/commits_wave${w}.txt
    echo "$(date '+%F %T') WAVE $w LAUNCH J=$J sha=$(git rev-parse --short=8 HEAD) jobs=$(wc -l < logs_v57/jobs_wave${w}.txt)" \
        >> logs_v57/status.txt
    nohup bash -c "xargs -a logs_v57/jobs_wave${w}.txt -d '\n' -P $J -n 1 bash; \
        echo \"\$(date '+%F %T') WAVE $w ENDED\" >> logs_v57/status.txt" > logs_v57/launcher_wave${w}.log 2>&1 &
    echo "wave $w launched in the background (pid $!). Progress: tail -f logs_v57/status.txt"
    ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
