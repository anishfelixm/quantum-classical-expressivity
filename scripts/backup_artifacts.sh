#!/usr/bin/env bash
# Everything needed to rebuild every table and figure, and to rerun any
# frozen-backbone experiment without a GPU cluster.
#   bash scripts/backup_artifacts.sh
set -u
cd "$(dirname "$0")/.."
D=$(date +%F); OUT="backup_$D"; mkdir -p "$OUT"

echo "== sizes ==";            du -sh artifacts/* 2>/dev/null | sort -h | tee "$OUT/sizes.txt"
echo "== free disk ==";        df -h . | tee -a "$OUT/sizes.txt"

RESULTS=()
for p in artifacts/shards artifacts/predictions artifacts/family_table_cache \
         artifacts/exploratory_cache artifacts/lr_selection.json \
         artifacts/family_table.json artifacts/family_table.tex \
         artifacts/exploratory_table.json; do
    [[ -e "$p" ]] && RESULTS+=("$p")
done
LOGS=( logs_*.txt )

echo "== results archive (shards, predictions, bootstrap caches, tables, logs) =="
tar czf "$OUT/results_$D.tgz" "${RESULTS[@]}" "${LOGS[@]}"
echo "== feature cache archive (lets frozen-backbone runs restart without the cluster) =="
[[ -d artifacts/feature_cache ]] && tar czf "$OUT/feature_cache_$D.tgz" artifacts/feature_cache

git rev-parse HEAD > "$OUT/git_head.txt"
( cd "$OUT" && sha256sum *.tgz > SHA256SUMS.txt && ls -lh )
echo "Done. Copy the whole $OUT/ folder off the cluster, then verify SHA256SUMS.txt."
