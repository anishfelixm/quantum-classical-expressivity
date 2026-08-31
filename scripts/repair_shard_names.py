"""
Repair the shard directory after over-aggressive archiving.

WHAT WENT WRONG
---------------
archive_legacy_shards.py moved every shard whose filename did not match
shards.shard_name(experiment, **its_keys). That rule is correct for detecting a
stale naming convention, but wrong as a reason to archive: almost all of those
files were the ONLY record of their keys. They were valid, unique data that
happened to be written before the naming was standardised.

Only a file with a CANONICAL TWIN is a duplicate. Everything else should have
been renamed, not removed.

    01_frozen           1,800 moved, 2 were duplicates
    06_premise            144 moved, 0 were duplicates

WHAT THIS DOES
--------------
For every archived file, recompute the canonical name from its own keys:

    canonical twin exists in the live directory
        -> genuine duplicate; LEAVE IT ARCHIVED. The live copy is newer and,
           in the two known cases, post-dates Amendment 2 (checkpoint selection
           changed from Macro-F1 to AUC), so the archived one is not merely
           redundant but incomparable.

    no twin
        -> restore it AND rename it to the canonical name, so the directory
           ends up with one naming convention and no data loss.

Predictions are keyed independently and were never touched, so a restored shard
still finds its own prediction file.

USAGE
-----
    python scripts/repair_shard_names.py            # report only
    python scripts/repair_shard_names.py --apply
"""
import argparse
import json
import os
import shutil
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import config      # noqa: E402
import shards      # noqa: E402

ARCHIVE = os.path.join(config.SHARD_DIR, "_legacy_naming")


def plan(experiment):
    """Returns (to_restore, true_duplicates) for one archived namespace."""
    src = os.path.join(ARCHIVE, experiment)
    live = os.path.join(config.SHARD_DIR, experiment)
    if not os.path.isdir(src):
        return [], []

    restore, duplicates = [], []
    for fn in sorted(os.listdir(src)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(src, fn)
        try:
            keys = json.load(open(path)).get("keys", {})
        except Exception as e:
            print(f"  UNREADABLE {fn}: {e}")
            continue

        canonical = shards.shard_name(experiment, **keys)
        if os.path.exists(os.path.join(live, canonical)):
            duplicates.append((fn, canonical))
        else:
            restore.append((fn, canonical))
    return restore, duplicates


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--apply", action="store_true")
    args = p.parse_args()

    if not os.path.isdir(ARCHIVE):
        print("Nothing archived. Nothing to repair.")
        return

    names = sorted(n for n in os.listdir(ARCHIVE)
                   if os.path.isdir(os.path.join(ARCHIVE, n)))

    total_restore, total_dup = 0, 0
    for exp in names:
        restore, duplicates = plan(exp)
        if not restore and not duplicates:
            continue

        print(f"\n=== {exp} ===")
        print(f"  restore + rename : {len(restore):>5d}   (unique data, old filename)")
        print(f"  keep archived    : {len(duplicates):>5d}   (canonical twin exists)")
        for fn, canonical in duplicates[:5]:
            print(f"      dup: {fn}")

        total_restore += len(restore)
        total_dup += len(duplicates)

        if args.apply and restore:
            live = os.path.join(config.SHARD_DIR, exp)
            os.makedirs(live, exist_ok=True)
            for fn, canonical in restore:
                shutil.move(os.path.join(ARCHIVE, exp, fn),
                            os.path.join(live, canonical))
            print(f"  restored to {live}")

    print(f"\n{total_restore} file(s) to restore, {total_dup} genuine duplicate(s).")
    if not args.apply:
        print("Nothing changed. Re-run with --apply.")
    else:
        print("Done. Re-run the analyses that touched these namespaces.")
        print("Every filename now matches shards.shard_name(), so "
              "archive_legacy_shards.py should report zero.")


if __name__ == "__main__":
    main()
