"""
Find shards whose FILENAME does not match the canonical name derived from their
own keys, and move them out of the way.

WHY THESE EXIST
---------------
Early runs used ad-hoc filenames like

    01_frozen__breastmnist__r20__d4__s123__quantum_vqc__aug0__fpall.json

before shards.shard_name() became the single naming authority. The keys INSIDE
those files are correct, so nothing was wrong with them at the time. But when a
cell is later re-run - for instance under --force after Amendment 2 changed
checkpoint selection from Macro-F1 to AUC - the re-run writes a canonically
named file alongside the old one. Two files, identical keys, different results.

04_statistical_analysis.collect() now raises on that, which is how these were
found. It should raise: silently keeping one of two contradictory records is
exactly the failure the shard system exists to prevent.

THE RULE - CORRECTED
--------------------
A shard is legacy-NAMED if shards.shard_name(experiment, **its_own_keys)
differs from its filename. That detects a stale convention, but it is NOT on
its own a reason to remove the file.

    legacy name, canonical twin EXISTS  ->  genuine duplicate, ARCHIVE it
    legacy name, no twin                ->  unique data, RENAME to canonical

The first version of this script archived on the name alone and moved 2,844
files when 2 were duplicates: 1,800 diagnostic runs and the entire 144-cell
premise check were valid, unique results that simply predated the naming
standard. Nothing was lost - they were moved, not deleted - but the analyses
run immediately afterwards were computed against a gutted directory.

Archived files are MOVED, never deleted: they are provenance for what the
project believed at the time, and the two known duplicates predate Amendment 2,
which makes them incomparable rather than merely redundant.

USAGE
-----
    python scripts/archive_legacy_shards.py                    # report only
    python scripts/archive_legacy_shards.py --apply            # move them
    python scripts/archive_legacy_shards.py --experiment 01_frozen --apply
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


def audit(experiment):
    """Returns (legacy, canonical_collisions) for one experiment namespace."""
    d = os.path.join(config.SHARD_DIR, experiment)
    if not os.path.isdir(d):
        return [], []

    legacy, by_canonical = [], {}
    for fn in sorted(os.listdir(d)):
        if not fn.endswith(".json"):
            continue
        path = os.path.join(d, fn)
        try:
            rec = json.load(open(path))
        except Exception as e:
            print(f"  UNREADABLE {fn}: {e}")
            continue

        keys = rec.get("keys", {})
        canonical = shards.shard_name(experiment, **keys)
        entry = {"file": fn, "path": path, "canonical": canonical,
                 "sha": rec.get("git_sha", "?")[:8],
                 "ts": rec.get("timestamp", "?"),
                 "auc": (rec.get("metrics") or {}).get("auc")}
        by_canonical.setdefault(canonical, []).append(entry)

    collisions = {c: v for c, v in by_canonical.items() if len(v) > 1}

    # Only a legacy-named file that SHARES its key-set with another file is a
    # duplicate. A legacy-named file that is alone is unique data with an old
    # filename, and gets renamed rather than archived.
    for canonical, entries in collisions.items():
        for e in entries:
            if e["file"] != canonical:
                legacy.append(e)

    rename = [e for entries in by_canonical.values() if len(entries) == 1
              for e in entries if e["file"] != e["canonical"]]
    return legacy, collisions, rename


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--experiment", default=None,
                   help="one namespace; default is every namespace present")
    p.add_argument("--apply", action="store_true",
                   help="actually move the files (default is report only)")
    args = p.parse_args()

    names = ([args.experiment] if args.experiment else
             sorted(n for n in os.listdir(config.SHARD_DIR)
                    if os.path.isdir(os.path.join(config.SHARD_DIR, n))
                    and not n.startswith("_")))

    total_legacy, total_rename = 0, 0
    for exp in names:
        legacy, collisions, rename = audit(exp)
        if not legacy and not collisions and not rename:
            continue

        print(f"\n=== {exp} ===")
        if collisions:
            print(f"{len(collisions)} key-set(s) claimed by more than one file:")
            for canonical, entries in sorted(collisions.items())[:10]:
                print(f"  {canonical}")
                for e in sorted(entries, key=lambda x: x["ts"]):
                    tag = "legacy" if e["file"] != canonical else "canonical"
                    auc = f"{e['auc']:.4f}" if e["auc"] is not None else "?"
                    print(f"    [{tag:9s}] {e['ts']}  sha {e['sha']}  auc {auc}")
                    print(f"                {e['file']}")

        if legacy:
            total_legacy += len(legacy)
            print(f"\n{len(legacy)} DUPLICATE shard(s) to archive.")
            if args.apply:
                dest = os.path.join(ARCHIVE, exp)
                os.makedirs(dest, exist_ok=True)
                for e in legacy:
                    shutil.move(e["path"], os.path.join(dest, e["file"]))
                print(f"Archived to {dest}")

        if rename:
            total_rename += len(rename)
            print(f"{len(rename)} unique shard(s) with a legacy filename - "
                  f"these get RENAMED, not archived.")
            if args.apply:
                d = os.path.join(config.SHARD_DIR, exp)
                for e in rename:
                    os.replace(e["path"], os.path.join(d, e["canonical"]))
                print(f"Renamed in place.")

        if (legacy or rename) and not args.apply:
            print("Re-run with --apply.")

    if total_legacy == 0 and total_rename == 0:
        print("\nClean: every filename matches its own keys, no duplicates.")
    elif not args.apply:
        print(f"\n{total_legacy} duplicate(s) would be archived, "
              f"{total_rename} file(s) renamed. Nothing changed.")
    else:
        print(f"\n{total_legacy} duplicate(s) archived under {ARCHIVE}; "
              f"{total_rename} renamed in place.")
        print("No unique result is ever moved out of a live namespace.")


if __name__ == "__main__":
    main()
