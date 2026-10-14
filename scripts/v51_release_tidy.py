"""
v51 release tidy: move the conference-version files into legacy/conference/,
then check every factual claim in README.md that can be checked against the repo.

    python scripts/v51_release_tidy.py            # check, then move, then verify
    python scripts/v51_release_tidy.py --dry-run  # report only, change nothing

WHY
---
The public repository is what a reviewer opens. It still carried three
conference figures named after claims the journal study withdrew
("bottleneck gap", "precision paradox") and four conference training logs. They
are kept for provenance rather than deleted, but moved under legacy/conference/
with a note saying what they are, so the repository no longer presents them as
results.

The README was rewritten from scratch. It makes concrete claims about the repo:
which files exist, which scripts run what, which versions were used. A claim
the repository contradicts is worse than no claim, so this script checks them
mechanically instead of relying on anyone's memory of the code.

WHAT IT DOES
------------
1. Refuses to move anything that code still refers to: it searches src/,
   scripts/ and tests/ for each file name. Mentions in docs/ and paper/ are
   reported but do not block, because they are text, not code paths.
2. `git mv` for each file. This preserves history and leaves the moves staged
   for review. Files already moved are skipped, so the script can be re-run.
3. Checks README.md:
   - every repository path it names exists;
   - every src/NN_*.py experiment script in the repo is listed in it, and it
     lists none that are absent;
   - the stated versions (torch, pennylane, numpy, medmnist) match what
     requirements.lock.txt / requirements.txt / environment.yml pin;
   - placeholders still to be filled before submission are listed as
     reminders.
Nothing outside the files named in MOVES is modified.
"""
import argparse
import glob
import os
import re
import subprocess
import sys

MOVES = [
    ("assets/fig01_bottleneck_gap.png", "legacy/conference/figures/fig01_bottleneck_gap.png"),
    ("assets/fig02_expressivity_dynamics.png", "legacy/conference/figures/fig02_expressivity_dynamics.png"),
    ("assets/fig03_precision_paradox.png", "legacy/conference/figures/fig03_precision_paradox.png"),
    ("results/frozen_ablation_logs.json", "legacy/conference/logs/frozen_ablation_logs.json"),
    ("results/end_to_end_logs.json", "legacy/conference/logs/end_to_end_logs.json"),
    ("results/02_end_to_end_logs.json", "legacy/conference/logs/02_end_to_end_logs.json"),
    ("results/robustness_e2e_logs.json", "legacy/conference/logs/robustness_e2e_logs.json"),
]

VERSIONS = {"torch": "2.4.1", "pennylane": "0.42.3", "numpy": "1.26.4", "medmnist": "3.0.2"}
README_MARKER = "Variational quantum vs. parameter-matched classical heads"
PLACEHOLDERS = ["archive DOI to be added", "will be added on publication"]
THIS = os.path.normpath("scripts/v51_release_tidy.py")


def text_files(roots, exts):
    for root in roots:
        for dp, _, fns in os.walk(root):
            if "__pycache__" in dp:
                continue
            for fn in fns:
                p = os.path.normpath(os.path.join(dp, fn))
                if p != THIS and fn.endswith(exts):
                    yield p


def references(name, roots, exts):
    hits = []
    for p in text_files(roots, exts):
        try:
            if name in open(p, encoding="utf-8", errors="ignore").read():
                hits.append(p)
        except OSError:
            pass
    return hits


def git(*args):
    return subprocess.run(["git", *args], capture_output=True, text=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    problems, notes = [], []

    if not os.path.exists(os.path.join("src", "config.py")):
        sys.exit("Run from the repository root (src/config.py not found).")
    readme = open("README.md", encoding="utf-8").read()
    if README_MARKER not in readme:
        sys.exit("README.md is not the v51 version. Extract v51.zip first.")

    # ---- 1. references -------------------------------------------------
    print("1. References to the files being moved")
    blocked = False
    for src, _ in MOVES:
        base = os.path.basename(src)
        code = references(base, ["src", "scripts", "tests"], (".py", ".sh"))
        text = references(base, ["docs", "paper"], (".md", ".tex", ".bib"))
        if code:
            blocked = True
            print(f"   BLOCK  {base} is referenced by code: {code}")
        elif text:
            print(f"   note   {base} is mentioned in text: {text}")
        else:
            print(f"   ok     {base}")
    if blocked:
        sys.exit("Nothing moved. Code refers to a file above; send that file.")

    # ---- 2. moves --------------------------------------------------------
    print("\n2. Moves")
    for src, dst in MOVES:
        if os.path.exists(dst) and not os.path.exists(src):
            print(f"   SKIP   already at {dst}")
            continue
        if not os.path.exists(src):
            problems.append(f"{src} not found and not yet moved")
            print(f"   MISSING {src}")
            continue
        if git("ls-files", "--error-unmatch", src).returncode != 0:
            problems.append(f"{src} is not tracked by git")
            print(f"   UNTRACKED {src}")
            continue
        if args.dry_run:
            print(f"   would move {src} -> {dst}")
            continue
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        r = git("mv", src, dst)
        if r.returncode != 0:
            problems.append(f"git mv {src}: {r.stderr.strip()}")
            print(f"   FAILED {src}: {r.stderr.strip()}")
        else:
            print(f"   moved  {src} -> {dst}")
    if not args.dry_run and os.path.isdir("assets") and not os.listdir("assets"):
        os.rmdir("assets")
        print("   removed the now-empty assets/")

    # ---- 3. README claims ----------------------------------------------
    print("\n3. README.md against the repository")
    named = set(re.findall(r"(?<![\w/])((?:src|tests|docs|scripts|legacy|paper)/[\w./-]*\w)", readme))
    named |= {f for f in ("LICENSE", "requirements.txt", "requirements.lock.txt",
                          "environment.yml") if f in readme}
    for path in sorted(named):
        exists = os.path.exists(path) or (args.dry_run and any(path == d or d.startswith(path + "/")
                                                                for _, d in MOVES))
        if not exists:
            problems.append(f"README names {path}, which does not exist")
        print(f"   {'ok     ' if exists else 'MISSING'} {path}")

    in_repo = {os.path.basename(p)[:-3] for p in glob.glob("src/[0-9][0-9]_*.py")}
    not_listed = sorted(n for n in in_repo if n not in readme)
    phantom = sorted(n for n in re.findall(r"^  (\d\d_[a-z_]+)\s", readme, re.M)
                     if n not in in_repo)
    for n in not_listed:
        problems.append(f"src/{n}.py exists but the README does not list it")
        print(f"   UNLISTED src/{n}.py")
    for n in phantom:
        problems.append(f"README lists {n}, which is not in src/")
        print(f"   PHANTOM  {n}")
    if not not_listed and not phantom:
        print(f"   ok      all {len(in_repo)} experiment scripts in src/ are listed, none extra")

    pins = ""
    for f in ("requirements.lock.txt", "requirements.txt", "environment.yml"):
        if os.path.exists(f):
            pins += open(f, encoding="utf-8", errors="ignore").read().lower() + "\n"
    for pkg, ver in VERSIONS.items():
        found = sorted(set(re.findall(rf"(?m)^[\s-]*{pkg}\s*[=<>~!]=\s*([\w.+]+)", pins)))
        if not found:
            notes.append(f"{pkg}: no pin found in the requirements files")
            print(f"   note    {pkg}: not pinned in requirements files (README says {ver})")
        elif any(v.split("+")[0] == ver for v in found):
            print(f"   ok      {pkg} {ver}  (pinned: {', '.join(found)})")
        else:
            problems.append(f"{pkg}: README says {ver}, requirements pin {found}")
            print(f"   MISMATCH {pkg}: README {ver}, requirements {found}")

    for ph in PLACEHOLDERS:
        if ph in readme:
            notes.append(f"placeholder to fill before submission: '{ph}'")

    # ---- summary ---------------------------------------------------------
    print()
    for n in notes:
        print(f"REMINDER  {n}")
    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  - {p}")
        print("\nDo not commit. Send this output.")
        sys.exit(1)
    print("\nAll checks passed." + ("" if args.dry_run else
          " Review with `git status`, then commit."))


if __name__ == "__main__":
    main()
