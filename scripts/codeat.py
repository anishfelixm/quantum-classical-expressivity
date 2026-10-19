"""
Run analysis code against the study code of a given commit.

The trained weights kept by Amendment 21 can only be loaded faithfully into the
model classes of the commit that trained them (heads, registry and the VQC
changed between commits). So a script that rebuilds trained models imports the
study code from that commit's git worktree, ../replay_trees/<sha12>, exactly as
scripts/replay.py does, while reading and writing artifacts in the main
repository's folder.

    import codeat
    SRC = codeat.setup(sha)          # BEFORE importing config, torch models, etc.

Must be called before `import config`.
"""
import os
import subprocess
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
TREES = os.path.join(os.path.dirname(REPO), "replay_trees")


def setup(sha=None):
    os.environ.setdefault("QML_ARTIFACT_ROOT", os.path.join(REPO, "artifacts"))
    if sha:
        # accept a short prefix; worktrees are named by the first 12 characters
        # of the full commit, as scripts/replay.py and the launcher name them
        sha = subprocess.run(["git", "-C", REPO, "rev-parse", sha], check=True,
                             capture_output=True, text=True).stdout.strip()
        root = os.path.join(TREES, sha[:12])
        if not os.path.isdir(os.path.join(root, "src")):
            os.makedirs(TREES, exist_ok=True)
            subprocess.run(["git", "-C", REPO, "worktree", "add", "--detach", root, sha], check=True)
    else:
        root = REPO
    src = os.path.join(root, "src")
    if src not in sys.path:
        sys.path.insert(0, src)
    return src


def load_by_path(name, path):
    """Import a file as a module (study scripts have names that start with digits)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod
