"""
Amendment 19: does the 100-epoch budget bind at n = 5?

    python -u scripts/run_budget_check.py --datasets pathmnist          # one dataset
    python -u scripts/run_budget_check.py --quick                       # timing: PathMNIST, seed 42
    python -u scripts/run_budget_check.py --summary-only                # table from 01's summariser

Runs src/01_frozen_backbone_ablation.py UNCHANGED, with exactly two constants
raised in-process before it starts:

    config.MAX_EPOCHS  100 -> 1000
    config.PATIENCE     30 ->  200

Everything else is the primary protocol: quantum_vqc and
matched_param_fullrank, frozen backbone, d = 4, learned projection, n = 5,
the tuned rate (1e-2, --use-tuned-lr), the first ten confirmatory seeds, all
four datasets. Namespace 30_budget_check.

WHY
---
At n = 5 the training set is one or two batches, so an epoch is one or two
optimisation steps, and the 100-epoch cap allows at most 200. The only evidence
that the cap does not bind is a check recorded in docs/analysis_plan.md on
27 August (PneumoniaMNIST, 3 seeds, three arms, at the then-default rate, run
by editing config). It has no script and no log in this repository, and it
predates the tuned rate. This repeats it under the final protocol, on all four
datasets, with records. Predictions: docs/analysis_plan.md, Amendment 19,
committed before this script first ran.

WHY NOT A NEW CLI FLAG ON 01
----------------------------
Adding --max-epochs to 01 would change the script that produced every reported
number, on the last day of compute. Setting the two constants from outside
leaves 01 byte-identical. The budget is not part of the shard key, so this
experiment MUST use its own namespace; the runner refuses any other.

The minimum-epoch floor follows MAX_EPOCHS (config.min_epochs_for clamps at
MAX_EPOCHS // 2), so under this budget it is 200 epochs with one training batch
and 100 with two, instead of 50. Recorded here because it changes when early
stopping can first fire.
"""
import argparse
import os
import runpy
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(REPO, "src")
sys.path.insert(0, SRC)

import config                      # noqa: E402
import train.loop as loop          # noqa: E402

NAMESPACE = "30_budget_check"
MAX_EPOCHS, PATIENCE = 1000, 200
ARMS = ["quantum_vqc", "matched_param_fullrank"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--datasets", nargs="+", default=list(config.DATASETS))
    p.add_argument("--seeds", nargs="+", type=int, default=list(config.ALL_SEEDS))
    p.add_argument("--quick", action="store_true",
                   help="timing only: PathMNIST, seed 42, both arms (a real cell, reused later)")
    p.add_argument("--summary-only", action="store_true")
    args = p.parse_args()

    if args.quick:
        args.datasets, args.seeds = ["pathmnist"], [42]

    # The override must reach the SAME config object that train.loop reads.
    config.MAX_EPOCHS = MAX_EPOCHS
    config.PATIENCE = PATIENCE
    assert loop.config is config, "train.loop imported a different config module"
    assert loop.config.MAX_EPOCHS == MAX_EPOCHS and loop.config.PATIENCE == PATIENCE
    print(f"budget check | MAX_EPOCHS={config.MAX_EPOCHS} PATIENCE={config.PATIENCE} "
          f"| min epochs at 1 batch = {config.min_epochs_for(1)}, "
          f"2 batches = {config.min_epochs_for(2)} | ns={NAMESPACE}", flush=True)

    argv = ["01_frozen_backbone_ablation.py", "--experiment", NAMESPACE]
    if args.summary_only:
        argv += ["--summary-only"]
    else:
        argv += ["--arms", *ARMS, "--dims", "4", "--regimes", "5",
                 "--datasets", *args.datasets,
                 "--seeds", *[str(s) for s in args.seeds],
                 "--use-tuned-lr"]
    assert NAMESPACE in argv
    sys.argv = argv
    runpy.run_path(os.path.join(SRC, "01_frozen_backbone_ablation.py"), run_name="__main__")


if __name__ == "__main__":
    main()
