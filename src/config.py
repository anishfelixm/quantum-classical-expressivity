"""
Central configuration. Single source of truth for the whole project.

No experiment script defines its own constants. If a number appears in the
manuscript, it is traceable to a line in this file.
"""
import os
import subprocess

import torch

# ---------------------------------------------------------------- paths
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ARTIFACT_ROOT = os.environ.get("QML_ARTIFACT_ROOT",
                               os.path.join(REPO_ROOT, "artifacts"))

DATA_CACHE = os.path.join(ARTIFACT_ROOT, "data_cache")
CHECKPOINT_DIR = os.path.join(ARTIFACT_ROOT, "checkpoints")
SHARD_DIR = os.path.join(ARTIFACT_ROOT, "shards")
LATENT_DIR = os.path.join(ARTIFACT_ROOT, "latents")
FEATURE_CACHE = os.path.join(ARTIFACT_ROOT, "feature_cache")
PREDICTION_DIR = os.path.join(ARTIFACT_ROOT, "predictions")

for _d in (DATA_CACHE, CHECKPOINT_DIR, SHARD_DIR, LATENT_DIR,
           FEATURE_CACHE, PREDICTION_DIR):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- datasets
DATASETS = ["breastmnist", "pneumoniamnist", "bloodmnist", "pathmnist"]
BINARY_DATASETS = ["breastmnist", "pneumoniamnist"]

# Verified train-split per-class counts (2026-08-09). The minimum determines
# the largest feasible n_per_class.
TRAIN_CLASS_COUNTS = {
    "breastmnist":    [147, 399],
    "pneumoniamnist": [1214, 3494],
    "bloodmnist":     [852, 2181, 1085, 2026, 849, 993, 2330, 1643],
    "pathmnist":      [9366, 9509, 10360, 10401, 8006, 12182, 7886, 9401, 12885],
}
VAL_CLASS_COUNTS = {
    "breastmnist":    [21, 57],
    "pneumoniamnist": [135, 389],
    "bloodmnist":     [122, 312, 155, 290, 122, 143, 333, 235],
    "pathmnist":      [1041, 1057, 1152, 1156, 890, 1354, 877, 1045, 1432],
}

# ---------------------------------------------------------------- scarcity
# Absolute shots per class, NOT fractions. Every dataset supports all five
# (breastmnist minority class = 147 > 100).
N_PER_CLASS = [5, 10, 20, 50, 100]

# Validation is subsampled to keep the scarcity claim honest, but capped by
# availability (breastmnist has only 21 in its minority val class).
VAL_MULTIPLIER = 2

# The full-data reference row uses the natural, imbalanced training split.
# PathMNIST is capped: 90k images x 100 epochs x all arms is ~12 GPU-days alone.
FULL_DATA_CAP = {"pathmnist": 10000}
FULL_DATA_SEEDS = 5
FULL_DATA_DIMS = [4]

# ---------------------------------------------------------------- sweep axes
BOTTLENECKS = [4, 8, 16]
ALL_SEEDS = [42, 123, 2026, 777, 888, 31337, 8, 271828, 161803, 1414]

# 40 seeds for the pre-registered confirmatory comparison only. Verified: no
# overlap with ALL_SEEDS beyond the first ten, which are intentionally shared so
# diagnostic and confirmatory runs are paired where they overlap.
CONFIRMATORY_SEEDS = ALL_SEEDS + [
    99, 1729, 6174, 2718281, 40, 1000003, 65537, 4321, 987654, 13,
    2027, 555, 9001, 314159, 11235, 77777, 60221, 33, 8191, 121393,
    17, 496, 8128, 28657, 1597, 46341, 5040, 720720, 2520, 10007,
]

# d=16 is the trainability showcase, not the headline; d=4 and d=8 carry the
# statistical weight.
SEEDS_BY_DIM = {4: ALL_SEEDS, 8: ALL_SEEDS, 16: ALL_SEEDS[:5]}

VQC_LAYERS = 2                 # default depth
VQC_LAYERS_PILOT = [1, 2, 4]   # depth sweep: manifold dim 3*L*d = 12/24/48
VQC_LAYERS_SWEEP = [1, 2, 4, 8]  # 8 gives 96 params > the 81-dim span at d=4

# ---------------------------------------------------------------- arms
ARMS = [
    "linear",            # minimum-capacity floor
    "mlp",               # non-linearity at zero extra parameters
    "deep_funnel",       # proves failure is not a depth problem
    "matched_param",     # rank-limited capacity control (diagnostic only)
    "matched_param_fullrank",   # full-rank capacity control; EXACT ONLY AT d=4
    "low_rank",          # full-rank capacity control at ANY d; capacity axis
    "fourier_rff",       # Q2 function-class control
    "fourier_exact",     # function-class ceiling (d <= 8 only)
    "quantum_vqc",       # treatment
    "quantum_reupload",  # Q4: same 24 params, spectrum {-2..2}^d instead of {-1,0,1}^d
]
QUANTUM_ARMS = ["quantum_vqc", "quantum_reupload"]

# Manuscript labels. The code names are frozen because ~2,000 shards are keyed
# on them, but several are inaccurate as prose: with USE_TANH the "linear" arm is
# tanh -> Linear(d,C), which is not a linear model. Tables and figures use these.
ARM_DISPLAY_NAMES = {
    "linear":                 "Identity head",
    "mlp":                    "GELU head",
    "deep_funnel":            "Deep funnel encoder",
    "matched_param":          "Matched-parameter (rank-limited)",
    "matched_param_fullrank": "Matched-parameter (full-rank)",
    "low_rank":               "Low-rank head",
    "fourier_rff":            "Random Fourier features",
    "fourier_exact":          "Exact Fourier basis",
    "quantum_vqc":            "VQC (single encoding)",
    "quantum_reupload":       "VQC (data re-uploading)",
    "pca_svm":                "PCA + SVM",
}

PRIMARY_COMPARISON = ("quantum_vqc", "matched_param_fullrank")  # Q1, H-P
DIAGNOSTIC_COMPARISON = ("quantum_vqc", "matched_param")        # rank-limited
SECONDARY_COMPARISON = ("quantum_vqc", "fourier_rff")           # Q2

PCA_SVM_REFERENCE = True       # reported, excluded from the test family
FOURIER_EXACT_MAX_DIM = 8      # 3^16 = 43M features is infeasible
FOURIER_RFF_MAX_FEATURES = 2048

# ---------------------------------------------------------------- low-rank head
# LowRankHead:  r = GELU( ((I + U V^T) z) * scale + bias ),  U,V in R^{d x rank}
#     params = 2*d*rank + 2*d
#
# Parity with the VQC's 3*L*d = 6d parameters (at L=2):
#     2*d*rank + 2*d = 6d   =>   rank = 2,   INDEPENDENT OF d
#
#     d=4  -> 24 params      d=8  -> 48 params      d=16 -> 96 params
#
# MatchedParamFullRankHead achieves parity only at d=4 (d^2 + 2d = 6d), and at
# d=8 a dense d x d matrix already costs 64 against a 48-parameter budget - so
# exact parity and full rank are not simultaneously reachable in dense form
# above d=4. rank=2 is what unblocks d=8 and d=16.
LOW_RANK_DEFAULT = 2
LOW_RANK_SWEEP = [0, 1, 2, 4, 8]   # 8, 16, 24, 40, 72 params at d=4

# ---------------------------------------------------------------- bottleneck
# HOW THE 256-d FEATURE VECTOR BECOMES d DIMENSIONS.
#
# "learned"  a trainable Linear(256, d). The default, and what every result so
#            far used.
# "pca"      the top-d principal directions of the TRAINING features, frozen.
#            Optimal linear compression in the mean-squared sense.
# "random"   a fixed Gaussian projection, frozen. Johnson-Lindenstrauss:
#            approximately distance-preserving and completely arm-agnostic.
#
# WHY THIS AXIS EXISTS. At d=4 with 2 classes the trainable parameter budget of
# the "frozen backbone" experiment is:
#
#     bottleneck Linear(256, 4)  1,028      97%
#     head                          24       2%
#     classifier Linear(4, 2)       10       1%
#
# So the experiment designed to isolate the HEAD's function class is dominated
# by a 1,028-parameter learned projection that can reshape the latent space to
# suit whichever head follows - the same absorption effect measured at the
# encoder level in Q3, one layer further down. Freezing the bottleneck is the
# only configuration in which the head is the dominant learner (24 of 34
# trainable parameters, ~70%).
#
# Both frozen variants are run: if the head ordering is the same under an
# optimal projection and a random one, the result is a property of the heads.
BOTTLENECK_POLICIES = ["learned", "pca", "random"]
BOTTLENECK_POLICY = "learned"

# ---------------------------------------------------------------- training
BATCH_SIZE = 32
MAX_EPOCHS = 100
PATIENCE = 30

LR_BACKBONE = 1e-4
LR_HEAD = 1e-3
LR_QUANTUM = 1e-3

WEIGHT_DECAY = 1e-4            # applied to EVERY arm, or to none - never split
WEIGHT_DECAY_VARIANTS = [0.0, 1e-4]   # pilot runs both as a sensitivity check

GRAD_CLIP_NORM = 20.0   # 2x the largest observed p95 (9.62, deep_funnel/bloodmnist);
                        # max observed 15.87. Clipping is a safety net against
                        # explosions, never a per-arm learning-rate multiplier.
CLASS_WEIGHT_CLIP = (0.1, 10.0)

# Checkpoints are selected on validation AUC (the primary endpoint) and, as a
# stated sensitivity check, also on Macro-F1. Selecting on F1 alone while
# reporting AUC let the selection criterion interact with the VQC's calibration
# failure - a confound, not a preference.
SELECTION_METRIC = "auc"


def min_epochs_for(n_batches: int) -> int:
    """
    Floor on epochs before early stopping may fire, clamped to stay reachable.

    The old code used max(20, 200 // len(train_loader)). At n_per_class=5 the
    loader has a single batch, giving min_epochs=200 against max_epochs=100 --
    so the early-stop branch was unreachable and every scarce run silently
    trained the full 100 epochs. The clamp to MAX_EPOCHS // 2 fixes that.
    """
    return min(max(20, 200 // max(n_batches, 1)), MAX_EPOCHS // 2)


AUGMENT_E2E = True             # identical across arms
AUGMENT_FROZEN = False         # must be off: caching requires deterministic features

# ---------------------------------------------------------------- noise
NOISE_LEVELS = [0.0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.08, 0.10, 0.15, 0.20]
SHOT_NOISE_SHOTS = 1024

# ---------------------------------------------------------------- statistics
BOOTSTRAP_RESAMPLES = 2000
ALPHA = 0.05
FDR_METHOD = "benjamini-hochberg"
DECLARED_FAMILY_SIZE = 17      # docs/analysis_plan.md; pass to 04 --family-size

# ---------------------------------------------------------------- quantum
QUANTUM_DEVICE = "default.qubit"
QUANTUM_DIFF_METHOD = "backprop"   # verified: propagates input gradients; 260x
                                   # faster than adjoint at d=16 on GPU

# z_tilde = tanh(z) * ANGLE_SCALE. RY is injective on [-pi, pi], so pi/2 uses only
# half the available range and bounds what the quantum arm can resolve. An untuned
# free parameter that handicaps one arm is a fairness objection, so it is swept.
ANGLE_SCALE = torch.pi / 2.0
ANGLE_SCALE_SWEEP = [torch.pi / 2.0, torch.pi]

# Ablation only. False feeds raw z to the head, removing the bounded squashing.
# CLASSICAL ARMS ONLY: RY encoding is 2*pi-periodic, so unbounded z makes the
# quantum encoding non-injective. build_arm raises if this is violated.
USE_TANH = True

# Q4 re-uploading. R=2 with n_layers=2 gives 1 layer per encoding block:
# identical 24 parameters to quantum_vqc, spectrum 5^d = 625 vs 3^d = 81 at d=4.
VQC_UPLOADS_REUPLOAD = 2

# ---------------------------------------------------------------- runtime
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def git_sha() -> str:
    """Commit that produced a result. Written into every shard."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
            stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def set_determinism(seed: int):
    """
    Seed every RNG the pipeline touches, and pin cuDNN to deterministic kernels.

    Called once at the top of every run cell. Without this the seeds are
    decorative: shuffling, augmentation, weight init, and the VQC's parameter
    init would all draw from an unseeded global stream, so 10 seeds would not be
    10 controlled repetitions.
    """
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seeds_for(bottleneck_dim: int):
    return SEEDS_BY_DIM[bottleneck_dim]


def arms_for(bottleneck_dim: int):
    return [a for a in ARMS
            if not (a == "fourier_exact" and bottleneck_dim > FOURIER_EXACT_MAX_DIM)]


def display_name(arm: str) -> str:
    """Manuscript label for an arm. Falls back to the code name."""
    return ARM_DISPLAY_NAMES.get(arm, arm)
