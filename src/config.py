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

for _d in (DATA_CACHE, CHECKPOINT_DIR, SHARD_DIR, LATENT_DIR, FEATURE_CACHE):
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

# d=16 is the trainability showcase, not the headline; d=4 and d=8 carry the
# statistical weight.
SEEDS_BY_DIM = {4: ALL_SEEDS, 8: ALL_SEEDS, 16: ALL_SEEDS[:5]}

VQC_LAYERS = 2                 # pilot sweeps [1, 2, 4]; best is carried forward
VQC_LAYERS_PILOT = [1, 2, 4]

# ---------------------------------------------------------------- arms
ARMS = [
    "linear",            # minimum-capacity floor
    "mlp",               # non-linearity at zero extra parameters
    "deep_funnel",       # proves failure is not a depth problem
    "matched_param",     # capacity control (parameter-matched to VQC)
    "fourier_rff",       # FUNCTION-CLASS control - the primary comparison
    "fourier_exact",     # function-class ceiling (d <= 8 only)
    "quantum_vqc",       # treatment
    "quantum_reupload",  # Q4: same 24 params, spectrum {-2..2}^d instead of {-1,0,1}^d
]
PRIMARY_COMPARISON = ("quantum_vqc", "matched_param")   # Q1: efficiency at equal params
SECONDARY_COMPARISON = ("quantum_vqc", "fourier_rff")   # Q2: dequantization over shared basis
PCA_SVM_REFERENCE = True       # reported, excluded from the test family

FOURIER_EXACT_MAX_DIM = 8      # 3^16 = 43M features is infeasible
FOURIER_RFF_MAX_FEATURES = 2048

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

# ---------------------------------------------------------------- quantum
QUANTUM_DEVICE = "default.qubit"
QUANTUM_DIFF_METHOD = "backprop"   # verified: propagates input gradients; 260x
                                   # faster than adjoint at d=16 on GPU

# z_tilde = tanh(z) * ANGLE_SCALE. RY is injective on [-pi, pi], so pi/2 uses only
# half the available range and bounds what the quantum arm can resolve. An untuned
# free parameter that handicaps one arm is a fairness objection, so it is swept.
ANGLE_SCALE = torch.pi / 2.0
ANGLE_SCALE_SWEEP = [torch.pi / 2.0, torch.pi]

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
    decorative: DataLoader shuffling, augmentation, weight init, and the VQC's
    parameter init would all draw from an unseeded global stream, so 10 seeds
    would not be 10 controlled repetitions.
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
