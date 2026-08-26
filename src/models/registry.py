"""
Model factory. Every arm is constructed HERE and nowhere else.

The previous codebase produced checkpoints named Classical_Deep_AE,
Classical_Deep_Bottleneck and Classical_Deep_Funnel for what was meant to be
the same arm - three names, three eras, no guarantee they were the same module.
A single factory makes that class of drift impossible.
"""
import math

import torch
import torch.nn as nn

from .backbone import TruncatedResNet18, FEATURE_DIM
from .heads import (IdentityHead, GELUHead, MatchedParamHead,
                    MatchedParamFullRankHead, LowRankHead,
                    DeepFunnelEncoder, init_weights)
from .classical_fourier import FourierExactHead, FourierRFFHead
from .quantum_vqc import VQCHead

# The angle scale was previously hardcoded to pi/2. It is a free hyperparameter
# that has never been tuned and it bounds absolute VQC performance, so it now
# lives in config and is swept. Fallback keeps this module importable standalone.
try:
    import config as _cfg
    DEFAULT_ANGLE_SCALE = float(getattr(_cfg, "ANGLE_SCALE", torch.pi / 2.0))
    DEFAULT_UPLOADS = int(getattr(_cfg, "VQC_UPLOADS_REUPLOAD", 2))
    DEFAULT_USE_TANH = bool(getattr(_cfg, "USE_TANH", True))
    DEFAULT_HEAD_RANK = int(getattr(_cfg, "LOW_RANK_DEFAULT", 2))
    DEFAULT_BOTTLENECK = str(getattr(_cfg, "BOTTLENECK_POLICY", "learned"))
except Exception:
    DEFAULT_ANGLE_SCALE = torch.pi / 2.0
    DEFAULT_UPLOADS = 2
    DEFAULT_USE_TANH = True
    DEFAULT_HEAD_RANK = 2
    DEFAULT_BOTTLENECK = "learned"

ARM_NAMES = ["linear", "mlp", "deep_funnel", "matched_param",
             "matched_param_fullrank", "low_rank",
             "fourier_rff", "fourier_exact",
             "quantum_vqc", "quantum_reupload",
             "quantum_rich", "quantum_rich_padded"]

QUANTUM_ARMS = ("quantum_vqc", "quantum_reupload",
                "quantum_rich", "quantum_rich_padded")
BOTTLENECK_POLICIES = ("learned", "pca", "random")


class BottleneckModel(nn.Module):
    """
    Shared pipeline:

        x -> backbone -> h (256) -> bottleneck -> z (d)
          -> z_tilde = tanh(z) * angle_scale        [if use_tanh]
          -> head -> r (d)
          -> classifier -> logits (C)

    The tanh rescaling is applied to EVERY arm, not just the quantum one. It
    exists to keep rotation angles inside the injective region of RY, but
    applying it to only one arm would confound the head with the input
    transform - so all arms receive identical inputs.

    NOTE FOR THE MANUSCRIPT: because of this, the arm named "linear" is really
    tanh-then-linear. Describe it as an "identity head" and state the shared
    rescaling once, rather than calling it a linear model.

    ON angle_scale
    --------------
    RY(theta)|0> = cos(theta/2)|0> + sin(theta/2)|1> is injective for
    theta in [-pi, pi], since theta/2 in [-pi/2, pi/2] keeps cos(theta/2) >= 0
    while sin(theta/2) sweeps [-1, 1] monotonically.

    pi/2 (the original choice) is therefore CONSERVATIVE: it uses only half the
    injective range, compressing angular separation between inputs and bounding
    what the quantum arm can resolve. pi is equally injective and doubles that
    separation. Leaving an untuned free parameter at a value that plausibly
    handicaps one arm is exactly what a reviewer calls an unfair comparison, so
    it is swept as a hyperparameter and the sweep is reported.

    ON use_tanh
    -----------
    Setting use_tanh=False feeds raw z to the head. This is an ABLATION FOR
    CLASSICAL ARMS ONLY, to check whether they were handicapped by a bounded
    squashing they do not need. It is mathematically invalid for quantum arms:
    RY is 2*pi-periodic, so an unbounded z maps distinct latents onto identical
    quantum states, destroying injectivity. build_arm() refuses the combination.

    ON bottleneck_policy - THE HEAD-ISOLATION CONTROL
    --------------------------------------------------
    At d=4 with 2 classes the trainable budget of the "frozen backbone"
    experiment is:

        bottleneck Linear(256, 4)   1,028      97%
        head                           24       2%
        classifier Linear(4, 2)        10       1%

    So the experiment meant to isolate the HEAD's function class is dominated by
    a learned projection forty times its size, which can reshape the latent
    space to suit whichever head follows. That is the same absorption effect
    measured at the encoder in Q3, one layer down, and it was never controlled.

        "learned"  trainable Linear(256, d). Default; all prior results.
        "pca"      top-d principal directions of the TRAINING features, frozen.
                   Optimal linear compression in the mean-squared sense, so no
                   "the projection was badly initialised" objection survives.
        "random"   fixed Gaussian projection, frozen. Johnson-Lindenstrauss:
                   approximately distance-preserving, and arm-agnostic by
                   construction.

    Under either frozen policy the head holds 24 of 34 trainable parameters
    (~70%) and is the dominant learner. If the head ordering is the same under
    an optimal projection and a random one, the result is about the heads.

    "pca" requires fit_bottleneck(features) to be called before training; the
    projection cannot be known at construction time.
    """

    def __init__(self, head, d, num_classes, freeze_policy="layer3_only",
                 deep_encoder=None, build_backbone=True, angle_scale=None,
                 use_tanh=None, bottleneck_policy=None, bottleneck_seed=42):
        super().__init__()
        self.d = d
        self.angle_scale = DEFAULT_ANGLE_SCALE if angle_scale is None else float(angle_scale)
        self.use_tanh = DEFAULT_USE_TANH if use_tanh is None else bool(use_tanh)
        self.bottleneck_policy = (DEFAULT_BOTTLENECK if bottleneck_policy is None
                                  else str(bottleneck_policy))
        if self.bottleneck_policy not in BOTTLENECK_POLICIES:
            raise ValueError(f"unknown bottleneck_policy "
                             f"'{self.bottleneck_policy}'; expected one of "
                             f"{BOTTLENECK_POLICIES}")

        self.backbone = TruncatedResNet18(freeze_policy) if build_backbone else None

        if deep_encoder is not None:
            if self.bottleneck_policy != "learned":
                raise ValueError(
                    "deep_funnel replaces the bottleneck, so it is incompatible "
                    f"with bottleneck_policy='{self.bottleneck_policy}'.")
            self.bottleneck = deep_encoder          # 256 -> 64 -> 16 -> d
        else:
            self.bottleneck = nn.Linear(FEATURE_DIM, d)
            self.bottleneck.apply(init_weights)

        self.pca_variance_retained = None
        self._configure_bottleneck(bottleneck_seed)

        self.head = head
        self.classifier = nn.Linear(head.out_dim, num_classes)
        self.classifier.apply(init_weights)

    # ---------------------------------------------------------------- bottleneck
    def _configure_bottleneck(self, seed):
        if self.bottleneck_policy == "learned":
            return

        if self.bottleneck_policy == "random":
            # Johnson-Lindenstrauss: entries ~ N(0, 1/FEATURE_DIM) keeps the
            # output scale comparable to the input scale, and pairwise distances
            # are preserved in expectation.
            g = torch.Generator().manual_seed(int(seed))
            with torch.no_grad():
                W = torch.randn(self.d, FEATURE_DIM, generator=g) / math.sqrt(FEATURE_DIM)
                self.bottleneck.weight.copy_(W)
                self.bottleneck.bias.zero_()

        # "pca" leaves the init in place until fit_bottleneck() is called;
        # either way the parameters are frozen now so no optimizer group forms.
        for p in self.bottleneck.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def fit_bottleneck(self, features):
        """
        Fill the frozen bottleneck with the top-d principal directions of
        `features` [N, 256]. Call BEFORE training, with TRAINING features only -
        fitting on validation or test would leak.

        z = W (h - mu) with W the top-d right singular vectors, so the map is the
        mean-squared-optimal linear compression to d dimensions. Implemented as a
        Linear with bias = -W mu, which is exactly the centred projection.
        """
        if self.bottleneck_policy != "pca":
            raise RuntimeError(
                f"fit_bottleneck() is only valid for bottleneck_policy='pca', "
                f"not '{self.bottleneck_policy}'")

        dev = self.bottleneck.weight.device
        X = features.detach().to(dev, torch.float32)
        if X.shape[0] < self.d:
            raise ValueError(
                f"PCA needs at least d={self.d} samples, got {X.shape[0]}")

        mu = X.mean(dim=0)
        _, S, Vh = torch.linalg.svd(X - mu, full_matrices=False)
        W = Vh[:self.d]                              # [d, 256], orthonormal rows

        self.bottleneck.weight.copy_(W)
        self.bottleneck.bias.copy_(-(W @ mu))

        total = float((S ** 2).sum())
        self.pca_variance_retained = (float((S[:self.d] ** 2).sum() / total)
                                      if total > 0 else None)
        return self.pca_variance_retained

    # ---------------------------------------------------------------- forward
    def latent(self, h):
        """h (256) -> z_tilde (d). Used by the latent-probe experiment."""
        z = self.bottleneck(h)
        return torch.tanh(z) * self.angle_scale if self.use_tanh else z

    def forward_from_features(self, h, return_latent=False):
        """Entry point for Experiment 1, which trains on cached frozen features."""
        z_t = self.latent(h)
        logits = self.classifier(self.head(z_t))
        return (logits, z_t) if return_latent else logits

    def forward(self, x, return_latent=False):
        if self.backbone is None:
            raise RuntimeError("model built without a backbone; use forward_from_features")
        return self.forward_from_features(self.backbone(x), return_latent)

    def set_bn_eval(self):
        if self.backbone is not None:
            self.backbone.set_bn_eval()

    # ---------------------------------------------------------------- optimiser
    def param_groups(self, lr_backbone, lr_head, lr_quantum, weight_decay):
        """
        Differential learning rates.

        weight_decay is passed identically to every group, including the quantum
        parameters. The previous codebase gave classical heads 1e-4 and quantum
        0.0, then concluded the quantum arm generalises better - that asymmetry
        is exactly the confound this signature removes. (Weight decay on rotation
        angles is not neutral: it biases them toward zero. The pilot therefore
        runs both wd=0 and wd=1e-4 across ALL arms as a sensitivity check.)

        A frozen bottleneck simply contributes no parameters here, because
        requires_grad is False.
        """
        quantum, backbone, head = [], [], []
        for name, p in self.named_parameters():
            if not p.requires_grad:
                continue
            if "q_layer" in name:
                quantum.append(p)
            elif name.startswith("backbone"):
                backbone.append(p)
            else:
                head.append(p)

        groups = []
        if backbone:
            groups.append({"params": backbone, "lr": lr_backbone, "weight_decay": weight_decay})
        if head:
            groups.append({"params": head, "lr": lr_head, "weight_decay": weight_decay})
        if quantum:
            groups.append({"params": quantum, "lr": lr_quantum, "weight_decay": weight_decay})
        return groups

    def trainable_state_dict(self):
        """
        Checkpoint only what actually changed. Saving full state dicts across
        thousands of runs would be tens of GB of identical frozen ImageNet
        weights.

        A frozen bottleneck is included anyway when it is not "learned": the
        projection is part of the experimental condition and a checkpoint that
        omitted it could not be reloaded faithfully.
        """
        trainable = {n for n, p in self.named_parameters() if p.requires_grad}
        keep_bottleneck = self.bottleneck_policy != "learned"
        return {k: v for k, v in self.state_dict().items()
                if k in trainable
                or "omega" in k                      # RFF frequencies are buffers
                or (keep_bottleneck and k.startswith("bottleneck"))}

    def capacity_report(self):
        """Trainable parameters by component. Goes in the manuscript's table."""
        out = {}
        for part in ("backbone", "bottleneck", "head", "classifier"):
            mod = getattr(self, part, None)
            out[part] = (0 if mod is None else
                         sum(p.numel() for p in mod.parameters() if p.requires_grad))
        total = sum(out.values())
        out["total"] = total
        out["head_share"] = (out["head"] / total) if total else 0.0
        out["bottleneck_policy"] = self.bottleneck_policy
        return out


def build_arm(arm, d, num_classes, n_layers=2, seed=42,
              freeze_policy="layer3_only", build_backbone=True,
              device_name="default.qubit", diff_method="backprop",
              angle_scale=None, n_uploads=None, use_tanh=None,
              head_rank=None, bottleneck_policy=None, bottleneck_seed=None):
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm '{arm}'; expected one of {ARM_NAMES}")

    # Checked BEFORE constructing anything: building a VQC only to reject it
    # wastes a device allocation, and the error should name the real problem.
    if use_tanh is False and arm in QUANTUM_ARMS:
        raise ValueError(
            f"use_tanh=False is invalid for '{arm}': RY encoding is 2*pi-periodic, "
            "so unbounded z collapses distinct latents onto identical quantum "
            "states. The no-tanh ablation is for classical arms only.")

    deep_encoder = None

    if arm == "linear":
        head = IdentityHead(d)
    elif arm == "mlp":
        head = GELUHead(d)
    elif arm == "matched_param":
        head = MatchedParamHead(d, n_layers=n_layers)
    elif arm == "matched_param_fullrank":
        head = MatchedParamFullRankHead(d, n_layers=n_layers)
    elif arm == "low_rank":
        # rank=2 gives 2*d*2 + 2*d = 6d = 3*L*d parameters at L=2, for ANY d.
        # Other ranks form the capacity axis; see LowRankHead's docstring.
        r = DEFAULT_HEAD_RANK if head_rank is None else int(head_rank)
        head = LowRankHead(d, rank=r, n_layers=n_layers)
    elif arm == "deep_funnel":
        head = IdentityHead(d)
        deep_encoder = DeepFunnelEncoder(FEATURE_DIM, d)
    elif arm == "fourier_rff":
        head = FourierRFFHead(d, seed=seed)
    elif arm == "fourier_exact":
        head = FourierExactHead(d)
    elif arm == "quantum_vqc":
        # R=1: spectrum {-1,0,1}^d, 3^d basis functions
        head = VQCHead(d, n_layers=n_layers, n_uploads=1,
                       device_name=device_name, diff_method=diff_method)
    elif arm == "quantum_rich":
        # SAME circuit and SAME 24 parameters as quantum_vqc. The only change is
        # that all 2-local <X_i X_j> terms are measured as well, so 10 numbers
        # are read out of the state instead of 4. Does not escape dequantization
        # - those terms live in the same 3^d span - but it reaches more of it.
        head = VQCHead(d, n_layers=n_layers, n_uploads=1,
                       device_name=device_name, diff_method=diff_method,
                       readout="pairs")
    elif arm == "quantum_rich_padded":
        # The control for quantum_rich. Identical width, so an identical shared
        # classifier, but the extra columns are repeats of the SAME d
        # expectations and carry no new information. rich - padded isolates
        # measurement richness from classifier capacity.
        head = VQCHead(d, n_layers=n_layers, n_uploads=1,
                       device_name=device_name, diff_method=diff_method,
                       readout="padded")
    elif arm == "quantum_reupload":
        # R=2 by default: spectrum {-2..2}^d, 5^d basis functions, SAME parameter
        # count as quantum_vqc. Isolates spectral richness from quantum-ness.
        R = DEFAULT_UPLOADS if n_uploads is None else int(n_uploads)
        head = VQCHead(d, n_layers=n_layers, n_uploads=R,
                       device_name=device_name, diff_method=diff_method)

    return BottleneckModel(head, d, num_classes,
                           freeze_policy=freeze_policy,
                           deep_encoder=deep_encoder,
                           build_backbone=build_backbone,
                           angle_scale=angle_scale,
                           use_tanh=use_tanh,
                           bottleneck_policy=bottleneck_policy,
                           bottleneck_seed=seed if bottleneck_seed is None
                           else bottleneck_seed)


def count_head_params(model):
    """Reported in the manuscript's parity table."""
    return {
        "head": sum(p.numel() for p in model.head.parameters() if p.requires_grad),
        "bottleneck": sum(p.numel() for p in model.bottleneck.parameters() if p.requires_grad),
        "classifier": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad),
    }
