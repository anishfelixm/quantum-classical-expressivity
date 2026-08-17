"""
Model factory. Every arm is constructed HERE and nowhere else.

The previous codebase produced checkpoints named Classical_Deep_AE,
Classical_Deep_Bottleneck and Classical_Deep_Funnel for what was meant to be
the same arm - three names, three eras, no guarantee they were the same module.
A single factory makes that class of drift impossible.
"""
import torch
import torch.nn as nn

from .backbone import TruncatedResNet18, FEATURE_DIM
from .heads import (IdentityHead, GELUHead, MatchedParamHead,
                    MatchedParamFullRankHead, DeepFunnelEncoder, init_weights)
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
except Exception:
    DEFAULT_ANGLE_SCALE = torch.pi / 2.0
    DEFAULT_UPLOADS = 2
    DEFAULT_USE_TANH = True

ARM_NAMES = ["linear", "mlp", "deep_funnel", "matched_param",
             "matched_param_fullrank", "fourier_rff", "fourier_exact",
             "quantum_vqc", "quantum_reupload"]

QUANTUM_ARMS = ("quantum_vqc", "quantum_reupload")


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
    """

    def __init__(self, head, d, num_classes, freeze_policy="layer3_only",
                 deep_encoder=None, build_backbone=True, angle_scale=None,
                 use_tanh=None):
        super().__init__()
        self.d = d
        self.angle_scale = DEFAULT_ANGLE_SCALE if angle_scale is None else float(angle_scale)
        self.use_tanh = DEFAULT_USE_TANH if use_tanh is None else bool(use_tanh)
        self.backbone = TruncatedResNet18(freeze_policy) if build_backbone else None

        if deep_encoder is not None:
            self.bottleneck = deep_encoder          # 256 -> 64 -> 16 -> d
        else:
            self.bottleneck = nn.Linear(FEATURE_DIM, d)
            self.bottleneck.apply(init_weights)

        self.head = head
        self.classifier = nn.Linear(head.out_dim, num_classes)
        self.classifier.apply(init_weights)

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

    def param_groups(self, lr_backbone, lr_head, lr_quantum, weight_decay):
        """
        Differential learning rates.

        weight_decay is passed identically to every group, including the quantum
        parameters. The previous codebase gave classical heads 1e-4 and quantum
        0.0, then concluded the quantum arm generalises better - that asymmetry
        is exactly the confound this signature removes. (Weight decay on rotation
        angles is not neutral: it biases them toward zero. The pilot therefore
        runs both wd=0 and wd=1e-4 across ALL arms as a sensitivity check.)
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
        """
        return {k: v for k, v in self.state_dict().items()
                if k in {n for n, p in self.named_parameters() if p.requires_grad}
                or "omega" in k}          # RFF frequencies are buffers, not params


def build_arm(arm, d, num_classes, n_layers=2, seed=42,
              freeze_policy="layer3_only", build_backbone=True,
              device_name="default.qubit", diff_method="backprop",
              angle_scale=None, n_uploads=None, use_tanh=None):
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
                           use_tanh=use_tanh)


def count_head_params(model):
    """Reported in the manuscript's parity table."""
    return {
        "head": sum(p.numel() for p in model.head.parameters() if p.requires_grad),
        "bottleneck": sum(p.numel() for p in model.bottleneck.parameters() if p.requires_grad),
        "classifier": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad),
    }
