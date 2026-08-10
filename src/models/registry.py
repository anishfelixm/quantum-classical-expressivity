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
                    DeepFunnelEncoder, init_weights)
from .classical_fourier import FourierExactHead, FourierRFFHead
from .quantum_vqc import VQCHead

ARM_NAMES = ["linear", "mlp", "deep_funnel", "matched_param",
             "fourier_rff", "fourier_exact", "quantum_vqc"]


class BottleneckModel(nn.Module):
    """
    Shared pipeline:

        x -> backbone -> h (256) -> bottleneck -> z (d)
          -> z_tilde = tanh(z) * pi/2
          -> head -> r (d)
          -> classifier -> logits (C)

    The tanh rescaling is applied to EVERY arm, not just the quantum one. It
    exists to keep rotation angles inside the injective region of RY, but
    applying it to only one arm would confound the head with the input
    transform - so all arms receive identical inputs.
    """

    def __init__(self, head, d, num_classes, freeze_policy="layer3_only",
                 deep_encoder=None, build_backbone=True):
        super().__init__()
        self.d = d
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
        return torch.tanh(z) * (torch.pi / 2.0)

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
        ~4800 runs would be ~40 GB of mostly-identical frozen ImageNet weights.
        """
        return {k: v for k, v in self.state_dict().items()
                if k in {n for n, p in self.named_parameters() if p.requires_grad}
                or "omega" in k}          # RFF frequencies are buffers, not params


def build_arm(arm, d, num_classes, n_layers=2, seed=42,
              freeze_policy="layer3_only", build_backbone=True,
              device_name="default.qubit", diff_method="backprop"):
    if arm not in ARM_NAMES:
        raise ValueError(f"unknown arm '{arm}'; expected one of {ARM_NAMES}")

    deep_encoder = None
    if arm == "linear":
        head = IdentityHead(d)
    elif arm == "mlp":
        head = GELUHead(d)
    elif arm == "matched_param":
        head = MatchedParamHead(d, n_layers=n_layers)
    elif arm == "deep_funnel":
        head = IdentityHead(d)
        deep_encoder = DeepFunnelEncoder(FEATURE_DIM, d)
    elif arm == "fourier_rff":
        head = FourierRFFHead(d, seed=seed)
    elif arm == "fourier_exact":
        head = FourierExactHead(d)
    elif arm == "quantum_vqc":
        head = VQCHead(d, n_layers=n_layers,
                       device_name=device_name, diff_method=diff_method)

    return BottleneckModel(head, d, num_classes,
                           freeze_policy=freeze_policy,
                           deep_encoder=deep_encoder,
                           build_backbone=build_backbone)


def count_head_params(model):
    """Reported in the manuscript's parity table."""
    return {
        "head": sum(p.numel() for p in model.head.parameters() if p.requires_grad),
        "bottleneck": sum(p.numel() for p in model.bottleneck.parameters() if p.requires_grad),
        "classifier": sum(p.numel() for p in model.classifier.parameters() if p.requires_grad),
    }
