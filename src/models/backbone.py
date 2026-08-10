"""
Shared truncated ResNet-18 backbone.

Every architecture in this study uses THIS module. Duplicating the backbone
definition across model files is how experimental parity silently breaks -
the previous codebase had three copies that had already drifted apart.
"""
import torch
import torch.nn as nn
import torchvision.models as tvm

FEATURE_DIM = 256          # channels at the output of layer3
LAYER3_INDEX = 6           # position of layer3 in resnet.children()


class TruncatedResNet18(nn.Module):
    """
    ImageNet-pretrained ResNet-18 truncated after layer3, globally pooled.

    forward(x: [B,3,224,224]) -> h: [B,256]

    freeze_policy:
        "all"          - entire backbone frozen (Experiment 1)
        "layer3_only"  - only layer3 trainable (Experiment 2)
        "none"         - everything trainable (not used; available for ablation)
    """

    def __init__(self, freeze_policy: str = "layer3_only"):
        super().__init__()
        if freeze_policy not in ("all", "layer3_only", "none"):
            raise ValueError(f"unknown freeze_policy: {freeze_policy}")
        self.freeze_policy = freeze_policy

        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        # drop layer4, avgpool, fc
        self.features = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self._apply_freeze()

    def _apply_freeze(self):
        for i, child in enumerate(self.features):
            trainable = (
                self.freeze_policy == "none"
                or (self.freeze_policy == "layer3_only" and i == LAYER3_INDEX)
            )
            for p in child.parameters():
                p.requires_grad = trainable

    def set_bn_eval(self):
        """
        Keep BatchNorm running statistics frozen in blocks we are not training.

        Must be called after model.train() on every epoch: .train() resets all
        submodules, silently re-enabling running-stat updates in frozen blocks
        and shifting the feature distribution under the heads.
        """
        for i, child in enumerate(self.features):
            trainable = (
                self.freeze_policy == "none"
                or (self.freeze_policy == "layer3_only" and i == LAYER3_INDEX)
            )
            if not trainable:
                child.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.features(x)
        h = self.pool(h)
        return torch.flatten(h, 1)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]
