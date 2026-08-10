"""
Classical classification heads.

PARITY CONTRACT
---------------
Every head is a map  z_tilde in R^d  ->  r in R^d,  followed by a SHARED
classifier Linear(d, C). The only thing that differs between arms is how
z_tilde becomes r. This is what makes the comparison interpretable: no arm
gets a wider final classifier, and no arm skips the tanh rescaling.

The one exception is DeepFunnelEncoder, which replaces the 256->d bottleneck
itself rather than acting after it - by design, since it exists to test
whether classical failure is a depth problem.
"""
import math
import torch
import torch.nn as nn


def init_weights(m):
    """Kaiming init on all custom layers, so no arm has an initialization edge."""
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)


class IdentityHead(nn.Module):
    """Linear arm: r = z_tilde. All capacity lives in the shared classifier."""
    def __init__(self, d: int):
        super().__init__()
        self.out_dim = d

    def forward(self, z):
        return z


class GELUHead(nn.Module):
    """MLP arm: r = GELU(z_tilde). Non-linearity at zero extra parameters."""
    def __init__(self, d: int):
        super().__init__()
        self.act = nn.GELU()
        self.out_dim = d

    def forward(self, z):
        return self.act(z)


class MatchedParamHead(nn.Module):
    """
    Capacity control. A generic non-linearity with the SAME trainable parameter
    count as the VQC, so any VQC advantage cannot be attributed to its extra
    24-96 parameters.

    VQC quantum parameters = 3 * L * d.
    Here: Linear(d, w, bias=False) -> GELU -> Linear(w, d, bias=False) = 2*d*w.
    Setting 2*d*w = 3*L*d gives w = 3L/2, which is exact for even L and rounds
    for odd L. The realised count is exposed for logging.
    """
    def __init__(self, d: int, n_layers: int = 2):
        super().__init__()
        target = 3 * n_layers * d
        w = max(1, int(round(3 * n_layers / 2)))
        self.fc1 = nn.Linear(d, w, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(w, d, bias=False)
        self.out_dim = d
        self.target_params = target
        self.actual_params = 2 * d * w
        self.apply(init_weights)

    def forward(self, z):
        return self.fc2(self.act(self.fc1(z)))


class DeepFunnelEncoder(nn.Module):
    """
    Deep classical compression: 256 -> 64 -> 16 -> d, replacing the single
    linear bottleneck.

    LayerNorm, not BatchNorm. At n_per_class=5 a batch can be smaller than the
    feature dimension, which makes BatchNorm statistics meaningless. Removing
    normalisation altogether would cripple the deep stack that this arm exists
    to make strong - i.e. it would bias the comparison toward the VQC, which is
    the opposite of what this arm is for. LayerNorm is batch-size independent.
    """
    def __init__(self, in_dim: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.GELU(), nn.LayerNorm(64),
            nn.Linear(64, 16), nn.GELU(), nn.LayerNorm(16),
            nn.Linear(16, d),
        )
        self.out_dim = d
        self.apply(init_weights)

    def forward(self, h):
        return self.net(h)
