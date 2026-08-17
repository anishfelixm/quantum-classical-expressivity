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
    Capacity control, BOTTLENECKED VARIANT.

    Linear(d, w, bias=False) -> GELU -> Linear(w, d, bias=False), so
    2*d*w parameters. Setting 2*d*w = 3*L*d gives w = 3L/2 = 3 at L=2.

    KNOWN LIMITATION - read before using this arm
    ----------------------------------------------
    w = 3 is INDEPENDENT OF d. The head therefore passes through a width-3
    hidden layer whatever the bottleneck dimension:

        d=4   ->  4 -> 3 -> 4   rank <= 3, loses one dimension
        d=8   ->  8 -> 3 -> 8   rank <= 3
        d=16  -> 16 -> 3 -> 16  rank <= 3, destroys most of the signal

    The VQC maps R^d -> R^d with no comparable rank constraint, so this arm is
    handicapped: mildly at d=4, severely above it.

    Consequences for results already collected with this arm:
      - A "tie" against the VQC UNDERSTATES the classical side, which
        strengthens rather than weakens a no-advantage conclusion.
      - The scarcity crossover survives: a constant handicap cannot create a
        trend in n. But its zero-point is shifted, so any measured VQC edge is
        an overestimate.
      - Comparisons at d>4 against this arm are NOT valid.

    Retained (not deleted) so the diagnostic remains reproducible. For the
    confirmatory analysis use MatchedParamFullRankHead.
    """

    def __init__(self, d: int, n_layers: int = 2):
        super().__init__()
        target = 3 * n_layers * d
        w = max(1, int(round(3 * n_layers / 2)))
        self.fc1 = nn.Linear(d, w, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(w, d, bias=False)
        self.out_dim = d
        self.hidden_width = w
        self.target_params = target
        self.actual_params = 2 * d * w
        self.rank_limited = (w < d)
        self.apply(init_weights)

    def forward(self, z):
        return self.fc2(self.act(self.fc1(z)))


class MatchedParamFullRankHead(nn.Module):
    """
    Capacity control, FULL-RANK VARIANT. Use this one for the primary comparison.

        Linear(d, d, bias=True) -> GELU -> elementwise scale

    Parameter count:  d^2 (weights) + d (bias) + d (scale)  =  d^2 + 2d
    VQC parameter count:                                       3*L*d

    These are equal exactly when d^2 + 2d = 3*L*d. At L=2 that is d^2 = 4d,
    i.e. d = 4, giving 24 = 24. Exact parity AND full rank simultaneously.

    The identity holds only at d=4. That is sufficient: the primary comparison
    and the confirmatory sweep are both d=4. At other dimensions the head is
    still constructed, the realised parameter count is exposed via
    `actual_params`, and any mismatch MUST be reported rather than glossed.

    Why an elementwise scale rather than a second full matrix: a second
    Linear(d,d) would cost another d^2 and overshoot the budget by a wide
    margin. The diagonal scale spends the remaining d parameters while keeping
    the map full-rank (it is invertible wherever no scale entry is zero).
    """

    def __init__(self, d: int, n_layers: int = 2):
        super().__init__()
        self.d = d
        self.fc = nn.Linear(d, d, bias=True)
        self.act = nn.GELU()
        self.scale = nn.Parameter(torch.ones(d))
        self.out_dim = d

        self.target_params = 3 * n_layers * d
        self.actual_params = d * d + 2 * d
        self.exact_match = (self.actual_params == self.target_params)
        self.rank_limited = False

        self.fc.apply(init_weights)

    def forward(self, z):
        return self.act(self.fc(z)) * self.scale


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
