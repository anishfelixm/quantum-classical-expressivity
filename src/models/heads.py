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
    confirmatory analysis use MatchedParamFullRankHead (d=4) or
    LowRankHead(rank=2) (any d).
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

    def describe(self):
        return {"d": self.out_dim, "hidden_width": self.hidden_width,
                "params": self.actual_params,
                "target_params": self.target_params,
                "exact_match": self.actual_params == self.target_params,
                "rank_limited": self.rank_limited}


class MatchedParamFullRankHead(nn.Module):
    """
    Capacity control, FULL-RANK VARIANT at d=4.

        Linear(d, d, bias=True) -> GELU -> elementwise scale

    Parameter count:  d^2 (weights) + d (bias) + d (scale)  =  d^2 + 2d
    VQC parameter count:                                       3*L*d

    Equal exactly when d^2 + 2d = 3*L*d. At L=2 that is d = 4, giving 24 = 24.

    THE IDENTITY HOLDS ONLY AT d=4:

        d=4   ->  24 vs 24    exact
        d=8   ->  80 vs 48    classical has 67% more
        d=16  -> 288 vs 96    classical has 3x more

    Worse, at d=8 a dense d x d matrix (64 parameters) already exceeds the
    48-parameter budget, so exact parity and full rank are not simultaneously
    achievable in this form above d=4 at all.

    Use LowRankHead(rank=2) for any comparison at d > 4.
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

    def describe(self):
        return {"d": self.d, "params": self.actual_params,
                "target_params": self.target_params,
                "exact_match": self.exact_match, "rank_limited": False}


class LowRankHead(nn.Module):
    """
    CAPACITY-CONTROLLED, FULL-RANK, EXACT PARITY AT EVERY d.

        r = GELU( ((I + U V^T) z) * scale + bias ),    U, V in R^{d x rank}

    Parameter count:  2*d*rank + 2*d

    WHY THIS HEAD EXISTS - TWO REASONS
    ----------------------------------
    1. EXACT PARITY AT ANY BOTTLENECK DIMENSION.

           2*d*rank + 2*d = 3*L*d = 6d   (at L=2)
       =>  2*d*rank = 4*d
       =>  rank = 2,  INDEPENDENT OF d

       d=4 -> 24 params, d=8 -> 48, d=16 -> 96, matching the VQC exactly in
       every case. MatchedParamFullRankHead manages this only at d=4, which
       confined the primary comparison to a single bottleneck dimension and
       invited the obvious "is this a d=4 artifact?" objection.

    2. A CLEAN CAPACITY AXIS THAT DOES NOT CONFOUND RANK.

       The paper's central claim is that the quantum head's advantage under
       extreme scarcity is a REGULARIZATION effect of its restricted function
       class - 24 parameters reaching a 24-dimensional manifold inside an
       81-dimensional trigonometric span - rather than anything quantum.

       That claim had no dedicated control. Every other arm varies capacity AND
       something else: MatchedParamHead loses rank as it shrinks, and a width-w
       MLP cannot go below 2*d*d parameters while staying full rank - so it
       cannot even reach the VQC's 24 at d=4, let alone the restricted end where
       the interesting behaviour is.

       Here `rank` varies capacity alone. I + U V^T is generically invertible at
       every rank including 0, so the map stays full-rank throughout:

           rank=0 ->  8 params (diagonal affine)   most restricted
           rank=1 -> 16
           rank=2 -> 24 params                     == VQC exactly
           rank=4 -> 40
           rank=8 -> 72                            least restricted

       If small-rank heads reproduce the quantum crossover - helping at
       n=5/class and hurting at n=100 - then restriction is the mechanism and it
       is classically reproducible. If they do not, the regularization
       explanation is wrong. Either outcome is informative; without this arm the
       mechanism is asserted rather than tested.

    INITIALISATION
    --------------
    U and V start small so I + U V^T begins near the identity: the head starts
    close to a diagonal affine map and grows its low-rank correction during
    training, rather than starting from a random near-singular matrix.
    """

    def __init__(self, d: int, rank: int = 2, n_layers: int = 2):
        super().__init__()
        if rank < 0:
            raise ValueError(f"rank must be >= 0, got {rank}")

        self.d = d
        self.rank = rank
        self.out_dim = d
        self.act = nn.GELU()

        if rank > 0:
            std = 0.1 / math.sqrt(d)
            self.U = nn.Parameter(torch.randn(d, rank) * std)
            self.V = nn.Parameter(torch.randn(d, rank) * std)
        else:
            self.register_parameter("U", None)
            self.register_parameter("V", None)

        self.scale = nn.Parameter(torch.ones(d))
        self.bias = nn.Parameter(torch.zeros(d))

        self.target_params = 3 * n_layers * d
        self.actual_params = 2 * d * rank + 2 * d
        self.exact_match = (self.actual_params == self.target_params)
        self.rank_limited = False          # I + U V^T is generically invertible

    def forward(self, z):
        h = z if self.U is None else z + (z @ self.V) @ self.U.t()
        return self.act(h * self.scale + self.bias)

    def describe(self):
        return {"d": self.d, "rank": self.rank,
                "params": self.actual_params,
                "target_params": self.target_params,
                "exact_match": self.exact_match, "rank_limited": False}


class DeepFunnelEncoder(nn.Module):
    """
    Deep classical compression: 256 -> 64 -> 16 -> d, replacing the single
    linear bottleneck.

    LayerNorm, not BatchNorm. At n_per_class=5 a batch can be smaller than the
    feature dimension, which makes BatchNorm statistics meaningless. Removing
    normalisation altogether would cripple the deep stack that this arm exists
    to make strong - i.e. it would bias the comparison toward the VQC, which is
    the opposite of what this arm is for. LayerNorm is batch-size independent.

    NOTE: this arm replaces the bottleneck, so it is incompatible with the
    frozen-bottleneck policies ("pca", "random"). build_arm() refuses that
    combination.
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
