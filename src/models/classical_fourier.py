"""
Classical trigonometric feature heads - the dequantization controls.

THEORY
------
With AngleEmbedding(rotation='Y') and no data re-uploading, the encoded state is

    |psi(z)> = tensor_j [ cos(z_j/2)|0> + sin(z_j/2)|1> ]

so every amplitude product psi_b * psi_b' factorises over qubits into terms from
{cos^2(u/2), sin^2(u/2), sin(u/2)cos(u/2)}, each of which is an affine
combination of {1, cos u, sin u} via the half-angle identities. Therefore, for
ANY circuit parameters Theta, the measured expectation value is exactly

    <X_i>(z) = sum_{s in {0,c,s}^d}  c_s(Theta) * prod_j f_{s_j}(z_j)

with f_0 = 1, f_c = cos, f_s = sin. The basis has exactly 3^d elements, and the
Fourier frequency support is {-1, 0, +1}^d. This reproduces, for this specific
architecture, the spectrum theorem of Schuld, Sweke & Meyer (2021): accessible
frequencies are the eigenvalue differences of the encoding generator, here Y/2
with eigenvalues +-1/2 giving differences {-1, 0, +1}.

WHAT THIS MEANS
---------------
The VQC does not compute anything outside this classically constructible span.
It does, however, reach only a low-dimensional MANIFOLD inside it: with 3*L*d
parameters it traces a 24-dimensional surface inside an 81-dimensional space at
d=4. The open question this study answers is whether that manifold is a useful
inductive bias - not whether the VQC is "quantum".

TWO ARMS, TWO DIFFERENT CONTROLS
--------------------------------
FourierExactHead - the full 3^d span with a linear fit. This is the CEILING of
the VQC's function class, and it deliberately has far more parameters than the
VQC. It is not a fair fight and must be labelled as such: it answers "does the
VQC exhaust its own class, or does a direct fit over the identical basis beat
the variational optimizer?"

FourierRFFHead - m frequency vectors sampled from {-1,0,1}^d, fixed at init.
This is the random-Fourier-features dequantization baseline.

WHY THESE ARE MATCHED ON BASIS DIMENSION, NOT PARAMETER COUNT
-------------------------------------------------------------
Solving for parameter parity in an RFF head gives 8m + 4 = 24, i.e. m ~ 2.5 -
two or three frequencies. That would be a rigged comparison: the VQC's 81-
function basis is FREE, obtained from the embedding, with parameters spent only
on steering within it. The RFF basis is equally free. Matching on parameters
would hand the VQC an 81-dimensional basis and cap its competitor at 5.

So: Fourier arms match basis dimension; MatchedParamHead matches parameter
count. Different arms, different parity axes, both stated explicitly. The
parameter-efficiency question is answered by matched_param, NOT by these arms.

FIX (2026-08-12): CANONICAL FREQUENCY SAMPLING
----------------------------------------------
The previous sampler drew uniformly from all 3^d frequency vectors, so it could
select both omega and -omega. Because

    cos(-omega . z) =  cos(omega . z)          (identical column)
    sin(-omega . z) = -sin(omega . z)          (linearly dependent column)

each such pair contributed two redundant features. Measured at d=4, seed 42:
12 of 40 sampled rows had their negation also present, so the head spanned 68
effective dimensions rather than the intended 80 - understating the control arm
that the dequantization claim rests on.

The fix samples from CANONICAL representatives: one per +-pair, defined as those
whose first non-zero entry is positive. The zero vector is excluded entirely,
since cos(0)=1 duplicates the classifier bias and sin(0)=0 is a dead feature.
At d=4 there are (3^4 - 1)/2 = 40 canonical frequencies, giving exactly 80
independent features.

NOTE: this changes fourier_rff behaviour. Results produced before this fix used
the 68-effective-dimension basis and must be regenerated before publication.
"""
import itertools

import torch
import torch.nn as nn

from .heads import init_weights


def exact_basis(z: torch.Tensor) -> torch.Tensor:
    """
    Full 3^d trigonometric basis: tensor_j [1, cos z_j, sin z_j].
    z: [B, d]  ->  [B, 3^d]

    Built by iterated Kronecker product, which is exact and avoids materialising
    the index set.
    """
    B, d = z.shape
    phi = torch.ones(B, 1, device=z.device, dtype=z.dtype)
    for j in range(d):
        block = torch.stack(
            [torch.ones_like(z[:, j]), torch.cos(z[:, j]), torch.sin(z[:, j])],
            dim=1,
        )                                          # [B, 3]
        phi = (phi.unsqueeze(2) * block.unsqueeze(1)).reshape(B, -1)
    return phi


def _is_canonical(omega) -> bool:
    """
    One representative per +-pair: the first non-zero entry must be positive.
    The all-zero vector is not canonical - it is excluded (see module docstring).
    """
    for x in omega:
        if x > 0:
            return True
        if x < 0:
            return False
    return False


def canonical_frequencies(d: int, max_freq: int = 1) -> torch.Tensor:
    """
    All ((2K+1)^d - 1)/2 canonical frequency vectors in {-K..K}^d, K = max_freq.

    K=1 is the single-encoding VQC's support {-1,0,1}^d. K=2 is the support of
    the re-uploading VQC (two RY encodings per qubit; the test suite verifies the
    circuit's output lies in this span to a residual below 1e-8). range(-1, 2)
    enumerates [-1, 0, 1] in the same order as before, so K=1 is unchanged.
    """
    vals = list(range(-max_freq, max_freq + 1))
    return torch.tensor(
        [w for w in itertools.product(vals, repeat=d) if _is_canonical(w)],
        dtype=torch.float32,
    )


def _sample_canonical_random(d: int, m: int, generator, max_freq: int = 1) -> torch.Tensor:
    """
    Rejection sampling for large d, where enumeration is infeasible
    (3^16 = 43M). Draws canonical vectors and rejects duplicates.
    """
    seen, out = set(), []
    while len(out) < m:
        batch = torch.randint(-max_freq, max_freq + 1, (4 * m, d), generator=generator)
        for row in batch:
            if len(out) >= m:
                break
            w = tuple(int(x) for x in row)
            if not _is_canonical(w):
                w = tuple(-x for x in w)           # flip into the canonical half
                if not _is_canonical(w):
                    continue                        # was the zero vector
            if w not in seen:
                seen.add(w)
                out.append(w)
    return torch.tensor(out, dtype=torch.float32)


class FourierExactHead(nn.Module):
    """Function-class ceiling. Feasible for d <= 8 (3^8 = 6561)."""

    def __init__(self, d: int, max_dim: int = 8):
        super().__init__()
        if d > max_dim:
            raise ValueError(
                f"FourierExactHead infeasible at d={d} (3^{d} features). "
                f"Use FourierRFFHead."
            )
        self.d = d
        self.n_features = 3 ** d
        self.proj = nn.Linear(self.n_features, d)
        self.out_dim = d
        self.apply(init_weights)

    def forward(self, z):
        return self.proj(exact_basis(z))


class FourierRFFHead(nn.Module):
    """
    Random Fourier features over the VQC's own frequency support {-1,0,1}^d.

    Frequencies are sampled once at construction from the given seed and stored
    as a non-trainable buffer, so they persist in checkpoints and are exactly
    reproducible. Uniform sampling over CANONICAL representatives (rather than a
    low-Hamming-weight bias) is used for the primary arm because a reviewer can
    regenerate it from the seed without argument.

    At d=16 the budget samples a vanishing fraction of the 43M available
    frequencies. That is the honest RFF setting - RFF is by definition a
    Monte-Carlo kernel approximation - and it is stated in the manuscript.
    """

    def __init__(self, d: int, seed: int, max_features: int = 2048,
                 max_freq: int = 1):
        """
        max_freq=1 (default) is the original arm, bit-identical: the same
        enumeration order, the same feasibility threshold (3^8), and the same
        random-number consumption.

        max_freq=2 covers the re-uploading VQC's support {-2..2}^d: at d=4 that
        is all 312 canonical frequencies, 624 features, Linear(624, 4) = 2,500
        parameters. Like the max_freq=1 arm against the single-encoding VQC
        (324 parameters vs 24), it is a direct fit over the circuit's function
        class, NOT a parameter-matched control.
        """
        super().__init__()
        self.d = d
        self.max_freq = max_freq
        base = 2 * max_freq + 1
        n_canonical = (base ** d - 1) // 2
        m = max(1, min(n_canonical, max_features // 2))

        g = torch.Generator().manual_seed(seed)
        if base ** d <= 3 ** 8:                    # enumerable: 6,561 vectors
            all_freqs = canonical_frequencies(d, max_freq)   # WITHOUT replacement
            idx = torch.randperm(all_freqs.shape[0], generator=g)[:m]
            omega = all_freqs[idx]
        else:
            omega = _sample_canonical_random(d, m, g, max_freq)

        self.register_buffer("omega", omega)        # [m, d], non-trainable
        self.m = omega.shape[0]
        self.n_features = 2 * self.m                # every column independent
        self.proj = nn.Linear(self.n_features, d)
        self.out_dim = d
        self.apply(init_weights)

    def forward(self, z):
        proj = z @ self.omega.t()                   # [B, m]
        phi = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)
        return self.proj(phi)
