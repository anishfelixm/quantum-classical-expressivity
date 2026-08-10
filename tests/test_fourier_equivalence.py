"""
THE DEQUANTIZATION TEST.

Numerically verifies the central mathematical claim of the paper: that for any
circuit parameters Theta, the VQC's output lies EXACTLY in the span of the 3^d
classical trigonometric basis functions.

Method: sample random z, evaluate both the VQC and the exact basis, solve the
least-squares system phi @ c = v, and check the residual is at machine
precision. If the residual is ~1e-6 or below, the VQC computes nothing outside
a classically constructible function space of dimension 3^d.

This test is cited in the manuscript appendix. It converts a hand-waved
theoretical argument into a reproducible numerical fact, and it disarms the
single most dangerous reviewer objection before it is raised.

Run:  python -m pytest tests/test_fourier_equivalence.py -v -s
"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.quantum_vqc import VQCHead                    # noqa: E402
from models.classical_fourier import exact_basis, FourierExactHead  # noqa: E402

TOL = 1e-5


@pytest.mark.parametrize("d,L", [(2, 1), (2, 3), (3, 2), (4, 1), (4, 2), (4, 4)])
def test_vqc_lies_in_exact_fourier_span(d, L):
    torch.manual_seed(0)
    vqc = VQCHead(d, n_layers=L).double()
    for p in vqc.parameters():
        torch.nn.init.normal_(p, 0.0, 1.0)     # deliberately large angles

    n_samples = max(4 * 3 ** d, 200)
    z = (torch.rand(n_samples, d, dtype=torch.float64) - 0.5) * torch.pi

    with torch.no_grad():
        v = vqc(z)                             # [N, d] expectation values
        phi = exact_basis(z)                   # [N, 3^d]

    assert phi.shape[1] == 3 ** d, f"expected {3**d} basis functions, got {phi.shape[1]}"

    sol = torch.linalg.lstsq(phi, v)
    residual = (phi @ sol.solution - v).abs().max().item()
    scale = v.abs().max().item()

    print(f"\n  d={d} L={L}: basis={3**d:6d}  max|residual|={residual:.3e}  "
          f"max|v|={scale:.3f}  quantum_params={vqc.n_quantum_params}")

    assert residual < TOL, (
        f"VQC output is NOT in the 3^{d} trigonometric span "
        f"(residual {residual:.3e}). Either the dequantization argument is "
        f"wrong or the circuit differs from the one analysed."
    )


def test_random_basis_does_not_fit():
    """
    Negative control. A basis of the WRONG frequencies must fail to fit, proving
    the positive result is not an artefact of over-parameterised least squares.
    """
    d, L = 3, 2
    torch.manual_seed(1)
    vqc = VQCHead(d, n_layers=L).double()
    for p in vqc.parameters():
        torch.nn.init.normal_(p, 0.0, 1.0)

    z = (torch.rand(600, d, dtype=torch.float64) - 0.5) * torch.pi
    with torch.no_grad():
        v = vqc(z)
        # frequencies 2 and 3 are NOT in the circuit's support {-1,0,1}
        omega = torch.tensor(
            [[2.0, 0, 0], [0, 2.0, 0], [0, 0, 2.0],
             [3.0, 0, 0], [2.0, 2.0, 0], [0, 2.0, 2.0]], dtype=torch.float64)
        proj = z @ omega.t()
        wrong = torch.cat([torch.cos(proj), torch.sin(proj)], dim=1)

    residual = (wrong @ torch.linalg.lstsq(wrong, v).solution - v).abs().max().item()
    print(f"\n  wrong-frequency basis: max|residual|={residual:.3e} (expected large)")
    assert residual > 1e-3, "wrong-frequency basis fit too well; test is not discriminative"


def test_exact_head_feature_count():
    for d in (2, 3, 4):
        head = FourierExactHead(d)
        assert head.n_features == 3 ** d
    with pytest.raises(ValueError):
        FourierExactHead(16)


if __name__ == "__main__":
    for d, L in [(2, 1), (3, 2), (4, 2), (4, 4)]:
        test_vqc_lies_in_exact_fourier_span(d, L)
    test_random_basis_does_not_fit()
    test_exact_head_feature_count()
    print("\nAll dequantization checks passed.")
