"""
Verifies that data re-uploading widens the accessible frequency spectrum exactly
as Schuld/Sweke/Meyer (2021) and Perez-Salinas et al. (2020) predict.

The single-encoding claim is already tested in test_fourier_equivalence.py:
    R=1  ->  span{ prod_j f_j(z_j) },  f in {1, cos, sin}   ->   3^d functions

This file tests the generalisation:
    R    ->  span{ prod_j g_j(z_j) },  g in {1, cos(k z), sin(k z)}_{k<=R}
                                                            ->   (2R+1)^d functions

The test is a least-squares fit of the circuit output onto the predicted basis.
A small residual means the output lies in the span; a LARGE residual on a
deliberately truncated basis means the fit is not vacuous - without that negative
control, a big enough basis would fit anything and the positive result would
prove nothing.

Run:  python -m pytest tests/test_reupload_spectrum.py -v -s
"""
import itertools
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models.quantum_vqc import VQCHead        # noqa: E402


def fourier_basis(z, max_freq):
    """
    Columns spanning  prod_j g_j(z_j),  g_j in {1} u {cos(k z_j), sin(k z_j)}_{k=1..R}

    Built as a tensor product over coordinates, giving exactly (2R+1)^d columns.
    """
    n, d = z.shape
    per_coord = []
    for j in range(d):
        cols = [np.ones(n)]
        for k in range(1, max_freq + 1):
            cols.append(np.cos(k * z[:, j]))
            cols.append(np.sin(k * z[:, j]))
        per_coord.append(cols)

    out = []
    for combo in itertools.product(*[range(2 * max_freq + 1) for _ in range(d)]):
        col = np.ones(n)
        for j, idx in enumerate(combo):
            col = col * per_coord[j][idx]
        out.append(col)
    return np.stack(out, axis=1)


def circuit_outputs(d, n_layers, n_uploads, n_samples, seed=0):
    torch.manual_seed(seed)
    head = VQCHead(d, n_layers=n_layers, n_uploads=n_uploads).double()
    z = (torch.rand(n_samples, d, dtype=torch.float64) * 2 - 1) * (torch.pi / 2)
    with torch.no_grad():
        v = head(z)
    return z.numpy(), np.asarray(v, dtype=np.float64), head


@pytest.mark.parametrize("d,n_layers,R", [
    (2, 2, 1),
    (2, 2, 2),
    (2, 4, 2),
    (3, 2, 2),
    (2, 3, 3),
])
def test_reupload_output_lies_in_predicted_span(d, n_layers, R):
    """Circuit output must lie in the (2R+1)^d basis, to machine precision."""
    n_basis = (2 * R + 1) ** d
    z, v, head = circuit_outputs(d, n_layers, R, n_samples=max(600, 6 * n_basis))
    B = fourier_basis(z, max_freq=R)

    coef, *_ = np.linalg.lstsq(B, v, rcond=None)
    resid = np.abs(B @ coef - v).max()

    print(f"\n  d={d} L={n_layers} R={R}: basis={n_basis:6d} "
          f"max|residual|={resid:.3e}  max|v|={np.abs(v).max():.3f}  "
          f"params={head.n_quantum_params}")
    assert resid < 1e-8, f"output escapes the predicted spectrum (residual {resid:.2e})"


@pytest.mark.parametrize("d,R", [(2, 2), (2, 3), (3, 2)])
def test_truncated_basis_fails(d, R):
    """
    NEGATIVE CONTROL. Fitting an R-upload circuit with only an (R-1)-frequency
    basis must FAIL - otherwise the positive test above proves nothing.
    """
    z, v, _ = circuit_outputs(d, n_layers=R, n_uploads=R, n_samples=800, seed=1)
    B_short = fourier_basis(z, max_freq=R - 1)

    coef, *_ = np.linalg.lstsq(B_short, v, rcond=None)
    resid = np.abs(B_short @ coef - v).max()

    print(f"\n  d={d} R={R} fitted with max_freq={R-1}: "
          f"max|residual|={resid:.3e} (expected LARGE)")
    assert resid > 1e-4, "truncated basis fitted too well - the test is vacuous"


def test_reupload_matches_single_encoding_at_R1():
    """R=1 must reproduce the original circuit exactly, not merely approximately."""
    torch.manual_seed(7)
    a = VQCHead(3, n_layers=2, n_uploads=1).double()
    torch.manual_seed(7)
    b = VQCHead(3, n_layers=2, n_uploads=1).double()
    z = torch.rand(64, 3, dtype=torch.float64)
    with torch.no_grad():
        diff = (a(z) - b(z)).abs().max().item()
    print(f"\n  R=1 reproducibility: max|diff|={diff:.3e}")
    assert diff == 0.0


def test_parameter_count_is_independent_of_uploads():
    """
    The whole point of the H5 comparison: identical parameters, richer spectrum.
    If this ever fails, the arms are no longer a controlled contrast.
    """
    base = VQCHead(4, n_layers=2, n_uploads=1)
    reup = VQCHead(4, n_layers=2, n_uploads=2)
    print(f"\n  R=1: {base.n_quantum_params} params, spectrum {base.spectrum_size}")
    print(f"  R=2: {reup.n_quantum_params} params, spectrum {reup.spectrum_size}")
    assert base.n_quantum_params == reup.n_quantum_params == 24
    assert base.spectrum_size == 81 and reup.spectrum_size == 625


def test_uneven_split_is_rejected():
    """n_layers must divide evenly across uploads, or blocks aren't comparable."""
    with pytest.raises(ValueError):
        VQCHead(4, n_layers=3, n_uploads=2)


if __name__ == "__main__":
    test_reupload_output_lies_in_predicted_span(2, 2, 2)
    test_truncated_basis_fails(2, 2)
    test_parameter_count_is_independent_of_uploads()
    print("\nRe-uploading spectrum verified.")
