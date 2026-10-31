"""
ENTANGLING RANGES (Amendment 26).

On 31 October 2026 it was found by reading the code, not by looking at any
result, that quantum_reupload and quantum_vqc differ in more than re-uploading.
PennyLane's StronglyEntanglingLayers sets the CNOT range of layer l to
(l mod (d-1)) + 1 when `ranges` is not given, and the count restarts at every
call. quantum_vqc calls it once with two layers (ranges 1, 2); quantum_reupload
calls it once per encoding block with one layer (ranges 1, 1).

These tests pin that down so it cannot be forgotten again, and check the arm
that separates the two effects:

1. The CNOT pattern of every circuit arm, read from the circuit PennyLane
   actually runs (device level), not from our own reading of its defaults.
2. quantum_vqc_range1 has exactly the entangling pattern of quantum_reupload.
3. Passing ranges=(1, 2) explicitly reproduces quantum_vqc's outputs bit for bit,
   which confirms that (1, 2) is what the default does.
4. Parameter parity: quantum_vqc_range1 has 3*L*d parameters, like quantum_vqc.
5. quantum_vqc_range1 still lies in the 3^d trigonometric span (the encoding,
   not the entangler, fixes the spectrum).

Run:  python -m pytest tests/test_entangling_ranges.py -v -s
"""
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pennylane as qml                                         # noqa: E402
from models.registry import build_arm, count_head_params        # noqa: E402
from models.quantum_vqc import VQCHead                          # noqa: E402
from models.classical_fourier import exact_basis                # noqa: E402

BASIC_OPS = {"CNOT", "Rot", "RY", "RX", "RZ"}


def device_ops(head):
    """The gate list PennyLane executes for this head, templates decomposed."""
    x = torch.zeros(head.d, dtype=torch.float64)
    w = head.q_layer.weights.detach().double()
    tape = qml.workflow.construct_tape(head.q_layer.qnode, level="device")(x, w)
    for _ in range(10):                          # decompose anything left over
        if all(op.name in BASIC_OPS for op in tape.operations):
            break
        tape = tape.expand(depth=1, stop_at=lambda op: op.name in BASIC_OPS)
    return tape.operations


def cnot_pairs(head):
    return [tuple(int(w) for w in op.wires) for op in device_ops(head) if op.name == "CNOT"]


def ring(d, r):
    return [(i, (i + r) % d) for i in range(d)]


def head_of(arm, d=4):
    return build_arm(arm, d=d, num_classes=2, seed=42, build_backbone=False).head


@pytest.mark.parametrize("d", [4, 8])
def test_cnot_patterns_of_every_circuit_arm(d):
    expected = {
        "quantum_vqc": ring(d, 1) + ring(d, 2),          # one call, ranges 1, 2
        "quantum_reupload": ring(d, 1) + ring(d, 1),     # one layer per call
        "quantum_vqc_range1": ring(d, 1) + ring(d, 1),
        "quantum_rich": ring(d, 1) + ring(d, 2),
        "quantum_rich_padded": ring(d, 1) + ring(d, 2),
    }
    for arm, want in expected.items():
        got = cnot_pairs(head_of(arm, d))
        print(f"\n  d={d} {arm:22s} CNOTs {got}")
        assert got == want, f"{arm} at d={d}: CNOTs {got}, expected {want}"


def test_range1_has_the_reupload_entangling_pattern():
    assert cnot_pairs(head_of("quantum_vqc_range1")) == cnot_pairs(head_of("quantum_reupload"))


def test_explicit_ranges_1_2_reproduce_the_default_vqc():
    torch.manual_seed(0)
    a = VQCHead(4, n_layers=2).double()
    b = VQCHead(4, n_layers=2, ranges=(1, 2)).double()
    b.q_layer.weights.data.copy_(a.q_layer.weights.data)
    z = (torch.rand(64, 4, dtype=torch.float64) - 0.5) * torch.pi
    assert torch.equal(a(z), b(z)), "ranges=(1, 2) is not what the default does"


@pytest.mark.parametrize("d,want", [(4, 24), (8, 48), (16, 96)])
def test_parameter_parity(d, want):
    m = build_arm("quantum_vqc_range1", d=d, num_classes=2, seed=42, build_backbone=False)
    n = count_head_params(m)["head"]
    print(f"\n  quantum_vqc_range1 d={d}: {n} head parameters")
    assert n == want


def test_range1_lies_in_the_single_encoding_span():
    torch.manual_seed(1)
    h = VQCHead(4, n_layers=2, ranges=(1, 1)).double()
    torch.nn.init.normal_(h.q_layer.weights, 0.0, 1.0)
    z = (torch.rand(400, 4, dtype=torch.float64) - 0.5) * torch.pi
    with torch.no_grad():
        v = h(z)
        phi = exact_basis(z).double()
        sol = torch.linalg.lstsq(phi, v)
        resid = (phi @ sol.solution - v).abs().max().item()
    print(f"\n  quantum_vqc_range1 in the 3^d span: max |residual| = {resid:.2e}")
    assert resid < 1e-8
