"""
ARM-PAIR AUDIT (Amendment 26): what ELSE differs between two compared arms?

    python scripts/audit_arm_pairs.py            # prints every pair; exit 1 on an unexpected difference

WHY THIS EXISTS
---------------
On 31 October 2026 a difference between quantum_reupload and quantum_vqc was
found by reading the code: besides re-uploading, their second entangling layer
uses a different CNOT range, set by a library default. Nothing in the pipeline
checked for it, because each arm was only ever checked on its own (parameter
counts, spectrum). This script checks PAIRS. For every pair of arms that a
reported contrast compares, it builds both models exactly as the experiments do
and lists every structural difference:

    projection   trainable count, policy (learned / pca / random)
    squash       tanh on or off, angle scale
    classifier   input and output width
    head         class, parameter count, parameter shapes
    circuit      the gates PennyLane actually executes, read from the device-level
                 tape (templates decomposed): number of encodings, rotation gates,
                 the CNOT pairs layer by layer, and the measured observables

Each pair declares the differences it is SUPPOSED to have. Anything else is
printed as UNEXPECTED and the script exits with code 1. A new contrast must be
added here before it is run; a new library version is re-checked by re-running
this file.

Training settings (learning rate, epochs, clipping, seeds) are not structural
and are checked elsewhere: the rate is in every run's key, and the rest is
identical by construction in the shared training loop.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from models.registry import build_arm, count_head_params        # noqa: E402

try:
    import pennylane as qml
except ImportError:                                             # classical-only audit
    qml = None

BASIC_OPS = {"CNOT", "Rot", "RY", "RX", "RZ"}

# Differences that are NOT intended but are known and disclosed in the
# manuscript. Printed as "disclosed", not failed, so the audit stays usable as a
# gate while still showing them every time.
DISCLOSED = {("re-uploading (H-S1)", "entangling pattern")}

# (name, (arm, kwargs), (arm, kwargs), d, intended differences, what the contrast is)
PAIRS = [
    ("primary", ("quantum_vqc", {}), ("matched_param_fullrank", {}), 4,
     {"head type", "parameter shapes"}, "VQC vs parameter-matched head (24 each)"),
    ("re-uploading (H-S1)", ("quantum_reupload", {}), ("quantum_vqc", {}), 4,
     {"encodings"},
     "re-upload VQC vs VQC, as registered; see DISCLOSED"),
    ("function class (H-S2)", ("quantum_vqc", {}), ("fourier_rff", {}), 4,
     {"head type", "parameter count", "parameter shapes"}, "VQC vs its 81-function fit"),
    ("readout, information (H-S7a)", ("quantum_rich", {}), ("quantum_rich_padded", {}), 4,
     {"observables"}, "10 distinct observables vs 4 repeated to width 10"),
    ("readout, headline (H-S7b)", ("quantum_rich", {}), ("quantum_vqc", {}), 4,
     {"observables", "classifier width"}, "10 outputs vs 4; the classifier widens"),
    ("restriction (H-S5)", ("low_rank", {"head_rank": 0}), ("low_rank", {"head_rank": 8}), 4,
     {"parameter count", "parameter shapes"}, "8 vs 72 parameters"),
    ("frozen projection, PCA (H-S6)",
     ("quantum_vqc", {"bottleneck_policy": "pca"}),
     ("matched_param_fullrank", {"bottleneck_policy": "pca"}), 4,
     {"head type", "parameter shapes"}, "primary contrast with the projection frozen"),
    ("frozen projection, random (H-S6)",
     ("quantum_vqc", {"bottleneck_policy": "random"}),
     ("matched_param_fullrank", {"bottleneck_policy": "random"}), 4,
     {"head type", "parameter shapes"}, "primary contrast with the projection frozen"),
    ("ansatz", ("quantum_basic", {}), ("quantum_vqc", {}), 4,
     {"rotation gates", "entangling pattern", "parameter shapes"},
     "6 basic layers vs 2 strongly entangling layers, 24 each"),
    ("re-upload function class (E7)", ("quantum_reupload", {}), ("fourier_rff_r2", {}), 4,
     {"head type", "parameter count", "parameter shapes"}, "re-upload VQC vs its 625-function fit"),
    ("re-upload vs matched", ("quantum_reupload", {}), ("matched_param_fullrank", {}), 4,
     {"head type", "parameter shapes"}, "exploratory"),
    ("d = 8", ("quantum_vqc", {}), ("low_rank", {"head_rank": 2}), 8,
     {"head type", "parameter shapes"}, "48 parameters each"),
    ("d = 16", ("quantum_vqc", {}), ("low_rank", {"head_rank": 2}), 16,
     {"head type", "parameter shapes"}, "96 parameters each"),
    ("depth L = 4", ("quantum_vqc", {"n_layers": 4}), ("quantum_vqc", {}), 4,
     {"parameter count", "parameter shapes", "rotation gates", "entangling pattern"},
     "exploratory depth sweep"),
    ("depth L = 1", ("quantum_vqc", {"n_layers": 1}), ("quantum_vqc", {}), 4,
     {"parameter count", "parameter shapes", "rotation gates", "entangling pattern"},
     "exploratory depth sweep"),
    ("entangling range (Amendment 26)", ("quantum_vqc_range1", {}), ("quantum_vqc", {}), 4,
     {"entangling pattern"}, "second layer CNOT range 1 vs 2"),
    ("re-uploading alone (Amendment 26)", ("quantum_reupload", {}), ("quantum_vqc_range1", {}), 4,
     {"encodings"}, "second encoding only"),
]


def build(arm, kw, d):
    return build_arm(arm, d=d, num_classes=2, seed=42, build_backbone=False, **kw)


def circuit_profile(head):
    """Gate-level description of a VQC head, from the tape PennyLane executes."""
    if qml is None or not hasattr(head, "q_layer"):
        return None
    x = torch.zeros(head.d, dtype=torch.float64)
    w = head.q_layer.weights.detach().double()
    tape = qml.workflow.construct_tape(head.q_layer.qnode, level="device")(x, w)
    for _ in range(10):
        if all(op.name in BASIC_OPS for op in tape.operations):
            break
        tape = tape.expand(depth=1, stop_at=lambda op: op.name in BASIC_OPS)
    ops = tape.operations
    return {
        "encodings": sum(op.name == "RY" for op in ops) // head.d,
        "rotation gates": tuple(sorted((n, sum(op.name == n for op in ops))
                                       for n in {"Rot", "RX", "RZ"})),
        "entangling pattern": tuple(tuple(int(q) for q in op.wires)
                                    for op in ops if op.name == "CNOT"),
        "observables": tuple(str(m.obs) for m in tape.measurements),
    }


def profile(model):
    head = model.head
    c = count_head_params(model)
    p = {
        "projection": (c["bottleneck"], getattr(model, "bottleneck_policy", None)),
        "squash": (getattr(model, "use_tanh", None), round(float(model.angle_scale), 6)),
        "classifier width": (model.classifier.in_features, model.classifier.out_features),
        "head type": type(head).__name__,
        "parameter count": c["head"],
        "parameter shapes": tuple(sorted((n, tuple(t.shape))
                                         for n, t in head.named_parameters() if t.requires_grad)),
    }
    circ = circuit_profile(head)
    if circ:
        p.update(circ)
    return p


def main():
    if qml is None:
        print("PennyLane not importable: circuit-level differences are NOT checked.")
    bad = 0
    for name, (arm_a, kw_a), (arm_b, kw_b), d, intended, what in PAIRS:
        pa, pb = profile(build(arm_a, kw_a, d)), profile(build(arm_b, kw_b, d))
        diffs = sorted(k for k in set(pa) | set(pb) if pa.get(k) != pb.get(k))
        unexpected = [k for k in diffs if k not in intended and (name, k) not in DISCLOSED]
        missing = [k for k in intended if k not in diffs]
        status = "UNEXPECTED DIFFERENCE" if unexpected else "ok"
        print(f"\n[{status}] {name}: {arm_a}{kw_a or ''} vs {arm_b}{kw_b or ''}, d={d}  ({what})")
        for k in diffs:
            tag = ("intended  " if k in intended else
                   "disclosed " if (name, k) in DISCLOSED else "UNEXPECTED")
            print(f"    {tag} {k}:\n        A: {pa.get(k)}\n        B: {pb.get(k)}")
        if missing:
            print(f"    note: declared but absent: {missing}")
        bad += bool(unexpected)
    print(f"\n{len(PAIRS)} pairs audited; {bad} with an unexpected difference.")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
