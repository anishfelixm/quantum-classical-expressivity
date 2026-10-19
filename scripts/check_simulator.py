"""
Before Amendment 23(b) runs: does the independent NumPy simulator of
tests/test_independent_theory.py follow PennyLane's conventions exactly?

    python scripts/check_simulator.py          # seconds, CPU

Two checks, on random weights (no trained model needed):
 1. noise-free: the simulator's <X_i> against the project's VQCHead
    (default.qubit) - must agree within 1e-6 for every input and output;
 2. per-layer depolarising: the simulator (where="layer") against PennyLane's
    default.mixed with a DepolarizingChannel on every wire after every
    StronglyEntanglingLayer - must agree within 1e-6.
Amendment 23(b) is computed with the simulator only if check 1 holds for the
trained weights themselves; this script finds a convention mismatch before the
trained weights exist.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import codeat                                          # noqa: E402


def main():
    codeat.setup(None)
    import numpy as np
    import pennylane as qml
    import torch
    from models.quantum_vqc import VQCHead
    sim = codeat.load_by_path("independent_theory",
                              os.path.join(codeat.REPO, "tests", "test_independent_theory.py"))
    worst1 = worst2 = 0.0
    for d, L in ((4, 2), (4, 1), (4, 4)):
        torch.manual_seed(7)
        head = VQCHead(d, n_layers=L, n_uploads=1)
        with torch.no_grad():
            head.q_layer.weights.normal_(0.0, 1.0)
        W = head.q_layer.weights.detach().double().numpy()
        Z = (np.random.default_rng(3).random((64, d)) * 2 - 1) * (np.pi / 2)
        with torch.no_grad():
            v_pl = head(torch.as_tensor(Z, dtype=torch.float32)).double().numpy()
        v_np = np.array([sim.run(z, W, d, L) for z in Z])
        e1 = float(np.abs(v_pl - v_np).max())
        worst1 = max(worst1, e1)

        dev = qml.device("default.mixed", wires=d)

        @qml.qnode(dev)
        def noisy(z, w, p):
            qml.AngleEmbedding(z, wires=range(d), rotation="Y")
            for l in range(L):
                # one layer at a time, so the noise can sit between layers; the
                # CNOT range must then be given explicitly, as the full template
                # would use it for layer l: (l mod (d-1)) + 1
                qml.StronglyEntanglingLayers(w[l:l + 1], wires=range(d), ranges=[(l % (d - 1)) + 1])
                for q in range(d):
                    qml.DepolarizingChannel(p, wires=q)
            return [qml.expval(qml.PauliX(i)) for i in range(d)]
        e2 = 0.0
        for p in (0.01, 0.05):
            for z in Z[:16]:
                a = np.array(noisy(z, W, p), dtype=float)
                b = sim.run(z, W, d, L, p=p, where="layer")
                e2 = max(e2, float(np.abs(a - b).max()))
        worst2 = max(worst2, e2)
        print(f"d={d} L={L}: noise-free max|diff| {e1:.2e}; per-layer depolarising max|diff| {e2:.2e}")
    ok = worst1 < 1e-6 and worst2 < 1e-6
    print(f"\nSIMULATOR CHECK {'PASSED' if ok else 'FAILED'} "
          f"(noise-free {worst1:.1e}, per-layer {worst2:.1e}; required < 1e-6)")


if __name__ == "__main__":
    main()
