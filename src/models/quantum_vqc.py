"""
Variational quantum circuit head.

    z_tilde -> AngleEmbedding(Y) -> StronglyEntanglingLayers(L) -> <X_i>  in R^d

Returns d expectation values, consumed by the same shared Linear(d, C)
classifier every other arm uses.

DIFF METHOD
-----------
diff_method="backprop" on default.qubit. Measured on the project hardware
(A100 MIG 3g.20gb, batch 32):

    d      backprop/GPU   adjoint/GPU   lightning/CPU
    4        0.026 s        1.79 s         4.07 s
    8        0.054 s        5.32 s         7.57 s
    16       0.357 s       >700   s        93.1  s

Backprop also builds a native torch graph, so gradient flow into the bottleneck
and layer3 is guaranteed by construction rather than by convention. (Adjoint was
verified to agree with backprop to ~3e-7 at PennyLane 0.42.3, so it is correct
here - it is simply far slower.)

Note that backprop requires default.qubit; lightning.qubit supports adjoint only.
"""
import torch
import torch.nn as nn
import pennylane as qml

from .heads import init_weights


class VQCHead(nn.Module):
    """
    n_qubits equals the bottleneck dimension d, so the quantum arm faces exactly
    the same information bottleneck as every classical arm.

    Trainable quantum parameters: 3 * L * d  (24 at d=4, L=2).
    Weights initialised N(0, 0.1): small variance delays barren plateaus, which
    matter at d=16 where gradient variance is ~62x smaller than at d=4.
    """

    def __init__(self, d: int, n_layers: int = 2,
                 device_name: str = "default.qubit",
                 diff_method: str = "backprop",
                 init_std: float = 0.1):
        super().__init__()
        self.d = d
        self.n_layers = n_layers
        self.out_dim = d
        self.n_quantum_params = 3 * n_layers * d

        dev = qml.device(device_name, wires=d)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(d), rotation="Y")
            qml.StronglyEntanglingLayers(weights, wires=range(d))
            return [qml.expval(qml.PauliX(i)) for i in range(d)]

        self.q_layer = qml.qnn.TorchLayer(
            circuit,
            {"weights": (n_layers, d, 3)},
            init_method={"weights": lambda t: nn.init.normal_(t, 0.0, init_std)},
        )

    def forward(self, z):
        return self.q_layer(z)

    def quantum_parameters(self):
        return list(self.q_layer.parameters())

    @torch.no_grad()
    def grad_variance(self):
        """Barren-plateau diagnostic. Call after backward()."""
        g = self.q_layer.weights.grad
        if g is None:
            return None
        return {"mean_abs": g.abs().mean().item(), "var": g.var().item()}
