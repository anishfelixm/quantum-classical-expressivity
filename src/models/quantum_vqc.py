"""
Variational quantum circuit head, with optional data re-uploading.

    z_tilde -> [AngleEmbedding(Y) -> StronglyEntanglingLayers]^R -> <X_i>  in R^d

Returns d expectation values, consumed by the same shared Linear(d, C)
classifier every other arm uses.

WHY RE-UPLOADING IS A SEPARATE ARM
----------------------------------
With a single encoding (R=1), RY(theta) = exp(-i*theta*Y/2) has generator Y/2
with eigenvalues +-1/2. Schuld, Sweke & Meyer (2021) show the accessible
frequency spectrum is the set of eigenvalue DIFFERENCES, here {-1, 0, +1}, so
the model can only represent degree-one trigonometric polynomials per input.
Across d features the joint spectrum is {-1,0,1}^d: 81 basis functions at d=4.

Perez-Salinas et al. (2020) establish that repeating the encoding R times widens
this to {-R..R} per coordinate, and that expressivity tracks spectral richness -
universality requires it.

So if the single-encoding VQC underperforms, two explanations compete:
    (a) the quantum coefficient manifold is a poor inductive bias, or
    (b) the frequency spectrum is simply too narrow.
Only (b) is testable by widening the spectrum, and this arm does exactly that.

THE PARAMETER BUDGET IS DELIBERATELY IDENTICAL
----------------------------------------------
    quantum_vqc      R=1, 2 layers          ->  3*2*d = 24 params, 3^d =  81 basis
    quantum_reupload R=2, 1 layer per block ->  3*2*d = 24 params, 5^d = 625 basis

Same trainable parameters, same readout width, same everything else. The only
difference is spectral richness. That is what makes this a clean control rather
than another confounded comparison.

DIFF METHOD
-----------
diff_method="backprop" on default.qubit. Measured on the project hardware
(A100 MIG 3g.20gb, batch 32):

    d      backprop/GPU   adjoint/GPU   lightning/CPU
    4        0.026 s        1.79 s         4.07 s
    8        0.054 s        5.32 s         7.57 s
    16       0.357 s       >700   s        93.1  s

Backprop builds a native torch graph, so gradient flow into the bottleneck and
layer3 is guaranteed by construction rather than by convention. (Adjoint was
verified to agree with backprop to ~3e-7 at PennyLane 0.42.3, so it is correct
here - simply far slower.) Backprop requires default.qubit; lightning.qubit
supports adjoint only.

Re-uploading roughly multiplies circuit depth by R, so expect ~R x the step time.
"""
import torch
import torch.nn as nn
import pennylane as qml

from .heads import init_weights


class VQCHead(nn.Module):
    """
    n_qubits equals the bottleneck dimension d, so the quantum arm faces exactly
    the same information bottleneck as every classical arm.

    Trainable quantum parameters: 3 * n_layers * d, independent of n_uploads.
    Weights initialised N(0, 0.1): small variance delays barren plateaus, which
    matter at d=16 where gradient variance is ~62x smaller than at d=4.

    Args:
        d:          qubits == bottleneck dimension
        n_layers:   total StronglyEntanglingLayers, split evenly across uploads
        n_uploads:  R. 1 reproduces the original single-encoding circuit exactly.
    """

    def __init__(self, d: int, n_layers: int = 2, n_uploads: int = 1,
                 device_name: str = "default.qubit",
                 diff_method: str = "backprop",
                 init_std: float = 0.1):
        super().__init__()
        if n_layers % n_uploads != 0:
            raise ValueError(
                f"n_layers ({n_layers}) must be divisible by n_uploads "
                f"({n_uploads}) so every encoding block gets equal depth - "
                f"otherwise the blocks are not comparable.")

        self.d = d
        self.n_layers = n_layers
        self.n_uploads = n_uploads
        self.out_dim = d
        self.n_quantum_params = 3 * n_layers * d
        # Accessible frequency spectrum: {-R..R}^d, i.e. (2R+1)^d basis functions.
        self.spectrum_size = (2 * n_uploads + 1) ** d

        layers_per_block = n_layers // n_uploads
        dev = qml.device(device_name, wires=d)

        @qml.qnode(dev, interface="torch", diff_method=diff_method)
        def circuit(inputs, weights):
            for r in range(n_uploads):
                qml.AngleEmbedding(inputs, wires=range(d), rotation="Y")
                lo = r * layers_per_block
                qml.StronglyEntanglingLayers(weights[lo:lo + layers_per_block],
                                             wires=range(d))
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

    def describe(self):
        """Reported in the manuscript's parity table."""
        return {
            "qubits": self.d,
            "layers": self.n_layers,
            "uploads": self.n_uploads,
            "quantum_params": self.n_quantum_params,
            "max_frequency": self.n_uploads,
            "spectrum_size": self.spectrum_size,
        }

    @torch.no_grad()
    def grad_variance(self):
        """Barren-plateau diagnostic. Call after backward()."""
        g = self.q_layer.weights.grad
        if g is None:
            return None
        return {"mean_abs": g.abs().mean().item(), "var": g.var().item()}
