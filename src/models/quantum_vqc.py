import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
import numpy as np

class QuantumHybridResNet(nn.Module):
    """
    Hybrid Quantum-Classical Architecture as defined in Section IV.D.
    
    Architecture:
    1. Pre-trained ResNet-18 Backbone (layer1 through layer3 strictly frozen).
    2. Information Bottleneck Layer (Linear projection from 512 -> 4).
    3. Quantum Interface: tanh(z) * pi bounding.
    4. 4-Qubit Variational Quantum Circuit (AngleEmbedding + StronglyEntanglingLayers).
    5. Classical Post-Processing (Linear projection of 4 Pauli-Z expectations to 1 logit).
    """
    def __init__(self, n_qubits: int = 4, n_layers: int = 2):
        super(QuantumHybridResNet, self).__init__()
        self.n_qubits = n_qubits
        
        # 1. Load Pre-trained ResNet-18 Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Enforce strict freezing rules as defined in Section IV.A
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True  # Allowed to learn domain-specific hierarchies
            else:
                param.requires_grad = False # Strictly frozen
                
        # 2. Information Bottleneck (Eq 5: z = W_comp * h + b_comp)
        self.bottleneck = nn.Linear(512, self.n_qubits)
        
        # 3 & 4. Setup the Variational Quantum Circuit (VQC)
        self.q_layer = self._build_quantum_circuit(n_qubits, n_layers)
        
        # 5. Classical Post-Processing
        self.post_process = nn.Linear(self.n_qubits, 1)

    def _build_quantum_circuit(self, n_qubits: int, n_layers: int) -> qml.qnn.TorchLayer:
        """
        Constructs the Continuous-Variable to Discrete-Qubit interface and VQC.
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Eq 8: Angle Embedding via independent single-qubit Pauli-X rotations
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            
            # Parameterized Ansatz: Strongly Entangling Layers
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # Eq 9: Expectation value of the Pauli-Z observable on all wires
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        # The total number of trainable quantum parameters is L * n * 3
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        
        # Initialize weights uniformly between 0 and 2*pi
        init_method = {"weights": lambda shape: torch.rand(shape) * 2 * np.pi}
        
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass outputting the pre-sigmoid logit.
        """
        # Feature Extraction
        h = self.backbone(x)
        h = torch.flatten(h, 1)          # Shape: (Batch, 512)
        
        # Compression to Bottleneck
        z = self.bottleneck(h)           # Shape: (Batch, 4)
        
        # Eq 7: Data Scaling and Bounding (Prevent phase-wrapping)
        z_scaled = torch.tanh(z) * np.pi # Shape: (Batch, 4), Range: [-pi, pi]
        
        # Quantum State Evolution and Measurement
        v_q = self.q_layer(z_scaled)     # Shape: (Batch, 4), Range: [-1, 1]
        
        # Final Classical Post-Processing logit
        y_hat = self.post_process(v_q)   # Shape: (Batch, 1)
        
        return y_hat