"""
Hybrid Quantum-Classical ResNet Architecture.
Integrates a PennyLane Variational Quantum Circuit (VQC) as the classification head 
to evaluate quantum expressivity in severe multi-class information constraint scenarios.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
import numpy as np

class QuantumHybridResNet(nn.Module):
    """
    End-to-End Hybrid Architecture.
    Maps compressed classical features into a Hilbert space using angle embedding, 
    entangles them, and generates multi-class decision logits via a projection head.
    """
    def __init__(self, num_classes: int, n_qubits: int = 4, n_layers: int = 4):
        super(QuantumHybridResNet, self).__init__()
        self.n_qubits = n_qubits
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Layer targeted freezing (Latent Reshaping tracking)
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "6." in name
                
        # Compress Layer 3 channels (256) down to the available qubit count (Information Bottleneck)
        self.bottleneck = nn.Linear(256, self.n_qubits)
        self.q_layer = self._build_quantum_circuit(n_qubits, n_layers)
        
        # Multi-class classification head mapping Pauli expectation values to target dimensions
        self.classifier = nn.Linear(self.n_qubits, num_classes)

    def _build_quantum_circuit(self, n_qubits, n_layers):
        """
        Constructs the PennyLane QNode utilizing Y-axis angle embedding 
        and strongly entangling topological layers.
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            # Encode classical data into quantum phase angles
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            # Variational sequence
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            # Measure expectation value along the X basis for all qubits
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        init_method = {"weights": lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1)}
        
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)          
        z = self.bottleneck(h)           
        
        # Scale continuous classical features to valid rotational angles [-pi/2, pi/2]
        z_scaled = torch.tanh(z) * (np.pi / 2)
        
        # Execute VQC -> returns tensor of shape (batch_size, n_qubits)
        v_q = self.q_layer(z_scaled) 
        
        # Compute final multi-class logits via linear mapping
        y_hat = self.classifier(v_q)
        
        return y_hat
