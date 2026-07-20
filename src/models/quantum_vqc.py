"""
Hybrid Quantum-Classical ResNet Architecture.
Integrates a PennyLane Variational Quantum Circuit (VQC) as the classification head 
to evaluate quantum expressivity in severe multi-class information constraint scenarios.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml

def init_weights(m):
    """
    Kaiming initialization to ensure strict structural parity with classical baselines.
    Prevents 'Initialization Bias' by standardizing the starting classical gradients.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class QuantumHybridResNet(nn.Module):
    """
    End-to-End Hybrid Architecture.
    Maps compressed classical features into a Hilbert space using angle embedding, 
    entangles them, and generates multi-class decision logits via a projection head.
    """
    def __init__(self, num_classes: int, n_qubits: int = 4, n_layers: int = 2):
        super(QuantumHybridResNet, self).__init__()
        self.n_qubits = n_qubits
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Slice off layer4, avgpool, and fc to retain mid-level hierarchical features
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Robust Freezing: Ensure parity with classical models (Unfreeze only layer3)
        for i, child in enumerate(self.backbone):
            for param in child.parameters():
                param.requires_grad = (i == 6)
                
        # Compress Layer 3 channels (256) down to the available qubit count (Information Bottleneck)
        self.bottleneck = nn.Linear(256, self.n_qubits)
        
        # Instantiate the VQC
        self.q_layer = self._build_quantum_circuit(n_qubits, n_layers)
        
        # Multi-class classification head mapping Pauli expectation values to target dimensions
        self.classifier = nn.Linear(self.n_qubits, num_classes)
        
        # Ensure classical layers here have the exact same initialization parity
        self.bottleneck.apply(init_weights)
        self.classifier.apply(init_weights)

    def _build_quantum_circuit(self, n_qubits, n_layers):
        """
        Constructs the PennyLane QNode.
        - AngleEmbedding(Y): Keeps probability amplitudes strictly real-valued.
        - diff_method="adjoint": O(1) memory scaling.
          when scaling to 16 qubits, avoiding the 2P overhead of parameter-shift.
        - default.qubit: CPU simulator. For N <= 16, CPU matrix multiplication is faster 
          than GPU kernel dispatch overhead. TorchLayer handles the CUDA-CPU tensor routing.
        """
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch", diff_method="adjoint")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        # Small variance initialization to prevent barren plateaus in deep Hilbert spaces
        init_method = {"weights": lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1)}
        
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)          
        z = self.bottleneck(h)           
        
        z_scaled = torch.tanh(z) * (torch.pi / 2.0)
        
        # Execute VQC -> returns tensor of shape (batch_size, n_qubits)
        v_q = self.q_layer(z_scaled) 
        
        # Compute final multi-class logits via linear mapping
        y_hat = self.classifier(v_q)
        
        return y_hat