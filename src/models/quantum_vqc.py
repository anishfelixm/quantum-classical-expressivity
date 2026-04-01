import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
import numpy as np

class QuantumHybridResNet(nn.Module):
    """
    Hybrid Quantum-Classical Architecture with Pure Quantum Readout.
    """
    def __init__(self, n_qubits=4, n_layers=2):
        super(QuantumHybridResNet, self).__init__()
        self.n_qubits = n_qubits
        
        # 1. Load Pre-trained ResNet-18 Backbone
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False
                
        # 2. Information Bottleneck
        self.bottleneck = nn.Linear(512, self.n_qubits)
        
        # 3. Setup the Variational Quantum Circuit
        self.q_layer = self._build_quantum_circuit(n_qubits, n_layers)
        
        # 4. Pure Quantum Readout Scalar (Replaces nn.Linear post_process)
        # Maps bounded expectation [-1, 1] to PyTorch logit space [-inf, inf]
        self.logit_scale = nn.Parameter(torch.tensor(5.0))
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def _build_quantum_circuit(self, n_qubits, n_layers):
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='X')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            
            # FORCE QUANTUM DECISION: Readout from all 4 qubits
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        init_method = {"weights": lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1)}
        
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = torch.flatten(h, 1)          
        z = self.bottleneck(h)           
        
        # Monotonic Phase Mapping
        z_scaled = torch.sigmoid(z) * np.pi 
        
        # Quantum State Evolution
        v_q = self.q_layer(z_scaled)

        # Aggregate the 4 qubit expectations into a single consensus prediction
        v_q_consensus = torch.mean(v_q, dim=1, keepdim=True)     
        
        # Pure Quantum Readout (Stretched to Logit Space)
        y_hat = (v_q_consensus * self.logit_scale + self.logit_bias).view(-1, 1)
        
        return y_hat