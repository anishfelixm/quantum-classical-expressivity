import torch
import torch.nn as nn
import torchvision.models as models
import pennylane as qml
import numpy as np

class QuantumHybridResNet(nn.Module):
    def __init__(self, n_qubits=4, n_layers=4):
        super(QuantumHybridResNet, self).__init__()
        self.n_qubits = n_qubits
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for name, param in self.backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True 
            else:
                param.requires_grad = False
                
        # Layer 3 outputs 256 channels
        self.bottleneck = nn.Linear(256, self.n_qubits)
        self.q_layer = self._build_quantum_circuit(n_qubits, n_layers)
        
        self.observable_weights = nn.Parameter(torch.ones(1, self.n_qubits)) 
        self.logit_bias = nn.Parameter(torch.tensor(0.0))

    def _build_quantum_circuit(self, n_qubits, n_layers):
        dev = qml.device("default.qubit", wires=n_qubits)
        
        @qml.qnode(dev, interface="torch")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits), rotation='Y')
            qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliX(i)) for i in range(n_qubits)]
            
        weight_shapes = {"weights": (n_layers, n_qubits, 3)}
        init_method = {"weights": lambda tensor: torch.nn.init.normal_(tensor, mean=0.0, std=0.1)}
        
        return qml.qnn.TorchLayer(circuit, weight_shapes, init_method=init_method)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)          
        z = self.bottleneck(h)           
        
        z_scaled = torch.tanh(z) * (np.pi / 2)
        v_q = self.q_layer(z_scaled) 
        v_q_weighted = v_q * self.observable_weights 
        y_hat = torch.sum(v_q_weighted, dim=1, keepdim=True) + self.logit_bias
        
        return y_hat