"""
Classical Baselines for Hybrid Quantum-Classical Image Classification.
Defines ResNet architectures truncated at Layer 3 for direct comparison 
against quantum latent projection.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalLinearResNet(nn.Module):
    """
    Linear Baseline Model.
    Projects the high-dimensional Layer 3 feature maps into a restricted latent space, 
    followed immediately by a linear decision boundary.
    """
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalLinearResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Slice off layer4, avgpool, and fc to retain mid-level hierarchical features
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Layer targeted freezing
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "6." in name

        # ResNet18 Layer 3 yields 256 channels
        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        return self.classifier(z)


class ClassicalMLPResNet(nn.Module):
    """
    Non-Linear Baseline Model (MLP).
    Introduces a GELU activation within the latent space to test classical non-linear 
    expressivity against the quantum variational circuit under data scarcity.
    """
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalMLPResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for name, param in self.backbone.named_parameters():
            param.requires_grad = "6." in name

        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        activated_z = self.activation(z)
        return self.classifier(activated_z)