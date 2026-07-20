"""
Classical Baselines for Hybrid Quantum-Classical Image Classification.
Defines ResNet architectures truncated at Layer 3 for direct comparison 
against quantum latent projection across multi-class medical datasets.
"""

import torch
import torch.nn as nn
import torchvision.models as models

def init_weights(m):
    """
    Kaiming initialization for better gradient flow in deep bottlenecks.
    Strictly applied only to custom heads to preserve pre-trained backbone weights.
    """
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

class ClassicalLinearResNet(nn.Module):
    """
    Linear Baseline Model.
    Projects the high-dimensional Layer 3 feature maps into a restricted latent space, 
    followed immediately by a multi-class linear decision boundary.
    """
    def __init__(self, num_classes: int, bottleneck_dim: int = 4):
        super(ClassicalLinearResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # ResNet18 Layer 3 (layer3) is index 6 in the children list
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Robust Freezing: Unfreeze only layer3
        for i, child in enumerate(self.backbone):
            for param in child.parameters():
                param.requires_grad = (i == 6)

        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        
        self.bottleneck.apply(init_weights)
        self.classifier.apply(init_weights)

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
    expressivity against the quantum variational circuit under data scarcity constraints.
    """
    def __init__(self, num_classes: int, bottleneck_dim: int = 4):
        super(ClassicalMLPResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for i, child in enumerate(self.backbone):
            for param in child.parameters():
                param.requires_grad = (i == 6)

        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        
        self.bottleneck.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        return self.classifier(self.activation(z))


class ClassicalDeepBottleneckResNet(nn.Module):
    """
    Deep Funnel Baseline Model (Autoencoder-style Encoder).
    Gradually compresses features using deep non-linear transformations 
    (256 -> 64 -> 16 -> 4) to provide the classical model with maximum 
    representational power before the final classification head.
    """
    def __init__(self, num_classes: int, bottleneck_dim: int = 4):
        super(ClassicalDeepBottleneckResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for i, child in enumerate(self.backbone):
            for param in child.parameters():
                param.requires_grad = (i == 6)

        # Deep Non-Linear Funnel
        self.encoder = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
            nn.BatchNorm1d(64),
            nn.Linear(64, 16),
            nn.GELU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, bottleneck_dim)
        )
        
        self.classifier = nn.Linear(bottleneck_dim, num_classes)
        
        self.encoder.apply(init_weights)
        self.classifier.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)
        
        z = self.encoder(h) 
        
        return self.classifier(z)