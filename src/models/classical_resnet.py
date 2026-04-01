import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalLinearResNet(nn.Module):
    """
    Baseline A: Strict Linear Probe (Testing pure representation quality).
    
    Architecture:
    1. Pre-trained ResNet-18 Backbone (layer1 through layer3 strictly frozen).
    2. Information Bottleneck Layer (Linear projection from 512 -> 4).
    3. Strict Linear Classification Head (No non-linearities).
    
    Purpose: To definitively test if the 4D bottleneck features are 
    linearly separable. Failure here proves the necessity of a non-linear 
    mapper (like the VQC or an MLP).
    """
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalLinearResNet, self).__init__()
        
        # Load Pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Enforce strict freezing rules
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Information Bottleneck
        self.bottleneck = nn.Linear(512, bottleneck_dim)
        
        # Pure Linear Classification
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        y_hat = self.classifier(z) 
        return y_hat


class ClassicalMLPResNet(nn.Module):
    """
    Baseline B: Standard Classical Head (Testing practical performance limits).
    
    Architecture:
    1. Pre-trained ResNet-18 Backbone (layer1 through layer3 strictly frozen).
    2. Information Bottleneck Layer (Linear projection from 512 -> 4).
    3. Non-Linear Classification Head (ReLU + Linear projection to 1 logit).
    
    Purpose: To serve as a fair, "best-effort" classical comparison against 
    the Quantum VQC under the exact same dimensional constraints.
    """
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalMLPResNet, self).__init__()
        
        # Load Pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Enforce strict freezing rules
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Information Bottleneck
        self.bottleneck = nn.Linear(512, bottleneck_dim)
        
        # Non-Linear Classification
        self.activation = nn.GELU()
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        activated_z = self.activation(z)
        y_hat = self.classifier(activated_z) 
        return y_hat