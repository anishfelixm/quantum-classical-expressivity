import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalResNetBottleneck(nn.Module):
    """
    Classical Baseline Architecture as defined in Section IV.C.
    
    Architecture:
    1. Pre-trained ResNet-18 Backbone (layer1 through layer3 strictly frozen).
    2. Information Bottleneck Layer (Linear projection from 512 -> 4).
    3. Classical Classification Head (ReLU activation + Linear layer to 1 logit).
    """
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalResNetBottleneck, self).__init__()
        
        # 1. Load Pre-trained ResNet-18
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Isolate the backbone (remove the original 1000-class fc layer)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        # Enforce strict freezing rules as defined in Section IV.A
        # layer1 through layer3 are permanently frozen. layer4 remains unfrozen.
        for name, param in self.backbone.named_parameters():
            if "layer4" in name:
                param.requires_grad = True  # Allowed to learn domain-specific hierarchies
            else:
                param.requires_grad = False # Strictly frozen to retain ImageNet filters

        # 2. Information Bottleneck (Eq 1: z = W_comp * h + b_comp)
        self.bottleneck = nn.Linear(512, bottleneck_dim)
        
        # 3. Classical Classification Head (Eq 2: y_c = W_clf * ReLU(z) + b_clf)
        self.activation = nn.ReLU()
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass outputting the pre-sigmoid logit for BCEWithLogitsLoss.
        """
        # Feature Extraction
        h = self.backbone(x)
        h = torch.flatten(h, 1)  # Shape: (Batch, 512)
        
        # Compression to Bottleneck
        z = self.bottleneck(h)   # Shape: (Batch, 4)
        
        # Classical Classification
        activated_z = self.activation(z)
        y_hat = self.classifier(activated_z) # Shape: (Batch, 1)
        
        return y_hat