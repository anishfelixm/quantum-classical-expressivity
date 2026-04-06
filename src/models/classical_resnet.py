import torch
import torch.nn as nn
import torchvision.models as models

class ClassicalLinearResNet(nn.Module):
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalLinearResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Slice off layer4, avgpool, and fc (stops exactly after layer3)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for name, param in self.backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Layer 3 outputs 256 channels
        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)         # Compress 14x14 spatial grid to 1x1
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        y_hat = self.classifier(z) 
        return y_hat


class ClassicalMLPResNet(nn.Module):
    def __init__(self, bottleneck_dim: int = 4):
        super(ClassicalMLPResNet, self).__init__()
        
        resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone = nn.Sequential(*list(resnet.children())[:-3])
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        
        for name, param in self.backbone.named_parameters():
            if "layer3" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        self.bottleneck = nn.Linear(256, bottleneck_dim)
        self.activation = nn.GELU()
        self.classifier = nn.Linear(bottleneck_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        h = self.pool(h)
        h = torch.flatten(h, 1)  
        z = self.bottleneck(h)   
        activated_z = self.activation(z)
        y_hat = self.classifier(activated_z) 
        return y_hat