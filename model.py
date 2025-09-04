# model.py
"""
Very small model factory:
- get_resnet18(num_classes): returns a ResNet18 (pretrained) with a 5-class head.
"""

import torch.nn as nn
from torchvision import models

def get_resnet18(num_classes: int = 5, pretrained: bool = True) -> nn.Module:
    """
    Create a ResNet18 model and replace its final layer for our #classes.
    """
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
    in_features = model.fc.in_features          # size of features coming into the final layer
    model.fc = nn.Linear(in_features, num_classes)  # new classifier head
    return model
