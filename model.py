import torch
import torch.nn as nn
from torchvision.models import googlenet

class PhishingModel(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.model = googlenet(pretrained=False, aux_logits=False)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        return self.model(x)

