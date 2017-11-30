import torch
import torch.nn as nn
from torchvision import models
# from torchvision.models import AlexNet


class AlexNet(nn.Module):

    def __init__(self, pretrained=True, include_fc7=True):
        super(AlexNet, self).__init__()
        original_model = models.alexnet(pretrained=pretrained)

        self.features = nn.Sequential(
            *list(original_model.features.children())
        )

        self.classifier = nn.Sequential(
            *list(original_model.classifier.children())
        )

    def extract_features(self, inputs):
        out = self.features(inputs)
        out = out.view(out.size(0), 256 * 6 * 6)
        return out

    def forward(self, inputs):
        out = self.extract_features(inputs)
        out = self.classifier(out)
        return out
