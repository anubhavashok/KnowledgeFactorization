import torch
import sys
sys.path.append('..')
from models import conv4
class ModifiedVGG16Model(torch.nn.Module):
        def __init__(self):
                super(ModifiedVGG16Model, self).__init__()

                model = torch.load('../models/conv4_5classes.net')#('cifar10_new.net')#models.vgg19_bn()
                self.features = model.features

                for param in self.features.parameters():
                        param.requires_grad = True

                self.classifier = model.classifier#nn.Sequential(
                    #nn.Linear(1024, 10))

        def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
