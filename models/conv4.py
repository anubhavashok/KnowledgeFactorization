from torch import nn


class CONV4(nn.Module):
    def __init__(self, num_outputs, num_channels=1):
        super(CONV4, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(num_channels, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 64, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),

            nn.AvgPool2d(6, 6)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, num_outputs)
        )

        self.name = 'conv4'

    def forward(self, inputs):
        out = self.features(inputs)
        out = out.view(-1, 128)
        out = self.classifier(out)
        return out



class FinetuneModel(nn.Module):

    def __init__(self, original_model, arch, num_outputs, freeze=True):
        super(FinetuneModel, self).__init__()

        self.arch = arch

        if arch == 'conv4':
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(0.2),
                nn.Linear(128, 64),
                nn.ReLU(True),
                nn.Linear(64, num_outputs)
            )
            # self.name = arch

        if freeze:
            # Freeze those weights
            for p in self.features.parameters():
                p.requires_grad = False

            for p in self.classifier.parameters():
                p.requires_grad = True

    def forward(self, x):
        f = self.features(x)
        if self.arch == 'conv4':
            f = f.view(-1, 128)
        y = self.classifier(f)
        return y

