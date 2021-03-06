from torch import nn


class CONV3(nn.Module):
    def __init__(self):
        super(CONV3, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 5, 1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            
            nn.Conv2d(32, 64, 3,  1, 1),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            
            nn.Conv2d(64, 128, 3,  1, 1),
            nn.MaxPool2d(2, 2),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            
            nn.AvgPool2d(6, 6)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 10)
        )


    def extract_features(self, inputs):
        out = self.features(inputs)
        out = out.view(-1, 128)
        return out

    def forward(self, inputs):
        out = self.extract_features(inputs)
        out = self.classifier(out)
        return out

