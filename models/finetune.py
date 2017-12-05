from torch import nn


class FinetuneModel(nn.Module):
    def __init__(self, original_model, arch, num_outputs, freeze_features=True):
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

        if arch == 'vgg19':
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Linear(1024, num_outputs)
            )

        if arch == 'alexnet':
            self.features = original_model.features
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * 6 * 6, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_outputs)
            )

        if freeze_features:
            # Freeze those weights
            for p in self.features.parameters():
                p.requires_grad = False

            for p in self.classifier.parameters():
                p.requires_grad = True

    def extract_features(self, inputs):
        # extracts the features before classifier layer and flattens them (for MDS training)
        out = self.features(inputs)
        if self.arch == 'conv4':
            out = out.view(-1, 128)
        if self.arch == 'vgg19':
            out = out.view(out.size(0), -1)
        if self.arch == 'alexnet':
            out = out.view(inputs.size(0), 256 * 6 * 6)
        return out

    def forward(self, inputs):
        out = self.extract_features(inputs)
        y = self.classifier(out)
        return y
