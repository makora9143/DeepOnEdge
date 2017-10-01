import torch.nn as nn
import torch.nn.functional as F

n_classes = 2

# CNN Model (2 conv layer)
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Linear(32, n_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #print(out.size())
        out = F.avg_pool2d(out, 56)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class DoE(nn.Module):
    def __init__(self):
        super(DoE, self).__init__()
        self.feature = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),

                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2))

        self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(32, 50),
                nn.ReLU(),
                nn.Dropout(),

                nn.Linear(50, 3))

    def forward(self, x):
        out = self.feature(x)
        out = F.avg_pool2d(out, 14).squeeze()
        out = self.classifier(out)
        return out


