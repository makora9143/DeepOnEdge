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
