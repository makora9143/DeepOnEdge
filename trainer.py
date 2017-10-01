import torch 
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

from model import CNN, DoE

class LineDataset(Dataset):
    def __init__(self, npz_file, transform):
        dataset = np.load(npz_file)
        self.images = dataset['data'] # matrix of flattened images
        self.labels = dataset['label'].astype(np.long)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].reshape((224, 224))
        image = Image.fromarray(image.astype(np.uint8))
        image = self.transform(image)

        return image, self.labels[idx]

# Hyper Parameters
num_epochs = 5
batch_size = 64
learning_rate = 0.001

preprocess = transforms.Compose([transforms.ToTensor()])

train_dataset = LineDataset(npz_file='/home/makora/src/drgn_project/SPWID/spwid_train.npz', transform=preprocess)
test_dataset = LineDataset(npz_file='/home/makora/src/drgn_project/SPWID/spwid_test.npz', transform=preprocess)

# Data Loader (Input Pipeline)
train_loader = DataLoader(dataset=train_dataset,
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = DataLoader(dataset=test_dataset,
                                          batch_size=batch_size, 
                                          shuffle=False)
        
cnn = DoE()
cnn.cuda()

# Loss and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate)

# Train the Model
for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
        images = Variable(images).cuda()
        labels = Variable(labels).cuda()
        
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = cnn(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print ('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(train_dataset)//batch_size, loss.data[0]))

# Test the Model
cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
correct = 0
total = 0
for images, labels in test_loader:
    images = Variable(images).cuda()
    outputs = cnn(images)
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted.cpu() == labels).sum()

print('Test Accuracy of the model on the 10000 test images: %d %%' % (100 * correct / total))

# Save the Trained Model

ckpt = {'model': cnn.state_dict()}
torch.save(ckpt, 'cnn.pkl')
