"""
Convolutional neural network for Face Emotion Recognition (FER) 2013 Dataset
"""

from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np


class Fer2013Dataset(Dataset):
    """Face Emotion Recognition dataset.

    Utility for loading FER into PyTorch. Dataset curated by Pierre-Luc Carrier
    and Aaron Courville in 2013.

    Each sample is 1 x 1 x 48 x 48, and each label is a scalar.
    """

    def __init__(self, path: str):
        """
        Args:
            path: Path to `.np` file containing sample nxd and label nx1
        """
        with np.load(path) as data:
            self._samples = data['X']
            self._labels = data['Y']
        self._samples = self._samples.reshape((-1, 1, 48, 48))

        self.X = Variable(torch.from_numpy(self._samples)).float()
        self.Y = Variable(torch.from_numpy(self._labels)).float()

    def __len__(self):
        return len(self._labels)

    def __getitem__(self, idx):
        return {'image': self._samples[idx], 'label': self._labels[idx]}


trainset = Fer2013Dataset('data/fer2013_train.npz')
trainloader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)

testset = Fer2013Dataset('data/fer2013_test.npz')
testloader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 6, 3)
        self.conv3 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 48)
        self.fc3 = nn.Linear(48, 3)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net().float()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        inputs = Variable(data['image'].float())
        labels = Variable(data['label'].long())
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.data[0]
        if i % 100 == 0:
            print('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / (i + 1)))
