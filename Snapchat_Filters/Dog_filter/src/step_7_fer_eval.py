from torch.utils.data import Dataset
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import torch
import cv2
import argparse

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

def evaluate(outputs: Variable, labels: Variable, normalized: bool=True
             ) -> float:
    """Evaluate neural network outputs against non-one-hotted labels."""
    Y = labels.data.numpy()
    Yhat = np.argmax(outputs.data.numpy(), axis=1)
    denom = Y.shape[0] if normalized else 1
    return float(np.sum(Yhat == Y) / denom)


def batch_evaluate(net: Net, dataset: Dataset, batch_size: int=500) -> float:
    """Evaluate neural network in batches, if dataset is too large."""
    score = 0.0
    n = dataset.X.shape[0]
    for i in range(0, n, batch_size):
        x = dataset.X[i: i + batch_size]
        y = dataset.Y[i: i + batch_size]
        score += evaluate(net(x), y, False)
    return score / n

def get_image_to_emotion_predictor(model_path='assets/model_best.pth'):
    """Returns predictor, from image to emotion index."""
    net = Net().float()
    pretrained_model = torch.load(model_path)
    net.load_state_dict(pretrained_model['state_dict'])

    def predictor(image: np.array):
        """Translates images into emotion indices."""
        if image.shape[2] > 1:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(image, (48, 48)).reshape((1, 1, 48, 48))
        X = Variable(torch.from_numpy(frame)).float()
        return np.argmax(net(X).data.numpy(), axis=1)[0]
    return predictor

def main():
    trainset = Fer2013Dataset('data/fer2013_train.npz')
    testset = Fer2013Dataset('data/fer2013_test.npz')
    net = Net().float()

    pretrained_model = torch.load("assets/model_best.pth")
    net.load_state_dict(pretrained_model['state_dict'])

    train_acc = batch_evaluate(net, trainset, batch_size=500)
    print('Training accuracy: %.3f' % train_acc)
    test_acc = batch_evaluate(net, testset, batch_size=500)
    print('Validation accuracy: %.3f' % test_acc)


if __name__ == '__main__':
    main()
