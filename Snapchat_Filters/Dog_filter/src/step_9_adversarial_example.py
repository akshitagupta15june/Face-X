"""
Adversarial Example generator for face emotion recognition neural network
"""

from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import cv2


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
        self.r = nn.Parameter(data=torch.zeros(1, 1, 48, 48), requires_grad=True)

    def forward(self, x):
        x = x + self.r
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net().float()
for param in net.parameters():
    param.requires_grad = False
net.r.requires_grad = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD([net.r], lr=0.05, momentum=0.9)

pretrained_model = torch.load('assets/model_best.pth')
net.load_state_dict(pretrained_model['state_dict'], strict=False)

target_label = 1
image = cv2.imread(
    'assets/example.png', cv2.IMREAD_GRAYSCALE).reshape(1, 1, 48, 48)
inputs = Variable(torch.from_numpy(image).float())
labels = Variable(torch.from_numpy(np.array([target_label])).long())

for i in range(2000):
    inputs.data.clamp_(0, 255)
    optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    # print statistics
    if i % 100 == 0:
        print('loss: %.3f' % loss.data[0])

    # if np.argmax(net(inputs).data.numpy()) == target_label:
    #     break

# print successful or not
outputs = net(inputs).data.numpy()
success = np.argmax(outputs) == target_label
print('Adversarial example generation %s' % (
    'successful' if success else 'failed'))

# save r
np.save('outputs/example_adversarial_r.npy', net.r.data.numpy())

# save new image
inputs.data.clamp_(0, 255)
image = inputs.data.numpy().reshape((48, 48, 1))
cv2.imwrite('outputs/example_adversarial.png', image)
print('Wrote adversarial example to outputs/example_adversarial.png')
