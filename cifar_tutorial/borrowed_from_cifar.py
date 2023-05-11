# set batch_size
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from qr_torch.qrdata import QrData, composed_transform, SubSet


class Net(nn.Module):
    """ Models a simple Convolutional Neural Network"""

    def __init__(self):
        """ initialize the network """
        super(Net, self).__init__()
        # 3 input image channel, 6 output channels,
        # 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(3, 6, 5)
        # Max pooling over a (2, 2) window
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        ''' the forward propagation algorithm '''
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


batch_size = 4
# set number of workers
num_workers = 2

trainset = QrData(SubSet.TRAIN, transform=composed_transform)
trainloader = DataLoader(trainset, batch_size=batch_size,
                         shuffle=True, num_workers=num_workers)

testset = QrData(SubSet.TEST, transform=composed_transform)
testloader = DataLoader(trainset, batch_size=batch_size,
                        shuffle=True, num_workers=num_workers)


net = Net()
print(net)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

n_epochs = 3
cost_list = []
accuracy_list = []
N_test = len(testset)
COST = 0


def train_model(n_epochs):
    for epoch in range(n_epochs):
        COST = 0
        for x, y in trainloader:
            optimizer.zero_grad()
            z = net(x)
            loss = criterion(z, y)
            loss.backward()
            optimizer.step()
            COST += loss.data

        cost_list.append(COST)
        correct = 0
        # perform a prediction on the validation  data
        for x_test, y_test in testloader:
            z = net(x_test)
            _, yhat = torch.max(z.data, 1)
            correct += (yhat == y_test).sum().item()
        accuracy = correct / N_test
        accuracy_list.append(accuracy)


train_model(n_epochs)