import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn
from torchvision.models import resnet18

from qr_torch.helpers import imshow
#from qr_torch.nets import CifarNet
from qr_torch.qrdata import QrData, SubSet, composed_transform, Label

# PATH = './cifar_net.pth'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 15
batch_size_valid = 10

# trainsetC10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = QrData(SubSet.TRAIN, transform=composed_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=0)

# trainsetC10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
validset = QrData(SubSet.VALID, transform=composed_transform)
validloader = torch.utils.data.DataLoader(validset, batch_size=batch_size_valid,
                                          shuffle=True, num_workers=0)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testset = QrData(SubSet.TEST, transform=composed_transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=1,
                                         shuffle=False, num_workers=0)

classes = [c.name for c in Label]

# get some random training images
dataiter = iter(trainloader)
images, labels = next(dataiter)

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))


import torch.optim as optim

net = resnet18(pretrained=True)
mean = [.485, .456, .406]
std = [.229, .224, .225]

composed = transforms.Compose([transforms.Resize(224),
                               transforms.ToTensor(),
                               transforms.Normalize(mean, std)])

for param in net.parameters():
    param.requires_grad=False

net.fc = nn.Linear(512, 2)

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = torch.optim.Adam([parameters for parameters in net.parameters() if parameters.requires_grad], lr=.003)

print("Training")

loss_list=[]
accuracy_list = []
correct = 0
n_test = len(validset)

for epoch in range(20):  # loop over the dataset multiple times
    loss_sublist = []
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        net.train()
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss_sublist.append(loss.data.item())
        loss.backward()
        optimizer.step()

    loss_list.append(np.mean(loss_list))

    for i, data in enumerate(validloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        net.eval()
        z = net(inputs)

        _, yhat = torch.max(z.data, 1)
        correct += (yhat == labels).sum().item()

    accuracy = correct / n_test
    accuracy_list.append(accuracy)

print('Finished Training')

torch.save(net.state_dict(), "resnet.pth")


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

"""
net = CifarNet()
net.load_state_dict(torch.load(CifarNet.pth_fname()))
"""
outputs = net(images)

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join(f'{classes[predicted[j]]:5s}'
                              for j in range(len(predicted))))

"""
correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for i_d, data in enumerate(testloader):
        images, labels = data
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
"""

total_correct = 0
total = 0
problematic = []
# since we're not training, we don't need to calculate the gradients for our outputs

err_img = {c: None for c in Label}

with torch.no_grad():
    for i_d, data in enumerate(testloader):
        image, label = data
        lbl = Label(label[0].item())
        # calculate outputs by running images through the network
        outputs = net(image)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        prd = Label(predicted[0].item())

        #print(lbl, prd, outputs.data)
        total += 1
        if not lbl == prd:
            #print(i_d, label[0].item(), predicted[0].item())
            problematic.append(i_d)

            if err_img[lbl] is None:
                err_img[lbl] = image
            else:
                err_img[lbl] = torch.cat([err_img[lbl],image], dim=0)
        else:
            total_correct += 1

print(f'Accuracy of the network on test images: {100 * total_correct // total} %')
plt.figure(1)
imshow(torchvision.utils.make_grid(err_img[Label.BOX]))
plt.title("label: BOX")

plt.figure(2)
imshow(torchvision.utils.make_grid(err_img[Label.DEVICE]))
plt.title("label: device")
