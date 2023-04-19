import torch
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from torch import nn

from qr_torch.helpers import imshow
from qr_torch.nets import CifarNet
from qr_torch.qrdata import QrData, SubSet, composed_transform, Label

# PATH = './cifar_net.pth'

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

# trainsetC10 = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainset = QrData(SubSet.TRAIN, transform=composed_transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
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

net = CifarNet()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

print("Training")
for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 100 == 99:    # print every 2000 mini-batches
            print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
            running_loss = 0.0

print('Finished Training')

torch.save(net.state_dict(), net.pth_fname())


dataiter = iter(testloader)
images, labels = next(dataiter)

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(len(labels))))

net = CifarNet()
net.load_state_dict(torch.load(CifarNet.pth_fname()))
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

        print(lbl, prd, outputs.data)
        total += 1
        if not lbl == prd:
            print(i_d, label[0].item(), predicted[0].item())
            problematic.append(i_d)

            if err_img[lbl] is None:
                err_img[lbl] = image
            else:
                err_img[lbl] = torch.cat([err_img[lbl],image], dim=0)
        else:
            total_correct += 1

print(f'Accuracy of the network on test images: {100 * total_correct // total} %')
imshow(torchvision.utils.make_grid(err_img[Label.BOX]))
plt.title("label: BOX")
imshow(torchvision.utils.make_grid(err_img[Label.DEVICE]))
plt.title("label: device")
