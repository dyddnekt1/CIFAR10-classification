import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import matplotlib.pyplot as plt
import numpy as np
import time

class block(nn.Module):
  def __init__(self, in_chanels, out_chanels, block_type=0):
    super(block, self).__init__()
    self.bn1 = nn.BatchNorm2d(in_chanels)
    self.bn2 = nn.BatchNorm2d(out_chanels)

    self.relu = nn.ReLU()
    if block_type == 0:
      self.conv1 = nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=1, padding=1)
    elif block_type == 1:
      self.conv1 = nn.Conv2d(in_chanels, out_chanels, kernel_size=3, stride=2, padding=1)
      self.conv_shortcut = nn.Conv2d(in_chanels, out_chanels, kernel_size=1, stride=2, padding=0)
    self.conv2 = nn.Conv2d(out_chanels, out_chanels, kernel_size=3, stride=1, padding=1)

    self.block_type = block_type

  def forward(self, x):
    shortcut = x

    out = self.bn1(x)
    out = self.relu(out)

    if self.block_type == 1:
      shortcut = self.conv_shortcut(out)
    out = self.conv1(out)

    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    out += shortcut
      
    return out

class IdentityResNet(nn.Module):
    
    # __init__ takes 4 parameters
    # nblk_stage1: number of blocks in stage 1, nblk_stage2.. similar
    def __init__(self, nblk_stage1, nblk_stage2, nblk_stage3, nblk_stage4):
        super(IdentityResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

        stage1 = []
        for n in range(nblk_stage1):
          stage1.append(block(64, 64))
        self.stage1 = nn.Sequential(*stage1)

        stage2 = [block(64, 128, block_type=1)]
        for n in range(nblk_stage2 - 1):
          stage2.append(block(128, 128))
        self.stage2 = nn.Sequential(*stage2)
        
        stage3 = [block(128, 256, block_type=1)]
        for n in range(nblk_stage3 - 1):
          stage3.append(block(256, 256))
        self.stage3 = nn.Sequential(*stage3)

        stage4 = [block(256, 512, block_type=1)]
        for n in range(nblk_stage4 - 1):
          stage4.append(block(512, 512))
        self.stage4 = nn.Sequential(*stage4)

        self.avgpool = nn.AvgPool2d(4, stride=4)

        self.fc = nn.Linear(512, 10)


    def forward(self, x):
        out = self.conv1(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)
        out = self.stage4(out)

        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)

        return out

# set device
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('current device: ', dev)


########################################
# data preparation: CIFAR10
########################################

# set batch size
batch_size = 4

# preprocessing
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# load training data
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True)

# load test data
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


# define network
net = IdentityResNet(nblk_stage1=2, nblk_stage2=2,
                     nblk_stage3=2, nblk_stage4=2)

# load model to GPU
net.to(dev)


# set loss function
criterion = nn.CrossEntropyLoss()

# set optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9) 

# start training
t_start = time.time()

for epoch in range(5):  # loop over the dataset multiple times
    
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(dev), data[1].to(dev)
        
        # set gradients to zero
        optimizer.zero_grad()

        # perform forward
        outputs = net(inputs)
        
        # set loss
        loss = criterion(outputs, labels)

        # perform backward
        loss.backward()

        # take SGC step
        optimizer.step()
        
        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0
            t_end = time.time()
            print('elapsed:', t_end-t_start, ' sec')
            t_start = t_end

print('Finished Training')


# now testing
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))

# test
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(dev), data[1].to(dev)
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1

# per-class accuracy
for i in range(10):
    print('Accuracy of %5s' %(classes[i]), ': ',
          100 * class_correct[i] / class_total[i],'%')

# overall accuracy
print('Overall Accurracy: ', (sum(class_correct)/sum(class_total))*100, '%')