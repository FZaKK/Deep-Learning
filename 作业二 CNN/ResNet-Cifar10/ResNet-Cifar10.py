import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random


seed = 10
random.seed(seed)
torch.manual_seed(seed)  # 为CPU设置种子用于生成随机数，以使得结果是确定的   　　
torch.cuda.manual_seed(seed) # 为当前GPU设置随机种子；
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# 也就是定义基础的ResBlock
class ResidualBlock(nn.Module):
    # BottleNeck有通道注意力机制
    # expansion = 1

    def __init__(self, inchannels, outchannels, stride=1):    # ResBlock, 输入channel和输出channel不一致需要1*1 conv
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannels, outchannels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannels),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannels, outchannels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannels)
        )
        # 如果维度并没有什么变化，就只需要skip connection即可
        self.downsample = nn.Sequential()
        # 无论是channel维度的变化，还是stride步长带来的子采样等都需要使用1*1 conv保持维度
        if stride != 1 or inchannels != outchannels:
            self.downsample = nn.Sequential(
                nn.Conv2d(inchannels, outchannels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannels)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.downsample(x)
        # 也可以在网络层之中添加relu层
        out = F.relu(out)
        return out


# 定义ResNet整个网络模型
class ResNet(nn.Module):
    # 输入的参数如果在pytorch的官网上可以是Block也可以是BottleNeck
    def __init__(self, ResidualBlock, num_classes=10):
        super(ResNet, self).__init__()
        self.inchannels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        # 对于Cifar10，图片仅有32*32的大小，就不再池化了
        # self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 2, stride=2)
        # modified
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = []
        strides.append(stride)
        for i in range(1, num_blocks):
            strides.append(1)
        # 第一个ResidualBlock的步幅由make_layer的函数参数stride指定
        # 后续的num_blocks-1个ResidualBlock的stride为1
        layers = []
        for i in range(num_blocks):
            layers.append(block(self.inchannels, channels, strides[i]))
            self.inchannels = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        # out = F.avg_pool2d(out, 4)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def ResNet18():
    return ResNet(ResidualBlock)


# 直接设置好默认的分类类别有10类
net = ResNet18().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)


def train(trainloader, epoch, log_interval=200):
    # Set model to training mode
    net.train()

    # Loop over each batch from the training set
    for batch_idx, (data, target) in enumerate(trainloader):
        # Copy data to GPU if needed
        data = data.to(device)
        target = target.to(device)

        # Zero gradient buffers
        optimizer.zero_grad()

        # Pass data through the network
        output = net(data)

        # Calculate loss
        loss = criterion(output, target)

        # Backpropagate
        loss.backward()

        # Update weights
        optimizer.step()  # w - alpha * dL / dw

        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                       100. * batch_idx / len(trainloader), loss.data.item()))


def validate(testloader, loss_vector, accuracy_vector):
    net.eval()
    val_loss, correct = 0, 0
    for data, target in testloader:
        data = data.to(device)
        target = target.to(device)
        output = net(data)
        val_loss += criterion(output, target).data.item()
        pred = output.data.max(1)[1]  # get the index of the max log-probability
        correct += pred.eq(target.data).cpu().sum()

    val_loss /= len(testloader)
    loss_vector.append(val_loss)

    accuracy = 100. * correct.to(torch.float32) / len(testloader.dataset)
    accuracy_vector.append(accuracy)

    print('\nValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(testloader.dataset), accuracy))


if __name__ == '__main__':
    # 设置好对应读取数据集和测试集的loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    batch_size = 64

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    print(net)

    epochs = 10
    lossv, accv = [], []
    # print("hello")
    for epoch in range(1, epochs + 1):
        train(trainloader, epoch)
        validate(testloader, lossv, accv)

    # print the figure
    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), lossv)
    plt.title('validation loss')

    plt.figure(figsize=(5, 3))
    plt.plot(np.arange(1, epochs + 1), accv)
    plt.title('validation accuracy')

    PATH = './cifar_net.pth'
    torch.save(net.state_dict(), PATH)
