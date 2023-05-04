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


seed = 3407
random.seed(seed)
torch.manual_seed(seed)  # 根据论文seed:3407 is all your need选择3407种子　　
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


# 搭建基于SENet的Conv Block和Identity Block网络结构
# 其中Conv Block与Identity Block的结构有点类似于Residual BlockNeck
class Block(nn.Module):
    # 这里的out_channels的是指block结构中的conv的可能的输出维度
    def __init__(self, in_channels, out_channels, stride=1, is_1x1conv=False):
        super(Block, self).__init__()
        # 各个Stage中的输出维度，第一个conv和第二个conv恒定相同
        out_channels1, out_channels2 = out_channels

        self.is_1x1conv = is_1x1conv  # 判断是否为Conv Block
        self.relu = nn.ReLU(inplace=True)

        # 第一个conv， stride = 1(stage = 1) or stride = 2(stage = 2, 3, 4)
        # 第一维的恒定维度为64，第二维的恒定维度为256（论文之中），我们可以自行设置
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU()
        )
        # 第二个conv
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels1),
            nn.ReLU()
        )
        # 第三个conv，不需要进行ReLu操作
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels1, out_channels2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels2),
        )

        # Conv Block的输入需要额外进行卷积和归一化操作
        # 类似于skip-connection
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels2, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels2)
            )

        # SENet结构
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(out_channels2, out_channels2 // 16, kernel_size=1),  # 16表示r，filter3//16表示C/r，全连接层
            nn.ReLU(),
            nn.Conv2d(out_channels2 // 16, out_channels2, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x_shortcut = x
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x2 = self.se(x1)  # 利用SENet计算出每个通道的权重大小
        x1 = x1 * x2  # 对原通道进行加权操作

        # Conv Block进行额外的卷积归一化操作
        if self.is_1x1conv:
            # x_shortcut = self.shortcut(x_shortcut)
            x_shortcut = self.shortcut(x)
            x1 = x1 + x_shortcut  # Add操作

        x1 = self.relu(x1)  # ReLU操作
        return x1


# SEResNet-50网络结构
# ResNet50[3, 4, 6, 3]；Conv Block和 Identity Block的个数
class SEResNet(nn.Module):
    def __init__(self, net_config=(3, 4, 6, 3), num_classes=10):
        super(SEResNet, self).__init__()

        # Stem Block
        self.conv1 = nn.Sequential(
            # 调整一下Cifar10的图片太小了
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage1
        channels = (64, 256)  # channel
        self.Stage1 = self._make_layer(in_channels=64, out_channels=channels, num=net_config[0], stride=1)
        # Stage2
        channels = (128, 512)  # channel
        self.Stage2 = self._make_layer(in_channels=256, out_channels=channels, num=net_config[1], stride=2)
        # Stage3
        channels = (256, 1024)  # channel
        self.Stage3 = self._make_layer(in_channels=512, out_channels=channels, num=net_config[2], stride=2)
        # Stage4
        channels = (512, 2048)  # channel
        self.Stage4 = self._make_layer(in_channels=1024, out_channels=channels, num=net_config[3], stride=2)

        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层 这里可理解为网络中四个Stage后的Subsequent Processing 环节
        self.fc = nn.Sequential(
            nn.Linear(2048, num_classes)
        )

    # 形成单个Stage的网络结构
    def _make_layer(self, in_channels, out_channels, num, stride=1):
        layers = []

        # Conv Block
        block_1 = Block(in_channels, out_channels, stride=stride, is_1x1conv=True)
        layers.append(block_1)

        # Identity Block结构叠加; 基于[3, 4, 6, 3]
        # 这里我们需要与前一个结构输出的维度接上，所以是out_channels[1]
        for i in range(1, num):
            layers.append(Block(out_channels[1], out_channels, stride=1, is_1x1conv=False))

        # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
        # 可以这么写layers为链表，nn.Sequential(*layers)
        return nn.Sequential(*layers)

    def forward(self, x):
        # Stem Block环节
        x = self.conv1(x)

        # 执行四个Stage环节
        x = self.Stage1(x)
        x = self.Stage2(x)
        x = self.Stage3(x)
        x = self.Stage4(x)

        # 执行Subsequent Processing环节
        x = self.global_average_pool(x)
        # x = torch.flatten(x, 1)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


# 直接设置好默认的分类类别有10类
net = SEResNet().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.8)


def train(trainloader, epoch, log_interval=1000):
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
        # get the index of the max log-probability
        pred = output.data.max(1)[1]
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
    batch_size = 4
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

    # save the module
    PATH = './cifar_SE_ResNet.pth'
    torch.save(net.state_dict(), PATH)
