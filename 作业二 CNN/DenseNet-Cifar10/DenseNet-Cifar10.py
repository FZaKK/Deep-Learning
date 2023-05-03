import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
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


# DenseNet之中的Dense结构，每一个DenseBlock之中可能有几个DenseLayer
# 加入了dropout层训练时防止过拟合
class DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate):
        super(DenseLayer, self).__init__()
        # 这里是pre-activation的结构
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module("conv1", nn.Conv2d(num_input_features, bn_size*growth_rate, kernel_size=1, stride=1, bias=False))
        self.add_module("norm2", nn.BatchNorm2d(bn_size*growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
        self.add_module("conv2", nn.Conv2d(bn_size*growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False))
        self.drop_rate = drop_rate
        self.add_module("dropout", nn.Dropout(drop_rate))

    def forward(self, x):
        new_features = super(DenseLayer, self).forward(x)
        # 在通道维上将输入和输出连结
        return torch.cat([x, new_features], 1)

# 这里就是DenseBlock结构，对于DenseNet之中应该都是加入了BottleNeck结构
# 一般bn_size为4
class DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate):
        super(DenseBlock, self).__init__()
        for i in range(num_layers):
            # i*growth_rate保证连接能够对的上维度
            layer = DenseLayer(num_input_features + i*growth_rate, growth_rate, bn_size, drop_rate)
            self.add_module("denselayer%d" % (i+1), layer)

# 不同的block之间可能会出现维度被我们调整的情形，这里就是采样降维转换层
class Transition(nn.Sequential):
    def __init__(self, num_input_feature, num_output_features):
        super(Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_feature))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module("conv", nn.Conv2d(num_input_feature, num_output_features, kernel_size=1, stride=1, bias=False))
        # Cifar的图片32*32，大小不用缩减应该
        self.add_module("pool", nn.AvgPool2d(2, stride=1, padding=1))

# 最后就是完成对DenseNet结构的定义，包含了Transition的压缩部分(DenseNet-C)
# 也包含了(DenseNet-B)也就是BottleNeck结构，这里给出的是DenseNet-121网络
class DenseNet(nn.Module):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64,
                 bn_size=4, compression_rate=0.5, drop_rate=0.2, num_classes=10):
        """
        :param growth_rate: 增长率，即K=32
        :param block_config: 每一个DenseBlock的layers数量，这里实现的是DenseNet-121
        :param num_init_features: 第一个卷积的通道数一般为64
        :param bn_size: bottleneck中1*1conv的factor=4，1*1conv输出的通道数一般为factor*K=128
        :param compression_rate: 压缩因子
        :param drop_rate: dropout层将神经元置0的概率，为0时表示不使用dropout层
        :param num_classes: 分类数
        """
        super(DenseNet, self).__init__()
        # first Conv2d
        self.features = nn.Sequential(OrderedDict([
            # 对于Cifar数据集改变卷积核大小
            ("conv0", nn.Conv2d(3, num_init_features, kernel_size=3, stride=1, padding=1, bias=False)),
            ("norm0", nn.BatchNorm2d(num_init_features)),
            ("relu0", nn.ReLU(inplace=True))
            # modified删除初始的池化层，因为图片大小不大
        ]))

        # DenseBlock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            # 对于每一个block来说都有不同层数的blocklayer
            block = DenseBlock(num_layers, num_features, bn_size, growth_rate, drop_rate)
            self.features.add_module("denseblock%d" % (i + 1), block)
            # 每多一层layer就会多growth_rate的特征
            num_features += num_layers*growth_rate
            if i != len(block_config) - 1:
                transition = Transition(num_features, int(num_features*compression_rate))
                self.features.add_module("transition%d" % (i + 1), transition)
                # 经过transition重设特征维度
                num_features = int(num_features * compression_rate)

        # final bn+ReLU
        self.features.add_module("norm5", nn.BatchNorm2d(num_features))
        self.features.add_module("relu5", nn.ReLU(inplace=True))

        # added
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # classification layer
        self.fc = nn.Linear(num_features, num_classes)

        # params initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # kaiming初始化
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.bias)

    def forward(self, x):
        out = self.features(x)
        # out = F.avg_pool2d(features, 7, stride=1).view(features.size(0), -1)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# 直接设置好默认的分类类别有10类
net = DenseNet().to(device)
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

    PATH = './cifar_DenseNet.pth'
    torch.save(net.state_dict(), PATH)