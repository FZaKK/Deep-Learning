## 导入第三方库
from torch import nn
import time
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.optim as optim


# 搭建基于SENet的Conv Block和Identity Block的网络结构
class Block(nn.Module):
    def __init__(self, in_channels, filters, stride=1, is_1x1conv=False):
        super(Block, self).__init__()

        # 各个Stage中的每一大块中每一小块的输出维度，即channel（filter1 = filter2 = filter3 / 4）
        filter1, filter2, filter3 = filters

        self.is_1x1conv = is_1x1conv  # 判断是否是Conv Block
        self.relu = nn.ReLU(inplace=True)  # RELU操作

        # 第一小块， stride = 1(stage = 1) or stride = 2(stage = 2, 3, 4)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, filter1, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(filter1),
            nn.ReLU()
        )

        # 中间小块
        self.conv2 = nn.Sequential(
            nn.Conv2d(filter1, filter2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(filter2),
            nn.ReLU()
        )

        # 最后小块，不需要进行ReLu操作
        self.conv3 = nn.Sequential(
            nn.Conv2d(filter2, filter3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(filter3),
        )

        # Conv Block的输入需要额外进行卷积和归一化操作（结合Conv Block网络图理解）
        if is_1x1conv:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, filter3, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(filter3)
            )

        # SENet(结合SENet的网络图理解)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 全局平均池化
            nn.Conv2d(filter3, filter3 // 16, kernel_size=1),  # 16表示r，filter3//16表示C/r，这里用卷积层代替全连接层
            nn.ReLU(),
            nn.Conv2d(filter3 // 16, filter3, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_shortcut = x
        x1 = self.conv1(x)  # 执行第一Block操作
        x1 = self.conv2(x1)  # 执行中间Block操作
        x1 = self.conv3(x1)  # 执行最后Block操作

        x2 = self.se(x1)  # 利用SENet计算出每个通道的权重大小
        x1 = x1 * x2  # 对原通道进行加权操作

        if self.is_1x1conv:  # Conv Block进行额外的卷积归一化操作
            x_shortcut = self.shortcut(x_shortcut)

        x1 = x1 + x_shortcut  # Add操作
        x1 = self.relu(x1)  # ReLU操作

        return x1


# 搭建SEResNet50
class SEResnet(nn.Module):
    def __init__(self, cfg):
        super(SEResnet, self).__init__()
        classes = cfg['classes']  # 分类的类别
        num = cfg['num']  # ResNet50[3, 4, 6, 3]；Conv Block和 Identity Block的个数

        # Stem Block
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        # Stage1
        filters = (64, 64, 256)  # channel
        self.Stage1 = self._make_layer(in_channels=64, filters=filters, num=num[0], stride=1)

        # Stage2
        filters = (128, 128, 512)  # channel
        self.Stage2 = self._make_layer(in_channels=256, filters=filters, num=num[1], stride=2)

        # Stage3
        filters = (256, 256, 1024)  # channel
        self.Stage3 = self._make_layer(in_channels=512, filters=filters, num=num[2], stride=2)

        # Stage4
        filters = (512, 512, 2048)  # channel
        self.Stage4 = self._make_layer(in_channels=1024, filters=filters, num=num[3], stride=2)

        # 自适应平均池化，(1, 1)表示输出的大小(H x W)
        self.global_average_pool = nn.AdaptiveAvgPool2d((1, 1))

        # 全连接层 这里可理解为网络中四个Stage后的Subsequent Processing 环节
        self.fc = nn.Sequential(
            nn.Linear(2048, classes)
        )

    # 形成单个Stage的网络结构
    def _make_layer(self, in_channels, filters, num, stride=1):
        layers = []

        # Conv Block
        block_1 = Block(in_channels, filters, stride=stride, is_1x1conv=True)
        layers.append(block_1)

        # Identity Block结构叠加; 基于[3, 4, 6, 3]
        for i in range(1, num):
            layers.append(Block(filters[2], filters, stride=1, is_1x1conv=False))

        # 返回Conv Block和Identity Block的集合，形成一个Stage的网络结构
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
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# SeResNet50的参数  （注意调用这个函数将间接调用SEResnet，这里单独编写一个函数是为了方便修改成其它ResNet网络的结构）
def SeResNet50():
    cfg = {
        'num': (3, 4, 6, 3),  # ResNet50，四个Stage中Block的个数（其中Conv Block为1个，剩下均为增加Identity Block）
        'classes': (10)  # 数据集分类的个数
    }

    return SEResnet(cfg)  # 调用SEResnet网络


## 导入数据集
def load_dataset(batch_size):
    # 下载训练集
    train_set = torchvision.datasets.CIFAR10(
        root="./data", train=True,
        download=True, transform=transforms.ToTensor()
    )

    # 下载测试集
    test_set = torchvision.datasets.CIFAR10(
        root="./data", train=False,
        download=True, transform=transforms.ToTensor()
    )

    train_iter = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_iter = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=True, num_workers=4
    )

    return train_iter, test_iter


# 训练模型
def train(net, train_iter, criterion, optimizer, num_epochs, device, num_print, lr_scheduler=None, test_iter=None):
    net.train()  # 训练模式
    record_train = list()  # 记录每一Epoch下训练集的准确率
    record_test = list()  # 记录每一Epoch下测试集的准确率

    for epoch in range(num_epochs):
        print("========== epoch: [{}/{}] ==========".format(epoch + 1, num_epochs))

        total, correct, train_loss = 0, 0, 0
        start = time.time()

        for i, (X, y) in enumerate(train_iter):
            X, y = X.to(device), y.to(device)  # GPU or CPU运行

            output = net(X)  # 计算输出
            loss = criterion(output, y)  # 计算损失

            optimizer.zero_grad()  # 梯度置0
            loss.backward()  # 计算梯度
            optimizer.step()  # 优化参数

            train_loss += loss.item()  # 累积损失
            total += y.size(0)  # 累积总样本数

            correct += (output.argmax(dim=1) == y).sum().item()  # 累积预测正确的样本数
            train_acc = 100.0 * correct / total  # 计算准确率

            if (i + 1) % num_print == 0:
                print("step: [{}/{}], train_loss: {:.3f} | train_acc: {:6.3f}% | lr: {:.6f}" \
                      .format(i + 1, len(train_iter), train_loss / (i + 1), \
                              train_acc, get_cur_lr(optimizer)))

        # 调整梯度下降算法的学习率
        if lr_scheduler is not None:
            lr_scheduler.step()

        # 输出训练的时间
        print("--- cost time: {:.4f}s ---".format(time.time() - start))

        if test_iter is not None:  # 判断测试集是否为空 (注意这里将调用test函数)
            record_test.append(test(net, test_iter, criterion, device))  # 每训练一个Epoch模型，使用测试集进行测试模型的准确度
        record_train.append(train_acc)

    # 返回每一个Epoch下测试集和训练集的准确率
    return record_train, record_test


# 验证模型
def test(net, test_iter, criterion, device):
    total, correct = 0, 0
    net.eval()  # 测试模式

    with torch.no_grad():  # 不计算梯度
        print("*************** test ***************")
        for X, y in test_iter:
            X, y = X.to(device), y.to(device)  # CPU or GPU运行

            output = net(X)  # 计算输出
            loss = criterion(output, y)  # 计算损失

            total += y.size(0)  # 计算测试集总样本数
            correct += (output.argmax(dim=1) == y).sum().item()  # 计算测试集预测准确的样本数

    test_acc = 100.0 * correct / total  # 测试集准确率

    # 输出测试集的损失
    print("test_loss: {:.3f} | test_acc: {:6.3f}%" \
          .format(loss.item(), test_acc))
    print("************************************\n")

    # 训练模式 （因为这里是因为每经过一个Epoch就使用测试集一次，使用测试集后，进入下一个Epoch前将模型重新置于训练模式）
    net.train()

    return test_acc


# 返回学习率lr的函数
def get_cur_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


# 画出每一个Epoch下测试集和训练集的准确率
def learning_curve(record_train, record_test=None):
    plt.style.use("ggplot")

    plt.plot(range(1, len(record_train) + 1), record_train, label="train acc")
    if record_test is not None:
        plt.plot(range(1, len(record_test) + 1), record_test, label="test acc")

    plt.legend(loc=4)
    plt.title("learning curve")
    plt.xticks(range(0, len(record_train) + 1, 5))
    plt.yticks(range(0, 101, 5))
    plt.xlabel("epoch")
    plt.ylabel("accuracy")

    plt.show()


BATCH_SIZE = 128  # 批大小
NUM_EPOCHS = 12  # Epoch大小
NUM_CLASSES = 10  # 分类的个数
LEARNING_RATE = 0.01  # 梯度下降学习率
MOMENTUM = 0.9  # 冲量大小
WEIGHT_DECAY = 0.0005  # 权重衰减系数
NUM_PRINT = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"  # GPU or CPU运行


def main():
    net = SeResNet50()
    net = net.to(DEVICE)  # GPU or CPU 运行

    train_iter, test_iter = load_dataset(BATCH_SIZE)  # 导入训练集和测试集

    criterion = nn.CrossEntropyLoss()  # 损失计算器

    # 优化器
    optimizer = optim.SGD(
        net.parameters(),
        lr=LEARNING_RATE,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=True
    )

    # 调整学习率 (step_size:每训练step_size个epoch，更新一次参数; gamma:更新lr的乘法因子)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    record_train, record_test = train(net, train_iter, criterion, optimizer, NUM_EPOCHS, DEVICE, NUM_PRINT,
                                      lr_scheduler, test_iter)

    learning_curve(record_train, record_test)  # 画出准确率曲线

if __name__ == '__main__':
    main()