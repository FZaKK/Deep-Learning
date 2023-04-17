import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from functools import partial
from einops.layers.torch import Rearrange, Reduce

n_epochs = 10
batch_size_train = 52
batch_size_test = 100
learning_rate = 0.01
momentum = 0.9
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed) # 设置好随机数种子
img_height = 28
img_width = 28


# 这里有点像残差网络那部分
class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    return nn.Sequential(
        dense(dim, dim * expansion_factor),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(dim * expansion_factor, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, patch_size, dim, depth, num_classes, expansion_factor = 4, dropout = 0.2):
    assert (image_size % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_size // patch_size) ** 2
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear  # 实现行列交替

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        # modified
        # nn.Linear((patch_size ** 2) * 3, dim),
        nn.Linear((patch_size ** 2) * 1, dim),   # 当图片channel为1,例如Mnist图片数据，则3要改为1
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, num_classes)
    )

# 模型定义为全局变量
model = MLPMixer(
        image_size=28,
        patch_size=7,
        dim=14,
        # depth=3,
        depth=3,
        num_classes=10
    )

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
mse = nn.MSELoss()
CrossEntropy = nn.CrossEntropyLoss()

# 定义训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        batch_size_train = data.shape[0]
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        pre_out = model(data)

        targ_out = torch.nn.functional.one_hot(target, num_classes=10)
        targ_out = targ_out.view((batch_size_train, 10)).float()
        # loss = mse(pre_out, targ_out)
        loss = CrossEntropy(pre_out, targ_out)

        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 300 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)] tLoss: {:.6f}'.format(
                epoch + 1, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.item()))

# 定义测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for data, target in test_loader:
            batch_size_test = data.shape[0]
            data, target = data.to(device), target.to(device)
            pre_out = model(data)

            targ_out = torch.nn.functional.one_hot(target, num_classes=10)
            targ_out = targ_out.view((batch_size_test, 10)).float()
            # test_loss += mse(pre_out, targ_out)  # 将一批的损失相加
            test_loss += CrossEntropy(pre_out, targ_out)

            # 获取准确率
            pred = pre_out.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct.to(torch.float32) / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset), accuracy))


if __name__ == '__main__':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    trainset = MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train, shuffle=True, num_workers=2)
    testset = MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test, shuffle=False, num_workers=2)

    # DEVICE = torch.device("cpu")
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(DEVICE)
    for epoch in range(n_epochs):
        train(model, DEVICE, train_loader, optimizer, epoch)
        test(model, DEVICE, test_loader)
        torch.save(model.state_dict(), 'model.pth')
        torch.save(optimizer.state_dict(), 'optimizer.pth')

