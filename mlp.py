import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import math

# 设置数据保存目录
save_dir = "mnist_data"
os.makedirs(save_dir, exist_ok=True)

# 加载并处理 MNIST 数据集
transform = transforms.Compose([transforms.ToTensor()])
mnist_train = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

# 将数据转换为 NumPy 数组并进行归一化
X_train = mnist_train.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_train = mnist_train.targets.numpy().astype(np.int32)
X_test = mnist_test.data.numpy().reshape(-1, 28 * 28).astype(np.float32) / 255.0
y_test = mnist_test.targets.numpy().astype(np.int32)

# 进行标准化
MEAN = 0.1307
STD = 0.3081
X_train = (X_train - MEAN) / STD
X_test = (X_test - MEAN) / STD

# 保存数据为二进制文件
X_train.tofile(os.path.join(save_dir, "X_train.bin"))
y_train.tofile(os.path.join(save_dir, "y_train.bin"))
X_test.tofile(os.path.join(save_dir, "X_test.bin"))
y_test.tofile(os.path.join(save_dir, "y_test.bin"))

# 创建 Tensor 数据集
train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))

# 创建数据加载器
batch_size = 64
train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

# 打印数据加载信息
print(f"Loaded {len(X_train)} MNIST samples")

# 验证数据
def validate_data():
    print("\n=== Data Validation ===")
    for i in range(3):  # 验证前3个样本
        print(f"Sample {i}: Label {y_train[i]}")
        pixel_min = X_train[i].min()
        pixel_max = X_train[i].max()
        print(f"Pixel range: [{pixel_min:.2f}, {pixel_max:.2f}]")
    print("======================\n")

validate_data()

# 定义 MLP 模型
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 256)  # 输入维度 784, 隐藏层大小 256
        self.fc2 = nn.Linear(256, 10)       # 输出 10 个类别

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 创建 MLP 模型实例
model = MLP()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()  # CrossEntropyLoss 自动处理标签的转换
optimizer = optim.Adam(model.parameters(), lr=3e-4)

# 训练模型
def train(model, train_loader, criterion, optimizer, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.view(-1, 28 * 28), target

            # 转换标签类型为 Long 类型
            target = target.long()

            # 清零梯度
            optimizer.zero_grad()

            # 前向传播
            output = model(data)
            loss = criterion(output, target)

            # 反向传播
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)

            if batch_idx % (len(train_loader) // 10) == 0 or batch_idx == len(train_loader) - 1:
                acc = 100. * correct / total
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f} | Acc: {acc:.2f}%")
        
        # Epoch结束后的输出
        avg_loss = total_loss / len(train_loader)
        final_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{epochs} complete | Avg Loss: {avg_loss:.4f} | Accuracy: {final_acc:.2f}%\n")

# 测试模型
def test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.view(-1, 28 * 28), target
            output = model(data)
            _, predicted = torch.max(output.data, 1)
            correct += (predicted == target).sum().item()
            total += target.size(0)
    
    final_acc = 100. * correct / total
    print(f"Test Accuracy: {final_acc:.2f}%")

# 训练和测试
print("Training with {} batches/epoch".format(len(train_loader)))
train(model, train_loader, criterion, optimizer, epochs=10)
test(model, test_loader)
