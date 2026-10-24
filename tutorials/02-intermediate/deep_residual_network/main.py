# ---------------------------------------------------------------------------- #
# An implementation of https://arxiv.org/pdf/1512.03385.pdf                    #
# See section 4.2 for the model architecture on CIFAR-10                       #
# Some part of the code was referenced from below                              #
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py   #
# ---------------------------------------------------------------------------- #

import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import matplotlib.pyplot as plt  # 新增：导入绘图库


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
num_epochs = 40
batch_size = 100
learning_rate = 0.001

# # Image preprocessing modules
# transform = transforms.Compose([
#     transforms.Pad(4),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomCrop(32),
#     transforms.ToTensor()])

# # CIFAR-10 dataset
# train_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                              train=True, 
#                                              transform=transform,
#                                              download=True)

# test_dataset = torchvision.datasets.CIFAR10(root='../../data/',
#                                             train=False, 
#                                             transform=transforms.ToTensor())

# # Data loader
# train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
#                                            batch_size=batch_size,
#                                            shuffle=True)

# test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
#                                           batch_size=batch_size,
#                                           shuffle=False)


#mission1：calculate the accuracy of the model on the test images of our dataset
# Arch28_split dataset
transform_train = transforms.Compose([
    transforms.Resize((128, 128)),      # 把图缩放到 128x128 (显存不够就改成 64x64)
    transforms.RandomHorizontalFlip(),  # 随机水平翻转
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) 
])

transform_test = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

data_dir = '../../data/Arch28_split'

train_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
test_dataset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True,
                                           num_workers=0)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False,
                                          num_workers=0)

# 3x3 convolution
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

# ResNet
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=28):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv = conv3x3(3, 16)
        self.bn = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self.make_layer(block, 16, layers[0])
        self.layer2 = self.make_layer(block, 32, layers[1], 2)
        self.layer3 = self.make_layer(block, 64, layers[2], 2)
        self.avg_pool = nn.AvgPool2d(8)
        self.fc = nn.Linear(1024, num_classes)
        
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out
    
model = ResNet(ResidualBlock, [2, 2, 2]).to(device)


# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# For updating learning rate
def update_lr(optimizer, lr):    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

# # Train the model
# total_step = len(train_loader)
# curr_lr = learning_rate
# for epoch in range(num_epochs):
#     for i, (images, labels) in enumerate(train_loader):
#         images = images.to(device)
#         labels = labels.to(device)
        
#         # Forward pass
#         outputs = model(images)
#         loss = criterion(outputs, labels)
        
#         # Backward and optimize
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
        
#         if (i+1) % 100 == 0:
#             print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
#                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

#     # Decay learning rate
#     if (epoch+1) % 20 == 0:
#         curr_lr /= 3
#         update_lr(optimizer, curr_lr)

# # Test the model
# model.eval()
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

#     print('Accuracy of the model on the test images: {} %'.format(100 * correct / total))


# mission2：绘图

# Train the model
total_step = len(train_loader)
curr_lr = learning_rate

# ================= 新增：用于记录画图数据的空列表 =================
train_losses = []
test_accuracies = []
# ==================================================================

for epoch in range(num_epochs):
    # ================= 新增：记录当前 epoch 的训练损失 =================
    model.train()  # 确保模型在训练模式
    running_loss = 0.0
    total_train_samples = 0
    # ==================================================================
    
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # ================= 新增：累加 loss =================
        running_loss += loss.item() * images.size(0)
        total_train_samples += images.size(0)
        # ===================================================
        
        if (i+1) % 100 == 0:
            print ("Epoch [{}/{}], Step [{}/{}] Loss: {:.4f}"
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # ================= 新增：计算并保存当前 epoch 的平均训练 Loss =================
    avg_train_loss = running_loss / total_train_samples
    train_losses.append(avg_train_loss)
    # ==============================================================================

    # ================= 新增：把测试代码挪到每个 epoch 里面 =================
    model.eval()  # 切换到测试模式
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    test_acc = 100 * correct / total
    test_accuracies.append(test_acc)
    
    # 每个 epoch 结束时打印一次结果，方便你观察进度
    print(f"====> Epoch [{epoch+1}/{num_epochs}] 总结: Train Loss: {avg_train_loss:.4f}, Test Acc: {test_acc:.2f}%")
    # =========================================================================

    # Decay learning rate
    if (epoch+1) % 20 == 0:
        curr_lr /= 3
        update_lr(optimizer, curr_lr)

# ================= 新增：训练结束后，绘制曲线 =================
epochs = range(1, num_epochs + 1)

plt.figure(figsize=(12, 5))

# 绘制训练损失曲线
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.title('Training Loss over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# 绘制测试准确率曲线
plt.subplot(1, 2, 2)
plt.plot(epochs, test_accuracies, 'r-', label='Test Accuracy')
plt.title('Test Accuracy over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig('training_curve_80.png') # 图片会保存在当前目录下
plt.show() # 弹窗显示


# Save the model checkpoint
torch.save(model.state_dict(), 'resnet.ckpt')
