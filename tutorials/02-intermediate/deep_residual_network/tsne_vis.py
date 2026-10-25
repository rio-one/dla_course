import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
from torch.utils.data import DataLoader, Subset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# ================= 1. 直接复制模型结构 =================
def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                     stride=stride, padding=1, bias=False)

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
# ======================================================

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载模型和权重
    model = ResNet(ResidualBlock, [2, 2, 2], num_classes=28).to(device)
    model.load_state_dict(torch.load('resnet.ckpt', map_location=device))
    model.eval()

    # 准备数据 (不需要随机翻转，只需调整大小和归一化)
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_dir = '../../data/Arch28_split/val'
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    
    # 🎯 核心逻辑：选取 10 个类，每个类挑 20 张图片
    # 这里我们随机打乱一下，保证能覆盖到不同的图
    class_indices = {i: [] for i in range(len(dataset.classes))}
    for idx in range(len(dataset)):
        # 直接读取标签，不加载图片，速度极快且绝对不会报图片损坏的错误
        label = dataset.targets[idx] 
        class_indices[label].append(idx)
    
    selected_indices = []
    num_classes = 10 # 想看几个类
    images_per_class = 90 # 每个类看几张
    for i in range(num_classes):
        # 如果这个类图片不够20张，就全拿
        chosen = np.random.choice(class_indices[i], min(images_per_class, len(class_indices[i])), replace=False)
        selected_indices.extend(chosen)
        
    subset_dataset = Subset(dataset, selected_indices)
    dataloader = DataLoader(subset_dataset, batch_size=16, shuffle=False)

    # 🎯 提取特征
    features = []
    labels = []
    
    # 由于你的模型 forward 直接出结果，我们需要把 fc 层拆掉，只要全连接层之前的特征
    # 这里我们直接用特征提取器截断法
    def extract_features(x):
        out = model.conv(x)
        out = model.bn(out)
        out = model.relu(out)
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = model.avg_pool(out)
        out = out.view(out.size(0), -1) # 这里就是 1024 维特征向量
        return out

    print(f"开始提取 {len(selected_indices)} 张图片的特征...")
    with torch.no_grad():
        for images, lbls in dataloader:
            images = images.to(device)
            feats = extract_features(images)
            features.append(feats.cpu().numpy())
            labels.extend(lbls.numpy())

    features = np.concatenate(features, axis=0) # 变成了 (N, 1024) 的矩阵
    labels = np.array(labels)

    # 🎯 t-SNE 降维
    print("正在进行 t-SNE 降维 (可能需要几秒钟)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(features) # 变成了 (N, 2) 的平面坐标

    # 🎯 画图
    plt.figure(figsize=(10, 8))
    # 使用 tab10 色板 (支持10种区别明显的颜色)
    cmap = plt.get_cmap('tab10')
    
    for i in range(num_classes):
        # 找出属于当前类别的点
        indices = np.where(labels == i)[0]
        plt.scatter(tsne_results[indices, 0], tsne_results[indices, 1], 
                    label=f'Class {dataset.classes[i]}', 
                    color=cmap(i), alpha=0.8, s=50)

    plt.title('t-SNE Visualization of CNN Features')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('tsne_visualization.png')
    print("✅ 可视化完成！已保存为 tsne_visualization.png")
    plt.show()