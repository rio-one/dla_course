import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

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
    # 2. 准备设备和加载模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ResNet(ResidualBlock, [2, 2, 2], num_classes=28).to(device)

    # 加载你训练好的权重
    model.load_state_dict(torch.load('resnet_80.ckpt', map_location=device))
    model.eval() # 必须切换到测试模式

    # 3. 指定你要看的图片路径 (⚠️请换成你本地实际的一张图片)
    img_path = r'../../data/Arch28_split/val/A02/A02_13.jpg'
    
    # 检查文件是否存在
    if not os.path.exists(img_path):
        print(f"❌ 找不到图片，请检查路径是否正确: {img_path}")
        exit()

    # 4. 图片预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img_pil = Image.open(img_path).convert('RGB')
    input_tensor = transform(img_pil).unsqueeze(0).to(device)

    # 获取预测结果
    outputs = model(input_tensor)
    _, predicted = torch.max(outputs, 1)
    print(f"✅ 模型预测该图片属于类别 ID: {predicted.item()}")

    # 5. 初始化 Grad-CAM (目标层选你的最后一个残差块 layer3)
    target_layers = [model.layer3] 
    cam = GradCAM(model=model, target_layers=target_layers)

    # 生成热力图
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)
    grayscale_cam = grayscale_cam[0, :]

    # 6. 将热力图和原图叠加
    img_np = input_tensor.cpu().squeeze().numpy().transpose(1, 2, 0)
    # 反标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)

    visualization = show_cam_on_image(img_np, grayscale_cam, use_rgb=True)

    # 7. 画图并保存
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f'Grad-CAM (Pred ID: {predicted.item()})')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('gradcam_result_80.png')
    print("✅ 可视化完成！已保存为 gradcam_result_80.png")
    plt.show()