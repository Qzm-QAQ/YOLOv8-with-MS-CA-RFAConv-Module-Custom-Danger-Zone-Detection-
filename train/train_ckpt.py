import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# ==========================
# 1. 模块定义
# ==========================

# 定义基本的卷积模块
class Conv(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, s=1, g=1, act=True):
        super(Conv, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=s,
            padding=k // 2,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 定义带有分组卷积的卷积模块
class Conv_L(nn.Module):
    def __init__(self, in_channels, out_channels, k=1, g=1, act=True):
        super(Conv_L, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=k,
            stride=1,
            padding=k // 2,
            groups=g,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU() if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# 定义通道注意力模块（CAConv）
class CAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CAConv, self).__init__()
        reduced_channels = max(out_channels // 16, 1)  # 确保至少为1
        self.conv = Conv(in_channels, out_channels, k=kernel_size, s=stride)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, reduced_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(reduced_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.ca(y)
        return x * y


# 定义 RFCAConv2 类
class RFCAConv2(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv_L(c1, 3 * c1, k=1, g=c1)
        self.group_conv2 = Conv_L(c1, 3 * c1, k=3, g=c1)
        self.group_conv3 = Conv_L(c1, 3 * c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 3 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c2, k=3, s=3, g=1)  # 修改输出通道数为 c2，groups=1
        self.CA = CAConv(c2, c2, kernel_size, stride)  # 确保 c2 = c2

    def forward(self, x):
        b, c1, _, _ = x.size()
        y = self.avg_pool(x)

        group1 = self.softmax(self.group_conv1(y))
        group2 = self.softmax(self.group_conv2(y))
        group3 = self.softmax(self.group_conv3(y))

        g1 = self.group_conv(x)

        out1 = g1 * group1
        out2 = g1 * group2
        out3 = g1 * group3

        out = torch.cat([out1, out2, out3], dim=1)  # (batch, 9*c1, h, w)

        # 获取输入特征图的形状
        batch_size, channels, height, width = out.shape

        # 计算输出特征图的通道数
        output_channels = c1  # c1

        if channels != output_channels * 9:
            raise ValueError(
                f"Expected channels = {output_channels * 9}, but got channels = {channels}"
            )

        out = out.view(batch_size, output_channels, 3, 3, height, width)
        out = out.permute(0, 1, 4, 2, 5, 3).reshape(
            batch_size, output_channels, 3 * height, 3 * width
        )

        out = self.convDown(out)
        out = self.CA(out)
        return out

# ==========================
# 2. 自定义 YOLO 模型
# ==========================

# 假设存在一个基类 YOLO，需要根据实际情况调整
class YOLO(nn.Module):
    def __init__(self, model_path=None):
        super(YOLO, self).__init__()
        # 这里应加载预训练模型的结构
        # 这里只是一个占位符
        self.model = nn.Sequential(
            nn.Conv2d(3, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # ... 其他层
        )
        if model_path and os.path.exists(model_path):
            self.load_state_dict(torch.load(model_path))

    def forward(self, x):
        return self.model(x)

    def train(self, mode=True):
        super(YOLO, self).train(mode)


# 自定义 YOLO 模型，替换部分卷积层为 RFCAConv2
class CustomYOLO(YOLO):
    def __init__(self, model_path=None, num_classes=1):
        super(CustomYOLO, self).__init__(model_path)
        self.num_classes = num_classes
        self.modify_model()

        # 获取模型中最后一个卷积层的 out_channels
        last_conv_out_channels = self.find_last_conv_out_channels(self.model)

        if last_conv_out_channels is None:
            raise AttributeError("模型中没有找到 Conv2d 层")

        # 定义预测层，根据类别数调整输出通道数
        # 这里假设使用单个锚点，每个锚点预测 (5 + num_classes) 个值
        self.prediction = nn.Conv2d(
            in_channels=last_conv_out_channels,  # 使用最后一个卷积层的 out_channels
            out_channels=5 + self.num_classes,
            kernel_size=1,
            stride=1,
            padding=0
        )

    def modify_model(self):
        # 遍历模型的所有层，找到需要替换的卷积层
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d) and module.kernel_size == (3, 3):
                in_channels = module.in_channels
                out_channels = module.out_channels
                stride = module.stride[0]
                # 创建 RFCAConv2 替换
                rfcaconv = RFCAConv2(in_channels, out_channels, kernel_size=3, stride=stride)
                # 递归替换模块
                self._replace_module(self.model, name, rfcaconv)

    def _replace_module(self, model, target_name, new_module):
        """递归地替换模型中的指定模块"""
        names = target_name.split(".")
        for name in names[:-1]:
            model = getattr(model, name)
        setattr(model, names[-1], new_module)

    def find_last_conv_out_channels(self, module):
        """
        递归查找模型中最后一个卷积层的 out_channels。
        """
        last_conv_out_channels = None
        # 递归遍历模型的所有子模块
        for child in module.children():
            if isinstance(child, nn.Conv2d):
                last_conv_out_channels = child.out_channels
            else:
                # 递归检查子模块
                out_channels = self.find_last_conv_out_channels(child)
                if out_channels is not None:
                    last_conv_out_channels = out_channels
        return last_conv_out_channels

    def forward(self, x):
        x = self.model(x)
        x = self.prediction(x)
        return x


# ==========================
# 3. 数据集类
# ==========================

class YOLODataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        if not os.path.exists(images_dir):
            raise FileNotFoundError(f"图像目录不存在: {images_dir}")
        if not os.path.exists(labels_dir):
            raise FileNotFoundError(f"标签目录不存在: {labels_dir}")

        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.transform = transform
        self.image_files = [
            f
            for f in os.listdir(images_dir)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]

        if not self.image_files:
            raise ValueError(f"图像目录为空: {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        label_path = os.path.join(self.labels_dir, os.path.splitext(img_name)[0] + ".txt")

        # 加载图像
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            raise IOError(f"无法打开图像文件 {img_path}: {e}")

        if self.transform:
            image = self.transform(image)

        # 加载标签
        targets = []
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue  # 跳过格式错误的行
                    class_id, x_center, y_center, width, height = map(float, parts)
                    targets.append([class_id, x_center, y_center, width, height])

        targets = (
            torch.tensor(targets, dtype=torch.float32)
            if targets
            else torch.zeros((0, 5))
        )

        return image, targets

# ==========================
# 4. YOLO 损失函数
# ==========================

# 请确保实现了完整的 YOLO 损失函数，以下仅为占位符
class YoloLoss(nn.Module):
    def __init__(self, num_classes=1, device="cuda"):
        super(YoloLoss, self).__init__()
        self.num_classes = num_classes
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self, predictions, targets):
        """
        简化版损失函数，仅计算预测与目标的均方误差。
        实际应用中，需要实现 YOLO 的复杂损失函数。
        
        predictions: 模型输出，形状 (batch_size, 5 + num_classes, H, W)
        targets: 目标标签，列表，每个元素形状为 (num_targets, 5)
        """
        # 检查是否有目标
        if len(targets) == 0:
            # 返回一个需要梯度的零张量
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        
        # 创建与 predictions 相同形状的目标张量
        # 初始化为零
        target_tensor = torch.zeros_like(predictions)

        for i, target in enumerate(targets):
            for obj in target:
                class_id, x_center, y_center, width, height = obj
                # 假设 H = W = grid_size, 例如 20
                grid_size = predictions.size(2)  # H
                grid_x = int(x_center * grid_size)
                grid_y = int(y_center * grid_size)
                if grid_x >= grid_size or grid_y >= grid_size:
                    continue  # 跳过无效的目标
                # 赋值
                target_tensor[i, 0:5, grid_y, grid_x] = torch.tensor([1.0, x_center, y_center, width, height], device=self.device)
                if self.num_classes > 0:
                    target_tensor[i, 5 + int(class_id), grid_y, grid_x] = 1.0
        
        # 计算均方误差损失
        loss = self.mse(predictions, target_tensor)
        return loss

# ==========================
# 5. 训练循环
# ==========================

def train():
    # 配置
    # 使用绝对路径或相对于当前脚本的相对路径
    images_dir_train = r'G:\\gra\\train\\dataset\\images\\train'  # 替换为您的训练图像目录
    labels_dir_train = r'G:\\gra\\train\\dataset\\labels\\train'  # 替换为您的训练标签目录

    images_dir_val = r'G:\\gra\\train\\dataset\\images\\val'      # 替换为您的验证图像目录
    labels_dir_val = r'G:\\gra\\train\\dataset\\labels\\val'      # 替换为您的验证标签目录

    model_path = None  # 如果有预训练模型路径，设置为该路径
    save_dir = r'G:\\gra\\train\\saved_models'  # 模型保存目录
    os.makedirs(save_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 打印路径以调试
    print(f"Images Directory (Train): {images_dir_train}")
    print(f"Labels Directory (Train): {labels_dir_train}")
    try:
        train_files = os.listdir(images_dir_train)
        print(f"Sample Train Image Files: {train_files[:5]}")  # 打印前5个文件
    except Exception as e:
        print(f"无法列出训练图像文件: {e}")
        return

    # 定义数据增强与预处理
    transform = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 初始化训练数据集和数据加载器
    try:
        dataset_train = YOLODataset(
            images_dir=images_dir_train, labels_dir=labels_dir_train, transform=transform
        )
        print(f"训练集样本数量: {len(dataset_train)}")
    except Exception as e:
        print(f"初始化训练数据集时出错: {e}")
        return

    try:
        dataloader_train = DataLoader(
            dataset_train,
            batch_size=16,
            shuffle=True,
            collate_fn=lambda x: tuple(zip(*x)),
        )
    except Exception as e:
        print(f"初始化训练数据加载器时出错: {e}")
        return

    # 初始化验证数据集和数据加载器（可选）
    try:
        dataset_val = YOLODataset(
            images_dir=images_dir_val, labels_dir=labels_dir_val, transform=transform
        )
        dataloader_val = DataLoader(
            dataset_val,
            batch_size=16,
            shuffle=False,
            collate_fn=lambda x: tuple(zip(*x)),
        )
        print(f"验证集样本数量: {len(dataset_val)}")
    except Exception as e:
        print(f"初始化验证数据集或数据加载器时出错: {e}")
        dataset_val = None
        dataloader_val = None

    # 初始化模型
    model = CustomYOLO(model_path=model_path, num_classes=1)
    model.to(device)

    # 打印模型结构以验证
    print(model)

    # 打印模型参数以验证设备一致性
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.shape)

    # 定义优化器
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 定义损失函数
    criterion = YoloLoss(num_classes=1, device=device)

    # 训练参数
    num_epochs = 50

    for epoch in range(num_epochs):
        model.train(True)
        epoch_loss = 0.0
        loop = tqdm(dataloader_train, desc=f"Epoch [{epoch+1}/{num_epochs}]")

        for batch_idx, (images, targets) in enumerate(loop):
            try:
                images = torch.stack(images).to(device)  # (batch_size, 3, 640, 640)
            except Exception as e:
                print(f"堆叠图像时出错: {e}")
                continue

            # 将 targets 转换为适合损失函数的格式
            try:
                targets = [t.to(device) for t in targets]
            except Exception as e:
                print(f"将目标移动到设备时出错: {e}")
                continue

            optimizer.zero_grad()

            try:
                outputs = model(images)
            except Exception as e:
                print(f"模型前向传播时出错: {e}")
                continue

            try:
                loss = criterion(outputs, targets)
            except Exception as e:
                print(f"计算损失时出错: {e}")
                # 创建一个需要梯度的零张量以避免报错
                loss = torch.tensor(0.0, device=device, requires_grad=True)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader_train)
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_epoch_loss:.4f}")

        # 保存模型
        torch.save(
            model.state_dict(),
            os.path.join(save_dir, f"custom_yolov_rfca_epoch_{epoch+1}.pth"),
        )

        # 可选：在每个 epoch 后进行验证
        if dataset_val and dataloader_val:
            model.train(False)
            val_loss = 0.0
            with torch.no_grad():
                for images, targets in tqdm(dataloader_val, desc="Validation"):
                    try:
                        images = torch.stack(images).to(device)
                        targets = [t.to(device) for t in targets]
                    except Exception as e:
                        print(f"处理验证数据时出错: {e}")
                        continue

                    try:
                        outputs = model(images)
                        loss = criterion(outputs, targets)
                        val_loss += loss.item()
                    except Exception as e:
                        print(f"验证时计算损失出错: {e}")
                        continue
            avg_val_loss = val_loss / len(dataloader_val)
            print(f"Epoch [{epoch+1}/{num_epochs}], Validation Loss: {avg_val_loss:.4f}")

# ==========================
# 6. 主函数
# ==========================

if __name__ == "__main__":
    train()
