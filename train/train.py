import torch
import torch.nn as nn
from ultralytics import YOLO

# ==========================
# 定义基础卷积模块
# ==========================
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


class CAConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(CAConv, self).__init__()
        reduced_channels = max(out_channels // 16, 1)
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

# ==========================
# 定义 RFCAConv2 模块
# ==========================
class RFCAConv2(nn.Module):
    def __init__(self, c1, c2, kernel_size, stride):
        super(RFCAConv2, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.group_conv1 = Conv_L(c1, 3 * c1, k=1, g=c1)
        self.group_conv2 = Conv_L(c1, 3 * c1, k=3, g=c1)
        self.group_conv3 = Conv_L(c1, 3 * c1, k=5, g=c1)

        self.softmax = nn.Softmax(dim=1)

        self.group_conv = Conv(c1, 3 * c1, k=3, g=c1)
        self.convDown = Conv(c1, c2, k=3, s=stride, g=1)  # 修改 stride
        self.CA = CAConv(c2, c2, kernel_size, stride)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

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

        out = torch.cat([out1, out2, out3], dim=1)

        batch_size, channels, height, width = out.shape
        output_channels = c1

        if channels != output_channels * 9:
            raise ValueError(
                f"Expected channels = {output_channels * 9}, but got channels = {channels}"
            )

        out = out.view(batch_size, output_channels, 3, 3, height, width)
        out = out.permute(0, 1, 4, 2, 5, 3).reshape(
            batch_size, output_channels, height, width  # 保持空间维度不变
        )

        out = self.convDown(out)
        out = self.CA(out)
        return out

# ==========================
# 递归替换 Conv2d 层的函数
# ==========================
def replace_conv_modules(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            in_channels = child.in_channels
            out_channels = child.out_channels
            kernel_size = child.kernel_size[0]
            stride = child.stride[0]
            padding = child.padding[0]
            dilation = child.dilation[0]
            groups = child.groups
            bias = child.bias is not None

            # 创建 RFCAConv2 替换 Conv2d
            rfca_conv = RFCAConv2(
                c1=in_channels,
                c2=out_channels,
                kernel_size=kernel_size,
                stride=stride
            )

            # 将替换后的模块赋值回去
            setattr(module, name, rfca_conv)
            print(f"Replaced Conv2d layer '{name}' with RFCAConv2")
        else:
            # 递归替换子模块
            replace_conv_modules(child)

# ==========================
# 自定义 YOLOv8 类
# ==========================
class CustomYOLOv8(YOLO):
    def __init__(self, model_path='yolov8n.pt', num_classes=80):
        super().__init__(model_path)
        self.num_classes = num_classes
        self.modify_model()

    def modify_model(self):
        """
        修改 YOLOv8 的模型结构，递归替换所有 Conv2d 层为 RFCAConv2
        """
        replace_conv_modules(self.model)
        print("Model modification complete.")

# ==========================
# 训练和评估自定义的 YOLOv8 模型
# ==========================
def train_and_evaluate():
    # 加载自定义的 YOLOv8 模型
    model = CustomYOLOv8('yolov8n.pt')

    # 打印修改后的模型结构（可选）
    print(model.model)


    model.train(
        data='data.yaml',
        epochs=160,  # 周期数越多提升越大 注意不要过拟合
        batch=38,
        imgsz=640,
        save_period=40,  # 每 40 个 epoch 保存一次
        project='models',
        name='custom_yolo'
    )

    # 评估模型
    results = model.val()

    # 打印 results_dict 来查看可用的评估结果
    print(results.results_dict)

    # 输出评估结果
    print(f"mAP@0.5: {results.results_dict.get('metrics/mAP50(B)', 'N/A')}")
    print(f"mAP@0.5:0.95: {results.results_dict.get('metrics/mAP50-95(B)', 'N/A')}")
    print(f"Precision: {results.results_dict.get('metrics/precision(B)', 'N/A')}")
    print(f"Recall: {results.results_dict.get('metrics/recall(B)', 'N/A')}")

# 主函数入口
if __name__ == "__main__":
    train_and_evaluate()
