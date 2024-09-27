import torch
from ultralytics import YOLO

# ==========================
# 训练和评估原始 YOLOv8 模型
# ==========================
class CustomYOLOv8(YOLO):
    def __init__(self, model_path='yolov8n.pt', num_classes=80):
        super().__init__(model_path)
        self.num_classes = num_classes
        # 不再修改模型的层结构

# ==========================
# 训练和评估 YOLOv8 模型
# ==========================
def train_and_evaluate():
    # 加载 YOLOv8 模型（不做修改）
    model = CustomYOLOv8('yolov8n.pt')

    # 开始训练，设置每 40 个 epoch 保存一次模型
    model.train(
        data='data.yaml',  # 使用上传的数据文件路径
        epochs=1,  # 训练 80 个 epoch
        batch=16, 
        imgsz=640, 
        save_period=1,  # 每 40 个 epoch 保存一次
        project='models', 
        name='original_yolo'
    )

    # 评估模型
    results = model.val()
    
    # 打印 results_dict 来查看可用的评估结果
    print(results.results_dict)

    # 输出评估结果
    print(f"mAP@0.5: {results.results_dict['metrics/mAP50(B)']}")
    print(f"mAP@0.5:0.95: {results.results_dict['metrics/mAP50-95(B)']}")
    print(f"Precision: {results.results_dict['metrics/precision(B)']}")
    print(f"Recall: {results.results_dict['metrics/recall(B)']}")

# 主函数入口
if __name__ == "__main__":
    train_and_evaluate()
