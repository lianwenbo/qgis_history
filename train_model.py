import os
import torch
from pathlib import Path
from functools import partial


# Monkey patch: 修复 PyTorch 2.6+ 的安全加载问题
original_torch_load = torch.load
torch.load = partial(original_torch_load, weights_only=False)


# 现在导入 ultralytics
from ultralytics import YOLO


def train_segmentation_model(dataset_yaml, model_size='n', epochs=100, imgsz=640):
    """
    训练YOLO实例分割模型
    
    Args:
        dataset_yaml: 数据集配置文件路径
        model_size: 模型大小 ('n', 's', 'm', 'l', 'x')
        epochs: 训练轮数
        imgsz: 输入图像大小
    """
    # 加载预训练模型
    model = YOLO(f'yolov8{model_size}-seg.pt')
    
    # 训练模型
    results = model.train(
        data=dataset_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=2,  # 数据少，用小batch
        device='mps' if os.name == 'posix' and hasattr(os, 'uname') and 'arm64' in os.uname().machine else 'cpu',
        patience=20,
        save=True,
        plots=True,
        val=True,
        # 针对线条检测的增强（减少会破坏线条结构的操作）
        hsv_h=0.01,
        hsv_s=0.5,
        hsv_v=0.3,
        degrees=0.0,  # 不旋转（经线应该是竖直的）
        translate=0.05,
        scale=0.3,
        shear=0.0,    # 不剪切（保持直线结构）
        perspective=0.0,
        flipud=0.0,
        fliplr=0.0,   # 不翻转（避免改变方向）
        mosaic=0.0,
        mixup=0.0
    )
    
    # 验证模型
    metrics = model.val()
    print(f"验证结果: {metrics}")
    
    return model


def load_trained_model(model_path):
    """加载训练好的模型"""
    model = YOLO(model_path)
    return model


if __name__ == '__main__':
    dataset_yaml = 'map_line_dataset/yolo_format/dataset.yaml'
    
    if Path(dataset_yaml).exists():
        print("开始训练...")
        print("使用10张标注地图，图像尺寸1024...")
        model = train_segmentation_model(
            dataset_yaml=dataset_yaml,
            model_size='n',
            epochs=100,
            imgsz=1024  # 大尺寸，保留线条细节
        )
        print("训练完成！")
    else:
        print(f"数据集配置文件不存在: {dataset_yaml}")
        print("请先运行 data_preparation.py 准备数据")
