"""
Colab 训练脚本 — 用于 Google Colab / CUDA 环境
自动检测 GPU，启用混合精度，训练 UNet 模型

使用方法:
    python colab_train.py [--epochs 30] [--batch_size 16] [--lr 1e-4]
"""

import os
import sys
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def check_environment():
    """检查环境并打印信息"""
    print("=" * 60)
    print("🚀 Colab 训练环境检查")
    print("=" * 60)
    
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"Python 版本: {sys.version.split()[0]}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")
        print(f"   CUDA 版本: {torch.version.cuda}")
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        print("⚠️  检测到 MPS (Apple Silicon)，将使用 MPS")
        print("   （建议在 Colab GPU 环境下运行以获得最佳性能）")
        device = torch.device('mps')
    else:
        print("❌ 未检测到 GPU，将使用 CPU（会非常慢）")
        device = torch.device('cpu')
    
    print(f"使用设备: {device}")
    print("=" * 60)
    return device


def main():
    parser = argparse.ArgumentParser(description='Colab UNet 训练')
    parser.add_argument('--data_dir', type=str, default='map_line_dataset/patch_data',
                        help='训练数据目录')
    parser.add_argument('--model_path', type=str, default='models/unet_map_lines_colab.pth',
                        help='模型保存路径')
    parser.add_argument('--epochs', type=int, default=30, help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=16, help='批大小')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--img_size', type=int, default=512, help='输入图像尺寸')
    parser.add_argument('--num_workers', type=int, default=2, help='DataLoader worker数')
    args = parser.parse_args()
    
    device = check_environment()
    
    sys.path.insert(0, str(Path(__file__).parent))
    from unet_model import UNet, MapLineDataset
    
    Path(args.model_path).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\n📂 加载数据: {args.data_dir}")
    dataset = MapLineDataset(
        args.data_dir, 
        img_size=args.img_size, 
        mode='patch_labelme'
    )
    
    use_pin_memory = (device.type == 'cuda')
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=(args.num_workers > 0)
    )
    print(f"   样本数: {len(dataset)}")
    print(f"   batch_size: {args.batch_size}")
    print(f"   每 epoch batch 数: {len(dataloader)}")
    
    print(f"\n🧠 初始化模型 (UNet, 4类)")
    model = UNet(n_channels=3, n_classes=4).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   参数量: {total_params/1e6:.1f} M")
    
    class_weights = torch.tensor([1.0, 50.0, 50.0, 80.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    use_amp = (device.type == 'cuda')
    scaler = None
    if use_amp:
        scaler = torch.amp.GradScaler('cuda')
        print("✅ 启用混合精度训练 (AMP)")
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )
    
    print(f"\n🏋️  开始训练 ({args.epochs} epochs)")
    print("-" * 60)
    
    best_loss = float('inf')
    start_time = time.time()
    loss_history = []
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        epoch_start = time.time()
        
        for batch_idx, (imgs, masks) in enumerate(dataloader):
            imgs = imgs.to(device, non_blocking=use_pin_memory)
            masks = masks.to(device, non_blocking=use_pin_memory, dtype=torch.long)
            
            optimizer.zero_grad(set_to_none=True)
            
            if use_amp and scaler:
                with torch.amp.autocast('cuda'):
                    outputs = model(imgs)
                    loss = criterion(outputs, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if (batch_idx + 1) % 20 == 0 or batch_idx == 0:
                elapsed = time.time() - epoch_start
                speed = (batch_idx + 1) / elapsed
                print(f"  Epoch {epoch+1:2d}/{args.epochs} "
                      f"[{batch_idx+1:3d}/{len(dataloader)}] "
                      f"loss={loss.item():.4f} "
                      f"speed={speed:.1f} it/s")
        
        scheduler.step()
        avg_loss = total_loss / max(num_batches, 1)
        loss_history.append(avg_loss)
        epoch_time = time.time() - epoch_start
        
        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'loss_history': loss_history,
            }, args.model_path)
        
        best_flag = " ← BEST" if is_best else ""
        print(f"  ── Epoch {epoch+1:2d} 完成 ── "
              f"avg_loss={avg_loss:.4f} "
              f"time={epoch_time:.1f}s "
              f"lr={scheduler.get_last_lr()[0]:.2e}{best_flag}")
    
    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("🎉 训练完成！")
    print(f"   总耗时: {total_time/60:.1f} 分钟")
    print(f"   最佳 Loss: {best_loss:.4f}")
    print(f"   模型保存: {args.model_path}")
    print("=" * 60)
    
    model_size = Path(args.model_path).stat().st_size / 1024**2
    print(f"   模型大小: {model_size:.1f} MB")
    print("\n💡 提示: 请及时下载模型文件，避免 Colab 会话断开丢失")


if __name__ == '__main__':
    main()
