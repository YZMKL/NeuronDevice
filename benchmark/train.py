"""
标准神经网络训练脚本 - Benchmark 对比
Standard Neural Network Training Script for Benchmark

用法:
    cd /home/zhc/Projects/neuromorphic
    
    # 训练标准 MLP (MNIST)
    python -m DeviceNeuron.benchmark.train --model mlp --dataset mnist
    
    # 训练标准 CNN (MNIST)
    python -m DeviceNeuron.benchmark.train --model cnn --dataset mnist --epochs 20
    
    # 训练标准 CNN (Fashion-MNIST)
    python -m DeviceNeuron.benchmark.train --model cnn --dataset fashion-mnist
    
    # 训练标准 CNN (CIFAR-10)
    python -m DeviceNeuron.benchmark.train --model cnn --dataset cifar10
    
    # 训练标准 VGG (CIFAR-10)
    python -m DeviceNeuron.benchmark.train --model vgg --dataset cifar10 --epochs 50 --lr 0.0005
    
    # 自定义参数
    python -m DeviceNeuron.benchmark.train --model mlp --hidden 512 256 128 --epochs 15 --lr 0.002

    # 禁用日志
    python -m DeviceNeuron.benchmark.train --model mlp --no-log
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import os
import sys
import time
from datetime import datetime
from typing import Optional, Dict, List, Any
from io import StringIO

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from .models import (
    MNISTStandardMLP,
    MNISTStandardCNN,
    FashionMNISTStandardCNN,
    CIFAR10StandardMLP,
    CIFAR10StandardCNN,
    CIFAR10StandardVGG
)


# ============================================================================
# 日志模块
# ============================================================================

class BenchmarkLogger:
    """
    Benchmark 训练日志记录器
    
    日志命名规则: {模型类型}_standard_{数据集}_{时间戳}.txt
    """
    
    def __init__(
        self,
        log_dir: str = "logs",
        model_type: str = "MLP",
        dataset: str = "mnist",
        enabled: bool = True
    ):
        self.enabled = enabled
        self.log_dir = log_dir
        self.model_type = model_type.upper()
        self.dataset = dataset.lower()
        
        self.log_file = None
        self.log_path = None
        
        if self.enabled:
            self._setup_log_file()
    
    def _setup_log_file(self):
        """设置日志文件"""
        os.makedirs(self.log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M")
        filename = f"{self.model_type}_standard_{self.dataset}_{timestamp}.txt"
        
        self.log_path = os.path.join(self.log_dir, filename)
        self.log_file = open(self.log_path, 'w', encoding='utf-8')
        
        # 写入日志头
        header = f"""
{'='*70}
Benchmark 训练日志 - 标准神经网络
{'='*70}
创建时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
模型类型: {self.model_type} (标准 PyTorch 实现)
数据集: {self.dataset.upper()}
日志文件: {self.log_path}
{'='*70}

注: 此为标准神经网络，无 Crossbar 仿真、无电导映射、无量化噪声
用于与 Crossbar 版本进行性能对比

"""
        self.log_file.write(header)
        self.log_file.flush()
    
    def print(self, *args, **kwargs):
        """同时打印到终端和日志文件"""
        output = StringIO()
        print(*args, file=output, **kwargs)
        message = output.getvalue()
        
        print(message, end='')
        
        if self.enabled and self.log_file:
            self.log_file.write(message)
            self.log_file.flush()
    
    def log_model_info(self, model: nn.Module, model_desc: str = ""):
        """记录模型信息"""
        self.print("\n" + "-"*50)
        self.print("模型信息 (标准神经网络):")
        self.print("-"*50)
        
        if model_desc:
            self.print(f"  结构: {model_desc}")
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.print(f"  总参数: {total_params:,}")
        self.print(f"  可训练参数: {trainable_params:,}")
        self.print(f"  类型: 标准 PyTorch (无 Crossbar 仿真)")
    
    def log_dataset_info(self, train_size: int, test_size: int, img_info: str = ""):
        """记录数据集信息"""
        self.print("\n" + "-"*50)
        self.print("数据集信息:")
        self.print("-"*50)
        self.print(f"  训练集: {train_size} 样本")
        self.print(f"  测试集: {test_size} 样本")
        if img_info:
            self.print(f"  图像: {img_info}")
    
    def log_training_params(
        self,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        optimizer: str = "Adam",
        device: str = "cpu"
    ):
        """记录训练参数"""
        self.print("\n" + "-"*50)
        self.print("训练参数:")
        self.print("-"*50)
        self.print(f"  Epochs: {epochs}")
        self.print(f"  Batch Size: {batch_size}")
        self.print(f"  Learning Rate: {learning_rate}")
        self.print(f"  Optimizer: {optimizer}")
        self.print(f"  Device: {device}")
    
    def log_epoch_start(self, epoch: int, total_epochs: int):
        """记录 epoch 开始"""
        self.print(f"\nEpoch {epoch}/{total_epochs}")
        self.print("-"*50)
    
    def log_epoch_result(
        self,
        epoch: int,
        train_loss: float,
        train_acc: float,
        val_loss: float,
        val_acc: float,
        is_best: bool = False
    ):
        """记录 epoch 结果"""
        if epoch == 0:
            # Epoch 0: 只显示验证结果（初始状态）
            self.print(f"Initial Val Loss: {val_loss:.4f}, Initial Val Acc: {val_acc:.2f}%")
        else:
            self.print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            self.print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
            if is_best:
                self.print(f"✓ Best model saved with accuracy: {val_acc:.2f}%")
    
    def log_final_result(
        self,
        best_acc: float,
        final_train_acc: float,
        total_time: float = None
    ):
        """记录最终结果"""
        self.print("\n" + "="*70)
        self.print("训练完成 - Benchmark 结果")
        self.print("="*70)
        self.print(f"  最佳验证准确率: {best_acc:.2f}%")
        self.print(f"  最终训练准确率: {final_train_acc:.2f}%")
        if total_time:
            self.print(f"  总训练时间: {total_time:.1f} 秒")
        self.print("\n" + "-"*50)
        self.print("对比说明:")
        self.print("-"*50)
        self.print("  此为标准神经网络的性能上限")
        self.print("  Crossbar 版本受到以下限制:")
        self.print("    - 电导态量化 (有限精度)")
        self.print("    - DAC/ADC 量化")
        self.print("    - 器件噪声")
        self.print("  预期 Crossbar 版本准确率会略低于此基准")
        self.print("="*70)
    
    def close(self):
        """关闭日志文件"""
        if self.log_file:
            self.print(f"\n日志已保存到: {self.log_path}")
            self.log_file.close()
            self.log_file = None
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False


def create_benchmark_logger(
    log_dir: str = None,
    model_type: str = "MLP",
    dataset: str = "mnist",
    enabled: bool = True
) -> BenchmarkLogger:
    """创建 Benchmark 日志记录器"""
    if log_dir is None:
        log_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "logs"
        )
    
    return BenchmarkLogger(
        log_dir=log_dir,
        model_type=model_type,
        dataset=dataset,
        enabled=enabled
    )


# ============================================================================
# 数据加载
# ============================================================================

def get_data_loaders(dataset='mnist', batch_size=64):
    """加载数据集"""
    if dataset == 'mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)
    
    elif dataset == 'fashion-mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
    
    elif dataset == 'cifar10':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


# ============================================================================
# 训练函数
# ============================================================================

def train_standard_model(
    model: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    epochs: int = 10,
    learning_rate: float = 0.001,
    device: str = 'cpu',
    logger: BenchmarkLogger = None,
    save_path: str = None
):
    """
    训练标准神经网络
    
    Args:
        model: PyTorch 模型
        train_loader: 训练数据加载器
        test_loader: 测试数据加载器
        epochs: 训练轮数
        learning_rate: 学习率
        device: 设备
        logger: 日志记录器
        save_path: 模型保存路径
        
    Returns:
        history: 训练历史
    """
    model = model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1) # 针对vgg
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0.0
    final_train_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    # Epoch 0: 初始验证（训练前）
    if logger:
        logger.log_epoch_start(0, epochs)
    
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            val_total += target.size(0)
            val_correct += predicted.eq(target).sum().item()
    
    val_loss /= len(test_loader)
    val_acc = 100. * val_correct / val_total
    
    # 记录初始验证结果
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    
    # 检查是否是最佳模型
    is_best = val_acc > best_acc
    if is_best:
        best_acc = val_acc
        if save_path:
            torch.save(model.state_dict(), save_path)
    
    if logger:
        logger.log_epoch_result(0, 0.0, 0.0, val_loss, val_acc, is_best)
    
    for epoch in range(1, epochs + 1):
        if logger:
            logger.log_epoch_start(epoch, epochs)
        
        # 训练
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc='Training', leave=True)
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()
            
            pbar.set_postfix({
                'loss': f'{train_loss/(batch_idx+1):.2f}',
                'acc': f'{100.*train_correct/train_total:.1f}'
            })
        
        train_loss /= len(train_loader)
        train_acc = 100. * train_correct / train_total
        final_train_acc = train_acc
        
        # 验证
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_loss /= len(test_loader)
        val_acc = 100. * val_correct / val_total
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 保存最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            if save_path:
                torch.save(model.state_dict(), save_path)
        
        if logger:
            logger.log_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, is_best)
        
        scheduler.step()
    
    history['best_acc'] = best_acc
    history['final_train_acc'] = final_train_acc
    
    return history


# ============================================================================
# 主函数
# ============================================================================

def main(args):
    """主训练函数"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 获取日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 创建日志记录器
    logger = create_benchmark_logger(
        log_dir=log_dir,
        model_type=args.model,
        dataset=args.dataset,
        enabled=not args.no_log
    )
    
    start_time = time.time()
    
    try:
        logger.print(f'Using device: {device}')
        
        # ========== 加载数据 ==========
        logger.print('\n' + '='*60)
        logger.print(f'加载 {args.dataset.upper()} 数据集')
        logger.print('='*60)
        
        train_loader, test_loader = get_data_loaders(
            dataset=args.dataset,
            batch_size=args.batch_size
        )
        
        dataset_info = {
            'mnist': '28x28x1',
            'fashion-mnist': '28x28x1',
            'cifar10': '32x32x3'
        }
        logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            test_size=len(test_loader.dataset),
            img_info=dataset_info[args.dataset]
        )
        
        # ========== 创建模型 ==========
        logger.print('\n' + '='*60)
        logger.print(f'创建标准模型 ({args.model} on {args.dataset.upper()})')
        logger.print('='*60)
        
        model_desc = ""
        
        if args.model == 'mlp':
            if args.dataset == 'mnist':
                model = MNISTStandardMLP(hidden_sizes=args.hidden, dropout=args.dropout)
                model_desc = f'Standard MLP: 784 → {args.hidden} → 10'
            elif args.dataset == 'fashion-mnist':
                model = MNISTStandardMLP(hidden_sizes=args.hidden, dropout=args.dropout)
                model_desc = f'Standard MLP: 784 → {args.hidden} → 10'
            else:
                model = CIFAR10StandardMLP(hidden_sizes=args.hidden, dropout=args.dropout)
                model_desc = f'Standard MLP: 3072 → {args.hidden} → 10'
        
        elif args.model == 'cnn':
            if args.dataset == 'mnist':
                model = MNISTStandardCNN(dropout=args.dropout)
                model_desc = 'Standard CNN: Conv[32,64] → FC[128] → 10'
            elif args.dataset == 'fashion-mnist':
                model = FashionMNISTStandardCNN(dropout=args.dropout)
                model_desc = 'Standard CNN: Conv[32,64] → FC[128] → 10'
            else:
                model = CIFAR10StandardCNN(dropout=args.dropout)
                model_desc = 'Standard CNN: Conv[64,128,256] → FC[512,256] → 10'
        
        elif args.model == 'vgg':
            if args.dataset in ['mnist', 'fashion-mnist']:
                raise ValueError("VGG 模型仅支持 CIFAR-10")
            model = CIFAR10StandardVGG(dropout=args.dropout)
            model_desc = 'Standard VGG: Conv[64,64]→[128,128]→[256,256]→FC[512]→10 (BN)'
        
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        logger.log_model_info(model, model_desc)
        
        # ========== 训练 ==========
        logger.print('\n' + '='*60)
        logger.print('开始训练 (标准神经网络)')
        logger.print('='*60)
        
        logger.log_training_params(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            optimizer="Adam",
            device=device
        )
        
        save_path = f'benchmark_{args.dataset}_{args.model}_best.pth'
        
        history = train_standard_model(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            learning_rate=args.lr,
            device=device,
            logger=logger,
            save_path=save_path
        )
        
        # ========== 最终结果 ==========
        total_time = time.time() - start_time
        logger.log_final_result(
            best_acc=history['best_acc'],
            final_train_acc=history['final_train_acc'],
            total_time=total_time
        )
        
    finally:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='标准神经网络 Benchmark 训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
示例:
  # MNIST MLP
  python -m DeviceNeuron.benchmark.train --model mlp --dataset mnist
  
  # Fashion-MNIST CNN
  python -m DeviceNeuron.benchmark.train --model cnn --dataset fashion-mnist
  
  # CIFAR-10 CNN
  python -m DeviceNeuron.benchmark.train --model cnn --dataset cifar10 --epochs 20
  
  # CIFAR-10 VGG
  python -m DeviceNeuron.benchmark.train --model vgg --dataset cifar10
'''
    )
    
    # 数据集
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='数据集 (default: mnist)')
    
    # 模型
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'vgg'],
                        help='模型类型 (default: mlp)')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=10,
                        help='训练轮数 (default: 10)')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='批大小 (default: 64)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='学习率 (default: 0.001)')
    parser.add_argument('--dropout', type=float, default=0.0,
                        help='Dropout 比例 (default: 0.0)')
    
    # 模型结构
    parser.add_argument('--hidden', type=int, nargs='+', default=[256, 128],
                        help='MLP 隐藏层大小 (default: 256 128)')
    
    # 其他
    parser.add_argument('--no-log', action='store_true',
                        help='禁用日志保存')
    
    args = parser.parse_args()
    main(args)

