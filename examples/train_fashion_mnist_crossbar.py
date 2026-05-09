"""
Fashion-MNIST 交叉阵列 + 自定义激活函数训练脚本
Fashion-MNIST Crossbar + Custom Activation Function Training Script

专门用于训练 Fashion-MNIST 数据集的交叉阵列 CNN 模型，使用自定义器件激活函数。

运行方式:
    cd /home/zhc/Projects/neuromorphic
    
    # 使用默认配置
    python -m DeviceNeuron.examples.train_fashion_mnist_crossbar
    
    # 使用真实器件数据配置
    python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --config real
    
    # 使用高精度配置
    python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --config high
    
    # 快速测试（较少epochs）
    python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --config fast
    
    # 自定义训练轮数
    python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --epochs 30

自定义激活函数:
    修改 DeviceNeuron/my_device_activation.py 中的 MyDeviceActivation 类
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DeviceNeuron import (
    FashionMNISTUnifiedCNNWithDeviceNeuron,
    # 配置模块
    CONFIG,
    DeviceNeuronConfig,
    create_real_device_config,
    create_high_precision_config,
    create_fast_test_config
)
from DeviceNeuron.logger import create_logger


def get_fashion_mnist_loaders(batch_size=64):
    """
    获取 Fashion-MNIST 数据加载器
    
    Args:
        batch_size: 批大小
        
    Returns:
        train_loader, test_loader
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    
    train_dataset = datasets.FashionMNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.FashionMNIST(
        './data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0
    )
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0
    )
    
    return train_loader, test_loader


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 获取日志目录
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )
    
    # 创建日志记录器
    logger = create_logger(
        log_dir=log_dir,
        model_type='cnn_device_neuron',
        dataset='fashion-mnist',
        use_device_neuron=True,
        enabled=not args.no_log
    )
    
    # 记录开始时间
    start_time = time.time()
    
    logger.print(f'Using device: {device}')
    
    try:
        # ========== Step 0: 选择配置 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 0: 加载配置 (DeviceNeuron/config.py)')
        logger.print('='*60)
        
        # 根据命令行参数选择配置
        if args.config == 'real':
            cfg = create_real_device_config()
            logger.print('使用真实器件数据配置')
        elif args.config == 'high':
            cfg = create_high_precision_config()
            logger.print('使用高精度配置')
        elif args.config == 'fast':
            cfg = create_fast_test_config()
            logger.print('使用快速测试配置')
        else:
            cfg = CONFIG
            logger.print('使用默认配置')
        
        # 打印配置信息
        logger.print('\n' + '='*70)
        logger.print('配置信息 - Configuration')
        logger.print('='*70)
        logger.print(cfg)
        
        # ========== Step 1: 创建Crossbar配置 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 1: 创建Crossbar配置（所有层共用）')
        logger.print('='*60)
        
        config = cfg.get_crossbar_config()
        
        logger.print('\n' + '-'*50)
        logger.print('Crossbar 配置:')
        logger.print('-'*50)
        logger.print(f"  电导态数量: {len(config.conductance_states.states)}")
        logger.print(f"  电导范围: [{config.conductance_states.states[0].mean:.2e}, "
                    f"{config.conductance_states.states[-1].mean:.2e}] S")
        logger.print(f"  DAC/ADC位宽: {config.dac_bits}/{config.adc_bits} bits")
        logger.print(f"  DAC噪声: {config.dac_noise}")
        logger.print(f"  ADC噪声: {config.adc_noise}")
        
        # ========== Step 2: 加载数据 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 2: 加载 FASHION-MNIST 数据集')
        logger.print('='*60)
        
        train_loader, test_loader = get_fashion_mnist_loaders(batch_size=cfg.BATCH_SIZE)
        
        logger.print('\n' + '-'*50)
        logger.print('数据集信息:')
        logger.print('-'*50)
        logger.print(f"  训练集: {len(train_loader.dataset)} 样本")
        logger.print(f"  测试集: {len(test_loader.dataset)} 样本")
        logger.print(f"  图像: 28x28x1")
        
        # ========== Step 3: 创建模型 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 3: 创建模型 (Fashion-MNIST CNN + Custom Device Activation)')
        logger.print('='*60)
        
        # 从配置获取器件激活函数
        device_activation = cfg.get_device_activation()
        
        model = FashionMNISTUnifiedCNNWithDeviceNeuron(
            config=config,
            device_activation=device_activation,
            conv_channels=cfg.MNIST_CNN_CONV_CHANNELS,  # 复用MNIST配置（相同结构）
            fc_sizes=cfg.MNIST_CNN_FC_SIZES,
            k_int_input=cfg.INTEGRATOR_K_INPUT,
            k_int_output=cfg.INTEGRATOR_K_OUTPUT,
            device_noise=cfg.DEVICE_NOISE_STD
        )
        
        model_desc = f'Fashion-MNIST DeviceNeuron CNN: Conv{cfg.MNIST_CNN_CONV_CHANNELS} → FC{cfg.MNIST_CNN_FC_SIZES} → 10 ({type(device_activation).__name__})'
        
        model = model.to(device)
        
        # 记录模型信息
        logger.log_model_info(model, model_desc)
        
        # ========== Step 4: 训练 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 4: 开始HIL训练')
        logger.print('='*60)
        
        # 使用命令行参数或配置中的epochs
        epochs = args.epochs if args.epochs > 0 else cfg.EPOCHS
        
        optimizer = optim.Adam(
            model.parameters(), 
            lr=cfg.LEARNING_RATE, 
            weight_decay=cfg.WEIGHT_DECAY
        )
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=cfg.LR_STEP_SIZE, 
            gamma=cfg.LR_GAMMA
        )
        
        logger.log_training_params(
            epochs=epochs,
            batch_size=cfg.BATCH_SIZE,
            learning_rate=cfg.LEARNING_RATE,
            optimizer="Adam",
            device=str(device)
        )
        
        save_path = f'fashion_mnist_crossbar_cnn_device_neuron_best.pth'
        best_acc = 0.0
        final_train_acc = 0.0
        
        # Epoch 0: 初始验证（训练前）
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
        
        # 检查是否是最佳模型
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            if cfg.SAVE_BEST_MODEL:
                torch.save(model.state_dict(), save_path)
        
        # 记录初始验证结果
        logger.log_epoch_result(0, 0.0, 0.0, val_loss, val_acc, is_best)
        
        # 训练循环
        for epoch in range(1, epochs + 1):
            logger.log_epoch_start(epoch, epochs)
            
            # 训练一个 epoch
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
            
            # 检查是否是最佳模型
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                if cfg.SAVE_BEST_MODEL:
                    torch.save(model.state_dict(), save_path)
            
            # 记录 epoch 结果
            logger.log_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, is_best)
            
            if scheduler:
                scheduler.step()
        
        # ========== Step 5: 最终评估 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 5: 最终评估')
        logger.print('='*60)
        
        if cfg.SAVE_BEST_MODEL and os.path.exists(save_path):
            logger.print(f'加载最佳模型: {save_path}')
            model.load_state_dict(torch.load(save_path))
        
        model.eval()
        final_val_loss = 0.0
        final_val_correct = 0
        final_val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                final_val_loss += criterion(output, target).item()
                _, predicted = output.max(1)
                final_val_total += target.size(0)
                final_val_correct += predicted.eq(target).sum().item()
        
        final_val_loss /= len(test_loader)
        final_val_acc = 100. * final_val_correct / final_val_total
        
        # ========== Step 6: 记录最终结果 ==========
        total_time = time.time() - start_time
        
        logger.log_final_result(
            best_acc=best_acc,
            final_train_acc=final_train_acc,
            total_time=total_time
        )
        
        # 额外记录最终验证准确率
        logger.print(f"  最终验证准确率: {final_val_acc:.2f}%")
        
        # 记录权重映射统计
        if hasattr(model, 'get_all_mapping_stats'):
            stats = model.get_all_mapping_stats()
            if stats:
                logger.log_mapping_stats(stats)
        
    finally:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Fashion-MNIST 交叉阵列 + 自定义激活函数训练',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
配置选项:
  default  使用默认配置 (DeviceNeuron/config.py 中的 CONFIG)
  real     使用真实器件数据 (expdata/device2.csv)
  high     高精度配置 (128态, 10-bit DAC/ADC)
  fast     快速测试配置 (32态, 较小网络, 3轮训练)
  
自定义激活函数:
  修改 DeviceNeuron/my_device_activation.py 中的 MyDeviceActivation 类

示例:
  # 使用真实器件数据配置
  python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --config real
  
  # 自定义训练轮数
  python -m DeviceNeuron.examples.train_fashion_mnist_crossbar --config real --epochs 30
'''
    )
    
    # 配置选择
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'real', 'high', 'fast'],
                        help='配置类型: default, real, high, fast (default: default)')
    
    # 训练轮数（0表示使用配置中的值）
    parser.add_argument('--epochs', type=int, default=0,
                        help='训练轮数，0表示使用配置中的值 (default: 0)')
    
    # 其他
    parser.add_argument('--no-log', action='store_true',
                        help='禁用日志保存')
    
    args = parser.parse_args()
    main(args)

