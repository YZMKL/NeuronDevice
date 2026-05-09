"""
统一Crossbar模型训练示例
Unified Crossbar Model Training Example

展示如何使用共享的CrossbarConfig来训练不同类型的网络。
所有配置参数集中在 DeviceNeuron/config.py 中管理。

支持数据集:
    - MNIST: 28x28 灰度手写数字 (10类)
    - Fashion-MNIST: 28x28 灰度服装图片 (10类)
    - CIFAR-10: 32x32 RGB彩色图片 (10类)

支持模型类型:
    标准模型 (使用 ReLU 激活):
    - mlp: 多层感知机
    - cnn: 卷积神经网络
    - vgg: VGG风格深层网络 (仅CIFAR-10)
    
    自定义器件神经元模型 (使用 my_device_activation.py 中的激活函数):
    - device_neuron: MLP + 自定义激活
    - cnn_device_neuron: CNN + 自定义激活
    - vgg_device_neuron: VGG + 自定义激活

运行方式:
    cd /home/zhc/Projects/neuromorphic
    
    # 标准模型
    python -m DeviceNeuron.examples.train_unified --model mlp --dataset mnist
    python -m DeviceNeuron.examples.train_unified --model cnn --dataset fashion-mnist
    python -m DeviceNeuron.examples.train_unified --model cnn --dataset cifar10
    python -m DeviceNeuron.examples.train_unified --model vgg --dataset cifar10
    
    # 自定义器件神经元 (激活函数在 my_device_activation.py 中定义)
    python -m DeviceNeuron.examples.train_unified --model device_neuron --dataset mnist
    python -m DeviceNeuron.examples.train_unified --model cnn_device_neuron --dataset fashion-mnist
    python -m DeviceNeuron.examples.train_unified --model cnn_device_neuron --dataset mnist
    python -m DeviceNeuron.examples.train_unified --model vgg_device_neuron --dataset cifar10 
    
    # 使用真实器件数据配置
    python -m DeviceNeuron.examples.train_unified --config real
    
自定义激活函数:
    修改 DeviceNeuron/my_device_activation.py 中的 MyDeviceActivation 类，
    所有 device_neuron/cnn_device_neuron/vgg_device_neuron 模型都会使用该激活函数。
    
配置文件位置:
    DeviceNeuron/config.py - 所有参数在此调整
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
    CrossbarConfig,
    ConductanceStates,
    UnifiedMLP,
    UnifiedCNN,
    UnifiedMLPWithDeviceNeuron,
    UnifiedCNNWithDeviceNeuron,
    UnifiedVGGWithDeviceNeuron,
    MNISTUnifiedMLP,
    MNISTUnifiedCNN,
    FashionMNISTUnifiedCNN,
    CIFAR10UnifiedMLP,
    CIFAR10UnifiedCNN,
    CIFAR10VGGStyleCNN,
    MNISTUnifiedMLPWithDeviceNeuron,
    CIFAR10UnifiedMLPWithDeviceNeuron,
    MNISTUnifiedCNNWithDeviceNeuron,
    FashionMNISTUnifiedCNNWithDeviceNeuron,
    CIFAR10UnifiedCNNWithDeviceNeuron,
    MNISTUnifiedVGGWithDeviceNeuron,
    CIFAR10UnifiedVGGWithDeviceNeuron,
    HILTrainer,
    # 配置模块
    CONFIG,
    DeviceNeuronConfig,
    create_real_device_config,
    create_high_precision_config,
    create_fast_test_config
)
from DeviceNeuron.custom_neuron import SigmoidDeviceActivation
from DeviceNeuron.logger import create_logger


def get_data_loaders(dataset='mnist', batch_size=64):
    """
    获取数据加载器
    
    Args:
        dataset: 'mnist', 'fashion-mnist' 或 'cifar10'
        batch_size: 批大小
        
    Returns:
        train_loader, test_loader
    """
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor()])
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
        # CIFAR-10 需要数据增强以获得更好的效果
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=test_transform)
    else:
        raise ValueError(f"Unknown dataset: {dataset}. Supported: 'mnist', 'fashion-mnist', 'cifar10'")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, test_loader


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 判断是否使用器件激活函数
    use_device_neuron = args.model in ['device_neuron', 'cnn_device_neuron', 'vgg_device_neuron']
    
    # 获取日志目录
    log_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "logs"
    )
    
    # 创建日志记录器
    logger = create_logger(
        log_dir=log_dir,
        model_type=args.model,
        dataset=args.dataset,
        use_device_neuron=use_device_neuron,
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
            cfg = CONFIG  # 使用默认配置
            logger.print('使用默认配置')
        
        # 记录配置信息
        logger.log_config(cfg)
        
        # ========== Step 1: 从配置创建Crossbar配置 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 1: 创建Crossbar配置（所有层共用）')
        logger.print('='*60)
        
        config = cfg.get_crossbar_config()
        
        # 记录 Crossbar 配置
        logger.log_crossbar_config(config)
        
        # ========== Step 2: 加载数据 ==========
        logger.print('\n' + '='*60)
        logger.print(f'Step 2: 加载 {args.dataset.upper()} 数据集')
        logger.print('='*60)
        
        train_loader, test_loader = get_data_loaders(dataset=args.dataset, batch_size=cfg.BATCH_SIZE)
        
        # 数据集信息
        dataset_info = {
            'mnist': {'input_size': 784, 'channels': 1, 'img_size': 28},
            'fashion-mnist': {'input_size': 784, 'channels': 1, 'img_size': 28},
            'cifar10': {'input_size': 3072, 'channels': 3, 'img_size': 32}
        }
        ds_info = dataset_info[args.dataset]
        logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            test_size=len(test_loader.dataset),
            img_info=f'{ds_info["img_size"]}x{ds_info["img_size"]}x{ds_info["channels"]}'
        )
        
        # ========== Step 3: 创建模型 ==========
        logger.print('\n' + '='*60)
        logger.print(f'Step 3: 创建模型 ({args.model} on {args.dataset.upper()})')
        logger.print('='*60)
        
        model_desc = ""
        
        if args.model == 'mlp':
            if args.dataset == 'mnist':
                model = MNISTUnifiedMLP(
                    config=config,
                    hidden_sizes=cfg.MNIST_MLP_HIDDEN_SIZES
                )
                model_desc = f'MLP: 784 → {cfg.MNIST_MLP_HIDDEN_SIZES} → 10'
            else:  # cifar10
                model = CIFAR10UnifiedMLP(
                    config=config,
                    hidden_sizes=cfg.CIFAR10_MLP_HIDDEN_SIZES
                )
                model_desc = f'MLP: 3072 → {cfg.CIFAR10_MLP_HIDDEN_SIZES} → 10'
            
        elif args.model == 'cnn':
            if args.dataset == 'mnist':
                model = MNISTUnifiedCNN(
                    config=config,
                    conv_channels=cfg.MNIST_CNN_CONV_CHANNELS,
                    fc_sizes=cfg.MNIST_CNN_FC_SIZES
                )
                model_desc = f'CNN: Conv{cfg.MNIST_CNN_CONV_CHANNELS} → FC{cfg.MNIST_CNN_FC_SIZES} → 10'
            elif args.dataset == 'fashion-mnist':
                model = FashionMNISTUnifiedCNN(
                    config=config,
                    conv_channels=cfg.MNIST_CNN_CONV_CHANNELS,  # 复用MNIST配置
                    fc_sizes=cfg.MNIST_CNN_FC_SIZES
                )
                model_desc = f'CNN: Conv{cfg.MNIST_CNN_CONV_CHANNELS} → FC{cfg.MNIST_CNN_FC_SIZES} → 10'
            else:  # cifar10
                model = CIFAR10UnifiedCNN(
                    config=config,
                    conv_channels=cfg.CIFAR10_CNN_CONV_CHANNELS,
                    fc_sizes=cfg.CIFAR10_CNN_FC_SIZES
                )
                model_desc = f'CNN: Conv{cfg.CIFAR10_CNN_CONV_CHANNELS} → FC{cfg.CIFAR10_CNN_FC_SIZES} → 10'
        
        elif args.model == 'vgg':
            if args.dataset in ['mnist', 'fashion-mnist']:
                raise ValueError("VGG模型仅支持CIFAR-10数据集，请使用 --dataset cifar10。Fashion-MNIST不支持VGG模型。")
            model = CIFAR10VGGStyleCNN(
                config=config,
                use_batchnorm=True,
                dropout=0.5
            )
            model_desc = 'VGG: Conv[64,64]→Pool→Conv[128,128]→Pool→Conv[256,256]→Pool→FC[512]→10 (BN+Dropout)'
            
        elif args.model == 'device_neuron':
            # 从配置获取器件激活函数 (MLP + Device Neuron)
            device_activation = cfg.get_device_activation()
            
            if args.dataset == 'mnist':
                model = MNISTUnifiedMLPWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    hidden_sizes=cfg.MNIST_MLP_HIDDEN_SIZES,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron MLP: 784 → {cfg.MNIST_MLP_HIDDEN_SIZES} → 10 ({type(device_activation).__name__})'
            else:  # cifar10
                model = CIFAR10UnifiedMLPWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    hidden_sizes=cfg.CIFAR10_MLP_HIDDEN_SIZES,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron MLP: 3072 → {cfg.CIFAR10_MLP_HIDDEN_SIZES} → 10 ({type(device_activation).__name__})'
        
        elif args.model == 'cnn_device_neuron':
            # CNN + Device Neuron
            device_activation = cfg.get_device_activation()
            
            if args.dataset == 'mnist':
                model = MNISTUnifiedCNNWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    conv_channels=cfg.MNIST_CNN_CONV_CHANNELS,
                    fc_sizes=cfg.MNIST_CNN_FC_SIZES,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron CNN: Conv{cfg.MNIST_CNN_CONV_CHANNELS} → FC{cfg.MNIST_CNN_FC_SIZES} → 10 ({type(device_activation).__name__})'
            elif args.dataset == 'fashion-mnist':
                model = FashionMNISTUnifiedCNNWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    conv_channels=cfg.MNIST_CNN_CONV_CHANNELS,  # 复用MNIST配置
                    fc_sizes=cfg.MNIST_CNN_FC_SIZES,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron CNN: Conv{cfg.MNIST_CNN_CONV_CHANNELS} → FC{cfg.MNIST_CNN_FC_SIZES} → 10 ({type(device_activation).__name__})'
            else:  # cifar10
                model = CIFAR10UnifiedCNNWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    conv_channels=cfg.CIFAR10_CNN_CONV_CHANNELS,
                    fc_sizes=cfg.CIFAR10_CNN_FC_SIZES,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron CNN: Conv{cfg.CIFAR10_CNN_CONV_CHANNELS} → FC{cfg.CIFAR10_CNN_FC_SIZES} → 10 ({type(device_activation).__name__})'
        
        elif args.model == 'vgg_device_neuron':
            # VGG + Device Neuron
            device_activation = cfg.get_device_activation()
            
            if args.dataset in ['mnist', 'fashion-mnist']:
                model = MNISTUnifiedVGGWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    use_batchnorm=True,
                    dropout=0.5,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron VGG: VGG-style (BN+Dropout) ({type(device_activation).__name__})'
            else:  # cifar10
                model = CIFAR10UnifiedVGGWithDeviceNeuron(
                    config=config,
                    device_activation=device_activation,
                    use_batchnorm=True,
                    dropout=0.5,
                    k_int_input=cfg.INTEGRATOR_K_INPUT,
                    k_int_output=cfg.INTEGRATOR_K_OUTPUT,
                    device_noise=cfg.DEVICE_NOISE_STD
                )
                model_desc = f'DeviceNeuron VGG: VGG-style (BN+Dropout) ({type(device_activation).__name__})'
        
        else:
            raise ValueError(f"Unknown model type: {args.model}")
        
        model = model.to(device)
        
        # 记录模型信息
        logger.log_model_info(model, model_desc)
        
        # ========== Step 4: 训练 ==========
        logger.print('\n' + '='*60)
        logger.print('Step 4: 开始HIL训练')
        logger.print('='*60)
        
        optimizer = optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
        criterion = nn.CrossEntropyLoss()
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.LR_STEP_SIZE, gamma=cfg.LR_GAMMA)
        
        # 记录训练参数
        logger.log_training_params(
            epochs=cfg.EPOCHS,
            batch_size=cfg.BATCH_SIZE,
            learning_rate=cfg.LEARNING_RATE,
            optimizer="Adam",
            device=str(device)
        )
        
        save_path = f'unified_{args.dataset}_{args.model}_best.pth'
        best_acc = 0.0
        final_train_acc = 0.0
        
        # Epoch 0: 初始验证（训练前）
        logger.log_epoch_start(0, cfg.EPOCHS)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                
                if args.model in ['mlp', 'device_neuron']:
                    data = data.view(data.size(0), -1)
                
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
        for epoch in range(1, cfg.EPOCHS + 1):
            logger.log_epoch_start(epoch, cfg.EPOCHS)
            
            # 训练一个 epoch
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc=f'Training', leave=True)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
                
                # 根据模型类型处理输入 (MLP 需要展平)
                if args.model in ['mlp', 'device_neuron']:
                    data = data.view(data.size(0), -1)
                
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
                    
                    if args.model in ['mlp', 'device_neuron']:
                        data = data.view(data.size(0), -1)
                    
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
        
        # 记录映射误差
        if hasattr(model, 'get_all_mapping_stats'):
            stats = model.get_all_mapping_stats()
            logger.log_mapping_stats(stats)
        
        # 记录最终结果
        total_time = time.time() - start_time
        logger.log_final_result(best_acc, final_train_acc, total_time)
        
    finally:
        # 确保日志文件被关闭
        logger.close()


def demo_usage():
    """演示统一Crossbar的使用方法"""
    print('''
================================================================================
统一Crossbar模块使用说明
================================================================================

核心概念:
  - 所有配置集中在 DeviceNeuron/config.py 中管理
  - CrossbarConfig: 统一的配置对象，定义器件电导态和DAC/ADC参数
  - 所有Crossbar层共用同一个配置，确保器件特性一致

================================================================================
配置文件位置: DeviceNeuron/config.py
================================================================================

所有参数都在这个文件中调整:

┌─────────────────────────────────────────────────────────────────────────────┐
│  # 电导态数据配置                                                          │
│  CONDUCTANCE_DATA_PATH = 'expdata/set2.csv'  # 设为 None 用默认值          │
│  CONDUCTANCE_PULSE_COL = 0                    # 脉冲数列                    │
│  CONDUCTANCE_MEAN_COL = 1                     # 平均电导列                  │
│  CONDUCTANCE_STD_COL = 2                      # 标准差列                    │
│                                                                             │
│  # DAC/ADC 配置                                                            │
│  DAC_BITS = 8                                                              │
│  ADC_BITS = 8                                                              │
│                                                                             │
│  # 器件神经元配置                                                          │
│  DEVICE_TRANSFER_CURVE_PATH = 'expdata/transfer+curve.xlsx'               │
│  INTEGRATOR_K_INPUT = 1.0                                                  │
│  INTEGRATOR_K_OUTPUT = 1.0                                                 │
│                                                                             │
│  # 模型结构                                                                │
│  MLP_HIDDEN_SIZES = [256, 128]                                             │
│  CNN_CONV_CHANNELS = [32, 64]                                              │
│                                                                             │
│  # 训练参数                                                                │
│  BATCH_SIZE = 64                                                           │
│  EPOCHS = 10                                                               │
│  LEARNING_RATE = 0.001                                                     │
└─────────────────────────────────────────────────────────────────────────────┘

================================================================================
代码使用方法
================================================================================

方法1: 使用预设配置
------------------
from DeviceNeuron import CONFIG, create_real_device_config

# 使用默认配置
config = CONFIG.get_crossbar_config()

# 使用真实器件数据配置
cfg = create_real_device_config()
config = cfg.get_crossbar_config()
curve = cfg.get_transfer_curve()

方法2: 自定义配置
------------------
from DeviceNeuron import DeviceNeuronConfig

my_cfg = DeviceNeuronConfig(
    CONDUCTANCE_DATA_PATH='my_device.csv',
    DAC_BITS=10,
    ADC_BITS=10,
    MLP_HIDDEN_SIZES=[512, 256, 128]
)
config = my_cfg.get_crossbar_config()

方法3: 从JSON加载
------------------
from DeviceNeuron import load_config_from_file, save_config_to_file

# 保存配置
save_config_to_file(my_cfg, 'my_config.json')

# 加载配置
cfg = load_config_from_file('my_config.json')

================================================================================
命令行使用
================================================================================

# 默认配置
python -m DeviceNeuron.examples.train_unified --model mlp

# 真实器件数据
python -m DeviceNeuron.examples.train_unified --model device_neuron --config real

# 快速测试
python -m DeviceNeuron.examples.train_unified --config fast

================================================================================
''')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='使用统一Crossbar训练神经网络',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
数据集选项:
  mnist         MNIST 手写数字 (28x28 灰度, 10类, 60000训练/10000测试)
  fashion-mnist Fashion-MNIST 服装图片 (28x28 灰度, 10类, 60000训练/10000测试)
  cifar10       CIFAR-10 彩色图片 (32x32 RGB, 10类, 50000训练/10000测试)

模型选项:
  标准模型 (ReLU 激活):
    mlp              多层感知机
    cnn              卷积神经网络
    vgg              VGG风格深层网络 (仅CIFAR-10)
    
  自定义激活函数模型 (使用 my_device_activation.py 中的 MyDeviceActivation):
    device_neuron         MLP + 自定义激活
    cnn_device_neuron     CNN + 自定义激活
    vgg_device_neuron     VGG + 自定义激活

配置选项:
  default  使用默认配置 (DeviceNeuron/config.py 中的 CONFIG)
  real     使用真实器件数据 (expdata/device2.csv)
  high     高精度配置 (128态, 10-bit DAC/ADC)
  fast     快速测试配置 (32态, 较小网络, 3轮训练)
  
自定义激活函数:
  修改 DeviceNeuron/my_device_activation.py 中的 MyDeviceActivation 类，
  所有带 _device_neuron 后缀的模型都会使用该激活函数。

示例:
  # 标准 MNIST MLP
  python -m DeviceNeuron.examples.train_unified --model mlp --dataset mnist
  
  # 标准 Fashion-MNIST CNN
  python -m DeviceNeuron.examples.train_unified --model cnn --dataset fashion-mnist
  
  # 自定义激活函数 MNIST MLP
  python -m DeviceNeuron.examples.train_unified --model device_neuron --dataset mnist
  
  # 自定义激活函数 Fashion-MNIST CNN
  python -m DeviceNeuron.examples.train_unified --model cnn_device_neuron --dataset fashion-mnist
  
  # 自定义激活函数 CNN (MNIST)
  python -m DeviceNeuron.examples.train_unified --model cnn_device_neuron --dataset mnist
  
  # 自定义激活函数 VGG (CIFAR-10)
  python -m DeviceNeuron.examples.train_unified --model vgg_device_neuron --dataset cifar10 --config real
'''
    )
    
    # 数据集选择
    parser.add_argument('--dataset', type=str, default='mnist',
                        choices=['mnist', 'fashion-mnist', 'cifar10'],
                        help='数据集: mnist, fashion-mnist, cifar10 (default: mnist)')
    
    # 模型选择
    parser.add_argument('--model', type=str, default='mlp',
                        choices=['mlp', 'cnn', 'vgg', 'device_neuron', 'cnn_device_neuron', 'vgg_device_neuron'],
                        help='模型类型: mlp, cnn, vgg, device_neuron (MLP+自定义激活), cnn_device_neuron (CNN+自定义激活), vgg_device_neuron (VGG+自定义激活)')
    
    # 配置选择
    parser.add_argument('--config', type=str, default='default',
                        choices=['default', 'real', 'high', 'fast'],
                        help='配置类型: default, real, high, fast')
    
    # 其他
    parser.add_argument('--demo', action='store_true', 
                        help='显示使用说明')
    parser.add_argument('--no-log', action='store_true',
                        help='禁用日志保存')
    
    args = parser.parse_args()
    
    if args.demo:
        demo_usage()
    else:
        main(args)

