"""
我的自定义配置
My Custom Configuration

整合:
1. 交叉阵列电导数据: trainingdata/device2.csv
2. 自定义器件激活函数: I(V) = 3.386 * (exp(-V / 44283.495) - 1)

使用方法:
    from DeviceNeuron.my_config import (
        MY_CONFIG,
        create_my_crossbar_config,
        create_my_mlp_model
    )
    
    # 创建模型
    model = create_my_mlp_model(hidden_sizes=[256, 128])
    
    # 训练
    from DeviceNeuron import HILTrainer
    trainer = HILTrainer(model, optimizer, criterion)
"""

import os
import torch
import torch.nn as nn
from dataclasses import dataclass, field
from typing import List, Optional

# 获取数据文件路径
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_TRAINING_DATA_DIR = os.path.join(_MODULE_DIR, 'trainingdata')
_DEVICE_DATA_PATH = os.path.join(_TRAINING_DATA_DIR, 'device2.csv')


@dataclass
class MyConfig:
    """
    我的自定义配置
    
    所有参数在这里调整
    """
    
    # ========================================================================
    # 交叉阵列电导数据
    # ========================================================================
    
    # 数据文件路径
    CONDUCTANCE_DATA_PATH: str = _DEVICE_DATA_PATH
    
    # CSV列配置 (device2.csv: average, STDEV.P, proportion)
    CONDUCTANCE_MEAN_COL: int = 0      # 平均电导在第0列
    CONDUCTANCE_STD_COL: int = 1       # 标准差在第1列
    CONDUCTANCE_SKIP_ROWS: int = 1     # 跳过表头
    
    # ========================================================================
    # DAC/ADC 配置
    # ========================================================================
    
    DAC_BITS: int = 8                  # DAC 位宽
    ADC_BITS: int = 8                  # ADC 位宽
    
    # ========================================================================
    # 器件激活函数配置
    # ========================================================================
    
    # 电压范围 (V)
    DEVICE_V_MIN: float = -4.0
    DEVICE_V_MAX: float = 0.0
    
    # 器件参数: I(V) = I_s * (exp(-V / R_Vt) - 1)
    DEVICE_I_S: float = 3.386
    DEVICE_R_VT: float = 44283.495     # 1.7127e+06 * 0.02585
    
    # 积分器参数
    INTEGRATOR_K_INPUT: float = 1.0
    INTEGRATOR_K_OUTPUT: float = 1.0
    
    # 器件噪声
    DEVICE_NOISE_STD: float = 0.01
    
    # ========================================================================
    # 模型结构配置
    # ========================================================================
    
    # MLP 配置 (MNIST)
    MLP_INPUT_SIZE: int = 784
    MLP_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [256, 128])
    MLP_OUTPUT_SIZE: int = 10
    
    # ========================================================================
    # 训练配置
    # ========================================================================
    
    BATCH_SIZE: int = 64
    EPOCHS: int = 50
    LEARNING_RATE: float = 0.001
    WEIGHT_DECAY: float = 0.0
    
    # 学习率调度
    LR_STEP_SIZE: int = 5
    LR_GAMMA: float = 0.5
    
    # ========================================================================
    # 方法
    # ========================================================================
    
    def get_crossbar_config(self):
        """
        从 device2.csv 创建 CrossbarConfig
        """
        from .crossbar_core import CrossbarConfig
        from .conductance_states import ConductanceStates
        import pandas as pd
        import numpy as np
        
        # 读取数据
        df = pd.read_csv(self.CONDUCTANCE_DATA_PATH)
        
        # 提取有效数据（过滤空行）
        g_mean = df.iloc[:, self.CONDUCTANCE_MEAN_COL].values
        g_std = df.iloc[:, self.CONDUCTANCE_STD_COL].values
        
        # 过滤 NaN
        valid_mask = ~(np.isnan(g_mean) | np.isnan(g_std))
        g_mean = g_mean[valid_mask]
        g_std = g_std[valid_mask]
        
        # 创建脉冲数（0, 1, 2, ...）
        pulse_count = np.arange(len(g_mean))
        
        print(f"[MyConfig] 加载交叉阵列数据:")
        print(f"  文件: {self.CONDUCTANCE_DATA_PATH}")
        print(f"  电导态数量: {len(g_mean)}")
        print(f"  电导范围: [{g_mean.min():.2e}, {g_mean.max():.2e}] S")
        print(f"  标准差范围: [{g_std.min():.2e}, {g_std.max():.2e}] S")
        
        # 创建 ConductanceStates
        states = ConductanceStates.from_arrays(
            pulse_count=pulse_count,
            g_mean=g_mean,
            g_std=g_std
        )
        
        # 创建 CrossbarConfig
        return CrossbarConfig(
            conductance_states=states,
            dac_bits=self.DAC_BITS,
            adc_bits=self.ADC_BITS
        )
    
    def get_device_activation(self):
        """
        获取自定义器件激活函数 (分段拟合版本)
        """
        from .my_device_activation import MyDeviceActivation
        
        # 新的分段拟合激活函数不需要参数
        return MyDeviceActivation()
    
    def print_config(self):
        """打印配置信息"""
        print("\n" + "="*70)
        print("我的自定义配置")
        print("="*70)
        
        print("\n【交叉阵列】")
        print(f"  数据文件: {self.CONDUCTANCE_DATA_PATH}")
        print(f"  DAC/ADC: {self.DAC_BITS}/{self.ADC_BITS} bits")
        
        print("\n【器件激活函数】(分段拟合 + 差分输入)")
        print(f"  输入: [-1,1] (差分归一化) → ×4 → [-4,4]V → 取反")
        print(f"  V∈[-4,-1]: I = -7.6108e-05·V - 3.6188e-05 (线性)")
        print(f"  V∈(-1,0) : I = 3.2084e-05·V² - 9.8649e-06·V - 5.8455e-07 (二次)")
        print(f"  V∈[0,4]  : I = 0 (死区)")
        print(f"  积分器增益: K_in={self.INTEGRATOR_K_INPUT}, K_out={self.INTEGRATOR_K_OUTPUT}")
        
        print("\n【模型结构】")
        print(f"  MLP: {self.MLP_INPUT_SIZE} → {self.MLP_HIDDEN_SIZES} → {self.MLP_OUTPUT_SIZE}")
        
        print("\n【训练参数】")
        print(f"  Batch: {self.BATCH_SIZE}, Epochs: {self.EPOCHS}")
        print(f"  LR: {self.LEARNING_RATE}")
        
        print("="*70 + "\n")


# ============================================================================
# 默认配置实例
# ============================================================================

MY_CONFIG = MyConfig()


# ============================================================================
# 便捷函数
# ============================================================================

def create_my_crossbar_config():
    """
    创建使用 device2.csv 数据的 CrossbarConfig
    """
    return MY_CONFIG.get_crossbar_config()


def create_my_device_activation():
    """
    创建自定义器件激活函数
    """
    return MY_CONFIG.get_device_activation()


def create_my_mlp_model(
    hidden_sizes: List[int] = None,
    input_size: int = None,
    output_size: int = None
):
    """
    创建使用自定义交叉阵列和激活函数的 MLP 模型
    
    Args:
        hidden_sizes: 隐藏层大小 (默认使用配置)
        input_size: 输入维度 (默认 784)
        output_size: 输出维度 (默认 10)
        
    Returns:
        UnifiedMLPWithDeviceNeuron 模型
        
    使用示例:
        model = create_my_mlp_model()
        model = create_my_mlp_model(hidden_sizes=[512, 256, 128])
    """
    from .unified_models import UnifiedMLPWithDeviceNeuron
    
    # 使用默认值或用户指定值
    hidden_sizes = hidden_sizes or MY_CONFIG.MLP_HIDDEN_SIZES
    input_size = input_size or MY_CONFIG.MLP_INPUT_SIZE
    output_size = output_size or MY_CONFIG.MLP_OUTPUT_SIZE
    
    # 创建配置
    config = MY_CONFIG.get_crossbar_config()
    activation = MY_CONFIG.get_device_activation()
    
    # 创建模型
    model = UnifiedMLPWithDeviceNeuron(
        input_size=input_size,
        hidden_sizes=hidden_sizes,
        output_size=output_size,
        config=config,
        device_activation=activation,
        k_int_input=MY_CONFIG.INTEGRATOR_K_INPUT,
        k_int_output=MY_CONFIG.INTEGRATOR_K_OUTPUT,
        device_noise=MY_CONFIG.DEVICE_NOISE_STD
    )
    
    return model


def create_my_cnn_model(
    dataset: str = 'mnist',
    conv_channels: List[int] = None,
    fc_sizes: List[int] = None
):
    """
    创建使用自定义交叉阵列和激活函数的 CNN 模型
    
    Args:
        dataset: 数据集 ('mnist' 或 'cifar10')
        conv_channels: 卷积通道数
        fc_sizes: 全连接层大小
        
    Returns:
        UnifiedCNNWithDeviceNeuron 模型
        
    使用示例:
        model = create_my_cnn_model()
        model = create_my_cnn_model(dataset='cifar10')
    """
    from .unified_models import MNISTUnifiedCNNWithDeviceNeuron, CIFAR10UnifiedCNNWithDeviceNeuron
    
    # 创建配置
    config = MY_CONFIG.get_crossbar_config()
    activation = MY_CONFIG.get_device_activation()
    
    if dataset.lower() == 'mnist':
        model_class = MNISTUnifiedCNNWithDeviceNeuron
        default_conv = [32, 64]
        default_fc = [128]
    else:  # cifar10
        model_class = CIFAR10UnifiedCNNWithDeviceNeuron
        default_conv = [64, 128, 256]
        default_fc = [512, 256]
    
    conv_channels = conv_channels or default_conv
    fc_sizes = fc_sizes or default_fc
    
    model = model_class(
        config=config,
        device_activation=activation,
        conv_channels=conv_channels,
        fc_sizes=fc_sizes,
        k_int_input=MY_CONFIG.INTEGRATOR_K_INPUT,
        k_int_output=MY_CONFIG.INTEGRATOR_K_OUTPUT,
        device_noise=MY_CONFIG.DEVICE_NOISE_STD
    )
    
    return model


def create_my_vgg_model(
    dataset: str = 'cifar10',
    use_batchnorm: bool = True,
    dropout: float = 0.5
):
    """
    创建使用自定义交叉阵列和激活函数的 VGG 模型
    
    Args:
        dataset: 数据集 ('mnist' 或 'cifar10')
        use_batchnorm: 是否使用 BatchNorm
        dropout: Dropout 率
        
    Returns:
        UnifiedVGGWithDeviceNeuron 模型
        
    使用示例:
        model = create_my_vgg_model()
        model = create_my_vgg_model(dataset='cifar10', dropout=0.3)
    """
    from .unified_models import MNISTUnifiedVGGWithDeviceNeuron, CIFAR10UnifiedVGGWithDeviceNeuron
    
    # 创建配置
    config = MY_CONFIG.get_crossbar_config()
    activation = MY_CONFIG.get_device_activation()
    
    if dataset.lower() == 'mnist':
        model_class = MNISTUnifiedVGGWithDeviceNeuron
    else:  # cifar10
        model_class = CIFAR10UnifiedVGGWithDeviceNeuron
    
    model = model_class(
        config=config,
        device_activation=activation,
        use_batchnorm=use_batchnorm,
        dropout=dropout,
        k_int_input=MY_CONFIG.INTEGRATOR_K_INPUT,
        k_int_output=MY_CONFIG.INTEGRATOR_K_OUTPUT,
        device_noise=MY_CONFIG.DEVICE_NOISE_STD
    )
    
    return model


def create_my_trainer(model, device='cuda'):
    """
    创建 HIL 训练器
    
    Args:
        model: 神经网络模型
        device: 设备 ('cuda' 或 'cpu')
        
    Returns:
        (trainer, optimizer, scheduler)
    """
    from .hil_trainer import HILTrainer
    import torch.optim as optim
    
    optimizer = optim.Adam(
        model.parameters(), 
        lr=MY_CONFIG.LEARNING_RATE,
        weight_decay=MY_CONFIG.WEIGHT_DECAY
    )
    
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=MY_CONFIG.LR_STEP_SIZE, 
        gamma=MY_CONFIG.LR_GAMMA
    )
    
    criterion = nn.CrossEntropyLoss()
    
    trainer = HILTrainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler
    )
    
    return trainer, optimizer, scheduler


# ============================================================================
# 统一模型创建接口
# ============================================================================

def create_my_model(
    model_type: str = 'mlp',
    dataset: str = 'mnist',
    **kwargs
):
    """
    统一模型创建接口
    
    只需修改 my_device_activation.py 中的 MyDeviceActivation 类，
    所有模型 (MLP, CNN, VGG) 都会使用相同的激活函数！
    
    Args:
        model_type: 模型类型 ('mlp', 'cnn', 'vgg')
        dataset: 数据集 ('mnist', 'cifar10')
        **kwargs: 传递给对应创建函数的参数
        
    Returns:
        创建的模型
        
    使用示例:
        # MLP
        model = create_my_model('mlp', 'mnist')
        model = create_my_model('mlp', 'mnist', hidden_sizes=[512, 256])
        
        # CNN
        model = create_my_model('cnn', 'mnist')
        model = create_my_model('cnn', 'cifar10', conv_channels=[64, 128])
        
        # VGG
        model = create_my_model('vgg', 'cifar10')
        model = create_my_model('vgg', 'cifar10', dropout=0.3)
    """
    model_type = model_type.lower()
    dataset = dataset.lower()
    
    if model_type == 'mlp':
        # 设置输入输出维度
        if dataset == 'mnist':
            kwargs.setdefault('input_size', 784)
            kwargs.setdefault('output_size', 10)
        else:  # cifar10
            kwargs.setdefault('input_size', 3072)
            kwargs.setdefault('output_size', 10)
        return create_my_mlp_model(**kwargs)
    
    elif model_type == 'cnn':
        return create_my_cnn_model(dataset=dataset, **kwargs)
    
    elif model_type == 'vgg':
        return create_my_vgg_model(dataset=dataset, **kwargs)
    
    else:
        raise ValueError(f"未知模型类型: {model_type}. 支持: mlp, cnn, vgg")


# ============================================================================
# 完整训练流程
# ============================================================================

def train_my_model(
    train_loader,
    test_loader,
    hidden_sizes: List[int] = None,
    epochs: int = None,
    device: str = None,
    save_path: str = 'my_model_best.pth',
    enable_log: bool = True
):
    """
    完整的训练流程
    
    Args:
        train_loader: 训练数据
        test_loader: 测试数据
        hidden_sizes: 隐藏层大小
        epochs: 训练轮数
        device: 设备
        save_path: 保存路径
        enable_log: 是否启用日志
        
    Returns:
        (model, history)
        
    使用示例:
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # 准备数据
        transform = transforms.ToTensor()
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./data', train=False, transform=transform)
        train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
        test_loader = DataLoader(test_data, batch_size=64)
        
        # 训练
        model, history = train_my_model(train_loader, test_loader)
    """
    import torch
    import torch.nn as nn
    import time
    from tqdm import tqdm
    from .logger import create_logger
    
    # 确定设备
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    epochs = epochs or MY_CONFIG.EPOCHS
    hidden_sizes = hidden_sizes or MY_CONFIG.MLP_HIDDEN_SIZES
    
    # 获取日志目录
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    
    # 创建日志记录器
    logger = create_logger(
        log_dir=log_dir,
        model_type="MLP",
        dataset="mnist",
        use_device_neuron=True,
        enabled=enable_log
    )
    
    start_time = time.time()
    
    try:
        logger.print("="*70)
        logger.print("开始训练 - 使用自定义交叉阵列和激活函数")
        logger.print("="*70)
        
        # 记录配置
        logger.log_config(MY_CONFIG)
        
        # 创建模型
        logger.print("\n创建模型...")
        model = create_my_mlp_model(hidden_sizes=hidden_sizes)
        model = model.to(device)
        
        model_desc = f"MLP: 784 → {hidden_sizes} → 10 (DeviceNeuron)"
        logger.log_model_info(model, model_desc)
        
        # 记录数据集信息
        logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            test_size=len(test_loader.dataset),
            img_info="28x28x1"
        )
        
        # 创建训练器
        trainer, optimizer, scheduler = create_my_trainer(model, device)
        
        # 记录训练参数
        logger.log_training_params(
            epochs=epochs,
            batch_size=MY_CONFIG.BATCH_SIZE,
            learning_rate=MY_CONFIG.LEARNING_RATE,
            optimizer="Adam",
            device=str(device)
        )
        
        # 训练循环
        logger.print(f"\n开始训练 ({epochs} epochs)...")
        best_acc = 0.0
        final_train_acc = 0.0
        criterion = nn.CrossEntropyLoss()
        
        # Epoch 0: 初始验证（训练前）
        logger.log_epoch_start(0, epochs)
        
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
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
            torch.save(model.state_dict(), save_path)
        
        # 记录初始验证结果
        logger.log_epoch_result(0, 0.0, 0.0, val_loss, val_acc, is_best)
        
        for epoch in range(1, epochs + 1):
            logger.log_epoch_start(epoch, epochs)
            
            # 训练
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            pbar = tqdm(train_loader, desc='Training', leave=True)
            for batch_idx, (data, target) in enumerate(pbar):
                data, target = data.to(device), target.to(device)
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
                    data = data.view(data.size(0), -1)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    val_total += target.size(0)
                    val_correct += predicted.eq(target).sum().item()
            
            val_loss /= len(test_loader)
            val_acc = 100. * val_correct / val_total
            
            # 保存最佳模型
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc
                torch.save(model.state_dict(), save_path)
            
            logger.log_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, is_best)
            
            if scheduler:
                scheduler.step()
        
        # 最终评估
        logger.print("\n最终评估...")
        model.load_state_dict(torch.load(save_path))
        
        # 记录映射误差
        if hasattr(model, 'get_all_mapping_stats'):
            stats = model.get_all_mapping_stats()
            logger.log_mapping_stats(stats)
        
        # 记录最终结果
        total_time = time.time() - start_time
        logger.log_final_result(best_acc, final_train_acc, total_time)
        
        return model, {'best_acc': best_acc, 'final_train_acc': final_train_acc}
        
    finally:
        logger.close()

