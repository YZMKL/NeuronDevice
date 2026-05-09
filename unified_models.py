"""
统一模型模块 - 使用共享Crossbar核心的网络模型
Unified Models - Network models using shared Crossbar core

所有模型共用同一个CrossbarConfig配置，确保器件特性一致。

使用方法:
    from DeviceNeuron import CrossbarConfig, UnifiedMLP, UnifiedCNN
    
    # 1. 创建配置（定义一次，所有层共用）
    config = CrossbarConfig.from_csv('device_data.csv')
    
    # 2. 创建模型
    mlp = UnifiedMLP(784, [256, 128], 10, config)
    cnn = UnifiedCNN(config, input_channels=1, num_classes=10)
"""

import torch
import torch.nn as nn
from typing import List, Optional

from .crossbar_core import (
    CrossbarConfig,
    CrossbarMVMCore,
    UnifiedCrossbarLinear,
    UnifiedCrossbarConv2d,
    UnifiedCrossbarLinearWithActivation,
    UnifiedCrossbarConv2dWithActivation
)
from .custom_neuron import (
    CustomDeviceNeuron, 
    UserDeviceActivation,
    SigmoidDeviceActivation
)


class DeviceNeuronAfterBN(nn.Module):
    """
    在 BatchNorm 之后应用器件非线性，使整体拓扑与标准 CNN 对齐：

        Crossbar 线性层 → BatchNorm → DeviceNeuronAfterBN

    这里不再做额外的 batch max 归一化（BN 已经稳定了尺度），仅将 BN 输出用 tanh
    映射到 (-1, 1)，再送入 CustomDeviceNeuron，并用可学习 output_scale 调整幅度。
    """

    def __init__(self, device_neuron: CustomDeviceNeuron):
        super().__init__()
        self.device_neuron = device_neuron
        self.output_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.tanh(x)
        shape_before = x.shape
        y = self.device_neuron(x.reshape(-1)).reshape(shape_before)
        return y * self.output_scale


class UnifiedMLP(nn.Module):
    """
    统一的Crossbar MLP网络
    所有层共用同一个CrossbarConfig
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        config: CrossbarConfig,
        activation: str = 'relu'
    ):
        """
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层大小列表
            output_size: 输出维度
            config: Crossbar配置（共享）
            activation: 激活函数类型
        """
        super().__init__()
        
        self.config = config
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(UnifiedCrossbarLinear(prev_size, hidden_size, config))
            layers.append(self._get_activation(activation))
            prev_size = hidden_size
        
        layers.append(UnifiedCrossbarLinear(prev_size, output_size, config))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, name: str) -> nn.Module:
        activations = {
            'relu': nn.ReLU(),
            'sigmoid': nn.Sigmoid(),
            'tanh': nn.Tanh(),
            'leaky_relu': nn.LeakyReLU(0.1),
        }
        return activations.get(name.lower(), nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_all_mapping_stats(self) -> List[dict]:
        """获取所有层的映射统计"""
        stats = []
        for module in self.modules():
            if isinstance(module, UnifiedCrossbarLinear):
                stats.append(module.get_mapping_stats())
        return stats


class UnifiedCNN(nn.Module):
    """
    统一的Crossbar CNN网络
    所有层共用同一个CrossbarConfig
    """
    
    def __init__(
        self,
        config: CrossbarConfig,
        input_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        conv_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_sizes: List[int] = [128]
    ):
        """
        Args:
            config: Crossbar配置（共享）
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            num_classes: 类别数
            conv_channels: 卷积层通道数
            kernel_size: 卷积核大小
            pool_size: 池化核大小
            fc_sizes: 全连接层大小
        """
        super().__init__()
        
        self.config = config
        
        # 卷积层
        conv_layers = []
        in_ch = input_channels
        current_size = input_size
        
        for out_ch in conv_channels:
            conv_layers.append(UnifiedCrossbarConv2d(
                in_ch, out_ch, kernel_size, config, padding=kernel_size // 2
            ))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(pool_size))
            in_ch = out_ch
            current_size = current_size // pool_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 全连接层
        flatten_size = conv_channels[-1] * current_size * current_size
        
        fc_layers = []
        prev_size = flatten_size
        
        for fc_size in fc_sizes:
            fc_layers.append(UnifiedCrossbarLinear(prev_size, fc_size, config))
            fc_layers.append(nn.ReLU())
            prev_size = fc_size
        
        fc_layers.append(UnifiedCrossbarLinear(prev_size, num_classes, config))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)  # 使用 reshape 处理非连续张量
        x = self.fc_layers(x)
        return x
    
    def get_all_mapping_stats(self) -> List[dict]:
        stats = []
        for module in self.modules():
            if isinstance(module, (UnifiedCrossbarLinear, UnifiedCrossbarConv2d)):
                stats.append(module.get_mapping_stats())
        return stats


class UnifiedMLPWithDeviceNeuron(nn.Module):
    """
    使用器件激活函数的统一MLP
    
    所有层共用 Crossbar 配置，激活函数使用用户自定义的 V→I 函数
    
    使用方法:
        # 方法1: 使用预定义激活函数
        from DeviceNeuron import SigmoidDeviceActivation
        model = UnifiedMLPWithDeviceNeuron(
            784, [256, 128], 10, config,
            device_activation=SigmoidDeviceActivation(k=12, v0=0.5)
        )
        
        # 方法2: 使用自定义函数
        def my_vi(v):
            return torch.sigmoid(10 * (v - 0.5))
        model = UnifiedMLPWithDeviceNeuron(
            784, [256, 128], 10, config,
            device_activation=FunctionDeviceActivation(my_vi)
        )
        
        # 方法3: 使用多项式拟合
        from DeviceNeuron import PolynomialDeviceActivation
        model = UnifiedMLPWithDeviceNeuron(
            784, [256, 128], 10, config,
            device_activation=PolynomialDeviceActivation([0.01, 0.5, 2.0])
        )
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0,
        # 向后兼容参数 (已弃用)
        transfer_curve = None,
    ):
        """
        Args:
            input_size: 输入维度
            hidden_sizes: 隐藏层大小
            output_size: 输出维度
            config: Crossbar 配置
            device_activation: 用户定义的 V→I 激活函数 (继承自 UserDeviceActivation)
            k_int_input: 输入积分器增益
            k_int_output: 输出积分器增益  
            device_noise: 器件噪声标准差
            transfer_curve: [已弃用] 使用 device_activation 代替
        """
        super().__init__()
        
        self.config = config
        
        # 确定使用的激活函数
        if device_activation is not None:
            self._activation_template = device_activation
        else:
            # 使用默认 Sigmoid 激活
            self._activation_template = SigmoidDeviceActivation(k=10.0, v0=0.5)
        
        self.k_int_input = k_int_input
        self.k_int_output = k_int_output
        self.device_noise = device_noise
        
        # 构建层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            # 每层创建独立的激活函数实例
            activation = self._create_activation_copy()
            
            layers.append(UnifiedCrossbarLinearWithActivation(
                prev_size, hidden_size, config,
                activation_module=CustomDeviceNeuron(
                    activation=activation,
                    k_int_input=k_int_input,
                    k_int_output=k_int_output,
                    device_noise=device_noise
                )
            ))
            prev_size = hidden_size
        
        # 输出层（无激活函数）
        layers.append(UnifiedCrossbarLinear(prev_size, output_size, config))
        
        self.network = nn.Sequential(*layers)
    
    def _create_activation_copy(self) -> UserDeviceActivation:
        """创建激活函数的副本（每层独立）"""
        # 对于大多数预定义激活函数，直接使用同一个实例是安全的
        # 因为它们没有可学习参数
        return self._activation_template
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = x.view(x.size(0), -1)
        return self.network(x)
    
    def get_all_mapping_stats(self) -> List[dict]:
        """获取所有层的映射统计"""
        stats = []
        for module in self.modules():
            if isinstance(module, UnifiedCrossbarLinear):
                stats.append(module.get_mapping_stats())
        return stats


class UnifiedCNNWithDeviceNeuron(nn.Module):
    """
    使用器件激活函数的统一CNN
    
    所有卷积层和全连接层都使用自定义器件激活函数
    
    使用方法:
        from DeviceNeuron import UnifiedCNNWithDeviceNeuron
        from DeviceNeuron.my_device_activation import MyDeviceActivation
        
        activation = MyDeviceActivation()
        model = UnifiedCNNWithDeviceNeuron(
            config=config,
            device_activation=activation,
            input_channels=1,
            num_classes=10
        )
    """
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        input_channels: int = 1,
        input_size: int = 28,
        num_classes: int = 10,
        conv_channels: List[int] = [32, 64],
        kernel_size: int = 3,
        pool_size: int = 2,
        fc_sizes: List[int] = [128],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        """
        Args:
            config: Crossbar配置
            device_activation: 用户定义的 V→I 激活函数
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            num_classes: 类别数
            conv_channels: 卷积层通道数
            kernel_size: 卷积核大小
            pool_size: 池化核大小
            fc_sizes: 全连接层大小
            k_int_input: 输入积分器增益
            k_int_output: 输出积分器增益
            device_noise: 器件噪声
        """
        super().__init__()
        
        self.config = config
        
        # 确定使用的激活函数
        if device_activation is not None:
            self._activation_template = device_activation
        else:
            self._activation_template = SigmoidDeviceActivation(k=10.0, v0=0.5)
        
        self.k_int_input = k_int_input
        self.k_int_output = k_int_output
        self.device_noise = device_noise
        
        # 卷积层
        conv_layers = []
        in_ch = input_channels
        current_size = input_size
        
        for out_ch in conv_channels:
            activation = self._create_activation_copy()
            conv_layers.append(UnifiedCrossbarConv2dWithActivation(
                in_ch, out_ch, kernel_size, config, 
                padding=kernel_size // 2,
                activation_module=CustomDeviceNeuron(
                    activation=activation,
                    k_int_input=k_int_input,
                    k_int_output=k_int_output,
                    device_noise=device_noise
                )
            ))
            conv_layers.append(nn.MaxPool2d(pool_size))
            in_ch = out_ch
            current_size = current_size // pool_size
        
        self.conv_layers = nn.Sequential(*conv_layers)
        
        # 全连接层
        flatten_size = conv_channels[-1] * current_size * current_size
        
        fc_layers = []
        prev_size = flatten_size
        
        for fc_size in fc_sizes:
            activation = self._create_activation_copy()
            fc_layers.append(UnifiedCrossbarLinearWithActivation(
                prev_size, fc_size, config,
                activation_module=CustomDeviceNeuron(
                    activation=activation,
                    k_int_input=k_int_input,
                    k_int_output=k_int_output,
                    device_noise=device_noise
                )
            ))
            prev_size = fc_size
        
        # 输出层（无激活函数）
        fc_layers.append(UnifiedCrossbarLinear(prev_size, num_classes, config))
        
        self.fc_layers = nn.Sequential(*fc_layers)
    
    def _create_activation_copy(self) -> UserDeviceActivation:
        """创建激活函数的副本"""
        return self._activation_template
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_layers(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc_layers(x)
        return x
    
    def get_all_mapping_stats(self) -> List[dict]:
        stats = []
        for module in self.modules():
            if isinstance(module, (UnifiedCrossbarLinear, UnifiedCrossbarConv2d, 
                                    UnifiedCrossbarLinearWithActivation, UnifiedCrossbarConv2dWithActivation)):
                stats.append(module.get_mapping_stats())
        return stats


class UnifiedVGGWithDeviceNeuron(nn.Module):
    """
    使用器件激活函数的VGG风格CNN
    
    结构与 CIFAR10VGGStyleCNN 相同，但使用自定义器件激活函数
    
    结构:
        Conv Block 1: 64 -> 64 -> Pool (32->16)
        Conv Block 2: 128 -> 128 -> Pool (16->8)
        Conv Block 3: 256 -> 256 -> Pool (8->4)
        FC: 4096 -> 512 -> 10
    
    使用方法:
        from DeviceNeuron import UnifiedVGGWithDeviceNeuron
        from DeviceNeuron.my_device_activation import MyDeviceActivation
        
        activation = MyDeviceActivation()
        model = UnifiedVGGWithDeviceNeuron(
            config=config,
            device_activation=activation,
            input_channels=3,
            num_classes=10
        )
    """
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        input_channels: int = 3,
        input_size: int = 32,
        num_classes: int = 10,
        use_batchnorm: bool = True,
        dropout: float = 0.5,
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        """
        Args:
            config: Crossbar配置
            device_activation: 用户定义的 V→I 激活函数
            input_channels: 输入通道数
            input_size: 输入图像尺寸
            num_classes: 类别数
            use_batchnorm: 是否使用BatchNorm
            dropout: Dropout率
            k_int_input: 输入积分器增益
            k_int_output: 输出积分器增益
            device_noise: 器件噪声
        """
        super().__init__()
        
        self.config = config
        
        # 确定使用的激活函数
        if device_activation is not None:
            self._activation_template = device_activation
        else:
            self._activation_template = SigmoidDeviceActivation(k=10.0, v0=0.5)
        
        self.k_int_input = k_int_input
        self.k_int_output = k_int_output
        self.device_noise = device_noise
        
        # Conv Block 1: 3 -> 64 -> 64, then pool（Conv → BN → 器件激活）
        self.conv1_1 = UnifiedCrossbarConv2d(input_channels, 64, 3, config, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.dn1_1 = DeviceNeuronAfterBN(self._create_device_neuron())

        self.conv1_2 = UnifiedCrossbarConv2d(64, 64, 3, config, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.dn1_2 = DeviceNeuronAfterBN(self._create_device_neuron())
        self.pool1 = nn.MaxPool2d(2)  # 32 -> 16
        
        # Conv Block 2: 64 -> 128 -> 128, then pool
        self.conv2_1 = UnifiedCrossbarConv2d(64, 128, 3, config, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.dn2_1 = DeviceNeuronAfterBN(self._create_device_neuron())

        self.conv2_2 = UnifiedCrossbarConv2d(128, 128, 3, config, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.dn2_2 = DeviceNeuronAfterBN(self._create_device_neuron())
        self.pool2 = nn.MaxPool2d(2)  # 16 -> 8
        
        # Conv Block 3: 128 -> 256 -> 256, then pool
        self.conv3_1 = UnifiedCrossbarConv2d(128, 256, 3, config, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.dn3_1 = DeviceNeuronAfterBN(self._create_device_neuron())

        self.conv3_2 = UnifiedCrossbarConv2d(256, 256, 3, config, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.dn3_2 = DeviceNeuronAfterBN(self._create_device_neuron())
        self.pool3 = nn.MaxPool2d(2)  # 8 -> 4
        
        # FC layers
        self.flatten_size = 256 * (input_size // 8) ** 2  # After 3 pooling layers
        self.fc1 = UnifiedCrossbarLinear(self.flatten_size, 512, config)
        self.bn_fc1 = nn.BatchNorm1d(512) if use_batchnorm else nn.Identity()
        self.dn_fc1 = DeviceNeuronAfterBN(self._create_device_neuron())
        self.dropout = nn.Dropout(dropout)
        self.fc2 = UnifiedCrossbarLinear(512, num_classes, config)  # 输出层无激活
    
    def _create_device_neuron(self) -> CustomDeviceNeuron:
        """创建器件神经元模块"""
        return CustomDeviceNeuron(
            activation=self._activation_template,
            k_int_input=self.k_int_input,
            k_int_output=self.k_int_output,
            device_noise=self.device_noise
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.dn1_1(self.bn1_1(self.conv1_1(x)))
        x = self.dn1_2(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.dn2_1(self.bn2_1(self.conv2_1(x)))
        x = self.dn2_2(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.dn3_1(self.bn3_1(self.conv3_1(x)))
        x = self.dn3_2(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # FC
        x = x.reshape(x.size(0), -1)
        x = self.dn_fc1(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_all_mapping_stats(self) -> List[dict]:
        stats = []
        for module in self.modules():
            if isinstance(module, (UnifiedCrossbarLinear, UnifiedCrossbarConv2d,
                                    UnifiedCrossbarLinearWithActivation, UnifiedCrossbarConv2dWithActivation)):
                stats.append(module.get_mapping_stats())
        return stats


# ============= MNIST专用模型 =============

class MNISTUnifiedMLP(UnifiedMLP):
    """MNIST专用的统一MLP"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        hidden_sizes: List[int] = [256, 128]
    ):
        super().__init__(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            config=config
        )


class MNISTUnifiedCNN(UnifiedCNN):
    """MNIST专用的统一CNN"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128]
    ):
        super().__init__(
            config=config,
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes
        )


class FashionMNISTUnifiedCNN(UnifiedCNN):
    """Fashion-MNIST专用的统一CNN"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128]
    ):
        super().__init__(
            config=config,
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes
        )


class CIFAR10UnifiedCNN(UnifiedCNN):
    """CIFAR-10专用的统一CNN (基础版)"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        conv_channels: List[int] = [64, 128, 256],
        fc_sizes: List[int] = [512, 256]
    ):
        super().__init__(
            config=config,
            input_channels=3,
            input_size=32,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes
        )


class CIFAR10VGGStyleCNN(nn.Module):
    """
    CIFAR-10专用的VGG风格CNN (更深的网络，更高的准确率)
    
    结构: 
        Conv Block 1: 64 -> 64 -> Pool (32->16)
        Conv Block 2: 128 -> 128 -> Pool (16->8)
        Conv Block 3: 256 -> 256 -> Pool (8->4)
        FC: 4096 -> 512 -> 10
    
    预期准确率: ~80-85%
    """
    
    def __init__(
        self,
        config: CrossbarConfig,
        use_batchnorm: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        self.config = config
        
        # Conv Block 1: 3 -> 64 -> 64, then pool
        self.conv1_1 = UnifiedCrossbarConv2d(3, 64, 3, config, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.conv1_2 = UnifiedCrossbarConv2d(64, 64, 3, config, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)  # 32 -> 16
        
        # Conv Block 2: 64 -> 128 -> 128, then pool
        self.conv2_1 = UnifiedCrossbarConv2d(64, 128, 3, config, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.conv2_2 = UnifiedCrossbarConv2d(128, 128, 3, config, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)  # 16 -> 8
        
        # Conv Block 3: 128 -> 256 -> 256, then pool
        self.conv3_1 = UnifiedCrossbarConv2d(128, 256, 3, config, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.conv3_2 = UnifiedCrossbarConv2d(256, 256, 3, config, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2)  # 8 -> 4
        
        # FC layers: 256*4*4 = 4096 -> 512 -> 10
        self.flatten_size = 256 * 4 * 4
        self.fc1 = UnifiedCrossbarLinear(self.flatten_size, 512, config)
        self.bn_fc1 = nn.BatchNorm1d(512) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = UnifiedCrossbarLinear(512, 10, config)
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.relu(self.bn1_1(self.conv1_1(x)))
        x = self.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        
        # Block 2
        x = self.relu(self.bn2_1(self.conv2_1(x)))
        x = self.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        
        # Block 3
        x = self.relu(self.bn3_1(self.conv3_1(x)))
        x = self.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        
        # FC
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_all_mapping_stats(self) -> List[dict]:
        stats = []
        for module in self.modules():
            if isinstance(module, (UnifiedCrossbarLinear, UnifiedCrossbarConv2d)):
                stats.append(module.get_mapping_stats())
        return stats


class CIFAR10UnifiedMLP(UnifiedMLP):
    """CIFAR-10专用的统一MLP"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        hidden_sizes: List[int] = [512, 256, 128]
    ):
        super().__init__(
            input_size=3072,  # 32*32*3
            hidden_sizes=hidden_sizes,
            output_size=10,
            config=config
        )


class MNISTUnifiedMLPWithDeviceNeuron(UnifiedMLPWithDeviceNeuron):
    """MNIST专用的器件神经元MLP"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        hidden_sizes: List[int] = [256, 128],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            config=config,
            device_activation=device_activation,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


class CIFAR10UnifiedMLPWithDeviceNeuron(UnifiedMLPWithDeviceNeuron):
    """CIFAR-10专用的器件神经元MLP"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        hidden_sizes: List[int] = [512, 256, 128],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            input_size=3072,  # 32*32*3
            hidden_sizes=hidden_sizes,
            output_size=10,
            config=config,
            device_activation=device_activation,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


# ============= 带器件神经元的CNN和VGG模型 =============

class MNISTUnifiedCNNWithDeviceNeuron(UnifiedCNNWithDeviceNeuron):
    """MNIST专用的器件神经元CNN"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            config=config,
            device_activation=device_activation,
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


class FashionMNISTUnifiedCNNWithDeviceNeuron(UnifiedCNNWithDeviceNeuron):
    """Fashion-MNIST专用的器件神经元CNN"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            config=config,
            device_activation=device_activation,
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


class CIFAR10UnifiedCNNWithDeviceNeuron(UnifiedCNNWithDeviceNeuron):
    """CIFAR-10专用的器件神经元CNN"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        conv_channels: List[int] = [64, 128, 256],
        fc_sizes: List[int] = [512, 256],
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            config=config,
            device_activation=device_activation,
            input_channels=3,
            input_size=32,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


class MNISTUnifiedVGGWithDeviceNeuron(UnifiedVGGWithDeviceNeuron):
    """MNIST专用的器件神经元VGG (需要将MNIST调整为32x32)"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        use_batchnorm: bool = True,
        dropout: float = 0.5,
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        # 注意: MNIST需要resize到32x32
        super().__init__(
            config=config,
            device_activation=device_activation,
            input_channels=1,
            input_size=32,
            num_classes=10,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )


class CIFAR10UnifiedVGGWithDeviceNeuron(UnifiedVGGWithDeviceNeuron):
    """CIFAR-10专用的器件神经元VGG"""
    
    def __init__(
        self,
        config: CrossbarConfig,
        device_activation: UserDeviceActivation = None,
        use_batchnorm: bool = True,
        dropout: float = 0.5,
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        device_noise: float = 0.0
    ):
        super().__init__(
            config=config,
            device_activation=device_activation,
            input_channels=3,
            input_size=32,
            num_classes=10,
            use_batchnorm=use_batchnorm,
            dropout=dropout,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )

