"""
标准神经网络模型 - 用于与 Crossbar 仿真结果对比
Standard Neural Network Models for Benchmark Comparison

这些模型的架构与 Crossbar 版本完全相同，但使用标准的 PyTorch 层，
没有电导映射、量化或器件噪声。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class StandardMLP(nn.Module):
    """
    标准多层感知机
    
    与 UnifiedMLP 对应，但使用标准 nn.Linear 层
    """
    
    def __init__(
        self,
        input_size: int,
        hidden_sizes: List[int],
        output_size: int,
        activation: str = 'relu',
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        
        # 构建层
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if activation == 'relu':
                layers.append(nn.ReLU())
            elif activation == 'sigmoid':
                layers.append(nn.Sigmoid())
            elif activation == 'tanh':
                layers.append(nn.Tanh())
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # 输出层
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 展平输入
        x = x.view(x.size(0), -1)
        return self.network(x)


class StandardCNN(nn.Module):
    """
    标准卷积神经网络
    
    与 UnifiedCNN 对应，但使用标准 PyTorch 层
    """
    
    def __init__(
        self,
        input_channels: int,
        input_size: int,
        num_classes: int,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128],
        kernel_size: int = 3,
        pool_size: int = 2,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.input_channels = input_channels
        self.input_size = input_size
        
        # 构建卷积层
        conv_layers = []
        in_channels = input_channels
        current_size = input_size
        
        for out_channels in conv_channels:
            conv_layers.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(pool_size)
            ])
            in_channels = out_channels
            current_size = current_size // pool_size
        
        self.conv = nn.Sequential(*conv_layers)
        
        # 计算展平后的大小
        self.flatten_size = conv_channels[-1] * current_size * current_size
        
        # 构建全连接层
        fc_layers = []
        prev_size = self.flatten_size
        
        for fc_size in fc_sizes:
            fc_layers.append(nn.Linear(prev_size, fc_size))
            fc_layers.append(nn.ReLU())
            if dropout > 0:
                fc_layers.append(nn.Dropout(dropout))
            prev_size = fc_size
        
        fc_layers.append(nn.Linear(prev_size, num_classes))
        
        self.fc = nn.Sequential(*fc_layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x


class StandardVGG(nn.Module):
    """
    标准 VGG 风格网络
    
    与 CIFAR10VGGStyleCNN 对应
    """
    
    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        use_batchnorm: bool = True,
        dropout: float = 0.5
    ):
        super().__init__()
        
        # Conv Block 1: 64 -> 64 -> Pool
        self.conv1_1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.conv1_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64) if use_batchnorm else nn.Identity()
        self.pool1 = nn.MaxPool2d(2)
        
        # Conv Block 2: 64 -> 128 -> 128 -> Pool
        self.conv2_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.conv2_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128) if use_batchnorm else nn.Identity()
        self.pool2 = nn.MaxPool2d(2)
        
        # Conv Block 3: 128 -> 256 -> 256 -> Pool
        self.conv3_1 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.conv3_2 = nn.Conv2d(256, 256, 3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256) if use_batchnorm else nn.Identity()
        self.pool3 = nn.MaxPool2d(2)
        
        # FC layers
        self.flatten_size = 256 * 4 * 4  # 32 -> 16 -> 8 -> 4
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.bn_fc1 = nn.BatchNorm1d(512) if use_batchnorm else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(512, num_classes)
        
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


# ============================================================================
# 预定义模型 - MNIST
# ============================================================================

class MNISTStandardMLP(StandardMLP):
    """MNIST 专用标准 MLP"""
    
    def __init__(self, hidden_sizes: List[int] = [256, 128], dropout: float = 0.0):
        super().__init__(
            input_size=784,
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation='relu',
            dropout=dropout
        )


class MNISTStandardCNN(StandardCNN):
    """MNIST 专用标准 CNN"""
    
    def __init__(
        self,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128],
        dropout: float = 0.0
    ):
        super().__init__(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            dropout=dropout
        )


class FashionMNISTStandardCNN(StandardCNN):
    """Fashion-MNIST 专用标准 CNN"""
    
    def __init__(
        self,
        conv_channels: List[int] = [32, 64],
        fc_sizes: List[int] = [128],
        dropout: float = 0.0
    ):
        super().__init__(
            input_channels=1,
            input_size=28,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            dropout=dropout
        )


# ============================================================================
# 预定义模型 - CIFAR-10
# ============================================================================

class CIFAR10StandardMLP(StandardMLP):
    """CIFAR-10 专用标准 MLP"""
    
    def __init__(self, hidden_sizes: List[int] = [512, 256, 128], dropout: float = 0.0):
        super().__init__(
            input_size=3072,  # 32*32*3
            hidden_sizes=hidden_sizes,
            output_size=10,
            activation='relu',
            dropout=dropout
        )


class CIFAR10StandardCNN(StandardCNN):
    """CIFAR-10 专用标准 CNN"""
    
    def __init__(
        self,
        conv_channels: List[int] = [64, 128, 256],
        fc_sizes: List[int] = [512, 256],
        dropout: float = 0.0
    ):
        super().__init__(
            input_channels=3,
            input_size=32,
            num_classes=10,
            conv_channels=conv_channels,
            fc_sizes=fc_sizes,
            dropout=dropout
        )


class CIFAR10StandardVGG(StandardVGG):
    """CIFAR-10 专用标准 VGG"""
    
    def __init__(self, use_batchnorm: bool = True, dropout: float = 0.5):
        super().__init__(
            input_channels=3,
            num_classes=10,
            use_batchnorm=use_batchnorm,
            dropout=dropout
        )

