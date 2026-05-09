# DeviceNeuron Benchmark 神经网络结构文档

本文档整理了 `DeviceNeuron/benchmark/` 目录下所有神经网络的具体结构，用于与 Crossbar 版本进行性能对比。

## 概述

所有模型都使用标准的 PyTorch 层实现，没有电导映射、量化噪声或器件仿真，用于提供性能上限基准。

## MLP 模型

### DiverseMLP (mlp_diverse.py)

用于多数据集训练的通用 MLP 结构：

- **Iris 数据集**:

  - 输入: 4 维特征
  - 隐藏层: [32, 16]
  - 输出: 3 类
  - 结构: `4 → Linear(32) → ReLU → Linear(16) → ReLU → Linear(3)`
- **Fashion-MNIST 数据集**:

  - 输入: 784 维 (28×28 灰度图像展平)
  - 隐藏层: [512, 256, 128]
  - 输出: 10 类
  - 结构: `784 → Linear(512) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Linear(10)`
- **UCI-HAR 数据集**:

  - 输入: 561 维特征
  - 隐藏层: [256, 128, 64]
  - 输出: 6 类 (人体活动识别)
  - 结构: `561 → Linear(256) → ReLU → Linear(128) → ReLU → Linear(64) → ReLU → Linear(6)`
- **ISOLET 数据集**:

  - 输入: 617 维特征
  - 隐藏层: [512, 256, 128]
  - 输出: 26 类 (英文字母识别)
  - 结构: `617 → Linear(512) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → Linear(26)`

### MNISTStandardMLP

专门为 MNIST 数据集设计的标准 MLP：

- **输入**: 784 维 (28×28 灰度图像展平)
- **隐藏层**: [256, 128] (默认，可配置)
- **输出**: 10 类 (数字 0-9)
- **激活函数**: ReLU
- **Dropout**: 可选 (默认 0.0)
- **结构**: `784 → Linear(256) → ReLU → [Dropout] → Linear(128) → ReLU → [Dropout] → Linear(10)`CNN 模型

## CNN模型

### MNISTStandardCNN

专门为 MNIST 数据集设计的标准 CNN：

- **输入**: 1×28×28 (灰度图像)
- **卷积层**:
  - Conv1: `1 → 32` (3×3, padding=1) → ReLU → MaxPool2d(2) → 32×14×14
  - Conv2: `32 → 64` (3×3, padding=1) → ReLU → MaxPool2d(2) → 64×7×7
- **全连接层**: [128] (默认，可配置)
- **输出**: 10 类
- **Dropout**: 可选 (默认 0.0)
- **结构**:
  ```
  Input(1×28×28)
  → Conv2d(32, 3×3) → ReLU → MaxPool2d(2) → (32×14×14)
  → Conv2d(64, 3×3) → ReLU → MaxPool2d(2) → (64×7×7)
  → Flatten → Linear(128) → ReLU → [Dropout] → Linear(10)
  ```

### FashionMNISTStandardCNN

专门为 Fashion-MNIST 数据集设计的标准 CNN：

- **输入**: 1×28×28 (灰度图像)
- **卷积层**:

  - Conv1: `1 → 32` (3×3, padding=1) → ReLU → MaxPool2d(2) → 32×14×14
  - Conv2: `32 → 64` (3×3, padding=1) → ReLU → MaxPool2d(2) → 64×7×7
- **全连接层**: [128] (默认，可配置)
- **输出**: 10 类 (服装类别)
- **Dropout**: 可选 (默认 0.0)
- **结构**: 与 MNISTStandardCNN 相同

## VGG 模型

### CIFAR10StandardVGG

专门为 CIFAR-10 数据集设计的 VGG 风格网络：

- **输入**: 3×32×32 (RGB 图像)
- **卷积块 1** (Block 1):
  - Conv1_1: `3 → 64` (3×3, padding=1) → [BatchNorm] → ReLU
  - Conv1_2: `64 → 64` (3×3, padding=1) → [BatchNorm] → ReLU
  - MaxPool2d(2) → 64×16×16
- **卷积块 2** (Block 2):
  - Conv2_1: `64 → 128` (3×3, padding=1) → [BatchNorm] → ReLU
  - Conv2_2: `128 → 128` (3×3, padding=1) → [BatchNorm] → ReLU
  - MaxPool2d(2) → 128×8×8
- **卷积块 3** (Block 3):
  - Conv3_1: `128 → 256` (3×3, padding=1) → [BatchNorm] → ReLU
  - Conv3_2: `256 → 256` (3×3, padding=1) → [BatchNorm] → ReLU
  - MaxPool2d(2) → 256×4×4
- **全连接层**:
  - FC1: `4096 → 512` → [BatchNorm] → ReLU → Dropout(0.5)
  - FC2: `512 → 10`
- **BatchNorm**: 默认启用
- **Dropout**: 0.5 (FC1 后)
- **结构**:
  ```
  Input(3×32×32)
  Block 1:
  → Conv2d(64, 3×3) → BatchNorm2d → ReLU
  → Conv2d(64, 3×3) → BatchNorm2d → ReLU
  → MaxPool2d(2) → (64×16×16)
  Block 2:
  → Conv2d(128, 3×3) → BatchNorm2d → ReLU
  → Conv2d(128, 3×3) → BatchNorm2d → ReLU
  → MaxPool2d(2) → (128×8×8)
  Block 3:
  → Conv2d(256, 3×3) → BatchNorm2d → ReLU
  → Conv2d(256, 3×3) → BatchNorm2d → ReLU
  → MaxPool2d(2) → (256×4×4)
  → Flatten → Linear(512) → BatchNorm1d → ReLU → Dropout(0.5) → Linear(10)
  ```

## 模型参数统计

| 模型                       | 参数数量 | 备注     |
| -------------------------- | -------- | -------- |
| DiverseMLP (Iris)          | ~2.7K    | 小型网络 |
| DiverseMLP (Fashion-MNIST) | ~1.2M    | 中等规模 |
| DiverseMLP (UCI-HAR)       | ~0.2M    | 中等规模 |
| MNISTStandardMLP           | ~0.3M    | 标准 MLP |
| MNISTStandardCNN           | ~0.1M    | 轻量 CNN |
| FashionMNISTStandardCNN    | ~0.1M    | 轻量 CNN |
| CIFAR10StandardVGG         | ~15M     | 大型 CNN |

## 训练配置

所有模型使用以下默认训练参数：

- **优化器**: Adam
- **学习率**: 0.001 (MLP), 0.0005 (CNN/VGG)
- **批大小**: 64
- **Epochs**: 10-20 (视数据集而定)
- **权重衰减**: 1e-4
- **学习率调度**: StepLR (step_size=5, gamma=0.5)

## 对比说明

这些标准模型提供性能上限，用于与 Crossbar 版本进行对比。Crossbar 版本会受到以下限制：

- 电导态量化 (有限精度)
- DAC/ADC 量化噪声
- 器件噪声
- 权重映射误差

预期 Crossbar 版本准确率会略低于这些基准。
