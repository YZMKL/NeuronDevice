# DeviceNeuron - 神经形态计算仿真框架

基于忆阻器交叉阵列的神经网络仿真框架，支持真实器件电导态数据、统一Crossbar MVM实现、HIL训练、自定义器件激活函数。

## 目录

- [功能特性](#功能特性)
- [安装](#安装)
- [快速开始](#快速开始)
- [核心概念](#核心概念)
- [API参考](#api参考)
- [运行示例](#运行示例)
- [项目结构](#项目结构)

---

## 功能特性

- **统一Crossbar核心** - MLP、CNN、VGG、自定义激活函数共用同一个MVM实现
- **真实器件电导态** - 支持加载CSV/Excel器件数据（脉冲数、平均电导、标准差）
- **完整权重映射** - Layer-wise归一化 + 差分拆分 + 最近态映射 + 高斯噪声采样
- **DAC/ADC量化** - 支持2-32bit可配置位宽，STE梯度传递
- **HIL训练** - STE反向传播，更新数字影子权重，支持学习率调度
- **自定义器件激活函数** - 使用器件I-V曲线作为激活函数，支持预设和自定义
- **统一配置系统** - `DeviceNeuronConfig` 集中管理所有硬件/训练/模型参数
- **标准基准对比** - `benchmark/` 模块提供标准NN实现用于对比
- **训练日志记录** - `TrainingLogger` 自动记录训练过程、模型信息、映射误差
- **多数据集支持** - MNIST、Fashion-MNIST、CIFAR-10

---

## 安装

```bash
cd neuromorphic
pip install -r DeviceNeuron/requirements.txt
```

验证安装：

```bash
python -c "from DeviceNeuron import CrossbarConfig; print('安装成功!')"
```

---

## 快速开始

### 1. 使用默认配置训练

```python
from DeviceNeuron import CrossbarConfig, UnifiedMLP, HILTrainer
import torch
import torch.nn as nn

# 创建Crossbar配置（定义一次，所有层共用）
config = CrossbarConfig.create_default(n_states=64, relative_std=0.05)

# 创建模型
model = UnifiedMLP(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    config=config
)

# 训练
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
trainer = HILTrainer(model, optimizer, criterion, device='cuda')
trainer.train(train_loader, val_loader, epochs=10)
```

### 2. 使用真实器件数据

```python
from DeviceNeuron import CrossbarConfig

# 从CSV加载电导态数据
config = CrossbarConfig.from_csv(
    'expdata/set2.csv',
    pulse_col=0, mean_col=1, std_col=2,
    dac_bits=8, adc_bits=8
)
```

### 3. 使用自定义器件激活函数

```python
from DeviceNeuron import UnifiedMLPWithDeviceNeuron
from DeviceNeuron.my_device_activation import MyDeviceActivation

config = CrossbarConfig.create_default(n_states=64)
activation = MyDeviceActivation()

model = UnifiedMLPWithDeviceNeuron(
    input_size=784,
    hidden_sizes=[256, 128],
    output_size=10,
    config=config,
    device_activation=activation
)
```

### 4. 使用统一配置系统

```python
from DeviceNeuron.config import CONFIG, DeviceNeuronConfig
from DeviceNeuron.my_config import MY_CONFIG, create_my_model, train_my_model

# 使用默认配置
config = CONFIG.get_crossbar_config()

# 使用自定义配置
model = create_my_model('mlp', 'mnist', hidden_sizes=[512, 256])

# 完整训练流程
model, history = train_my_model(train_loader, test_loader)
```

---

## 核心概念

### 统一架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CrossbarConfig                           │
│  电导态数据 + DAC/ADC参数（定义一次，所有层共用）            │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    CrossbarMVMCore                          │
│  统一MVM计算: DAC → 权重映射 → I = V·(G⁺-G⁻) → ADC          │
└─────────────────────────────────────────────────────────────┘
                              │
          ┌───────────────────┼───────────────────┐
          ▼                   ▼                   ▼
   UnifiedCrossbar     UnifiedCrossbar     UnifiedCrossbar
   Linear              Conv2d              LinearWithActivation
          │                   │                   │
          ▼                   ▼                   ▼
     UnifiedMLP          UnifiedCNN      UnifiedMLPWithDeviceNeuron
     UnifiedCNNWithDeviceNeuron           UnifiedVGGWithDeviceNeuron
```

### 权重映射流程

W → 归一化 → Ŵ → 差分拆分 → (Ŵ⁺, Ŵ⁻) → 最近态映射 → (Gk⁺, Gk⁻) → 高斯采样 → G ~ N(Gk, σk²)

### 器件激活函数数据流

```
Crossbar输出电流 → 积分器(归一化到[-1,1]) → 用户激活函数(V→I) → 积分器(归一化到[0,1]) → 下一层
```

---

## API参考

### 配置类

| 类名 | 描述 |
| --- | --- |
| `CrossbarConfig` | Crossbar统一配置（电导态+DAC/ADC） |
| `DeviceNeuronConfig` | 统一配置类（硬件+训练+模型参数） |
| `MyConfig` | 用户自定义配置（整合交叉阵列+激活函数） |
| `ConductanceStates` | 器件电导态数据管理 |

```python
# 从CSV加载
config = CrossbarConfig.from_csv('data.csv', dac_bits=8, adc_bits=8)

# 从Excel加载
config = CrossbarConfig.from_excel('data.xlsx')

# 使用默认值
config = CrossbarConfig.create_default(n_states=64, relative_std=0.05)

# 使用统一配置
from DeviceNeuron.config import CONFIG, create_real_device_config
config = CONFIG.get_crossbar_config()
config = create_real_device_config().get_crossbar_config()
```

### 量化模块

| 类名 | 描述 |
| --- | --- |
| `DAC` | 数模转换器（电压量化） |
| `ADC` | 模数转换器（电流量化） |
| `DynamicADC` | 动态范围ADC |
| `StraightThroughEstimator` | STE梯度传递 |

### 层类

| 类名 | 描述 |
| --- | --- |
| `CrossbarMVMCore` | 统一MVM计算核心 |
| `UnifiedCrossbarLinear` | 统一全连接层 |
| `UnifiedCrossbarConv2d` | 统一卷积层 |
| `UnifiedCrossbarLinearWithActivation` | 带自定义激活的全连接层 |
| `UnifiedCrossbarConv2dWithActivation` | 带自定义激活的卷积层 |

### 模型类

| 类名 | 数据集 | 描述 |
| --- | --- | --- |
| `UnifiedMLP` | 通用 | 通用MLP |
| `UnifiedCNN` | 通用 | 通用CNN |
| `UnifiedMLPWithDeviceNeuron` | 通用 | 器件激活MLP |
| `UnifiedCNNWithDeviceNeuron` | 通用 | 器件激活CNN |
| `UnifiedVGGWithDeviceNeuron` | 通用 | 器件激活VGG |
| `MNISTUnifiedMLP` | MNIST | MNIST专用MLP |
| `MNISTUnifiedCNN` | MNIST | MNIST专用CNN |
| `FashionMNISTUnifiedCNN` | Fashion-MNIST | Fashion-MNIST CNN |
| `CIFAR10UnifiedMLP` | CIFAR-10 | CIFAR-10 MLP |
| `CIFAR10UnifiedCNN` | CIFAR-10 | CIFAR-10 CNN |
| `CIFAR10VGGStyleCNN` | CIFAR-10 | CIFAR-10 VGG风格CNN |
| `MNISTUnifiedMLPWithDeviceNeuron` | MNIST | MNIST 器件激活MLP |
| `MNISTUnifiedCNNWithDeviceNeuron` | MNIST | MNIST 器件激活CNN |
| `FashionMNISTUnifiedCNNWithDeviceNeuron` | Fashion-MNIST | Fashion-MNIST 器件激活CNN |
| `CIFAR10UnifiedMLPWithDeviceNeuron` | CIFAR-10 | CIFAR-10 器件激活MLP |
| `CIFAR10UnifiedCNNWithDeviceNeuron` | CIFAR-10 | CIFAR-10 器件激活CNN |
| `MNISTUnifiedVGGWithDeviceNeuron` | MNIST | MNIST 器件激活VGG |
| `CIFAR10UnifiedVGGWithDeviceNeuron` | CIFAR-10 | CIFAR-10 器件激活VGG |

### 器件激活函数

| 类名 | 描述 |
| --- | --- |
| `UserDeviceActivation` | 用户自定义激活函数基类（V→I） |
| `FunctionDeviceActivation` | 基于函数的激活 |
| `CustomDeviceNeuron` | 完整器件神经元（含积分器） |
| `Integrator` | 电流-电压积分器 |
| `SigmoidDeviceActivation` | Sigmoid-like器件激活 |
| `ReLUDeviceActivation` | ReLU-like器件激活 |
| `TanhDeviceActivation` | Tanh-like器件激活 |
| `ThresholdDeviceActivation` | 阈值器件激活 |
| `PolynomialDeviceActivation` | 多项式器件激活 |
| `PiecewiseLinearDeviceActivation` | 分段线性器件激活 |
| `SigmoidDeviceNeuron` | 预置Sigmoid神经元 |
| `ReLUDeviceNeuron` | 预置ReLU神经元 |
| `TanhDeviceNeuron` | 预置Tanh神经元 |
| `ThresholdDeviceNeuron` | 预置阈值神经元 |
| `MyDeviceActivation` | 用户自定义分段拟合激活 |
| `MyDeviceActivationV2` | 用户自定义激活V2 |

### 训练与日志

| 类名 | 描述 |
| --- | --- |
| `HILTrainer` | HIL训练器（含训练/验证循环） |
| `HILInference` | 推理器（含噪声模拟） |
| `TrainingLogger` | 训练日志记录器 |

### 工具函数

| 函数 | 描述 |
| --- | --- |
| `load_device_data` | 加载器件数据 |
| `extract_conductance_states` | 提取电导态 |
| `analyze_device_characteristics` | 分析器件特性 |
| `calculate_weight_mapping_error` | 计算权重映射误差 |
| `print_model_crossbar_info` | 打印模型Crossbar信息 |

### 用户自定义配置（my_config）

| 函数 | 描述 |
| --- | --- |
| `create_my_crossbar_config` | 创建自定义Crossbar配置 |
| `create_my_device_activation` | 创建自定义器件激活 |
| `create_my_mlp_model` | 创建自定义MLP |
| `create_my_cnn_model` | 创建自定义CNN |
| `create_my_vgg_model` | 创建自定义VGG |
| `create_my_model` | 统一模型创建接口 |
| `create_my_trainer` | 创建训练器 |
| `train_my_model` | 完整训练流程 |

### 基准对比（benchmark）

| 类名 | 描述 |
| --- | --- |
| `StandardMLP` | 标准MLP（与UnifiedMLP架构对齐） |
| `StandardCNN` | 标准CNN |
| `StandardVGG` | 标准VGG |
| `train_standard_model` | 标准模型训练函数 |
| `BenchmarkLogger` | 基准测试日志记录器 |

---

## 运行示例

```bash
cd neuromorphic

# 运行完整测试
python -m DeviceNeuron.examples.test_all

# 训练MLP（MNIST）
python -m DeviceNeuron.examples.train_unified --model mlp

# 训练CNN（MNIST）
python -m DeviceNeuron.examples.train_unified --model cnn

# 训练器件神经元MLP
python -m DeviceNeuron.examples.train_unified --model device_neuron

# 训练CIFAR-10 Crossbar模型
python -m DeviceNeuron.examples.train_cifar10_corssbar

# 训练Fashion-MNIST Crossbar模型
python -m DeviceNeuron.examples.train_fashion_mnist_crossbar

# 使用自定义配置训练
python -m DeviceNeuron.examples.train_my_model

# 多样化MLP训练
python -m DeviceNeuron.examples.mlp_diverse
```

---

## 项目结构

```
DeviceNeuron/
├── __init__.py                  # 包入口（v0.2.0）
├── README.md                    # 文档
├── requirements.txt             # 依赖
├── workflow.md                  # 工作流程说明
│
├── config.py                    # 统一配置（DeviceNeuronConfig）
├── my_config.py                 # 用户自定义配置（MyConfig）
├── conductance_states.py        # 电导态管理
├── crossbar_core.py             # 统一Crossbar核心（MVM实现）
├── unified_models.py            # 统一模型（MLP、CNN、VGG）
├── custom_neuron.py             # 自定义器件神经元（激活函数）
├── my_device_activation.py      # 用户自定义器件激活函数
├── quantization.py              # DAC/ADC量化
├── hil_trainer.py               # HIL训练框架
├── logger.py                    # 训练日志记录
├── utils.py                     # 工具函数
│
├── examples/                    # 示例脚本
│   ├── test_all.py              # 完整测试
│   ├── train_unified.py         # 统一训练脚本
│   ├── train_my_model.py        # 自定义模型训练
│   ├── train_fashion_mnist_crossbar.py  # Fashion-MNIST训练
│   ├── train_cifar10_corssbar.py        # CIFAR-10训练
│   └── mlp_diverse.py           # 多样化MLP训练
│
├── benchmark/                   # 标准NN基准对比
│   ├── models.py                # 标准模型（MLP/CNN/VGG）
│   ├── train.py                 # 标准训练流程
│   └── mlp_diverse.py           # 多样化MLP基准
│
├── analysis/                    # 分析与可视化
│   ├── features/                # 特征提取与分析
│   │   ├── cifar10_deviceneuron_features.py
│   │   ├── cifar10_deviceneuron_confusion.py
│   │   └── cifar10_preview.py
│   └── visualizations/          # 可视化输出
│       ├── crossbar_cnn/
│       ├── crossbar_device_neuron_cnn/
│       └── standard_cnn/
│
├── expdata/                     # 器件实验数据
│   ├── set2.csv
│   ├── reset2.csv
│   ├── device2.csv / device2.xlsx
│   ├── device300.csv
│   ├── transfer+curve.xlsx
│   ├── conductance+endurance.xlsx
│   ├── rectification.xlsx
│   └── dG_decreasing.txt / dG_increasing.txt
│
└── logs/                        # 训练日志
```

---

## 自定义器件激活函数

继承 `UserDeviceActivation` 实现自己的V→I特性：

```python
from DeviceNeuron import UserDeviceActivation, CustomDeviceNeuron

class MyActivation(UserDeviceActivation):
    def forward(self, voltage):
        """
        Args:
            voltage: 输入电压 ∈ [-1, 1]
        Returns:
            current: 输出电流 ∈ [0, 1]
        """
        # 用户定义的 V→I 曲线
        v_abs = voltage.abs()
        current = 1.0 / (1.0 + torch.exp(-10.0 * (voltage - 0.5)))
        return current

# 创建神经元
neuron = CustomDeviceNeuron(activation=MyActivation())
```

---

## 常见问题

### Q: 如何准备电导态数据？

CSV格式：

```csv
pulse,G_mean,G_std
0,1.00e-09,5.00e-11
1,2.15e-09,1.08e-10
...
```

- 第1列：脉冲数（用于标识电导态）
- 第2列：平均电导值（单位S）
- 第3列：标准差（单位S）

### Q: 如何自定义网络结构？

```python
from DeviceNeuron import CrossbarConfig, UnifiedCrossbarLinear, UnifiedCrossbarConv2d
import torch.nn as nn

config = CrossbarConfig.create_default(n_states=64)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = UnifiedCrossbarConv2d(1, 32, 3, config, padding=1)
        self.fc1 = UnifiedCrossbarLinear(32*14*14, 10, config)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc1(x)
```

### Q: 如何与标准NN对比？

```python
from DeviceNeuron.benchmark import MNISTStandardMLP, train_standard_model

# 创建与UnifiedMLP架构相同的标准MLP
model = MNISTStandardMLP(hidden_sizes=[256, 128])

# 训练并记录基准结果
history = train_standard_model(model, train_loader, test_loader, epochs=50)
```

---

## License

MIT License
