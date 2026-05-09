"""
DeviceNeuron - 神经形态计算仿真框架
Neuromorphic Computing Simulation Framework

支持功能:
- 权重到交叉阵列电导的映射（真实器件电导态数据）
- MLP和CNN网络的Crossbar层实现
- HIL (Hardware-in-the-Loop) 训练
- DAC/ADC量化仿真
- 器件噪声建模
- 自定义器件神经元（使用真实器件I-V特性作为激活函数）

使用示例:
    from DeviceNeuron import CrossbarConfig, UnifiedMLP, HILTrainer
    
    # 1. 创建配置（定义一次，所有层共用）
    config = CrossbarConfig.from_csv(
        'device_data.csv',   # 电导态数据：脉冲数, 平均电导, 标准差
        dac_bits=8, adc_bits=8
    )
    
    # 2. 创建模型
    model = UnifiedMLP(784, [256, 128], 10, config)
    
    # 3. 训练
    trainer = HILTrainer(model, optimizer, criterion)
    trainer.train(train_loader, val_loader, epochs=10)
"""

# 量化模块
from .quantization import DAC, ADC, DynamicADC, StraightThroughEstimator

# 电导态管理
from .conductance_states import (
    ConductanceStates,
    ConductanceStateData,
    ConductanceStatesTorchLUT,
    WeightToConductanceMapper
)

# 统一Crossbar核心模块
from .crossbar_core import (
    CrossbarConfig,
    CrossbarMVMCore,
    UnifiedCrossbarLinear,
    UnifiedCrossbarConv2d,
    UnifiedCrossbarLinearWithActivation,
    UnifiedCrossbarConv2dWithActivation
)

# 统一模型
from .unified_models import (
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
    CIFAR10UnifiedVGGWithDeviceNeuron
)

# 自定义器件神经元（激活函数）
from .custom_neuron import (
    # 核心接口
    UserDeviceActivation,
    FunctionDeviceActivation,
    CustomDeviceNeuron,
    Integrator,
    # 预定义激活函数
    SigmoidDeviceActivation,
    ReLUDeviceActivation,
    TanhDeviceActivation,
    ThresholdDeviceActivation,
    PolynomialDeviceActivation,
    PiecewiseLinearDeviceActivation,
    # 预设神经元
    SigmoidDeviceNeuron,
    ReLUDeviceNeuron,
    TanhDeviceNeuron,
    ThresholdDeviceNeuron,
    # 向后兼容 (已弃用)
    DeviceTransferCurve,
    DeviceTransferCurveTorch,
    CustomDeviceNeuronFromFile,
)

# 用户自定义器件激活函数
from .my_device_activation import (
    MyDeviceActivation,
    MyDeviceActivationV2,
    create_my_device_neuron
)

# 用户自定义配置（整合交叉阵列 + 激活函数）
from .my_config import (
    MyConfig,
    MY_CONFIG,
    create_my_crossbar_config,
    create_my_device_activation,
    create_my_mlp_model,
    create_my_cnn_model,
    create_my_vgg_model,
    create_my_model,
    create_my_trainer,
    train_my_model
)

# 训练框架
from .hil_trainer import HILTrainer, HILInference

# 配置模块
from .config import (
    DeviceNeuronConfig,
    CONFIG,
    create_real_device_config,
    create_high_precision_config,
    create_fast_test_config,
    load_config_from_file,
    save_config_to_file
)

# 工具函数
from .utils import (
    load_device_data,
    extract_conductance_states,
    analyze_device_characteristics,
    calculate_weight_mapping_error,
    print_model_crossbar_info
)

# 日志模块
from .logger import TrainingLogger, create_logger

__version__ = '0.2.0'
__author__ = 'Neuromorphic Computing Lab'

__all__ = [
    # 量化模块
    'DAC',
    'ADC',
    'DynamicADC',
    'StraightThroughEstimator',
    # 电导态管理
    'ConductanceStates',
    'ConductanceStateData',
    'ConductanceStatesTorchLUT',
    'WeightToConductanceMapper',
    # Crossbar核心
    'CrossbarConfig',
    'CrossbarMVMCore',
    # 统一Crossbar层
    'UnifiedCrossbarLinear',
    'UnifiedCrossbarConv2d',
    'UnifiedCrossbarLinearWithActivation',
    'UnifiedCrossbarConv2dWithActivation',
    # 统一模型
    'UnifiedMLP',
    'UnifiedCNN',
    'UnifiedMLPWithDeviceNeuron',
    'UnifiedCNNWithDeviceNeuron',
    'UnifiedVGGWithDeviceNeuron',
    'MNISTUnifiedMLP',
    'MNISTUnifiedCNN',
    'FashionMNISTUnifiedCNN',
    'CIFAR10UnifiedMLP',
    'CIFAR10UnifiedCNN',
    'CIFAR10VGGStyleCNN',
    'MNISTUnifiedMLPWithDeviceNeuron',
    'CIFAR10UnifiedMLPWithDeviceNeuron',
    'MNISTUnifiedCNNWithDeviceNeuron',
    'FashionMNISTUnifiedCNNWithDeviceNeuron',
    'CIFAR10UnifiedCNNWithDeviceNeuron',
    'MNISTUnifiedVGGWithDeviceNeuron',
    'CIFAR10UnifiedVGGWithDeviceNeuron',
    # 自定义器件神经元 - 核心接口
    'UserDeviceActivation',
    'FunctionDeviceActivation',
    'CustomDeviceNeuron',
    'Integrator',
    # 预定义激活函数
    'SigmoidDeviceActivation',
    'ReLUDeviceActivation',
    'TanhDeviceActivation',
    'ThresholdDeviceActivation',
    'PolynomialDeviceActivation',
    'PiecewiseLinearDeviceActivation',
    # 预设神经元
    'SigmoidDeviceNeuron',
    'ReLUDeviceNeuron',
    'TanhDeviceNeuron',
    'ThresholdDeviceNeuron',
    # 向后兼容
    'DeviceTransferCurve',
    'DeviceTransferCurveTorch',
    'CustomDeviceNeuronFromFile',
    # 用户自定义激活函数
    'MyDeviceActivation',
    'MyDeviceActivationV2',
    'create_my_device_neuron',
    # 用户自定义配置
    'MyConfig',
    'MY_CONFIG',
    'create_my_crossbar_config',
    'create_my_device_activation',
    'create_my_mlp_model',
    'create_my_cnn_model',
    'create_my_vgg_model',
    'create_my_model',
    'create_my_trainer',
    'train_my_model',
    # 训练框架
    'HILTrainer',
    'HILInference',
    # 配置模块
    'DeviceNeuronConfig',
    'CONFIG',
    'create_real_device_config',
    'create_high_precision_config',
    'create_fast_test_config',
    'load_config_from_file',
    'save_config_to_file',
    # 工具函数
    'load_device_data',
    'extract_conductance_states',
    'analyze_device_characteristics',
    'calculate_weight_mapping_error',
    'print_model_crossbar_info',
    # 日志
    'TrainingLogger',
    'create_logger',
]
