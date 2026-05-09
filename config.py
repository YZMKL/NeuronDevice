"""
DeviceNeuron 统一配置文件
=========================

所有配置参数集中在此文件中，方便统一管理和调整。

使用方法:
    from DeviceNeuron.config import CONFIG
    
    # 访问电导态数据路径
    data_path = CONFIG.CONDUCTANCE_DATA_PATH
    
    # 获取预配置的CrossbarConfig
    config = CONFIG.get_crossbar_config()
    
    # 获取训练参数
    lr = CONFIG.LEARNING_RATE
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ============================================================================
# 路径配置
# ============================================================================

# 获取当前模块目录
_MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
_EXPDATA_DIR = os.path.join(_MODULE_DIR, 'expdata')


@dataclass
class DeviceNeuronConfig:
    """
    DeviceNeuron 统一配置类
    
    所有硬件参数、训练参数、模型结构都在这里配置。
    """
    
    # ========================================================================
    # 电导态数据配置
    # ========================================================================
    
    # 电导态数据文件路径 (CSV 或 Excel)
    # 设置为 None 则使用默认生成的电导态
    CONDUCTANCE_DATA_PATH: Optional[str] = None
    
    # CSV/Excel 文件列配置
    CONDUCTANCE_PULSE_COL: int = 0      # 脉冲数列索引
    CONDUCTANCE_MEAN_COL: int = 1       # 平均电导列索引
    CONDUCTANCE_STD_COL: int = 2        # 电导标准差列索引
    CONDUCTANCE_SKIP_ROWS: int = 1      # 跳过的表头行数
    CONDUCTANCE_SHEET_NAME: int = 0     # Excel工作表索引
    
    # 默认电导态参数 (当 CONDUCTANCE_DATA_PATH 为 None 时使用)
    DEFAULT_N_STATES: int = 64          # 电导态数量
    DEFAULT_G_MIN: float = 1e-9         # 最小电导值 (S)
    DEFAULT_G_MAX: float = 1e-5         # 最大电导值 (S)
    DEFAULT_RELATIVE_STD: float = 0.05  # 相对标准差
    
    # ========================================================================
    # DAC/ADC 配置
    # ========================================================================
    
    DAC_BITS: int = 16                   # DAC 位宽
    ADC_BITS: int = 16                   # ADC 位宽
    DAC_NOISE_STD: float = 0.0          # DAC 噪声标准差
    ADC_NOISE_STD: float = 0.0          # ADC 噪声标准差
    
    # ========================================================================
    # 器件神经元配置 (自定义激活函数)
    # ========================================================================
    
    # 器件 I-V 传输曲线文件路径
    # 设置为 None 则使用默认的 sigmoid-like 曲线
    DEVICE_TRANSFER_CURVE_PATH: Optional[str] = None
    
    # 传输曲线数据列配置
    TRANSFER_VOLTAGE_COL: int = 0       # 电压列索引
    TRANSFER_CURRENT_COL: int = 1       # 电流列索引
    TRANSFER_SKIP_ROWS: int = 1         # 跳过的表头行数
    
    # 默认传输曲线类型 ('sigmoid', 'relu', 'tanh', 'custom')
    # 设置为 'custom' 时使用 my_device_activation.py 中的 MyDeviceActivation
    DEFAULT_TRANSFER_TYPE: str = 'sigmoid'
    
    # 是否使用自定义激活函数 (my_device_activation.py 中的 MyDeviceActivation)
    USE_MY_DEVICE_ACTIVATION: bool = False
    
    # 积分器参数
    INTEGRATOR_K_INPUT: float = 1.0     # 输入积分器增益 (I -> V)
    INTEGRATOR_K_OUTPUT: float = 1.0    # 输出积分器增益 (I -> V)
    
    # 器件噪声
    DEVICE_NOISE_STD: float = 0.01      # 器件输出噪声标准差
    
    # ========================================================================
    # 模型结构配置
    # ========================================================================
    
    # ========================================================================
    # MNIST 模型配置
    # ========================================================================
    
    MNIST_MLP_INPUT_SIZE: int = 784      # 输入维度 (28*28)
    MNIST_MLP_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [256, 128])
    MNIST_MLP_OUTPUT_SIZE: int = 10      # 输出类别数
    
    MNIST_CNN_INPUT_CHANNELS: int = 1    # 输入通道数 (灰度图)
    MNIST_CNN_INPUT_SIZE: int = 28       # 输入图像尺寸
    MNIST_CNN_CONV_CHANNELS: List[int] = field(default_factory=lambda: [32, 64])
    MNIST_CNN_FC_SIZES: List[int] = field(default_factory=lambda: [128])
    
    # ========================================================================
    # CIFAR-10 模型配置
    # ========================================================================
    
    CIFAR10_MLP_INPUT_SIZE: int = 3072   # 输入维度 (32*32*3)
    CIFAR10_MLP_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [512, 256, 128])
    CIFAR10_MLP_OUTPUT_SIZE: int = 10    # 输出类别数
    
    CIFAR10_CNN_INPUT_CHANNELS: int = 3  # 输入通道数 (RGB)
    CIFAR10_CNN_INPUT_SIZE: int = 32     # 输入图像尺寸
    CIFAR10_CNN_CONV_CHANNELS: List[int] = field(default_factory=lambda: [64, 128, 256])
    CIFAR10_CNN_FC_SIZES: List[int] = field(default_factory=lambda: [512, 256])
    
    # 向后兼容的别名 (默认使用 MNIST)
    MLP_INPUT_SIZE: int = 784            # 输入维度 (MNIST: 28*28)
    MLP_HIDDEN_SIZES: List[int] = field(default_factory=lambda: [256, 128])
    MLP_OUTPUT_SIZE: int = 10            # 输出类别数
    
    CNN_INPUT_CHANNELS: int = 1          # 输入通道数 (灰度图: 1)
    CNN_CONV_CHANNELS: List[int] = field(default_factory=lambda: [32, 64])
    CNN_FC_SIZES: List[int] = field(default_factory=lambda: [128])
    CNN_OUTPUT_SIZE: int = 10            # 输出类别数
    CNN_KERNEL_SIZE: int = 3             # 卷积核大小
    
    # ========================================================================
    # 训练配置
    # ========================================================================
    
    BATCH_SIZE: int = 64                # 批大小
    EPOCHS: int = 50                    # 训练轮数
    LEARNING_RATE: float = 0.0005       # 学习率 (自定义激活函数建议较低)
    WEIGHT_DECAY: float = 1e-4          # L2正则化 (防止VGG过拟合)
    
    # 学习率调度
    LR_STEP_SIZE: int = 15              # 学习率衰减步长 (更温和的衰减)
    LR_GAMMA: float = 0.1               # 学习率衰减系数
    
    
    # 保存配置
    SAVE_BEST_MODEL: bool = True        # 是否保存最佳模型
    SAVE_PATH_TEMPLATE: str = 'unified_{model}_best.pth'
    
    # ========================================================================
    # 硬件仿真配置
    # ========================================================================
    
    # 是否在训练时模拟硬件噪声
    ENABLE_TRAINING_NOISE: bool = True
    
    # 推理时的电导采样
    ENABLE_INFERENCE_NOISE: bool = True
    
    # ========================================================================
    # 方法
    # ========================================================================
    
    def get_crossbar_config(self):
        """
        获取预配置的 CrossbarConfig 对象
        
        Returns:
            CrossbarConfig: 配置好的 Crossbar 配置对象
        """
        from .crossbar_core import CrossbarConfig
        
        if self.CONDUCTANCE_DATA_PATH is not None:
            path = self.CONDUCTANCE_DATA_PATH
            if path.endswith('.xlsx') or path.endswith('.xls'):
                return CrossbarConfig.from_excel(
                    path,
                    sheet_name=self.CONDUCTANCE_SHEET_NAME,
                    pulse_col=self.CONDUCTANCE_PULSE_COL,
                    mean_col=self.CONDUCTANCE_MEAN_COL,
                    std_col=self.CONDUCTANCE_STD_COL,
                    skip_rows=self.CONDUCTANCE_SKIP_ROWS,
                    dac_bits=self.DAC_BITS,
                    adc_bits=self.ADC_BITS
                )
            else:
                return CrossbarConfig.from_csv(
                    path,
                    pulse_col=self.CONDUCTANCE_PULSE_COL,
                    mean_col=self.CONDUCTANCE_MEAN_COL,
                    std_col=self.CONDUCTANCE_STD_COL,
                    skip_rows=self.CONDUCTANCE_SKIP_ROWS,
                    dac_bits=self.DAC_BITS,
                    adc_bits=self.ADC_BITS
                )
        else:
            return CrossbarConfig.create_default(
                n_states=self.DEFAULT_N_STATES,
                g_min=self.DEFAULT_G_MIN,
                g_max=self.DEFAULT_G_MAX,
                relative_std=self.DEFAULT_RELATIVE_STD,
                dac_bits=self.DAC_BITS,
                adc_bits=self.ADC_BITS
            )
    
    def get_device_activation(self):
        """
        获取器件激活函数
        
        Returns:
            UserDeviceActivation: 用户定义的 V→I 激活函数
            
        注意:
            - 设置 USE_MY_DEVICE_ACTIVATION=True 使用自定义的 MyDeviceActivation
            - 或设置 DEFAULT_TRANSFER_TYPE='custom' 也会使用自定义激活函数
            - 预定义激活函数: 'sigmoid', 'relu', 'tanh'
        """
        # 检查是否使用自定义激活函数
        if self.USE_MY_DEVICE_ACTIVATION or self.DEFAULT_TRANSFER_TYPE == 'custom':
            from .my_device_activation import MyDeviceActivation
            return MyDeviceActivation()
        
        # 使用预定义激活函数
        from .custom_neuron import (
            SigmoidDeviceActivation,
            ReLUDeviceActivation,
            TanhDeviceActivation
        )
        
        activation_map = {
            'sigmoid': SigmoidDeviceActivation(k=10.0, v0=0.5),
            'relu': ReLUDeviceActivation(k=2.0, vth=0.2),
            'tanh': TanhDeviceActivation(k=6.0, v0=0.5),
        }
        
        return activation_map.get(self.DEFAULT_TRANSFER_TYPE, SigmoidDeviceActivation())
    
    # 向后兼容
    def get_transfer_curve(self):
        """[已弃用] 请使用 get_device_activation()"""
        import warnings
        warnings.warn(
            "get_transfer_curve() 已弃用，请使用 get_device_activation()",
            DeprecationWarning
        )
        return self.get_device_activation()
    
    def get_device_data_path(self, filename: str) -> str:
        """获取 expdata 目录下的文件完整路径"""
        return os.path.join(_EXPDATA_DIR, filename)
    
    def print_config(self):
        """打印当前配置"""
        print('\n' + '='*70)
        print('DeviceNeuron 配置')
        print('='*70)
        
        print('\n【电导态数据】')
        if self.CONDUCTANCE_DATA_PATH:
            print(f'  数据文件: {self.CONDUCTANCE_DATA_PATH}')
        else:
            print(f'  使用默认配置: {self.DEFAULT_N_STATES} states, '
                  f'[{self.DEFAULT_G_MIN:.0e}, {self.DEFAULT_G_MAX:.0e}] S')
        
        print('\n【DAC/ADC】')
        print(f'  DAC: {self.DAC_BITS} bits, 噪声 σ={self.DAC_NOISE_STD}')
        print(f'  ADC: {self.ADC_BITS} bits, 噪声 σ={self.ADC_NOISE_STD}')
        
        print('\n【器件激活函数】')
        if self.USE_MY_DEVICE_ACTIVATION or self.DEFAULT_TRANSFER_TYPE == 'custom':
            print(f'  类型: MyDeviceActivation (my_device_activation.py)')
        else:
            print(f'  类型: {self.DEFAULT_TRANSFER_TYPE}')
        print(f'  积分器增益: K_in={self.INTEGRATOR_K_INPUT}, K_out={self.INTEGRATOR_K_OUTPUT}')
        print(f'  器件噪声: σ={self.DEVICE_NOISE_STD}')
        
        print('\n【模型结构 - MNIST】')
        print(f'  MLP: {self.MNIST_MLP_INPUT_SIZE} → {self.MNIST_MLP_HIDDEN_SIZES} → {self.MNIST_MLP_OUTPUT_SIZE}')
        print(f'  CNN: Ch{self.MNIST_CNN_CONV_CHANNELS} → FC{self.MNIST_CNN_FC_SIZES} → 10')
        
        print('\n【模型结构 - CIFAR-10】')
        print(f'  MLP: {self.CIFAR10_MLP_INPUT_SIZE} → {self.CIFAR10_MLP_HIDDEN_SIZES} → {self.CIFAR10_MLP_OUTPUT_SIZE}')
        print(f'  CNN: Ch{self.CIFAR10_CNN_CONV_CHANNELS} → FC{self.CIFAR10_CNN_FC_SIZES} → 10')
        
        print('\n【训练参数】')
        print(f'  Batch: {self.BATCH_SIZE}, Epochs: {self.EPOCHS}')
        print(f'  LR: {self.LEARNING_RATE}, Decay: {self.LR_GAMMA}@{self.LR_STEP_SIZE}')
        
        print('='*70 + '\n')


# ============================================================================
# 预设配置
# ============================================================================

# 默认配置实例
CONFIG = DeviceNeuronConfig()


# 使用真实器件数据的配置示例
def create_real_device_config() -> DeviceNeuronConfig:
    """
    创建使用真实器件数据的配置
    
    使用 expdata/device2.csv 中的实测电导态数据
    使用 my_device_activation.py 中的自定义激活函数
    
    device2.csv 格式:
        列0: average (平均电导)
        列1: STDEV.P (标准差)
        列2: proportion (占比，不使用)
    """
    return DeviceNeuronConfig(
        # 使用 device2.csv 中的电导态数据
        CONDUCTANCE_DATA_PATH=os.path.join(_EXPDATA_DIR, 'device300.csv'),
        CONDUCTANCE_MEAN_COL=0,   # 平均电导在第0列
        CONDUCTANCE_STD_COL=1,    # 标准差在第1列
        CONDUCTANCE_SKIP_ROWS=1,  # 跳过表头
        # 使用自定义激活函数 (my_device_activation.py 中的 MyDeviceActivation)
        USE_MY_DEVICE_ACTIVATION=True
    )


# 高精度配置
def create_high_precision_config() -> DeviceNeuronConfig:
    """创建高精度配置 (更多电导态，更高DAC/ADC位宽)"""
    return DeviceNeuronConfig(
        DEFAULT_N_STATES=128,
        DAC_BITS=10,
        ADC_BITS=10,
        DEFAULT_RELATIVE_STD=0.02,
        DEVICE_NOISE_STD=0.005
    )


# 快速测试配置
def create_fast_test_config() -> DeviceNeuronConfig:
    """创建快速测试配置 (较少参数，快速验证)"""
    return DeviceNeuronConfig(
        DEFAULT_N_STATES=32,
        DAC_BITS=6,
        ADC_BITS=6,
        MLP_HIDDEN_SIZES=[64, 32],
        CNN_CONV_CHANNELS=[16, 32],
        CNN_FC_SIZES=[64],
        BATCH_SIZE=128,
        EPOCHS=3,
        LEARNING_RATE=0.003
    )


# ============================================================================
# 便捷函数
# ============================================================================

def load_config_from_file(filepath: str) -> DeviceNeuronConfig:
    """
    从 JSON 文件加载配置
    
    Args:
        filepath: JSON 配置文件路径
        
    Returns:
        DeviceNeuronConfig: 配置对象
    """
    import json
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return DeviceNeuronConfig(**data)


def save_config_to_file(config: DeviceNeuronConfig, filepath: str):
    """
    保存配置到 JSON 文件
    
    Args:
        config: 配置对象
        filepath: 保存路径
    """
    import json
    from dataclasses import asdict
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(asdict(config), f, indent=2, ensure_ascii=False)

