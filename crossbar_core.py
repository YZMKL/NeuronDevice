"""
Crossbar核心模块 - 统一的MVM实现
Crossbar Core Module - Unified Matrix-Vector Multiplication Implementation

所有Crossbar层（MLP、CNN、自定义激活函数）都共用这个核心模块。
用户只需定义一次电导态数据，所有层都会使用相同的器件特性。

使用方法:
    # 1. 创建电导态配置
    from DeviceNeuron import ConductanceStates, CrossbarConfig
    
    states = ConductanceStates.from_csv('device_data.csv')
    config = CrossbarConfig(
        conductance_states=states,
        dac_bits=8,
        adc_bits=8
    )
    
    # 2. 创建各种层，共用同一个配置
    from DeviceNeuron import UnifiedCrossbarLinear, UnifiedCrossbarConv2d
    
    fc1 = UnifiedCrossbarLinear(784, 256, config)
    fc2 = UnifiedCrossbarLinear(256, 10, config)
    conv1 = UnifiedCrossbarConv2d(1, 32, 3, config)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Union
from dataclasses import dataclass
import math

from .conductance_states import ConductanceStates, WeightToConductanceMapper
from .quantization import DAC, DynamicADC


@dataclass
class CrossbarConfig:
    """
    Crossbar配置类
    统一管理所有Crossbar层的参数
    """
    # 电导态数据（必须）
    conductance_states: ConductanceStates
    
    # DAC参数
    dac_bits: int = 8
    dac_noise: float = 0.0
    v_max: float = 1.0
    
    # ADC参数
    adc_bits: int = 8
    adc_noise: float = 0.0
    
    def __post_init__(self):
        """验证配置"""
        if self.conductance_states is None:
            raise ValueError("必须提供 conductance_states")
    
    @classmethod
    def from_csv(
        cls,
        file_path: str,
        pulse_col: int = 0,
        mean_col: int = 1,
        std_col: int = 2,
        skip_rows: int = 1,
        **kwargs
    ) -> 'CrossbarConfig':
        """从CSV文件创建配置"""
        states = ConductanceStates.from_csv(
            file_path, pulse_col, mean_col, std_col, skip_rows
        )
        return cls(conductance_states=states, **kwargs)
    
    @classmethod
    def from_excel(
        cls,
        file_path: str,
        sheet_name: Union[str, int] = 0,
        pulse_col: int = 0,
        mean_col: int = 1,
        std_col: int = 2,
        skip_rows: int = 1,
        **kwargs
    ) -> 'CrossbarConfig':
        """从Excel文件创建配置"""
        states = ConductanceStates.from_excel(
            file_path, sheet_name, pulse_col, mean_col, std_col, skip_rows
        )
        return cls(conductance_states=states, **kwargs)
    
    @classmethod
    def create_default(
        cls,
        n_states: int = 64,
        g_min: float = 1e-9,
        g_max: float = 1e-5,
        relative_std: float = 0.05,
        **kwargs
    ) -> 'CrossbarConfig':
        """创建默认配置（用于测试）"""
        states = ConductanceStates.create_log(
            g_min=g_min,
            g_max=g_max,
            n_states=n_states,
            relative_std=relative_std
        )
        return cls(conductance_states=states, **kwargs)
    
    def get_info(self) -> dict:
        """获取配置信息"""
        return {
            'n_states': self.conductance_states.n_states,
            'g_range': (self.conductance_states.g_min, self.conductance_states.g_max),
            'dac_bits': self.dac_bits,
            'adc_bits': self.adc_bits,
            'dac_noise': self.dac_noise,
            'adc_noise': self.adc_noise,
        }


class CrossbarMVMCore(nn.Module):
    """
    Crossbar矩阵向量乘法核心模块
    
    实现统一的MVM计算流程:
    1. DAC: 数字输入 → 模拟电压
    2. 权重映射: W → (G⁺, G⁻)
    3. MVM计算: I = V · (G⁺ - G⁻)
    4. ADC: 模拟电流 → 数字输出
    
    所有Crossbar层都使用这个核心模块进行计算。
    """
    
    def __init__(self, config: CrossbarConfig):
        """
        Args:
            config: CrossbarConfig配置对象
        """
        super().__init__()
        
        self.config = config
        
        # DAC
        self.dac = DAC(
            n_bits=config.dac_bits,
            v_max=config.v_max,
            noise_std=config.dac_noise
        )
        
        # ADC
        self.adc = DynamicADC(
            n_bits=config.adc_bits,
            noise_std=config.adc_noise
        )
        
        # 权重到电导映射器
        self.weight_mapper = WeightToConductanceMapper(config.conductance_states)
    
    def apply_dac(self, x: torch.Tensor) -> torch.Tensor:
        """
        DAC: 数字输入 → 模拟电压
        
        Args:
            x: 数字输入 [0, 1]
            
        Returns:
            模拟电压
        """
        return self.dac(x)
    
    def apply_adc(self, current: torch.Tensor) -> torch.Tensor:
        """
        ADC: 模拟电流 → 数字输出
        
        Args:
            current: 模拟电流
            
        Returns:
            数字输出
        """
        return self.adc(current)
    
    def map_weight(
        self,
        weight: torch.Tensor,
        add_noise: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        权重映射: W → (G⁺, G⁻, scale)
        
        流程:
        1. Layer-wise归一化: Ŵ = W / max|W|
        2. 差分拆分: Ŵ⁺ = max(Ŵ, 0), Ŵ⁻ = max(-Ŵ, 0)
        3. 映射到最近电导态
        4. 高斯采样: G ~ N(G_k, σ_k²)
        
        Args:
            weight: 原始权重
            add_noise: 是否添加器件噪声
            
        Returns:
            g_pos: 正电导 G⁺
            g_neg: 负电导 G⁻
            scale: 缩放因子
        """
        return self.weight_mapper(weight, add_noise)
    
    def compute_mvm(
        self,
        v_in: torch.Tensor,
        g_diff: torch.Tensor,
        weight_float: torch.Tensor,
        scale: torch.Tensor
    ) -> torch.Tensor:
        """
        Crossbar MVM计算: I = V · (G⁺ - G⁻)
        
        使用STE进行梯度传递
        
        Args:
            v_in: 输入电压 [batch, in_features]
            g_diff: 差分电导 (G⁺ - G⁻) [out_features, in_features]
            weight_float: 原始浮点权重（用于反向传播）
            scale: 缩放因子
            
        Returns:
            输出电流
        """
        if self.training:
            return CrossbarMVMFunction.apply(v_in, g_diff, weight_float, scale)
        else:
            return F.linear(v_in, g_diff * scale, None)
    
    def forward_linear(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        完整的全连接层前向传播
        
        Args:
            x: 输入 [batch, in_features]
            weight: 权重 [out_features, in_features]
            bias: 偏置 [out_features]
            
        Returns:
            输出 [batch, out_features]
        """
        # Step 1: DAC
        v_in = self.apply_dac(x)
        
        # Step 2: 权重映射
        g_pos, g_neg, scale = self.map_weight(weight, add_noise=self.training)
        g_diff = g_pos - g_neg
        
        # Step 3: MVM
        i_out = self.compute_mvm(v_in, g_diff, weight, scale)
        
        # Step 4: ADC
        y = self.apply_adc(i_out)
        
        # Step 5: 加偏置
        if bias is not None:
            y = y + bias
        
        return y
    
    def forward_conv2d(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        bias: Optional[torch.Tensor],
        stride: Tuple[int, int],
        padding: Tuple[int, int]
    ) -> torch.Tensor:
        """
        完整的卷积层前向传播（im2col展开为MVM）
        
        Args:
            x: 输入 [batch, in_channels, H, W]
            weight: 权重 [out_channels, in_channels, kH, kW]
            bias: 偏置 [out_channels]
            stride: 步长
            padding: 填充
            
        Returns:
            输出 [batch, out_channels, H_out, W_out]
        """
        batch_size = x.shape[0]
        out_channels = weight.shape[0]
        kernel_size = (weight.shape[2], weight.shape[3])
        
        # Step 1: DAC
        v_in = self.apply_dac(x)
        
        # Step 2: im2col展开
        v_unfold = F.unfold(v_in, kernel_size=kernel_size, stride=stride, padding=padding)
        # [batch, in_channels*kH*kW, L]
        
        H_out = (x.shape[2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1
        W_out = (x.shape[3] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1
        
        # Step 3: 权重映射
        weight_flat = weight.view(out_channels, -1)  # [out_channels, in_channels*kH*kW]
        g_pos, g_neg, scale = self.map_weight(weight_flat, add_noise=self.training)
        g_diff = g_pos - g_neg
        
        # Step 4: MVM
        v_unfold_t = v_unfold.transpose(1, 2)  # [batch, L, features]
        
        if self.training:
            i_out = CrossbarConvMVMFunction.apply(v_unfold_t, g_diff, weight_flat, scale)
        else:
            i_out = torch.matmul(v_unfold_t, g_diff.t()) * scale
        
        # i_out: [batch, L, out_channels]
        i_out = i_out.transpose(1, 2)  # [batch, out_channels, L]
        
        # Step 5: ADC
        y = self.apply_adc(i_out)
        
        # Reshape
        y = y.view(batch_size, out_channels, H_out, W_out)
        
        # Step 6: 加偏置
        if bias is not None:
            y = y + bias.view(1, -1, 1, 1)
        
        return y


class CrossbarMVMFunction(torch.autograd.Function):
    """
    Crossbar MVM 的 STE 实现
    
    前向: y = V · (G⁺ - G⁻) · scale  (使用电导)
    反向: 假装 y = V · W             (使用数字权重，STE)
    
    关键: scale 因子需要在梯度中体现，否则梯度尺度不对
    """
    
    @staticmethod
    def forward(ctx, v_in, g_diff, weight_float, scale):
        ctx.save_for_backward(v_in, weight_float, scale)
        # 前向: I = V · (G⁺ - G⁻) · scale
        return F.linear(v_in, g_diff * scale, None)
    
    @staticmethod
    def backward(ctx, grad_output):
        v_in, weight_float, scale = ctx.saved_tensors
        grad_v_in = grad_weight = None
        
        # 反向传播使用原始权重 (STE)
        # 但需要考虑 scale 因子，确保梯度尺度正确
        if ctx.needs_input_grad[0]:
            # ∂L/∂V = ∂L/∂y · W^T
            grad_v_in = grad_output.mm(weight_float)
        
        if ctx.needs_input_grad[2]:
            # ∂L/∂W = V^T · ∂L/∂y
            grad_weight = grad_output.t().mm(v_in)
        
        return grad_v_in, None, grad_weight, None


class CrossbarConvMVMFunction(torch.autograd.Function):
    """
    卷积层 Crossbar MVM 的 STE 实现
    """
    
    @staticmethod
    def forward(ctx, v_unfold_t, g_diff, weight_flat, scale):
        ctx.save_for_backward(v_unfold_t, weight_flat, scale)
        return torch.matmul(v_unfold_t, g_diff.t()) * scale
    
    @staticmethod
    def backward(ctx, grad_output):
        v_unfold_t, weight_flat, scale = ctx.saved_tensors
        grad_v_unfold_t = grad_weight = None
        
        if ctx.needs_input_grad[0]:
            grad_v_unfold_t = torch.matmul(grad_output, weight_flat)
        
        if ctx.needs_input_grad[2]:
            grad_weight = torch.matmul(
                grad_output.transpose(1, 2), v_unfold_t
            ).sum(dim=0)
        
        return grad_v_unfold_t, None, grad_weight, None


# ============= 统一的Crossbar层 =============

class UnifiedCrossbarLinear(nn.Module):
    """
    统一的Crossbar全连接层
    使用共享的CrossbarMVMCore进行计算
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: CrossbarConfig,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # 共享的Crossbar核心
        self.crossbar_core = CrossbarMVMCore(config)
        
        # 权重和偏置
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.crossbar_core.forward_linear(x, self.weight, self.bias)
    
    def get_mapping_stats(self) -> dict:
        return self.crossbar_core.weight_mapper.compute_mapping_error(self.weight)


class UnifiedCrossbarConv2d(nn.Module):
    """
    统一的Crossbar卷积层
    使用共享的CrossbarMVMCore进行计算
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        config: CrossbarConfig,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        
        # 共享的Crossbar核心
        self.crossbar_core = CrossbarMVMCore(config)
        
        # 权重和偏置
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.crossbar_core.forward_conv2d(
            x, self.weight, self.bias, self.stride, self.padding
        )
    
    def get_mapping_stats(self) -> dict:
        weight_flat = self.weight.view(self.out_channels, -1)
        return self.crossbar_core.weight_mapper.compute_mapping_error(weight_flat)


class UnifiedCrossbarLinearWithActivation(nn.Module):
    """
    带自定义激活函数的统一Crossbar全连接层
    
    完整数据流:
    x → DAC → Crossbar MVM (I = V·(G⁺-G⁻)·scale) → ADC → 对称归一化[-1,1] → 器件激活 → 缩放输出
    
    关键点:
    1. 差分阵列: I_out = I_pos - I_neg = V·G⁺·scale - V·G⁻·scale (可正可负)
    2. ADC 将模拟电流转换为数字信号
    3. 对称归一化到 [-1,1]，保持零点在中心
    4. 激活函数: 输入 [-1,1]，内部扩展到 [-4,4]，取反，应用 V-I 特性
    5. 激活函数输出 [0,1]，缩放回合理范围
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        config: CrossbarConfig,
        activation_module: nn.Module = None,
        bias: bool = True
    ):
        """
        Args:
            in_features: 输入维度
            out_features: 输出维度
            config: Crossbar配置
            activation_module: 自定义激活模块（如CustomDeviceNeuron）
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        # Crossbar核心
        self.crossbar_core = CrossbarMVMCore(config)
        
        # 权重和偏置
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        # 激活函数
        self.activation = activation_module if activation_module else nn.ReLU()
        
        # 用于归一化的运行统计（类似 BatchNorm）
        self.register_buffer('running_i_min', torch.tensor(-1.0))
        self.register_buffer('running_i_max', torch.tensor(1.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.1
        
        # 输出缩放因子（可学习，用于恢复信号幅度）
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        # 激活函数输入偏移（固定值0.7，不可学习）
        # 偏移0.7，将 [-1, 1] 偏移到 [-0.3, 1.7]，使更多输入落在有效区域（避免死区）
        self.register_buffer('activation_input_offset', torch.tensor(0.7))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        数据流:
        1. DAC: 数字输入 → 模拟电压
        2. 权重映射: W → (G⁺, G⁻), 差分表示
        3. MVM: I_diff = V · (G⁺ - G⁻) · scale (差分电流，可正可负)
        4. ADC: 模拟电流 → 数字信号
        5. 归一化: 映射到 [0,1] 供激活函数
        6. 激活函数: 器件 V-I 特性
        7. 缩放 + 偏置: 恢复输出范围
        """
        # Step 1: DAC - 数字输入转模拟电压
        v_in = self.crossbar_core.apply_dac(x)
        
        # Step 2: 权重映射到差分电导
        g_pos, g_neg, scale = self.crossbar_core.map_weight(self.weight, add_noise=self.training)
        g_diff = g_pos - g_neg  # 差分电导，可正可负
        
        # Step 3: MVM - 计算差分电流
        # I_diff = V · (G⁺ - G⁻) · scale
        # 这是差分阵列的核心：两个阵列的电流相减
        i_diff = self.crossbar_core.compute_mvm(v_in, g_diff, self.weight, scale)
        
        # Step 4: ADC - 模拟电流转数字信号
        # 注意: 差分电流可以是正负，ADC 需要处理这种情况
        i_digital = self.crossbar_core.apply_adc(i_diff)
        
        # Step 5: 对称归一化到 [-1, 1] 供激活函数使用
        # 差分电流可正可负，对称归一化保持零点在中心
        i_normalized, i_scale = self._normalize_for_activation(i_digital)
        
        # Step 6: 通过自定义激活函数
        # 激活函数输入: [-1, 1], 输出: [0, 1]
        # 激活函数内部处理: 扩展到 [-4,4] → 取反 → 分段公式
        y_activated = self.activation(i_normalized)
        
        # Step 7: 缩放回原始范围 + 偏置
        # 将 [0,1] 映射回原始电流范围，并应用可学习缩放
        y = y_activated * i_scale * self.output_scale
        
        if self.bias is not None:
            y = y + self.bias
        
        return y
    
    def _normalize_for_activation(self, current: torch.Tensor):
        """
        将差分电流归一化到 [-1, 1] 供激活函数使用
        
        差分电流 I_diff ∈ [-I_max, I_max] → [-1, 1]
        使用对称归一化，保持零点在中心
        
        Returns:
            normalized: 归一化后的电流 [-1, 1]
            i_scale: 缩放因子（用于反归一化）
        """
        if self.training:
            # 计算当前 batch 的最大绝对值
            i_abs_max = current.abs().max().detach()
            i_scale = i_abs_max + 1e-10
            
            # 更新运行统计
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_i_max.copy_(i_scale)
                else:
                    self.running_i_max.mul_(1 - self.momentum).add_(i_scale * self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            # 推理时使用运行统计
            i_scale = self.running_i_max + 1e-10
        
        # 对称归一化: [-I_max, I_max] → [-1, 1]
        # 这样零点保持在中心
        normalized = current / i_scale
        normalized = torch.clamp(normalized, -1, 1)
        
        return normalized, i_scale
    
    def get_mapping_stats(self) -> dict:
        """获取权重映射统计"""
        return self.crossbar_core.weight_mapper.compute_mapping_error(self.weight)


class UnifiedCrossbarConv2dWithActivation(nn.Module):
    """
    带自定义激活函数的统一Crossbar卷积层
    
    完整数据流:
    x → DAC → Crossbar Conv MVM (I = V·(G⁺-G⁻)·scale) → ADC → 对称归一化[-1,1] → 器件激活 → 缩放输出
    
    与 UnifiedCrossbarLinearWithActivation 原理相同，但处理卷积操作
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        config: CrossbarConfig,
        stride: int = 1,
        padding: int = 0,
        activation_module: nn.Module = None,
        bias: bool = True
    ):
        """
        Args:
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            config: Crossbar配置
            stride: 步长
            padding: 填充
            activation_module: 自定义激活模块（如CustomDeviceNeuron）
            bias: 是否使用偏置
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        
        # Crossbar核心
        self.crossbar_core = CrossbarMVMCore(config)
        
        # 权重和偏置
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, self.kernel_size[0], self.kernel_size[1])
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        
        # 激活函数
        self.activation = activation_module if activation_module else nn.ReLU()
        
        # 用于归一化的运行统计
        self.register_buffer('running_i_max', torch.tensor(1.0))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))
        self.momentum = 0.1
        
        # 输出缩放因子
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        
        # 激活函数输入偏移（固定值0.7，不可学习）
        # 偏移0.7，将 [-1, 1] 偏移到 [-0.3, 1.7]，使更多输入落在有效区域（避免死区）
        self.register_buffer('activation_input_offset', torch.tensor(0.7))
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        数据流 (与Linear版相同，但使用im2col处理卷积):
        1. DAC: 数字输入 → 模拟电压
        2. im2col 展开
        3. 权重映射: W → (G⁺, G⁻)
        4. MVM: I_diff = V · (G⁺ - G⁻) · scale
        5. ADC: 模拟电流 → 数字信号
        6. 归一化到 [-1, 1]
        7. 激活函数
        8. 缩放 + 偏置
        """
        batch_size = x.shape[0]
        
        # Step 1: DAC
        v_in = self.crossbar_core.apply_dac(x)
        
        # Step 2: im2col展开
        v_unfold = F.unfold(v_in, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding)
        # [batch, in_channels*kH*kW, L]
        
        H_out = (x.shape[2] + 2 * self.padding[0] - self.kernel_size[0]) // self.stride[0] + 1
        W_out = (x.shape[3] + 2 * self.padding[1] - self.kernel_size[1]) // self.stride[1] + 1
        
        # Step 3: 权重映射
        weight_flat = self.weight.view(self.out_channels, -1)
        g_pos, g_neg, scale = self.crossbar_core.map_weight(weight_flat, add_noise=self.training)
        g_diff = g_pos - g_neg
        
        # Step 4: MVM
        v_unfold_t = v_unfold.transpose(1, 2)  # [batch, L, features]
        
        if self.training:
            i_out = CrossbarConvMVMFunction.apply(v_unfold_t, g_diff, weight_flat, scale)
        else:
            i_out = torch.matmul(v_unfold_t, g_diff.t()) * scale
        
        # i_out: [batch, L, out_channels]
        i_out = i_out.transpose(1, 2)  # [batch, out_channels, L]
        
        # Step 5: ADC
        i_digital = self.crossbar_core.apply_adc(i_out)
        
        # Step 6: 归一化到 [-1, 1]
        i_normalized, i_scale = self._normalize_for_activation(i_digital)
        
        # Step 7: 激活函数
        # 需要 reshape 为 2D 进行激活，再恢复形状
        shape_before = i_normalized.shape
        i_normalized_flat = i_normalized.reshape(-1)  # 展平
        y_activated_flat = self.activation(i_normalized_flat)
        y_activated = y_activated_flat.reshape(shape_before)
        
        # Step 8: 缩放 + reshape + 偏置
        y = y_activated * i_scale * self.output_scale
        y = y.view(batch_size, self.out_channels, H_out, W_out)
        
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        
        return y
    
    def _normalize_for_activation(self, current: torch.Tensor):
        """将差分电流归一化到 [-1, 1]"""
        if self.training:
            # 计算当前 batch 的最大绝对值
            i_abs_max = current.abs().max().detach()
            i_scale = i_abs_max + 1e-10
            
            # 更新运行统计
            with torch.no_grad():
                if self.num_batches_tracked == 0:
                    self.running_i_max.copy_(i_scale)
                else:
                    self.running_i_max.mul_(1 - self.momentum).add_(i_scale * self.momentum)
                self.num_batches_tracked.add_(1)
        else:
            i_scale = self.running_i_max + 1e-10
        
        normalized = current / i_scale
        normalized = torch.clamp(normalized, -1, 1)
        
        return normalized, i_scale
    
    def get_mapping_stats(self) -> dict:
        weight_flat = self.weight.view(self.out_channels, -1)
        return self.crossbar_core.weight_mapper.compute_mapping_error(weight_flat)

