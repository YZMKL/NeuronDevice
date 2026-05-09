"""
自定义器件神经元模块
Custom Device Neuron Module

简化接口：用户只需要提供一个 V→I 拟合函数

数据流程：
    Crossbar输出电流 
    → 积分器(归一化到-1 to 1电压) 
    → 用户自定义激活函数(-1 to 1电压 → 0-1电流)
    → 积分器(归一化到0-1电压) 
    → 下一层Crossbar输入

用户接口：
    1. 继承 UserDeviceActivation 基类
    2. 实现 forward(voltage) -> current 方法
    3. 输入是 -1 to 1 的电压，输出是 0-1 的电流
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Callable, Optional, Tuple, Union


# =============================================================================
# 积分器
# =============================================================================

class Integrator(nn.Module):
    """
    电流-电压积分器
    
    将电流归一化转换为电压：V = normalize(I * k_int)
    
    在实际硬件中对应: V = (1/C) * ∫I dt
    在数字仿真中简化为比例因子 + 归一化
    """
    
    def __init__(
        self,
        k_int: float = 1.0,
        v_min: float = 0.0,
        v_max: float = 1.0,
        noise_std: float = 0.0,
        learnable: bool = False
    ):
        """
        Args:
            k_int: 积分常数 (比例因子)
            v_min: 输出电压最小值
            v_max: 输出电压最大值
            noise_std: 噪声标准差
            learnable: k_int是否可学习
        """
        super().__init__()
        
        if learnable:
            self.k_int = nn.Parameter(torch.tensor(k_int))
        else:
            self.register_buffer('k_int', torch.tensor(k_int))
        
        self.v_min = v_min
        self.v_max = v_max
        self.noise_std = noise_std
    
    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        Args:
            current: 输入电流 (归一化)
            
        Returns:
            输出电压 (归一化到 [v_min, v_max])
        """
        voltage = current * self.k_int
        
        # 裁剪到有效范围
        voltage = torch.clamp(voltage, self.v_min, self.v_max)
        
        # 添加噪声
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(voltage) * self.noise_std
            voltage = voltage + noise
            voltage = torch.clamp(voltage, self.v_min, self.v_max)
        
        return voltage


# =============================================================================
# 用户自定义激活函数接口
# =============================================================================

class UserDeviceActivation(nn.Module):
    """
    用户自定义器件激活函数基类
    
    用户需要继承这个类，实现 forward 方法
    
    输入: 归一化电压 (-1 to 1)
    输出: 归一化电流 (0-1)
    
    示例:
        class MyDeviceActivation(UserDeviceActivation):
            def forward(self, voltage):
                # 用户的 V-I 拟合函数
                # voltage: -1 to 1 之间的电压
                # 返回: 0-1 之间的电流
                current = 1 / (1 + torch.exp(-10 * (voltage - 0.5)))
                return current
    """
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        """
        用户需要实现的 V→I 转换函数
        
        Args:
            voltage: 归一化电压 (-1 to 1)
            
        Returns:
            归一化电流 (0-1)
        """
        raise NotImplementedError("请实现 forward 方法")


class FunctionDeviceActivation(UserDeviceActivation):
    """
    从用户提供的函数创建激活函数
    
    用于用户不想创建类的情况
    
    示例:
        def my_vi_function(voltage):
            return torch.sigmoid(10 * (voltage - 0.5))
        
        activation = FunctionDeviceActivation(my_vi_function)
    """
    
    def __init__(self, vi_function: Callable[[torch.Tensor], torch.Tensor]):
        """
        Args:
            vi_function: V→I 函数，接收 0-1 电压，返回 0-1 电流
        """
        super().__init__()
        self.vi_function = vi_function
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        return self.vi_function(voltage)


# =============================================================================
# 预定义的常用激活函数
# =============================================================================

class SigmoidDeviceActivation(UserDeviceActivation):
    """
    Sigmoid 器件激活函数
    
    I = 1 / (1 + exp(-k * (V - V0)))
    """
    
    def __init__(self, k: float = 10.0, v0: float = 0.5):
        """
        Args:
            k: 斜率参数
            v0: 阈值电压 (中心点)
        """
        super().__init__()
        self.k = k
        self.v0 = v0
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        current = torch.sigmoid(self.k * (voltage - self.v0))
        return current


class ReLUDeviceActivation(UserDeviceActivation):
    """
    ReLU 器件激活函数
    
    I = max(0, k * (V - Vth))
    """
    
    def __init__(self, k: float = 2.0, vth: float = 0.2):
        """
        Args:
            k: 线性增益
            vth: 阈值电压
        """
        super().__init__()
        self.k = k
        self.vth = vth
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        current = F.relu(self.k * (voltage - self.vth))
        # 归一化到 0-1
        current = torch.clamp(current, 0, 1)
        return current


class TanhDeviceActivation(UserDeviceActivation):
    """
    Tanh 器件激活函数
    
    I = 0.5 * (1 + tanh(k * (V - V0)))
    """
    
    def __init__(self, k: float = 6.0, v0: float = 0.5):
        """
        Args:
            k: 斜率参数
            v0: 中心点
        """
        super().__init__()
        self.k = k
        self.v0 = v0
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        # 映射到 0-1
        current = 0.5 * (1 + torch.tanh(self.k * (voltage - self.v0)))
        return current


class ThresholdDeviceActivation(UserDeviceActivation):
    """
    阈值开关器件激活函数
    
    I = I_on if V > Vth else I_off
    (使用 sigmoid 近似以保持可微分)
    """
    
    def __init__(self, vth: float = 0.5, sharpness: float = 50.0, i_off: float = 0.01, i_on: float = 1.0):
        """
        Args:
            vth: 阈值电压
            sharpness: 边缘锐度 (越大越接近阶跃)
            i_off: 关态电流
            i_on: 开态电流
        """
        super().__init__()
        self.vth = vth
        self.sharpness = sharpness
        self.i_off = i_off
        self.i_on = i_on
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        # 使用 sigmoid 近似阶跃函数
        switch = torch.sigmoid(self.sharpness * (voltage - self.vth))
        current = self.i_off + (self.i_on - self.i_off) * switch
        return current


class PolynomialDeviceActivation(UserDeviceActivation):
    """
    多项式器件激活函数
    
    I = sum(a_i * V^i) for i in 0..n
    用户可以提供多项式系数来拟合实验数据
    """
    
    def __init__(self, coefficients: list):
        """
        Args:
            coefficients: 多项式系数 [a0, a1, a2, ...], I = a0 + a1*V + a2*V^2 + ...
        """
        super().__init__()
        self.register_buffer('coeffs', torch.tensor(coefficients, dtype=torch.float32))
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        current = torch.zeros_like(voltage)
        v_power = torch.ones_like(voltage)
        
        for coeff in self.coeffs:
            current = current + coeff * v_power
            v_power = v_power * voltage
        
        # 归一化到 0-1
        current = torch.clamp(current, 0, 1)
        return current


class PiecewiseLinearDeviceActivation(UserDeviceActivation):
    """
    分段线性器件激活函数
    
    根据用户提供的 (V, I) 点进行线性插值
    """
    
    def __init__(self, v_points: list, i_points: list):
        """
        Args:
            v_points: 电压点列表 (必须升序排列)
            i_points: 对应的电流点列表
        """
        super().__init__()
        assert len(v_points) == len(i_points), "V和I点数必须相同"
        self.register_buffer('v_pts', torch.tensor(v_points, dtype=torch.float32))
        self.register_buffer('i_pts', torch.tensor(i_points, dtype=torch.float32))
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        # 使用 searchsorted 找到区间
        # 然后线性插值
        v_flat = voltage.flatten()
        
        # 找到插入位置
        idx = torch.searchsorted(self.v_pts, v_flat)
        idx = torch.clamp(idx, 1, len(self.v_pts) - 1)
        
        # 获取左右边界
        v_left = self.v_pts[idx - 1]
        v_right = self.v_pts[idx]
        i_left = self.i_pts[idx - 1]
        i_right = self.i_pts[idx]
        
        # 线性插值
        frac = (v_flat - v_left) / (v_right - v_left + 1e-10)
        frac = torch.clamp(frac, 0, 1)
        current = i_left + frac * (i_right - i_left)
        
        return current.reshape(voltage.shape)


# =============================================================================
# 核心：自定义器件神经元
# =============================================================================

class DeviceNeuronSTE(torch.autograd.Function):
    """
    器件神经元的 STE (Straight-Through Estimator) 实现
    
    前向：使用用户定义的器件非线性特性
    反向：使用 Sigmoid 梯度近似（保证可训练）
    """
    
    @staticmethod
    def forward(ctx, voltage, activation_fn, use_sigmoid_grad, sigmoid_k):
        ctx.save_for_backward(voltage)
        ctx.use_sigmoid_grad = use_sigmoid_grad
        ctx.sigmoid_k = sigmoid_k
        
        # 调用用户的激活函数
        current = activation_fn(voltage)
        return current
    
    @staticmethod
    def backward(ctx, grad_output):
        voltage, = ctx.saved_tensors
        
        if ctx.use_sigmoid_grad:
            # 使用 Sigmoid 导数作为梯度近似
            # sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
            k = ctx.sigmoid_k
            sigmoid_v = torch.sigmoid(k * (voltage - 0.5))
            grad_input = grad_output * sigmoid_v * (1 - sigmoid_v) * k
        else:
            # 直通估计器 (STE)
            grad_input = grad_output
        
        return grad_input, None, None, None


class CustomDeviceNeuron(nn.Module):
    """
    自定义器件神经元 (简化接口)
    
    完整流程:
        I_crossbar → 积分器(I→V, 0-1) → 用户激活函数(V→I, 0-1) → 积分器(I→V, 0-1) → V_next
    
    用户只需要提供一个接收 0-1 电压、输出 0-1 电流的激活函数
    
    使用示例:
        # 方法1: 使用预定义激活函数
        neuron = CustomDeviceNeuron(SigmoidDeviceActivation(k=12, v0=0.5))
        
        # 方法2: 使用自定义函数
        def my_vi(v):
            return torch.sigmoid(10 * (v - 0.5))
        neuron = CustomDeviceNeuron.from_function(my_vi)
        
        # 方法3: 使用多项式拟合
        coeffs = [0.01, 0.1, 0.5, 2.0]  # I = 0.01 + 0.1*V + 0.5*V^2 + 2*V^3
        neuron = CustomDeviceNeuron.from_polynomial(coeffs)
    """
    
    def __init__(
        self,
        activation: UserDeviceActivation,
        # 积分器参数
        k_int_input: float = 1.0,
        k_int_output: float = 1.0,
        # 噪声参数
        integrator_noise: float = 0.0,
        device_noise: float = 0.0,
        # STE 参数
        use_sigmoid_grad: bool = True,
        sigmoid_grad_k: float = 10.0,
    ):
        """
        Args:
            activation: 用户定义的 V→I 激活函数 (继承自 UserDeviceActivation)
            k_int_input: 输入积分器比例因子
            k_int_output: 输出积分器比例因子
            integrator_noise: 积分器噪声标准差
            device_noise: 器件输出噪声标准差
            use_sigmoid_grad: 反向传播时是否使用 Sigmoid 梯度近似
            sigmoid_grad_k: Sigmoid 梯度的斜率参数
        """
        super().__init__()
        
        self.use_sigmoid_grad = use_sigmoid_grad
        self.sigmoid_grad_k = sigmoid_grad_k
        self.device_noise = device_noise
        
        # 输入积分器: I_crossbar → V_device_input (-1 to 1)
        self.input_integrator = Integrator(
            k_int=k_int_input,
            v_min=-1.0,
            v_max=1.0,
            noise_std=integrator_noise
        )
        
        # 用户定义的器件激活函数: V (-1 to 1) → I (0-1)
        self.activation = activation
        
        # 输出积分器: I_device → V_next_layer (0-1)
        self.output_integrator = Integrator(
            k_int=k_int_output,
            v_min=0.0,
            v_max=1.0,
            noise_std=integrator_noise
        )
    
    def forward(self, current: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            current: Crossbar 输出电流 (归一化到 -1 to 1，差分电流)
            
        Returns:
            下一层输入电压 (归一化到 0-1)
        """
        # Step 1: 电流 → 电压 (输入积分器，归一化到 -1 to 1)
        v_device_in = self.input_integrator(current)
        
        # Step 2: 电压 → 电流 (用户的器件激活函数)
        if self.training:
            # 使用 STE 进行梯度传递
            i_device_out = DeviceNeuronSTE.apply(
                v_device_in,
                self.activation,
                self.use_sigmoid_grad,
                self.sigmoid_grad_k
            )
        else:
            i_device_out = self.activation(v_device_in)
        
        # 添加器件噪声
        if self.training and self.device_noise > 0:
            noise = torch.randn_like(i_device_out) * self.device_noise
            i_device_out = torch.clamp(i_device_out + noise, 0, 1)
        
        # Step 3: 电流 → 电压 (输出积分器，归一化到 0-1)
        v_next = self.output_integrator(i_device_out)
        
        return v_next
    
    @classmethod
    def from_function(
        cls,
        vi_function: Callable[[torch.Tensor], torch.Tensor],
        **kwargs
    ) -> 'CustomDeviceNeuron':
        """
        从函数创建器件神经元
        
        Args:
            vi_function: V→I 函数，接收 0-1 电压，返回 0-1 电流
            **kwargs: 传递给 CustomDeviceNeuron 的其他参数
            
        Returns:
            CustomDeviceNeuron 实例
            
        示例:
            def my_activation(v):
                return torch.sigmoid(10 * (v - 0.5))
            
            neuron = CustomDeviceNeuron.from_function(my_activation)
        """
        activation = FunctionDeviceActivation(vi_function)
        return cls(activation=activation, **kwargs)
    
    @classmethod
    def from_polynomial(
        cls,
        coefficients: list,
        **kwargs
    ) -> 'CustomDeviceNeuron':
        """
        从多项式系数创建器件神经元
        
        Args:
            coefficients: 多项式系数 [a0, a1, a2, ...], I = a0 + a1*V + a2*V^2 + ...
            **kwargs: 传递给 CustomDeviceNeuron 的其他参数
            
        Returns:
            CustomDeviceNeuron 实例
            
        示例:
            # I = 0.01 + 0.5*V + 2*V^2 - V^3
            neuron = CustomDeviceNeuron.from_polynomial([0.01, 0.5, 2.0, -1.0])
        """
        activation = PolynomialDeviceActivation(coefficients)
        return cls(activation=activation, **kwargs)
    
    @classmethod
    def from_lookup_table(
        cls,
        v_points: list,
        i_points: list,
        **kwargs
    ) -> 'CustomDeviceNeuron':
        """
        从查找表创建器件神经元
        
        Args:
            v_points: 电压点列表 (0-1, 升序)
            i_points: 对应的电流点列表 (0-1)
            **kwargs: 传递给 CustomDeviceNeuron 的其他参数
            
        Returns:
            CustomDeviceNeuron 实例
            
        示例:
            v_pts = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
            i_pts = [0.01, 0.02, 0.1, 0.5, 0.9, 0.99]
            neuron = CustomDeviceNeuron.from_lookup_table(v_pts, i_pts)
        """
        activation = PiecewiseLinearDeviceActivation(v_points, i_points)
        return cls(activation=activation, **kwargs)


# =============================================================================
# 便捷预设神经元
# =============================================================================

class SigmoidDeviceNeuron(CustomDeviceNeuron):
    """使用 Sigmoid 特性的器件神经元"""
    
    def __init__(self, k: float = 10.0, v0: float = 0.5, **kwargs):
        super().__init__(activation=SigmoidDeviceActivation(k=k, v0=v0), **kwargs)


class ReLUDeviceNeuron(CustomDeviceNeuron):
    """使用 ReLU 特性的器件神经元"""
    
    def __init__(self, k: float = 2.0, vth: float = 0.2, **kwargs):
        super().__init__(activation=ReLUDeviceActivation(k=k, vth=vth), **kwargs)


class TanhDeviceNeuron(CustomDeviceNeuron):
    """使用 Tanh 特性的器件神经元"""
    
    def __init__(self, k: float = 6.0, v0: float = 0.5, **kwargs):
        super().__init__(activation=TanhDeviceActivation(k=k, v0=v0), **kwargs)


class ThresholdDeviceNeuron(CustomDeviceNeuron):
    """使用阈值开关特性的器件神经元"""
    
    def __init__(self, vth: float = 0.5, sharpness: float = 50.0, **kwargs):
        super().__init__(activation=ThresholdDeviceActivation(vth=vth, sharpness=sharpness), **kwargs)


# =============================================================================
# 向后兼容：保留旧接口 (Deprecated)
# =============================================================================

class DeviceTransferCurve:
    """
    [已弃用] 请使用新的 UserDeviceActivation 接口
    
    保留此类是为了向后兼容
    """
    
    def __init__(self, voltage: np.ndarray = None, current: np.ndarray = None, **kwargs):
        import warnings
        warnings.warn(
            "DeviceTransferCurve 已弃用。请使用 UserDeviceActivation 接口。",
            DeprecationWarning
        )
        if voltage is not None and current is not None:
            self.v_points = voltage.tolist()
            self.i_points = current.tolist()
        else:
            self.v_points = [0, 1]
            self.i_points = [0, 1]
    
    @classmethod
    def from_excel(cls, file_path: str, **kwargs) -> 'DeviceTransferCurve':
        """从 Excel 加载 (向后兼容)"""
        import warnings
        warnings.warn(
            "DeviceTransferCurve.from_excel 已弃用。请使用 PiecewiseLinearDeviceActivation。",
            DeprecationWarning
        )
        return cls()
    
    @classmethod
    def create_default(cls, curve_type: str = 'sigmoid') -> 'DeviceTransferCurve':
        """创建默认曲线 (向后兼容)"""
        return cls()


class DeviceTransferCurveTorch(nn.Module):
    """[已弃用] 请使用 UserDeviceActivation"""
    
    def __init__(self, **kwargs):
        super().__init__()
        import warnings
        warnings.warn("DeviceTransferCurveTorch 已弃用。请使用 UserDeviceActivation。", DeprecationWarning)
    
    def forward(self, x):
        return torch.sigmoid(10 * (x - 0.5))


class CustomDeviceNeuronFromFile(CustomDeviceNeuron):
    """[已弃用] 请使用 CustomDeviceNeuron.from_lookup_table"""
    
    def __init__(self, file_path: str = None, **kwargs):
        import warnings
        warnings.warn(
            "CustomDeviceNeuronFromFile 已弃用。请使用 CustomDeviceNeuron.from_lookup_table。",
            DeprecationWarning
        )
        # 使用默认 Sigmoid
        super().__init__(activation=SigmoidDeviceActivation(), **kwargs)
