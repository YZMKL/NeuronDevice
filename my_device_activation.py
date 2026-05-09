"""
自定义器件激活函数
Custom Device Activation Function

用户的器件 V-I 特性 (分段拟合):
    数据流: 输入 → 映射到 [-4,4] → 取反 → 分段公式 → 归一化电流 (0-1)
    
    分段公式 (取反后的电压 V):
        V ∈ [-4, -1.0] : I = -7.6108e-05·V - 3.6188e-05  (线性)
        V ∈ (-1.0, 0)  : I = 3.2084e-05·V² - 9.8649e-06·V - 5.8455e-07  (二次)
        V ∈ [0, 4]     : I = 0
"""

import torch
import torch.nn as nn
import numpy as np
from .custom_neuron import UserDeviceActivation


class MyDeviceActivation(UserDeviceActivation):
    """
    自定义器件激活函数 (分段拟合版本)
    
    数据流 (差分交叉阵列):
        1. 输入: 差分电流归一化后的电压 v_in ∈ [-1, 1]
        2. 扩展到 [-4, 4]: v_scaled = v_in * 4
        3. 取反: v = -v_scaled
        4. 分段计算电流:
           - V ∈ [-4, -1.0] : I = -7.6108e-05·V - 3.6188e-05
           - V ∈ (-1.0, 0)  : I = 3.2084e-05·V² - 9.8649e-06·V - 5.8455e-07
           - V ∈ [0, 4]     : I = 0
        5. 归一化电流到 [0, 1]
    
    映射关系 (差分逻辑):
        v_in = -1  → v_scaled = -4 → v = 4   → I = 0
        v_in = 0   → v_scaled = 0  → v = 0   → I ≈ 0 (边界)
        v_in = 1   → v_scaled = 4  → v = -4  → I = I_max
    """
    
    # 分段函数参数
    # 线性段: V ∈ [-4, -1], I = a1*V + b1
    A1 = -7.6108e-05
    B1 = -3.6188e-05
    
    # 二次段: V ∈ (-1, 0), I = a2*V^2 + b2*V + c2
    A2 = 3.2084e-05
    B2 = -9.8649e-06
    C2 = -5.8455e-07
    
    def __init__(self):
        """初始化激活函数"""
        super().__init__()
        
        # 预计算电流范围用于归一化
        # 输入 [-1, 1] → 扩展到 [-4, 4] → 取反 → [-4, 4]
        # 所以实际电压范围是 [-4, 4]
        v_samples = torch.linspace(-4.0, 4.0, 10000)
        i_samples = self._compute_current_raw(v_samples)
        
        self.I_min = i_samples.min().item()
        self.I_max = i_samples.max().item()
        
        print(f"[MyDeviceActivation] 分段拟合激活函数初始化完成:")
        print(f"  输入范围: [-1, 1] (差分归一化)")
        print(f"  映射: [-1,1] → ×4 → [-4,4]V → 取反 → [-4,4]V")
        print(f"  电流范围: [{self.I_min:.6e}, {self.I_max:.6e}] A")
        print(f"  分段公式:")
        print(f"    V ∈ [-4, -1]: I = {self.A1:.4e}·V + {self.B1:.4e}")
        print(f"    V ∈ (-1, 0) : I = {self.A2:.4e}·V² + {self.B2:.4e}·V + {self.C2:.4e}")
        print(f"    V ∈ [0, 4]  : I = 0")
    
    def _compute_current_raw(self, v: torch.Tensor) -> torch.Tensor:
        """
        计算原始电流 (分段函数，未归一化)
        
        Args:
            v: 取反后的电压 (范围 [-4, 4])
            
        Returns:
            原始电流
        """
        current = torch.zeros_like(v)
        
        # 线性段: V ∈ [-4, -1]
        mask_linear = (v >= -4.0) & (v <= -1.0)
        current[mask_linear] = self.A1 * v[mask_linear] + self.B1
        
        # 二次段: V ∈ (-1, 0)
        mask_quad = (v > -1.0) & (v < 0.0)
        v_quad = v[mask_quad]
        current[mask_quad] = self.A2 * v_quad**2 + self.B2 * v_quad + self.C2
        
        # V ∈ [0, 4]: I = 0 (已经初始化为0)
        
        return current
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            voltage: 差分归一化电压，范围 [-1, 1]
            
        Returns:
            归一化电流 (0-1)，输出给框架
        """
        # Step 1: 将 [-1, 1] 扩展到 [-4, 4]
        # voltage=-1 → v_scaled=-4, voltage=0 → v_scaled=0, voltage=1 → v_scaled=4
        v_scaled = voltage * 4.0
        
        # Step 2: 取反
        v = -v_scaled
        # 现在: voltage=-1 → v=4, voltage=0 → v=0, voltage=1 → v=-4
        # 正的差分电流(权重为正) → v=-4 → 高电流输出
        # 负的差分电流(权重为负) → v=4  → 零电流输出
        
        # Step 3: 计算器件电流 (分段函数)
        # 注意：如果输入超出 [-1, 1]，v 可能超出 [-4, 4]，需要裁剪
        v_clamped = torch.clamp(v, -4.0, 4.0)
        current_raw = self._compute_current_raw(v_clamped)
        
        # Step 4: 归一化电流到 [0, 1]
        if abs(self.I_max - self.I_min) > 1e-15:
            current_normalized = (current_raw - self.I_min) / (self.I_max - self.I_min)
        else:
            current_normalized = torch.zeros_like(current_raw)
        
        # 确保在 [0, 1] 范围内
        current_normalized = torch.clamp(current_normalized, 0.0, 1.0)
        
        return current_normalized


class MyDeviceActivationV2(UserDeviceActivation):
    """
    自定义器件激活函数 (旧版本，基于二极管方程)
    
    已弃用，保留供参考
    """
    
    def __init__(
        self,
        v_min: float = -4.0,
        v_max: float = 0.0,
        I_s: float = 3.386,
        a: float = None,
    ):
        super().__init__()
        
        self.v_min = v_min
        self.v_max = v_max
        self.I_s = I_s
        
        if a is None:
            R = 1.7127e+06
            V_t = 0.02585
            self.a = -1.0 / (R * V_t)
        else:
            self.a = a
        
        v_samples = torch.linspace(v_min, v_max, 1000)
        i_samples = self.I_s * (torch.exp(self.a * v_samples) - 1)
        self.I_min = i_samples.min().item()
        self.I_max = i_samples.max().item()
        
        print(f"[MyDeviceActivationV2] 电流范围: [{self.I_min:.6e}, {self.I_max:.6e}]")
    
    def forward(self, voltage: torch.Tensor) -> torch.Tensor:
        v_real = self.v_min + (self.v_max - self.v_min) * voltage
        current = self.I_s * (torch.exp(self.a * v_real) - 1)
        current = (current - self.I_min) / (self.I_max - self.I_min + 1e-15)
        return torch.clamp(current, 0.0, 1.0)


# ============================================================================
# 便捷函数
# ============================================================================

def create_my_device_neuron(**kwargs):
    """
    创建使用自定义激活函数的器件神经元
    
    使用示例:
        from DeviceNeuron.my_device_activation import create_my_device_neuron
        
        neuron = create_my_device_neuron(k_int_input=1.0, k_int_output=1.0)
    """
    from .custom_neuron import CustomDeviceNeuron
    
    activation = MyDeviceActivation()
    return CustomDeviceNeuron(activation=activation, **kwargs)


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("测试自定义器件激活函数 (分段拟合 + 差分输入)")
    print("="*60)
    
    # 创建激活函数
    activation = MyDeviceActivation()
    
    # 测试输入 (差分归一化电压 [-1, 1])
    v_in = torch.tensor([-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0])
    
    # 计算中间变量 (映射: [-1,1] → ×4 → [-4,4] → 取反 → [-4,4])
    v_scaled = v_in * 4.0          # 扩展到 [-4, 4]
    v_neg = -v_scaled              # 取反
    
    # 计算输出
    i_norm = activation(v_in)
    
    print("\n测试结果:")
    print("-" * 70)
    print(f"{'V_in':>10} | {'V_scaled':>10} | {'V_neg':>10} | {'I_norm':>10}")
    print("-" * 70)
    for vi, vs, vg, i in zip(v_in.tolist(), v_scaled.tolist(), v_neg.tolist(), i_norm.tolist()):
        print(f"{vi:>10.2f} | {vs:>10.2f} | {vg:>10.2f} | {i:>10.4f}")
    print("-" * 70)
    
    print("\n分析 (差分逻辑):")
    print(f"  V_in=-1 (负权重主导) → V_neg=4   → I=0 (死区)")
    print(f"  V_in=0  (平衡)       → V_neg=0   → I≈0 (边界)")
    print(f"  V_in=1  (正权重主导) → V_neg=-4  → I=I_max")
    print(f"")
    print(f"  → 正权重激活，负权重抑制，符合差分逻辑！")
    
    print("\n测试完成!")

