# HIL (Hardware-in-the-Loop) 训练框架 - 详细工作流程

本文档详细描述两种训练模式的完整实现逻辑，包括前向传播、反向传播和权重更新的物理模型细节。

---

## 目录

1. [模式一：标准 Crossbar + ReLU（ADC/DAC 量化）](#模式一标准-crossbar--relu)
2. [模式二：差分 Crossbar + 自定义器件激活函数](#模式二差分-crossbar--自定义器件激活函数)
3. [核心组件详解](#核心组件详解)
4. [物理模型对应关系](#物理模型对应关系)

---

## 模式一：标准 Crossbar + ReLU

### 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          前向传播 (Forward Pass)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  x_digital ──→ [DAC] ──→ V_in ──→ [Crossbar MVM] ──→ I_out ──→ [ADC] ──→ y     │
│      │                      │            │                          │           │
│      │                      │            ↓                          │           │
│      │                      │    W_float → 映射 → (G⁺, G⁻)          │           │
│      │                      │            │                          │           │
│      │                      │            ↓                          │           │
│      │                      └───→ I = V · (G⁺-G⁻) · scale ←────────┘           │
│      │                                                                          │
│      ↓                                                                          │
│  [ReLU] ──→ 下一层                                                              │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: DAC（数模转换）

**物理含义**: 将数字输入信号转换为模拟电压

**实现代码** (`quantization.py`):
```python
def forward(self, x):
    # 1. 确保输入在 [0, 1]
    x = torch.clamp(x, 0, 1)
    
    # 2. 量化到 2^n_bits 级
    n_levels = 2 ** self.n_bits - 1
    x_quant = torch.round(x * n_levels) / n_levels
    
    # 3. 映射到电压范围 [0, V_max]
    v = x_quant * self.v_max
    
    # 4. 添加噪声（可选）
    if self.training and self.noise_std > 0:
        v = v + torch.randn_like(v) * self.noise_std * self.v_max
    
    return v
```

**物理模型**:
- 输入: 数字信号 `x ∈ [0, 1]`
- 输出: 模拟电压 `V ∈ [0, V_max]`
- 量化: `V = round(x × (2^n - 1)) / (2^n - 1) × V_max`

**STE（直通估计器）**:
- 前向: 执行量化
- 反向: 梯度直接传递 `∂L/∂x = ∂L/∂v`（忽略量化的不可微性）

---

### Step 2: 权重到电导映射

**物理含义**: 将数字权重映射到忆阻器的可编程电导态

**实现代码** (`conductance_states.py`):

#### Step 2.1: Layer-wise 归一化
```python
def normalize_weights(self, weight):
    # Ŵ = W / max|W|
    scale = weight.abs().max() + 1e-10
    normalized_weight = weight / scale
    return normalized_weight, scale  # Ŵ ∈ [-1, 1], scale 用于恢复输出尺度
```

**物理意义**: 
- 将权重缩放到 [-1, 1]，匹配电导的相对范围
- `scale` 保存原始尺度，用于输出恢复

#### Step 2.2: 差分拆分
```python
def differential_split(self, normalized_weight):
    # Ŵ⁺ = max(Ŵ, 0)  正权重
    # Ŵ⁻ = max(-Ŵ, 0) 负权重
    w_pos = torch.clamp(normalized_weight, min=0)
    w_neg = torch.clamp(-normalized_weight, min=0)
    return w_pos, w_neg  # 都在 [0, 1]
```

**物理意义**:
- 忆阻器电导只能是正值，无法直接表示负权重
- 使用两个交叉阵列分别存储正负权重
- 最终电流 `I = V·G⁺ - V·G⁻`

#### Step 2.3: 映射到最近电导态
```python
def map_weight_to_state_index(self, normalized_weight):
    # G(Ŵ) = argmin_{G_k ∈ G} |Ŵ - (G_k - G_min)/(G_max - G_min)|
    w_flat = normalized_weight.view(-1, 1)
    distances = torch.abs(w_flat - self.g_normalized.unsqueeze(0))
    indices = torch.argmin(distances, dim=1)
    return indices
```

**物理意义**:
- 器件只能达到有限个稳定电导态 `G = {G_1, G_2, ..., G_K}`
- 将理想权重映射到**最近的可达电导态**
- 这是硬件约束导致的量化误差的来源

#### Step 2.4: 高斯采样（Forward 噪声）
```python
def get_conductance(self, w_normalized, add_noise=True):
    indices = self.map_weight_to_state_index(w_normalized)
    g_mean = self.g_mean[indices]
    
    if add_noise:
        # G ~ N(G_k, σ_k²) - 模拟器件涨落
        g_std = self.g_std[indices]
        noise = torch.randn_like(g_mean) * g_std
        conductance = g_mean + noise
        conductance = torch.clamp(conductance, self.g_min, self.g_max)
    else:
        conductance = g_mean
    
    return conductance, indices
```

**物理意义**:
- 每个电导态有固定的均值 `G_k` 和标准差 `σ_k`
- 训练时采样 `G ~ N(G_k, σ_k²)` 模拟器件噪声
- 推理时可以使用均值 `G_k`

---

### Step 3: Crossbar MVM（矩阵向量乘法）

**物理含义**: 交叉阵列执行模拟计算

**实现代码** (`crossbar_core.py`):
```python
def compute_mvm(self, v_in, g_diff, weight_float, scale):
    if self.training:
        return CrossbarMVMFunction.apply(v_in, g_diff, weight_float, scale)
    else:
        return F.linear(v_in, g_diff * scale, None)
```

**物理模型**:
```
I_j = Σ_i (V_i × G_{ij}) = V^T · G
```

对于差分阵列:
```
I_j = Σ_i V_i × (G⁺_{ij} - G⁻_{ij}) × scale
```

**实际计算**:
```python
# 前向
I_out = torch.mm(V_in, G_diff.T) * scale
# 其中 G_diff = G⁺ - G⁻
```

---

### Step 4: ADC（模数转换）

**物理含义**: 将模拟电流转换为数字信号

**实现代码** (`quantization.py`):
```python
def forward(self, current):
    # 1. 动态范围归一化
    if self.training:
        i_max = current.max().detach()
        i_min = current.min().detach()
        # 更新运行时统计
        self.running_max = (1-m)*self.running_max + m*i_max
        self.running_min = (1-m)*self.running_min + m*i_min
    else:
        i_max = self.running_max
        i_min = self.running_min
    
    # 2. 归一化到 [0, 1]
    normalized = (current - i_min) / (i_max - i_min + 1e-10)
    
    # 3. 量化
    n_levels = 2 ** self.n_bits - 1
    quantized = torch.round(normalized * n_levels) / n_levels
    
    # 4. 反归一化（保持原始尺度）
    output = quantized * (i_max - i_min) + i_min
    
    return output
```

**物理意义**:
- 将电流量化到 `2^n_bits` 个离散电平
- 动态范围 ADC 自动适应不同层的输出范围
- 反归一化确保输出尺度正确

---

### Step 5: 反向传播（STE）

**关键问题**: 量化和电导映射是不可微的，如何传递梯度？

**解决方案**: Straight-Through Estimator (STE)

**实现代码** (`crossbar_core.py`):
```python
class CrossbarMVMFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_in, g_diff, weight_float, scale):
        ctx.save_for_backward(v_in, weight_float, scale)
        # 前向: 使用电导 I = V · (G⁺ - G⁻) · scale
        return F.linear(v_in, g_diff * scale, None)
    
    @staticmethod
    def backward(ctx, grad_output):
        v_in, weight_float, scale = ctx.saved_tensors
        
        # 反向: 假装使用原始数字权重 W_float（不是 G）
        # ∂L/∂V = ∂L/∂y · W^T
        if ctx.needs_input_grad[0]:
            grad_v_in = grad_output.mm(weight_float)
        
        # ∂L/∂W = V^T · ∂L/∂y
        if ctx.needs_input_grad[2]:
            grad_weight = grad_output.t().mm(v_in)
        
        return grad_v_in, None, grad_weight, None
```

**物理意义**:
```
前向传播: y = V · (G⁺ - G⁻) · scale     ← 使用映射后的电导
反向传播: ∂L/∂W = V^T · ∂L/∂y           ← 假装 y = V · W_float
```

**为什么这样做？**
- `G` 是 `W_float` 的量化版本
- 直接对 `G` 求梯度没有意义（离散值）
- STE 让梯度通过量化操作传递到数字权重 `W_float`

---

### Step 6: 权重更新

**实现**:
```python
# 标准 PyTorch 优化器
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 训练循环
for epoch in range(epochs):
    for x, y in dataloader:
        optimizer.zero_grad()
        
        # 前向（使用电导）
        output = model(x)
        loss = criterion(output, y)
        
        # 反向（梯度传到 W_float）
        loss.backward()
        
        # 更新 W_float
        optimizer.step()
        
        # 下次前向时，W_float 会被重新映射到新的电导
```

**关键点**:
- `W_float` 是可学习参数（数字影子权重）
- 每次前向传播时，`W_float → (G⁺, G⁻)` 重新映射
- 优化器更新的是 `W_float`，不是 `G`

---

## 模式二：差分 Crossbar + 自定义器件激活函数

### 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                          前向传播 (Forward Pass)                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  x ──→ [DAC] ──→ V_in                                                          │
│                   │                                                             │
│                   ↓                                                             │
│          W_float → 归一化 → 差分拆分 → 映射电导 → G⁺, G⁻                          │
│                   │                           │                                 │
│                   ↓                           ↓                                 │
│          I_diff = V_in · (G⁺ - G⁻) · scale   (差分电流，可正可负)                │
│                   │                                                             │
│                   ↓                                                             │
│          [ADC] ──→ I_digital                                                    │
│                   │                                                             │
│                   ↓                                                             │
│          [对称归一化] ──→ V_norm ∈ [-1, 1]   (差分信号保持零点)                  │
│                   │                                                             │
│                   ↓                                                             │
│          [扩展] ──→ V_scaled = V_norm × 4 ∈ [-4, 4]                             │
│                   │                                                             │
│                   ↓                                                             │
│          [取反] ──→ V_device = -V_scaled ∈ [-4, 4]                              │
│                   │                                                             │
│                   ↓                                                             │
│          [分段 V-I 函数] ──→ I_device ∈ [I_min, I_max]                          │
│                   │                                                             │
│                   ↓                                                             │
│          [电流归一化] ──→ y ∈ [0, 1]                                            │
│                   │                                                             │
│                   ↓                                                             │
│          [输出缩放 + 偏置] ──→ 下一层                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

### Step 1-4: 与模式一相同

DAC、权重映射、Crossbar MVM、ADC 步骤与模式一完全相同。

### Step 5: 对称归一化（差分信号）

**物理含义**: 差分交叉阵列输出的电流可正可负，需要对称归一化

**实现代码** (`crossbar_core.py`):
```python
def _normalize_for_activation(self, current):
    if self.training:
        # 计算最大绝对值
        i_abs_max = current.abs().max().detach()
        i_scale = i_abs_max + 1e-10
        
        # 更新运行统计
        self.running_i_max = (1-m)*self.running_i_max + m*i_scale
    else:
        i_scale = self.running_i_max
    
    # 对称归一化: [-I_max, I_max] → [-1, 1]
    normalized = current / i_scale
    normalized = torch.clamp(normalized, -1, 1)
    
    return normalized, i_scale
```

**物理意义**:
- 差分电流: `I_diff = V · (G⁺ - G⁻)`
  - `G⁺ > G⁻` → 正电流 → 正权重主导
  - `G⁺ < G⁻` → 负电流 → 负权重主导
  - `G⁺ = G⁻` → 零电流 → 权重为零
- 对称归一化保持零点在中心: `I=0 → V_norm=0`

---

### Step 6: 自定义器件激活函数

**物理含义**: 使用真实器件的 V-I 特性作为激活函数

**实现代码** (`my_device_activation.py`):
```python
class MyDeviceActivation(nn.Module):
    # 分段函数参数（来自器件测量）
    # 线性段: V ∈ [-4, -1], I = a1*V + b1
    A1 = -7.6108e-05
    B1 = -3.6188e-05
    
    # 二次段: V ∈ (-1, 0), I = a2*V² + b2*V + c2
    A2 = 3.2084e-05
    B2 = -9.8649e-06
    C2 = -5.8455e-07
    
    def forward(self, voltage):
        """
        输入: voltage ∈ [-1, 1] (差分归一化)
        """
        # Step 1: 扩展到 [-4, 4]
        v_scaled = voltage * 4.0
        
        # Step 2: 取反
        v = -v_scaled
        # 现在: voltage=1 (正权重) → v=-4 → 高电流输出
        #       voltage=-1 (负权重) → v=4 → 零电流输出
        
        # Step 3: 分段 V-I 计算
        current = self._compute_current_raw(v)
        
        # Step 4: 归一化到 [0, 1]
        current_normalized = (current - self.I_min) / (self.I_max - self.I_min)
        current_normalized = torch.clamp(current_normalized, 0.0, 1.0)
        
        return current_normalized
    
    def _compute_current_raw(self, v):
        current = torch.zeros_like(v)
        
        # 线性段: V ∈ [-4, -1]
        mask_linear = (v >= -4.0) & (v <= -1.0)
        current[mask_linear] = self.A1 * v[mask_linear] + self.B1
        
        # 二次段: V ∈ (-1, 0)
        mask_quad = (v > -1.0) & (v < 0.0)
        v_quad = v[mask_quad]
        current[mask_quad] = self.A2 * v_quad**2 + self.B2 * v_quad + self.C2
        
        # V ∈ [0, 4]: I = 0 (死区)
        
        return current
```

**物理意义**:
```
差分电流 I_diff:
  正权重主导 (I > 0) → V_norm > 0 → V_device < 0 → 高电流输出 → 激活
  负权重主导 (I < 0) → V_norm < 0 → V_device > 0 → 零电流输出 → 抑制
  平衡 (I = 0)      → V_norm = 0 → V_device = 0 → 边界          → 阈值
```

**激活函数特性**:
```
I_norm
1.0 ┤                          ● (正权重 → 激活)
    │                        ╱
    │                     ╱
0.5 ┤                  ╱
    │               ╱
    │            ╱
0.0 ┼━━━━━━━━━━●
    └────┬────┬────┬────┬────┬
       -1  -0.5   0   0.5   1   V_norm
    负权重←      →正权重
    (抑制)      (激活)
```

---

### Step 7: 反向传播（自定义激活函数）

**关键点**: 分段函数的导数

**实现**（PyTorch 自动微分）:
```python
# V_norm ∈ [-1, 1] → V_device ∈ [-4, 4]
v_device = -voltage * 4.0

# 分段函数的梯度:
# 1. 线性段 V ∈ [-4, -1]: ∂I/∂V = A1 = -7.6108e-05
# 2. 二次段 V ∈ (-1, 0):  ∂I/∂V = 2*A2*V + B2
# 3. 死区   V ∈ [0, 4]:   ∂I/∂V = 0

# 链式法则:
# ∂I_norm/∂V_norm = (∂I/∂V_device) × (-4) / (I_max - I_min)
```

**物理意义**:
- 线性段梯度恒定
- 二次段梯度随电压变化
- 死区梯度为零（类似 ReLU 的死神经元问题）

---

### 完整前向-反向流程图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              完整训练循环                                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  ┌─────────── 前向传播 ───────────┐                                            │
│  │                                │                                            │
│  │  x → DAC → V_in                │                                            │
│  │           ↓                    │                                            │
│  │  W_float → G⁺,G⁻,scale         │  ← 权重映射（每次前向重新执行）             │
│  │           ↓                    │                                            │
│  │  I_diff = V·(G⁺-G⁻)·scale     │  ← Crossbar MVM                            │
│  │           ↓                    │                                            │
│  │  ADC → I_digital               │                                            │
│  │           ↓                    │                                            │
│  │  归一化 → [-1,1]               │  ← 对称归一化（差分）                       │
│  │           ↓                    │                                            │
│  │  器件激活函数 → [0,1]          │  ← 分段 V-I 特性                            │
│  │           ↓                    │                                            │
│  │  缩放 + 偏置 → y               │                                            │
│  │                                │                                            │
│  └────────────────────────────────┘                                            │
│                    │                                                            │
│                    ↓ loss = CrossEntropy(y, label)                             │
│                                                                                 │
│  ┌─────────── 反向传播 (STE) ─────┐                                            │
│  │                                │                                            │
│  │  ∂L/∂y (从 loss 开始)          │                                            │
│  │           ↓                    │                                            │
│  │  ∂L/∂I_norm (激活函数导数)     │  ← 分段函数梯度                            │
│  │           ↓                    │                                            │
│  │  ∂L/∂I_diff (归一化梯度)       │                                            │
│  │           ↓                    │                                            │
│  │  ∂L/∂W_float (STE!)            │  ← 关键: 用 W_float 计算梯度，不是 G       │
│  │                                │                                            │
│  │  公式: ∂L/∂W = V_in^T · ∂L/∂I  │                                            │
│  │                                │                                            │
│  └────────────────────────────────┘                                            │
│                    │                                                            │
│                    ↓                                                            │
│  ┌─────────── 优化器更新 ─────────┐                                            │
│  │                                │                                            │
│  │  W_float ← W_float - lr × ∂L/∂W│  ← Adam/SGD 更新数字权重                   │
│  │                                │                                            │
│  │  (下次前向时 W_float 重新映射)  │  ← 新的 W_float → 新的 (G⁺, G⁻)           │
│  │                                │                                            │
│  └────────────────────────────────┘                                            │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 核心组件详解

### 1. ConductanceStates（电导态管理）

**数据结构**:
```python
G = {(G_k, σ_k)}_{k=1}^K
```

**来源**: 从真实器件测量数据加载（CSV/Excel）
- `G_k`: 第 k 个稳定电导态的均值
- `σ_k`: 第 k 个电导态的标准差（器件涨落）

**使用**:
- 量化: 将任意权重映射到最近的 `G_k`
- 噪声: 前向时采样 `G ~ N(G_k, σ_k²)`

### 2. WeightToConductanceMapper（权重映射器）

**完整流程**:
```
W_float                           ← 原始数字权重
    │
    ↓ normalize_weights()
Ŵ = W / max|W| ∈ [-1, 1]         ← Layer-wise 归一化
    │
    ↓ differential_split()
Ŵ⁺ = max(Ŵ, 0)                   ← 正权重 [0, 1]
Ŵ⁻ = max(-Ŵ, 0)                  ← 负权重 [0, 1]
    │
    ↓ map_to_conductance()
G⁺ = argmin|Ŵ⁺ - G_norm|         ← 映射到最近电导态
G⁻ = argmin|Ŵ⁻ - G_norm|
    │
    ↓ add_noise (if training)
G⁺ ~ N(G_k⁺, σ_k⁺²)              ← 高斯采样
G⁻ ~ N(G_k⁻, σ_k⁻²)
```

### 3. StraightThroughEstimator（STE）

**核心思想**:
```
前向: y = quantize(x)            ← 执行量化
反向: ∂L/∂x = ∂L/∂y              ← 梯度直通（假装量化不存在）
```

**应用场景**:
- DAC 量化
- ADC 量化
- 权重到电导映射

---

## 物理模型对应关系

| 代码模块 | 物理含义 | 关键参数 |
|----------|----------|----------|
| `DAC` | 数模转换器 | `n_bits`, `v_max`, `noise_std` |
| `ADC/DynamicADC` | 模数转换器 | `n_bits`, `noise_std`, 动态范围 |
| `ConductanceStates` | 忆阻器电导态 | `G_mean[]`, `G_std[]` |
| `WeightToConductanceMapper` | 权重编程 | 归一化 + 差分 + 映射 |
| `CrossbarMVMCore.compute_mvm` | 交叉阵列 MVM | `I = V·G` |
| `MyDeviceActivation` | 器件 V-I 特性 | 分段函数参数 |
| `STE` | 梯度传递 | 忽略量化的不可微性 |

### 物理假设

1. **欧姆定律**: `I = V × G`
2. **线性叠加**: `I_total = Σ V_i × G_i`
3. **高斯噪声**: 器件涨落服从正态分布
4. **差分消除**: `I_diff = I⁺ - I⁻` 消除公共模式噪声
5. **量化有限精度**: DAC/ADC 引入量化误差

### 训练收敛原理

虽然前向使用量化/噪声的电导，但：
1. STE 让梯度传递到连续的数字权重 `W_float`
2. 优化器更新 `W_float` 最小化 loss
3. 下次前向时，更好的 `W_float` 映射到更好的电导组合
4. 迭代收敛到硬件约束下的最优解

---

## 代码文件对应

| 文件 | 功能 |
|------|------|
| `quantization.py` | DAC、ADC、STE 实现 |
| `conductance_states.py` | 电导态管理、权重映射 |
| `crossbar_core.py` | Crossbar MVM、统一层实现 |
| `my_device_activation.py` | 自定义器件激活函数 |
| `unified_models.py` | MLP、CNN 模型定义 |
| `hil_trainer.py` | HIL 训练框架 |
| `config.py` | 配置管理 |
