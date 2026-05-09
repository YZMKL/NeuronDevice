# DeviceNeuron/analysis/features/cifar10_deviceneuron_features.py
# python -m DeviceNeuron.analysis.features.cifar10_deviceneuron_features

import torch
import torch.nn as nn
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from DeviceNeuron import CIFAR10UnifiedVGGWithDeviceNeuron
from DeviceNeuron.config import CONFIG, create_real_device_config
from DeviceNeuron.my_device_activation import MyDeviceActivation


# ── 1. Hook 注册器 ────────────────────────────────────────────────────
class Visualizer:
    def __init__(self, model):
        self.model = model
        self.feature_maps = []
        self.layer_names = []
        self.fc_vectors = []
        self.fc_layer_names = []
        self.hooks = []

    def _hook_2d(self, name):
        def hook(module, input, output):
            self.feature_maps.append(output.detach().cpu())
            self.layer_names.append(name)
        return hook

    def _hook_1d(self, name):
        def hook(module, input, output):
            self.fc_vectors.append(output.detach().cpu().numpy()[0])
            self.fc_layer_names.append(name)
        return hook

    def register_hooks(self):
        m = self.model

        # Block 1
        self.hooks.append(m.conv1_1.register_forward_hook(self._hook_2d("Conv1_1")))
        self.hooks.append(m.dn1_1.register_forward_hook(self._hook_2d("Act1_1")))
        self.hooks.append(m.conv1_2.register_forward_hook(self._hook_2d("Conv1_2")))
        self.hooks.append(m.dn1_2.register_forward_hook(self._hook_2d("Act1_2")))
        self.hooks.append(m.pool1.register_forward_hook(self._hook_2d("Pool1")))

        # Block 2
        self.hooks.append(m.conv2_1.register_forward_hook(self._hook_2d("Conv2_1")))
        self.hooks.append(m.dn2_1.register_forward_hook(self._hook_2d("Act2_1")))
        self.hooks.append(m.conv2_2.register_forward_hook(self._hook_2d("Conv2_2")))
        self.hooks.append(m.dn2_2.register_forward_hook(self._hook_2d("Act2_2")))
        self.hooks.append(m.pool2.register_forward_hook(self._hook_2d("Pool2")))

        # Block 3
        self.hooks.append(m.conv3_1.register_forward_hook(self._hook_2d("Conv3_1")))
        self.hooks.append(m.dn3_1.register_forward_hook(self._hook_2d("Act3_1")))
        self.hooks.append(m.conv3_2.register_forward_hook(self._hook_2d("Conv3_2")))
        self.hooks.append(m.dn3_2.register_forward_hook(self._hook_2d("Act3_2")))
        self.hooks.append(m.pool3.register_forward_hook(self._hook_2d("Pool3")))

        # FC
        self.hooks.append(m.fc1.register_forward_hook(self._hook_1d("01_FC1_512")))
        self.hooks.append(m.dn_fc1.register_forward_hook(self._hook_1d("02_Act_FC")))
        self.hooks.append(m.fc2.register_forward_hook(self._hook_1d("03_FC2_output")))

    def remove_hooks(self):
        for h in self.hooks:
            h.remove()


# ── 2. 主可视化函数 ───────────────────────────────────────────────────
def visualize_features(model_path, target_class="airplane", class_index=1, top_k=6):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ── 数据集 ──
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=tf_test)
    classes = test_dataset.classes

    # 找指定类别的第 class_index 张图
    target_label = classes.index(target_class)
    matching_indices = [i for i, l in enumerate(test_dataset.targets) if l == target_label]
    image_index = matching_indices[class_index]
    print(f"类别 '{target_class}' 第{class_index}张，全局索引: {image_index}")

    img_tensor, label = test_dataset[image_index]
    img_input = img_tensor.unsqueeze(0).to(device)

    # ── 加载模型 ──
    cfg = create_real_device_config()
    config = cfg.get_crossbar_config()
    device_activation = MyDeviceActivation()

    model = CIFAR10UnifiedVGGWithDeviceNeuron(
        config=config,
        device_activation=device_activation,
        use_batchnorm=True,
        dropout=0.5,
        k_int_input=cfg.INTEGRATOR_K_INPUT,
        k_int_output=cfg.INTEGRATOR_K_OUTPUT,
        device_noise=cfg.DEVICE_NOISE_STD
    ).to(device)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # ── 注册 Hook 并推理 ──
    viz = Visualizer(model)
    viz.register_hooks()
    with torch.no_grad():
        output = model(img_input)
    pred = output.argmax(1).item()
    viz.remove_hooks()

    print(f"True: {classes[label]}  |  Pred: {classes[pred]}")

    # ── 还原原图 ──
    inv_normalize = transforms.Normalize(
        mean=[-0.4914/0.2470, -0.4822/0.2435, -0.4465/0.2616],
        std=[1/0.2470, 1/0.2435, 1/0.2616]
    )
    img_show = inv_normalize(img_tensor).permute(1, 2, 0).numpy()
    img_show = np.clip(img_show, 0, 1)

    # ── 创建输出目录 ──
    base_dir = f"DeviceNeuron/analysis/features/figures/features/deviceneuron_{target_class}{class_index}"
    os.makedirs(base_dir, exist_ok=True)
    plt.imsave(os.path.join(base_dir, "00_input_image.png"), img_show)
    print(f"保存目录: {base_dir}/")

    # ════════════════════════════════════════════════════
    # Part A: 2D 特征图
    # ════════════════════════════════════════════════════
    COLOR_MAP = {"Conv": "black", "Act": "blue", "Pool": "red"}

    num_layers = len(viz.feature_maps)
    fig, axes = plt.subplots(num_layers, top_k + 1,
                             figsize=(3 * (top_k + 1), 2.2 * num_layers))
    fig.suptitle(
        f"Top-{top_k} Channels (Conv → Act → Pool)\nTrue: {classes[label]} | Pred: {classes[pred]}",
        fontsize=16, y=0.99
    )

    for i, fmap in enumerate(viz.feature_maps):
        layer_name = viz.layer_names[i]
        layer_dir = os.path.join(base_dir, layer_name)
        os.makedirs(layer_dir, exist_ok=True)

        ax0 = axes[i, 0]
        if i == 0:
            ax0.imshow(img_show)
            ax0.set_title(f"Input Image\n{layer_name}", fontsize=10)
        else:
            prefix = layer_name.split("_")[0]  # Conv / Act / Pool
            color = COLOR_MAP.get(prefix, "black")
            ax0.text(0.5, 0.5, layer_name, fontsize=11,
                     ha='center', va='center', fontweight='bold', color=color,
                     transform=ax0.transAxes)
        ax0.axis('off')

        c = fmap.shape[1]
        channel_max = fmap[0].view(c, -1).max(dim=1)[0]
        topk_idx = torch.topk(channel_max, min(top_k, c))[1]

        for j in range(top_k):
            ax = axes[i, j + 1]
            if j >= len(topk_idx):
                ax.axis('off')
                continue
            idx = topk_idx[j].item()
            ch = fmap[0, idx].numpy()
            ch = (ch - ch.min()) / (ch.max() - ch.min() + 1e-8)

            plt.imsave(os.path.join(layer_dir, f"ch_{idx:03d}.png"), ch, cmap='viridis')
            ax.imshow(ch, cmap='viridis')
            ax.set_title(f"Ch {idx}", fontsize=9)
            ax.axis('off')

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    summary_path = f"DeviceNeuron/analysis/features/figures/flow_summary_deviceneuron_{target_class}{class_index}.png"
    plt.savefig(summary_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"汇总特征图已保存: {summary_path}")

    # ════════════════════════════════════════════════════
    # Part B: 1D FC 热力图
    # ════════════════════════════════════════════════════
    fc_dir = os.path.join(base_dir, "FC_Layers")
    os.makedirs(fc_dir, exist_ok=True)

    for i, vec in enumerate(viz.fc_vectors):
        fc_name = viz.fc_layer_names[i]
        dim = len(vec)
        heatmap = vec.reshape(dim, 1)

        fig_fc, ax_fc = plt.subplots(figsize=(2, 10))
        ax_fc.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')

        if dim == 10:
            ax_fc.set_title(f"{fc_name}\n(True:{classes[label]})", fontsize=10)
            ax_fc.set_xticks([])
            ax_fc.set_yticks(range(10))
            ax_fc.set_yticklabels(classes, fontsize=9)
        else:
            ax_fc.set_title(f"{fc_name}\n({dim})", fontsize=10)
            ax_fc.set_xticks([])
            ax_fc.set_yticks([0, dim - 1])
            ax_fc.set_yticklabels([0, dim - 1], fontsize=9)

        fc_path = os.path.join(fc_dir, f"{fc_name}_vertical.png")
        fig_fc.tight_layout()
        fig_fc.savefig(fc_path, dpi=300, bbox_inches='tight')
        plt.close(fig_fc)

    print(f"FC热力图已保存: {fc_dir}/")


# ── 3. 入口 ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    MODEL_FILE = "unified_cifar10_vgg_device_neuron_best.pth"

    visualize_features(
        model_path=MODEL_FILE,
        target_class="ship",  # CIFAR-10类别：airplane/automobile/bird/cat/deer/dog/frog/horse/ship/truck
        class_index=12,            # 该类别中第几张（0开始）
        top_k=6
    )