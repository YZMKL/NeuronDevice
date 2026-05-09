"""
MLP 多数据集训练脚本 - Iris, Fashion-MNIST, UCI HAR, ISOLET, Letter Recognition
Standard MLP Training on Diverse Datasets

用法:
    cd /home/zhc/Projects/neuromorphic
    python -m DeviceNeuron.benchmark.mlp_diverse
    python -m DeviceNeuron.benchmark.mlp_diverse --datasets Letter-Recognition
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms
import os
import sys
import time
import argparse
from tqdm import tqdm
import numpy as np
from typing import List

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DeviceNeuron.benchmark.train import create_benchmark_logger

class DiverseMLP(nn.Module):
    """标准多层感知机 - 使用 ReLU 激活函数"""
    def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int):
        super().__init__()
        layers = []
        prev_size = input_size
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_size, h))
            layers.append(nn.ReLU())
            prev_size = h
        layers.append(nn.Linear(prev_size, output_size))
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 展平输入
        x = x.view(x.size(0), -1)
        return self.network(x)

# ============================================================================
# 数据加载函数
# ============================================================================

def get_iris_loader(data_dir: str):
    """加载 Iris 数据集 (使用 sklearn 辅助加载)"""
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    
    return DataLoader(train_ds, batch_size=16, shuffle=True), DataLoader(test_ds, batch_size=16)

def get_fashion_mnist_loader(data_dir: str):
    """加载 Fashion-MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
    return DataLoader(train_dataset, batch_size=64, shuffle=True), DataLoader(test_dataset, batch_size=64)

def get_letter_recognition_loader(data_dir: str):
    """加载 UCI Letter Recognition 数据集"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from urllib.request import urlretrieve
    from urllib.error import URLError, HTTPError

    candidate_files = [
        os.path.join(data_dir, 'letter-recognition.data'),
        os.path.join(data_dir, 'LetterRecognition', 'letter-recognition.data'),
        os.path.join(data_dir, 'Letter-Recognition', 'letter-recognition.data'),
        os.path.join(data_dir, 'UCI Letter Recognition', 'letter-recognition.data'),
        os.path.join(data_dir, 'uci-letter', 'letter-recognition.data'),
    ]
    data_path = next((path for path in candidate_files if os.path.exists(path)), None)
    if data_path is None:
        for root, _, files in os.walk(data_dir):
            if 'letter-recognition.data' in files:
                data_path = os.path.join(root, 'letter-recognition.data')
                break

    if data_path is None:
        download_dir = os.path.join(data_dir, 'LetterRecognition')
        os.makedirs(download_dir, exist_ok=True)
        data_path = os.path.join(download_dir, 'letter-recognition.data')
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/letter-recognition/letter-recognition.data'
        print(f"[Letter-Recognition] 未找到本地数据，尝试从 UCI 下载到 {data_path} ...")
        try:
            urlretrieve(url, data_path)
            print(f"[Letter-Recognition] 下载完成: {data_path}")
        except (URLError, HTTPError) as exc:
            raise FileNotFoundError(
                f"无法下载 Letter Recognition 数据集: {exc}. 请检查网络或手动下载到 {download_dir}."
            )
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"下载后未找到文件: {data_path}。请手动检查下载目录。"
            )

    df = pd.read_csv(data_path, header=None, sep=',', engine='python', skip_blank_lines=True)
    df = df.dropna(how='all')
    if df.shape[1] != 17:
        raise ValueError(
            f"Letter Recognition 数据集应包含 17 列，但发现 {df.shape[1]} 列。"
        )

    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    y = df.iloc[:, 0].astype(str).values
    y = LabelEncoder().fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(test_ds, batch_size=64)


def get_isolet_loader(data_dir: str):
    """加载 UCI ISOLET 数据集"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from urllib.request import urlretrieve
    from urllib.error import URLError, HTTPError

    def find_file(name: str):
        for root, _, files in os.walk(data_dir):
            if name in files:
                return os.path.join(root, name)
        return None

    train_path = find_file('isolet1+2+3+4.data')
    test_path = find_file('isolet5.data')
    if train_path is None or test_path is None:
        part_paths = [find_file(f'isolet{i}.data') for i in range(1, 5)]
        if all(part_paths) and test_path is not None:
            train_df = pd.concat([pd.read_csv(p, header=None) for p in part_paths], ignore_index=True)
            test_df = pd.read_csv(test_path, header=None)
        else:
            download_dir = os.path.join(data_dir, 'ISOLET')
            os.makedirs(download_dir, exist_ok=True)
            url_base = 'https://archive.ics.uci.edu/ml/machine-learning-databases/isolet'
            train_path = os.path.join(download_dir, 'isolet1+2+3+4.data')
            test_path = os.path.join(download_dir, 'isolet5.data')
            if not os.path.exists(train_path):
                print(f"[ISOLET] 未找到本地训练数据，尝试下载 {train_path} ...")
                try:
                    urlretrieve(f"{url_base}/isolet1+2+3+4.data", train_path)
                except (URLError, HTTPError) as exc:
                    raise FileNotFoundError(
                        f"无法下载 ISOLET 训练数据: {exc}. 请手动下载到 {download_dir}."
                    )
            if not os.path.exists(test_path):
                print(f"[ISOLET] 未找到本地测试数据，尝试下载 {test_path} ...")
                try:
                    urlretrieve(f"{url_base}/isolet5.data", test_path)
                except (URLError, HTTPError) as exc:
                    raise FileNotFoundError(
                        f"无法下载 ISOLET 测试数据: {exc}. 请手动下载到 {download_dir}."
                    )
            train_df = pd.read_csv(train_path, header=None)
            test_df = pd.read_csv(test_path, header=None)
    else:
        train_df = pd.read_csv(train_path, header=None)
        test_df = pd.read_csv(test_path, header=None)

    if train_df.shape[1] != 618 or test_df.shape[1] != 618:
        raise ValueError(
            f"ISOLET 数据集应包含 618 列，但训练集 {train_df.shape[1]} 列，测试集 {test_df.shape[1]} 列。"
        )

    X_train = train_df.iloc[:, :-1].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    y_train = train_df.iloc[:, -1].astype(int).to_numpy() - 1
    X_test = test_df.iloc[:, :-1].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    y_test = test_df.iloc[:, -1].astype(int).to_numpy() - 1

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))

    return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(test_ds, batch_size=64)


def get_uci_har_loader(data_dir: str):
    """加载本地 UCI HAR 数据集 (自动适配 CSV 或 TXT 格式)"""
    import pandas as pd
    
    # 方案 A: 优先尝试加载 HAR1 中的 CSV 格式
    har1_path = os.path.join(data_dir, "HAR1")
    if os.path.exists(os.path.join(har1_path, "train.csv")):
        print(f"\n[UCI-HAR] 检测到 HAR1 (CSV 格式): {har1_path}")
        train_df = pd.read_csv(os.path.join(har1_path, "train.csv"))
        test_df = pd.read_csv(os.path.join(har1_path, "test.csv"))

        # 1. 自动识别标签列 (查找常见名称或取最后一列)
        label_names = ['Activity', 'label', 'y', 'activity', 'Activity_Label']
        label_col = next((c for c in label_names if c in train_df.columns), train_df.columns[-1])
        train_y = train_df[label_col].values
        test_y = test_df[label_col].values
        
        # 2. 提取数值特征并对齐到 561 维 (UCI HAR 标准特征数)
        # 排除非数值列（如 Subject）和标签列
        train_x_df = train_df.select_dtypes(include=['number'])
        if label_col in train_x_df.columns:
            train_x_df = train_x_df.drop(columns=[label_col])
        
        test_x_df = test_df.select_dtypes(include=['number'])
        if label_col in test_x_df.columns:
            test_x_df = test_x_df.drop(columns=[label_col])
            
        # 截取最后的 561 列（防止前面有 Index 或 ID 列）
        train_x = train_x_df.values[:, -561:]
        test_x = test_x_df.values[:, -561:]
        
        # 3. 处理非数值标签 (如果是字符串则编码)
        if not np.issubdtype(train_y.dtype, np.number):
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            train_y = le.fit_transform(train_y)
            test_y = le.transform(test_y)

        print(f"  加载成功: 特征维度 {train_x.shape[1]}, 标签列 '{label_col}'")
    else:
        # 方案 B: 递归探测 TXT 格式 (兼容 HAR/raw/ 等嵌套结构)
        extract_path = None
        for cand in ["HAR", "HAR1", "UCI HAR Dataset", "UCI-HAR-Dataset"]:
            base_path = os.path.join(data_dir, cand)
            if not os.path.exists(base_path):
                continue
            for root, _, files in os.walk(base_path):
                if "X_train.txt" in files:
                    extract_path = root
                    break
            if extract_path:
                break

        if not extract_path:
            raise FileNotFoundError(f"在 {data_dir} 下未找到有效的 HAR 数据集。请确保已解压或存在 HAR1 文件夹。")

        print(f"\n[UCI-HAR] 检测到标准 TXT 格式: {extract_path}")
        train_x = pd.read_csv(os.path.join(extract_path, "X_train.txt"), sep=r'\s+', header=None).values
        train_y = pd.read_csv(os.path.join(extract_path, "y_train.txt"), sep=r'\s+', header=None).values.flatten()
        
        # 自动定位测试集目录
        test_path = extract_path.replace("train", "test")
        if not os.path.exists(os.path.join(test_path, "X_test.txt")):
            test_path = os.path.join(os.path.dirname(extract_path), "test")
            
        test_x = pd.read_csv(os.path.join(test_path, "X_test.txt"), sep=r'\s+', header=None).values
        test_y = pd.read_csv(os.path.join(test_path, "y_test.txt"), sep=r'\s+', header=None).values.flatten()

    # 统一处理标签: 确保是从 0 开始的整数
    train_y, test_y = train_y.astype(int), test_y.astype(int)
    if train_y.min() == 1:
        train_y -= 1
        test_y -= 1
    
    train_ds = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    test_ds = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
    
    return DataLoader(train_ds, batch_size=64, shuffle=True), DataLoader(test_ds, batch_size=64)

# ============================================================================
# 训练逻辑
# ============================================================================

def normalize_dataset_name(name: str) -> str:
    return name.strip().lower().replace('_', '-').replace(' ', '-')


def parse_args():
    parser = argparse.ArgumentParser(description='Train MLP on one or more benchmark datasets.')
    parser.add_argument(
        '--datasets',
        nargs='+',
        default=None,
        help='要训练的数据集名称，例如 Iris Fashion-MNIST UCI-HAR Letter-Recognition。默认运行所有可用数据集。'
    )
    return parser.parse_args()


def train_task(name, model, train_loader, test_loader, epochs, lr, device, logger):
    """训练单个任务"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1) if epochs > 10 else None
    criterion = nn.CrossEntropyLoss()
    
    logger.print(f"\n{'='*20} 正在训练数据集: {name} {'='*20}")
    logger.log_model_info(model, f"Standard MLP for {name}")
    
    best_acc = 0.0
    final_train_acc = 0.0
    start_time = time.time()
    
    for epoch in range(1, epochs + 1):
        logger.log_epoch_start(epoch, epochs)
        model.train()
        total_loss, correct, total = 0, 0, 0
        pbar = tqdm(train_loader, desc=f'Training {name}', leave=False)
        for data, target in pbar:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            pbar.set_postfix({'loss': f'{total_loss/len(train_loader):.4f}', 'acc': f'{100.*correct/total:.2f}%'})
        
        train_acc = 100. * correct / total
        final_train_acc = train_acc
        
        model.eval()
        val_correct, val_total, val_loss = 0, 0, 0.0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item()
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_loss /= len(test_loader)
        best_acc = max(best_acc, val_acc)
        
        logger.log_epoch_result(epoch, total_loss/len(train_loader), train_acc, val_loss, val_acc, val_acc >= best_acc)
        
        if scheduler:
            scheduler.step()
    
    logger.log_final_result(best_acc, final_train_acc, time.time() - start_time)

def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # 确定 data 目录路径 (与 DeviceNeuron 同级)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_dir = os.path.join(project_root, 'data')
    
    # 定义不同数据集的参数: (名称, 加载器, 输入维度, 隐藏层结构, 输出类别数, Epochs, LR)
    tasks = [
        ('Iris', get_iris_loader, 4, [32, 16], 3, 20, 0.01),
        ('Fashion-MNIST', get_fashion_mnist_loader, 784, [512, 256, 128], 10, 20, 0.001),
        ('UCI-HAR', get_uci_har_loader, 561, [256, 128, 64], 6, 20, 0.001),
        ('Letter-Recognition', get_letter_recognition_loader, 16, [256, 128], 26, 20, 0.001),
        ('ISOLET', get_isolet_loader, 617, [512, 256, 128], 26, 20, 0.001),
    ]

    available_task_names = [task[0] for task in tasks]
    if args.datasets:
        selected_keys = set()
        name_aliases = {
            'iris': 'iris',
            'fashion-mnist': 'fashion-mnist',
            'fashion': 'fashion-mnist',
            'mnist-fashion': 'fashion-mnist',
            'uci-har': 'uci-har',
            'har': 'uci-har',
            'letter-recognition': 'letter-recognition',
            'letter': 'letter-recognition',
            'uci-letter': 'letter-recognition',
            'letter-recog': 'letter-recognition',
            'isolet': 'isolet',
            'uci-isolet': 'isolet',
        }
        for raw_name in args.datasets:
            key = normalize_dataset_name(raw_name)
            selected_keys.add(name_aliases.get(key, key))

        tasks = [task for task in tasks if normalize_dataset_name(task[0]) in selected_keys]
        if not tasks:
            raise ValueError(
                f"未找到匹配的任务：{args.datasets}。可用任务为：{', '.join(available_task_names)}"
            )

    for name, loader_fn, in_dim, hiddens, out_dim, epochs, lr in tasks:
        logger = create_benchmark_logger(model_type=f"MLP_{name}", dataset=name)
        try:
            train_loader, test_loader = loader_fn(data_dir)
            model = DiverseMLP(in_dim, hiddens, out_dim)
            train_task(name, model, train_loader, test_loader, epochs, lr, device, logger)
        except Exception as e:
            logger.print(f"训练 {name} 时出错: {e}")
        finally:
            logger.close()

if __name__ == "__main__":
    main()