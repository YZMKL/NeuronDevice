"""
Crossbar MLP 多数据集训练脚本
支持: crossbar 和 crossbar+deviceactivation 两种模式
数据集: iris, fashion-mnist, uci-har, letter-recognition, satimage, isolet

运行方式:
    cd /home/zhc/Projects/neuromorphic
    python -m DeviceNeuron.examples.mlp_diverse --mode crossbar --dataset all
    python -m DeviceNeuron.examples.mlp_diverse --mode crossbar --dataset all --config real
    python -m DeviceNeuron.examples.mlp_diverse --mode crossbar-device --dataset all --config real
    python -m DeviceNeuron.examples.mlp_diverse --mode crossbar-device --dataset letter-recognition --config real
    python -m DeviceNeuron.examples.mlp_diverse --mode crossbar --dataset satimage --config real
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from DeviceNeuron import (
    UnifiedMLP,
    UnifiedMLPWithDeviceNeuron,
    CONFIG,
    create_real_device_config,
    create_high_precision_config,
    create_fast_test_config,
)
from DeviceNeuron.logger import create_logger


def get_iris_loaders(batch_size=64):
    """加载 Iris 数据集"""
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
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_fashion_mnist_loaders(batch_size=64, data_dir='./data'):
    """加载 Fashion-MNIST 数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])

    train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def get_uci_har_loaders(batch_size=64, data_dir='./data'):
    """加载本地 UCI HAR 数据集"""
    candidate_dirs = [
        os.path.join(data_dir, 'UCI-HAR-Dataset'),
        os.path.join(data_dir, 'UCI HAR Dataset'),
        os.path.join(data_dir, 'HAR'),
        os.path.join(data_dir, 'HAR1'),
    ]

    extract_dir = None
    for cand in candidate_dirs:
        if not os.path.exists(cand):
            continue
        if os.path.exists(os.path.join(cand, 'train', 'X_train.txt')):
            extract_dir = os.path.join(cand, 'train')
            break
        if os.path.exists(os.path.join(cand, 'X_train.txt')):
            extract_dir = cand
            break

    if extract_dir is None:
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到 UCI-HAR 数据集，请检查目录是否包含 UCI-HAR-Dataset 或 UCI HAR Dataset。"
        )

    if os.path.basename(extract_dir) == 'train':
        train_dir = extract_dir
        test_dir = os.path.join(os.path.dirname(extract_dir), 'test')
    else:
        train_dir = extract_dir
        test_dir = os.path.join(extract_dir, 'test')

    train_x = pd.read_csv(os.path.join(train_dir, 'X_train.txt'), sep=r'\s+', header=None).values
    train_y = pd.read_csv(os.path.join(train_dir, 'y_train.txt'), sep=r'\s+', header=None).values.flatten()
    test_x = pd.read_csv(os.path.join(test_dir, 'X_test.txt'), sep=r'\s+', header=None).values
    test_y = pd.read_csv(os.path.join(test_dir, 'y_test.txt'), sep=r'\s+', header=None).values.flatten()

    train_y = train_y.astype(int)
    test_y = test_y.astype(int)
    if train_y.min() == 1:
        train_y -= 1
        test_y -= 1

    train_ds = TensorDataset(torch.FloatTensor(train_x), torch.LongTensor(train_y))
    test_ds = TensorDataset(torch.FloatTensor(test_x), torch.LongTensor(test_y))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_letter_recognition_loaders(batch_size=64, data_dir='./data'):
    """加载 UCI Letter Recognition 数据集"""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler

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
        raise FileNotFoundError(
            f"在 {data_dir} 下未找到 Letter Recognition 文件 'letter-recognition.data'。"
        )

    df = pd.read_csv(data_path, header=None, sep=',', engine='python', skip_blank_lines=True)
    df = df.dropna(how='all')
    if df.shape[1] != 17:
        raise ValueError(
            f"Letter Recognition 数据集应包含 17 列，但发现 {df.shape[1]} 列。"
        )

    X = df.iloc[:, 1:].apply(pd.to_numeric, errors='raise').to_numpy(dtype=np.float32)
    y = LabelEncoder().fit_transform(df.iloc[:, 0].astype(str).values)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_satimage_loaders(batch_size=64, data_dir='./data'):
    """加载 UCI Satimage 数据集"""
    from sklearn.preprocessing import StandardScaler
    from urllib.request import urlretrieve
    from urllib.error import URLError, HTTPError

    candidate_dirs = [
        os.path.join(data_dir, 'Satimage'),
        os.path.join(data_dir, 'satimage'),
        os.path.join(data_dir, 'UCI Satimage'),
        os.path.join(data_dir, 'uci-satimage'),
    ]
    data_dir_found = next((d for d in candidate_dirs if os.path.exists(d)), None)
    if data_dir_found is None:
        data_dir_found = os.path.join(data_dir, 'Satimage')
        os.makedirs(data_dir_found, exist_ok=True)

    train_path = os.path.join(data_dir_found, 'sat.trn')
    test_path = os.path.join(data_dir_found, 'sat.tst')

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"[Satimage] 未找到本地数据，尝试从 UCI 下载到 {data_dir_found} ...")
        try:
            urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.trn', train_path)
            urlretrieve('https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/satimage/sat.tst', test_path)
            print(f"[Satimage] 下载完成: {train_path}, {test_path}")
        except (URLError, HTTPError) as exc:
            raise FileNotFoundError(
                f"无法下载 Satimage 数据集: {exc}. 请检查网络或手动下载到 {data_dir_found}."
            )

    train_df = pd.read_csv(train_path, sep=r'\s+', header=None)
    test_df = pd.read_csv(test_path, sep=r'\s+', header=None)

    if train_df.shape[1] != 37 or test_df.shape[1] != 37:
        raise ValueError(
            f"Satimage 数据集应包含 37 列，但发现训练集 {train_df.shape[1]} 列，测试集 {test_df.shape[1]} 列。"
        )

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(int)
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(int)

    # 检查并修正标签范围：Satimage 标签应为 1-6，转为 0-5
    print(f"[Satimage] 训练集标签范围: {y_train.min()} - {y_train.max()}")
    print(f"[Satimage] 测试集标签范围: {y_test.min()} - {y_test.max()}")

    if y_train.min() == 1 and y_train.max() == 6:
        y_train -= 1
        y_test -= 1
    else:
        # 如果不是 1-6，强制映射到 0-5
        y_train = np.clip(y_train - 1, 0, 5)
        y_test = np.clip(y_test - 1, 0, 5)

    print(f"[Satimage] 修正后训练集标签范围: {y_train.min()} - {y_train.max()}")
    print(f"[Satimage] 修正后测试集标签范围: {y_test.min()} - {y_test.max()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_isolet_loaders(batch_size=64, data_dir='./data'):
    """加载 UCI ISOLET 数据集"""
    from sklearn.preprocessing import StandardScaler
    from urllib.request import urlretrieve
    from urllib.error import URLError, HTTPError
    import zipfile
    import subprocess
    import shutil

    candidate_dirs = [
        os.path.join(data_dir, 'ISOLET'),
        os.path.join(data_dir, 'isolet'),
        os.path.join(data_dir, 'UCI ISOLET'),
        os.path.join(data_dir, 'uci-isolet'),
    ]
    data_dir_found = next((d for d in candidate_dirs if os.path.exists(d)), None)
    if data_dir_found is None:
        data_dir_found = os.path.join(data_dir, 'ISOLET')
        os.makedirs(data_dir_found, exist_ok=True)

    def decompress_z(path):
        target = path[:-2]
        if os.path.exists(target):
            return target
        uncompress = shutil.which('uncompress')
        if uncompress is None:
            raise FileNotFoundError(
                f"未找到 uncompress 工具，无法解压 {path}。请安装 ncompress 或手动将 .Z 文件解压至相同目录。"
            )
        subprocess.run([uncompress, '-f', path], check=True)
        return target

    def find_isolet_files():
        file_map = {}
        for root, _, files in os.walk(data_dir):
            for f in files:
                file_map[f] = os.path.join(root, f)

        combined_train = file_map.get('isolet1+2+3+4.data')
        combined_train_z = file_map.get('isolet1+2+3+4.data.Z')
        combined_zip = file_map.get('isolet1+2+3+4+5.data.zip')
        test_data = file_map.get('isolet5.data')
        test_z = file_map.get('isolet5.data.Z')
        parts = [file_map.get(f'isolet{i}.data') for i in range(1, 5)]
        parts_z = [file_map.get(f'isolet{i}.data.Z') for i in range(1, 5)]

        if combined_train is not None and test_data is not None:
            return [combined_train], test_data
        if combined_train_z is not None:
            combined_train = decompress_z(combined_train_z)
            if test_data is None and test_z is not None:
                test_data = decompress_z(test_z)
            return [combined_train], test_data

        if all(parts) and test_data is not None:
            return parts, test_data
        if all(parts_z) and test_data is not None:
            parts = [decompress_z(p) for p in parts_z]
            return parts, test_data
        if all(parts) and test_z is not None:
            test_data = decompress_z(test_z)
            return parts, test_data
        if all(parts_z) and test_z is not None:
            parts = [decompress_z(p) for p in parts_z]
            test_data = decompress_z(test_z)
            return parts, test_data

        if combined_zip is not None:
            with zipfile.ZipFile(combined_zip, 'r') as zip_ref:
                zip_ref.extractall(data_dir_found)
            return find_isolet_files()

        return None, None

    train_paths, test_file = find_isolet_files()
    if train_paths is None or test_file is None:
        raise FileNotFoundError(
            f"未找到 ISOLET 数据文件。请将 isolet1+2+3+4.data 或 isolet1.data..isolet4.data 和 isolet5.data 放在 {data_dir} 下。"
        )

    train_dfs = [pd.read_csv(path, header=None) for path in train_paths]
    train_df = pd.concat(train_dfs, ignore_index=True)

    if not os.path.exists(test_file):
        raise FileNotFoundError(f"未找到 ISOLET 测试文件: {test_file}")
    test_df = pd.read_csv(test_file, header=None)

    if train_df.shape[1] != 618 or test_df.shape[1] != 618:
        raise ValueError(
            f"ISOLET 数据集应包含 618 列，但发现训练集 {train_df.shape[1]} 列，测试集 {test_df.shape[1]} 列。"
        )

    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values.astype(int) - 1
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values.astype(int) - 1

    print(f"[ISOLET] 训练集标签范围: {y_train.min()} - {y_train.max()}")
    print(f"[ISOLET] 测试集标签范围: {y_test.min()} - {y_test.max()}")

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    train_ds = TensorDataset(torch.FloatTensor(X_train), torch.LongTensor(y_train))
    test_ds = TensorDataset(torch.FloatTensor(X_test), torch.LongTensor(y_test))
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0), DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)


DATASET_CONFIG = {
    'iris': {
        'loader': get_iris_loaders,
        'input_size': 4,
        'hidden_sizes': [32, 16],
        'output_size': 3,
        'img_info': '4 features'
    },
    'fashion-mnist': {
        'loader': get_fashion_mnist_loaders,
        'input_size': 784,
        'hidden_sizes': [512, 256, 128],
        'output_size': 10,
        'img_info': '28x28 grayscale'
    },
    'uci-har': {
        'loader': get_uci_har_loaders,
        'input_size': 561,
        'hidden_sizes': [256, 128, 64],
        'output_size': 6,
        'img_info': '561 features'
    },
    'letter-recognition': {
        'loader': get_letter_recognition_loaders,
        'input_size': 16,
        'hidden_sizes': [256, 128],
        'output_size': 26,
        'img_info': '16 features',
        'learning_rate': 0.001,
        'scheduler_step_size': 6,
        'scheduler_gamma': 0.1,
    },
    'satimage': {
        'loader': get_satimage_loaders,
        'input_size': 36,
        'hidden_sizes': [256, 128, 64],
        'output_size': 6,
        'img_info': '36 features',
        'learning_rate': 0.001,
        'scheduler_step_size': 6,
        'scheduler_gamma': 0.1,    },
    'isolet': {
        'loader': get_isolet_loaders,
        'input_size': 617,
        'hidden_sizes': [512, 256, 128],
        'output_size': 26,
        'img_info': '617 features',
        'learning_rate': 0.001,
        'scheduler_step_size': 6,
        'scheduler_gamma': 0.1,    }
}


def build_model(mode: str, dataset: str, crossbar_config, device_activation=None, k_int_input=1.0, k_int_output=1.0, device_noise=0.0):
    """创建 Crossbar MLP 或 Crossbar + Device Activation MLP"""
    ds_cfg = DATASET_CONFIG[dataset]
    if mode == 'crossbar':
        return UnifiedMLP(
            input_size=ds_cfg['input_size'],
            hidden_sizes=ds_cfg['hidden_sizes'],
            output_size=ds_cfg['output_size'],
            config=crossbar_config,
            activation='relu'
        )

    if mode == 'crossbar-device':
        return UnifiedMLPWithDeviceNeuron(
            input_size=ds_cfg['input_size'],
            hidden_sizes=ds_cfg['hidden_sizes'],
            output_size=ds_cfg['output_size'],
            config=crossbar_config,
            device_activation=device_activation,
            k_int_input=k_int_input,
            k_int_output=k_int_output,
            device_noise=device_noise
        )

    raise ValueError(f'Unknown mode: {mode}')


def train_model(name, model, train_loader, test_loader, cfg, device, logger, epochs, model_desc=None, learning_rate=None, scheduler_step_size=None, scheduler_gamma=None):
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate or cfg.LEARNING_RATE, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=scheduler_step_size if scheduler_step_size is not None else cfg.LR_STEP_SIZE,
        gamma=scheduler_gamma if scheduler_gamma is not None else cfg.LR_GAMMA
    )
    criterion = nn.CrossEntropyLoss()

    logger.print(f"\n{'='*20} 训练任务: {name} {'='*20}")
    logger.log_model_info(model, model_desc or '')
    logger.log_training_params(
        epochs=epochs,
        batch_size=cfg.BATCH_SIZE,
        learning_rate=learning_rate or cfg.LEARNING_RATE,
        optimizer='Adam',
        device=device
    )

    best_acc = 0.0
    final_train_acc = 0.0
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        logger.log_epoch_start(epoch, epochs)

        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * data.size(0)
            _, predicted = output.max(1)
            train_total += target.size(0)
            train_correct += predicted.eq(target).sum().item()

        train_loss /= len(train_loader.dataset)
        train_acc = 100.0 * train_correct / train_total
        final_train_acc = train_acc

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                val_loss += loss.item() * data.size(0)
                _, predicted = output.max(1)
                val_total += target.size(0)
                val_correct += predicted.eq(target).sum().item()

        val_loss /= len(test_loader.dataset)
        val_acc = 100.0 * val_correct / val_total
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            if cfg.SAVE_BEST_MODEL:
                save_name = f'{name.replace(" ", "_")}_{logger.model_type}_best.pth'
                torch.save(model.state_dict(), save_name)

        logger.log_epoch_result(epoch, train_loss, train_acc, val_loss, val_acc, is_best)
        scheduler.step()

    logger.log_final_result(best_acc, final_train_acc, time.time() - start_time)


def parse_args():
    parser = argparse.ArgumentParser(description='MLP Crossbar 多数据集训练脚本')
    parser.add_argument('--mode', type=str, default='crossbar', choices=['crossbar', 'crossbar-device'], help='训练模式: crossbar or crossbar-device')
    parser.add_argument('--dataset', type=str, default='all', choices=['all', 'iris', 'fashion-mnist', 'uci-har', 'letter-recognition', 'satimage', 'isolet'], help='数据集')
    parser.add_argument('--config', type=str, default='default', choices=['default', 'real', 'high', 'fast'], help='配置类型')
    parser.add_argument('--epochs', type=int, default=0, help='训练轮数，默认使用配置中的EPOCHS')
    parser.add_argument('--data-dir', type=str, default='./data', help='数据目录')
    parser.add_argument('--no-log', action='store_true', help='禁用日志文件输出')
    return parser.parse_args()


def main():
    args = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.config == 'real':
        cfg = create_real_device_config()
    elif args.config == 'high':
        cfg = create_high_precision_config()
    elif args.config == 'fast':
        cfg = create_fast_test_config()
    else:
        cfg = CONFIG

    crossbar_config = cfg.get_crossbar_config()
    device_activation = None
    if args.mode == 'crossbar-device':
        device_activation = cfg.get_device_activation()

    datasets_to_run = ['iris', 'fashion-mnist', 'uci-har', 'letter-recognition', 'satimage', 'isolet'] if args.dataset == 'all' else [args.dataset]

    for dataset_name in datasets_to_run:
        ds_cfg = DATASET_CONFIG[dataset_name]
        loader_fn = ds_cfg['loader']

        train_loader, test_loader = loader_fn(batch_size=cfg.BATCH_SIZE, data_dir=args.data_dir) if dataset_name in ['uci-har', 'letter-recognition', 'satimage', 'isolet'] else loader_fn(batch_size=cfg.BATCH_SIZE)

        model = build_model(
            args.mode,
            dataset_name,
            crossbar_config,
            device_activation=device_activation,
            k_int_input=cfg.INTEGRATOR_K_INPUT,
            k_int_output=cfg.INTEGRATOR_K_OUTPUT,
            device_noise=cfg.DEVICE_NOISE_STD
        )

        model_type = f'mlp_{args.mode}'
        logger = create_logger(
            log_dir=os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'logs'),
            model_type=model_type,
            dataset=dataset_name,
            use_device_neuron=(args.mode == 'crossbar-device'),
            enabled=not args.no_log
        )

        logger.print(f'Using device: {device}')
        logger.log_config(cfg)
        logger.log_crossbar_config(crossbar_config)
        logger.log_dataset_info(
            train_size=len(train_loader.dataset),
            test_size=len(test_loader.dataset),
            img_info=ds_cfg['img_info']
        )

        epochs = args.epochs if args.epochs > 0 else cfg.EPOCHS
        model_desc = f"{dataset_name} MLP: {ds_cfg['input_size']} → {ds_cfg['hidden_sizes']} → {ds_cfg['output_size']}"
        if args.mode == 'crossbar-device':
            model_desc += ' (device activation)'

        train_model(
            name=f'{dataset_name} {args.mode}',
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            cfg=cfg,
            device=device,
            logger=logger,
            epochs=epochs,
            model_desc=model_desc,
            learning_rate=ds_cfg.get('learning_rate'),
            scheduler_step_size=ds_cfg.get('scheduler_step_size'),
            scheduler_gamma=ds_cfg.get('scheduler_gamma')
        )
        logger.close()


if __name__ == '__main__':
    main()
