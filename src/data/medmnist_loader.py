import os
import torch
import numpy as np
import medmnist
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

# -----------------------------------------------------------------------------
# Explicit Normalization Constants
# Exported globally so that Robustness Evaluation can import them
# to perform mathematically sound inverse-normalization before noise injection.
# -----------------------------------------------------------------------------
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]


def get_medmnist_loaders(
    dataset_name: str, 
    batch_size: int = 32, 
    train_frac: float = 1.0, 
    data_root: str = "data",
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Universal MedMNIST loader for QML Journal Expansion.
    Dynamically handles both 1-channel (Grayscale) and 3-channel (RGB) datasets
    while adapting them for the ResNet-18 feature extractor.
    
    Args:
        dataset_name (str): Name of the MedMNIST dataset (e.g., 'breastmnist').
        batch_size (int): Batch size for training and evaluation.
        train_frac (float): Scarcity regime constraint (e.g., 0.01 for 1%).
        data_root (str): Caching directory for downloaded datasets.
        seed (int): Deterministic seed for identical subset extraction.
        
    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: train, val, and test loaders.
    """
    # 1. Enforce strict reproducibility for subset sampling
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(data_root, exist_ok=True)

    # 2. Resolve dataset info to check native image channels
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']

    # 3. Dynamically build transformations based on image channels
    transform_list = [transforms.ToTensor()]
    
    # Only repeat channels if the native image is grayscale (1-channel)
    if n_channels == 1:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
    transform_list.extend([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD)
    ])
    
    data_transform = transforms.Compose(transform_list)

    # 4. Load datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root=data_root)
    val_dataset   = DataClass(split='val', transform=data_transform, download=True, root=data_root)
    test_dataset  = DataClass(split='test', transform=data_transform, download=True, root=data_root)
    
    # 5. Enforce deterministic data scarcity regime
    if train_frac < 1.0:
        total_len = len(train_dataset)
        n_samples = max(int(total_len * train_frac), 2) # Min 2 samples to avoid BatchNorm crash
        
        # If the final batch will have exactly 1 sample, BatchNorm1d will crash.
        # We adjust the subset size by 1 to prevent this without dropping whole batches.
        if n_samples % batch_size == 1:
            if n_samples < total_len:
                n_samples += 1
            else:
                n_samples -= 1
                
        indices = np.random.choice(total_len, n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"[{dataset_name}] Applied {train_frac*100}% data scarcity. Training Samples: {n_samples}")
    else:
        print(f"[{dataset_name}] Utilizing full 100% dataset. Training Samples: {len(train_dataset)}")
    
    # 6. Construct DataLoaders (drop_last=False to preserve all scarce data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    datasets = [
        'breastmnist', 'pneumoniamnist',
        'bloodmnist', 'pathmnist',
    ]
    
    # DYNAMIC PATH RESOLUTION: 
    # Automatically creates 'data_cache' in the project root directory
    PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    CACHE_DIR = os.path.join(PROJECT_ROOT, "data_cache")
    os.makedirs(CACHE_DIR, exist_ok=True)
    
    for ds in datasets:
        print(f"\n--- Processing {ds} ---")
        train_dl, val_dl, test_dl = get_medmnist_loaders(
            dataset_name=ds, 
            batch_size=32, 
            train_frac=0.01,
            data_root=CACHE_DIR
        )
        
        imgs, labels = next(iter(train_dl))
        print(f"[Success] {ds} Output Shape: {imgs.shape} (Expected: Bx3x224x224)")