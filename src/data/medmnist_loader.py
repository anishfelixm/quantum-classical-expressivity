import os
import torch
import numpy as np
import medmnist
from torchvision import transforms
from torch.utils.data import DataLoader, Subset
from typing import Tuple

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
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.makedirs(data_root, exist_ok=True)

    # Resolve dataset info to check channels
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    n_channels = info['n_channels']

    # Dynamically build transformations based on image channels
    transform_list = [
        transforms.ToTensor(),
    ]
    
    # Only repeat channels if the image is grayscale
    if n_channels == 1:
        transform_list.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
        
    transform_list.extend([
        transforms.Resize((224, 224), antialias=True),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    data_transform = transforms.Compose(transform_list)

    # Load datasets
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root=data_root)
    val_dataset   = DataClass(split='val', transform=data_transform, download=True, root=data_root)
    test_dataset  = DataClass(split='test', transform=data_transform, download=True, root=data_root)
    
    # Enforce data scarcity regime
    if train_frac < 1.0:
        total_len = len(train_dataset)
        n_samples = max(int(total_len * train_frac), batch_size) 
        
        indices = np.random.choice(total_len, n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"[{dataset_name}] Applied {train_frac*100}% data scarcity. Samples: {n_samples}")
    else:
        print(f"[{dataset_name}] Utilizing full 100% dataset. Samples: {len(train_dataset)}")
    
    # Construct DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test and cache all 6 datasets for the Journal run
    datasets = [
        'breastmnist', 'pneumoniamnist', # The Conference baselines (1-channel)
        'bloodmnist', 'pathmnist',       # New Journal datasets (3-channel)
        'dermamnist', 'octmnist'         # New Journal datasets (3-channel / 1-channel)
    ]
    
    os.makedirs("/home/jovyan/qml_exp_2026/data_cache", exist_ok=True)
    
    for ds in datasets:
        print(f"\n--- Processing {ds} ---")
        train_dl, val_dl, test_dl = get_medmnist_loaders(
            dataset_name=ds, 
            batch_size=16, 
            train_frac=0.1,  # Testing the scarcity fraction
            data_root="/home/jovyan/qml_exp_2026/data_cache"
        )
        
        imgs, labels = next(iter(train_dl))
        print(f"[Success] {ds} Output Shape: {imgs.shape} (Should be Bx3x224x224)")
