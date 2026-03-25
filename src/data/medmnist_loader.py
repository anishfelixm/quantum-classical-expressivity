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
    Loads MedMNIST datasets and adapts them for the ResNet-18 feature extractor.
    
    Preprocessing steps applied:
    1. 1-channel grayscale is repeated to 3-channel RGB.
    2. 28x28 spatial resolution is interpolated to 224x224.
    3. Normalized using standard ImageNet mean and standard deviation.

    Args:
        dataset_name: 'breastmnist' or 'pneumoniamnist'
        batch_size: Batch size for the dataloaders
        train_frac: Fraction of training data to use (e.g., 0.1 for 10% scarcity regime)
        data_root: Directory to cache the downloaded .npz files
        seed: Random seed for deterministic subsampling

    Returns:
        train_loader, val_loader, test_loader
    """
    # Ensure reproducibility for data subsampling
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Ensure data directory exists
    os.makedirs(data_root, exist_ok=True)

    # Define transformations mandated by the pre-trained ResNet-18 backbone
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1)), # 1-channel to 3-channel RGB
        transforms.Resize((224, 224), antialias=True),  # Standardize to ResNet input dimensions
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Resolve the specific MedMNIST dataset class dynamically
    info = medmnist.INFO[dataset_name]
    DataClass = getattr(medmnist, info['python_class'])
    
    # Load the pristine datasets (Downloading directly to the specified data_root)
    train_dataset = DataClass(split='train', transform=data_transform, download=True, root=data_root)
    val_dataset   = DataClass(split='val', transform=data_transform, download=True, root=data_root)
    test_dataset  = DataClass(split='test', transform=data_transform, download=True, root=data_root)
    
    # Enforce severe data scarcity regime if train_frac < 1.0
    if train_frac < 1.0:
        total_len = len(train_dataset)
        n_samples = int(total_len * train_frac)
        n_samples = max(n_samples, batch_size) # Ensure at least one full batch exists
        
        indices = np.random.choice(total_len, n_samples, replace=False)
        train_dataset = Subset(train_dataset, indices)
        print(f"[Data Loader] Applied {train_frac*100}% data scarcity constraint. Training samples: {n_samples}")
    else:
        print(f"[Data Loader] Utilizing full 100% dataset. Training samples: {len(train_dataset)}")
    
    # Construct DataLoaders
    # Note: Validation and Test sets remain unmodified as per Section V.A specifications
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader