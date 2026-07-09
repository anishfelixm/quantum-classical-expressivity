import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, f1_score
import numpy as np
import argparse
from tqdm import tqdm
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data')))
from medmnist_loader import get_medmnist_loaders

def extract_features(loader, device):
    """Passes images through ResNet18 Layer 3 to get 256-D features."""
    resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    backbone = nn.Sequential(*list(resnet.children())[:-3], nn.AdaptiveAvgPool2d((1, 1))).to(device)
    backbone.eval()
    
    features, labels = [], []
    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="Extracting Features", leave=False):
            out = backbone(inputs.to(device))
            out = torch.flatten(out, 1)
            features.append(out.cpu().numpy())
            labels.append(targets.squeeze().numpy())
            
    return np.vstack(features), np.concatenate(labels)

def run_pca_baseline(dataset, fraction, bottleneck_dim=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[*] Running PCA + SVM Baseline on {dataset} (Fraction: {fraction})")

    train_loader, val_loader, test_loader = get_medmnist_loaders(
        dataset_name=dataset, train_frac=fraction, batch_size=32, data_root="/home/jovyan/qml_exp_2026/data_cache"
    )

    # 1. Extract 256-dimensional features
    print("[*] Extracting ResNet Features...")
    X_train, y_train = extract_features(train_loader, device)
    X_test, y_test = extract_features(test_loader, device)

    # 2. Run PCA Compression
    print(f"[*] Compressing from 256 to {bottleneck_dim} dimensions using PCA...")
    pca = PCA(n_components=bottleneck_dim)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    variance_retained = np.sum(pca.explained_variance_ratio_)
    print(f"[*] Variance Retained in top {bottleneck_dim} components: {variance_retained*100:.2f}%")

    # 3. Train SVM on compressed features
    print("[*] Training SVM Classifier...")
    svm = SVC(kernel='rbf', class_weight='balanced')
    svm.fit(X_train_pca, y_train)

    # 4. Evaluate
    y_pred = svm.predict(X_test_pca)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    acc = np.mean(y_pred == y_test)
    
    print(f"\n[PCA + SVM RESULTS] | Accuracy: {acc*100:.2f}% | Macro-F1: {macro_f1:.4f}")
    return macro_f1

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="bloodmnist")
    parser.add_argument("--fraction", type=float, default=0.1)
    args = parser.parse_args()
    
    run_pca_baseline(args.dataset, args.fraction)
