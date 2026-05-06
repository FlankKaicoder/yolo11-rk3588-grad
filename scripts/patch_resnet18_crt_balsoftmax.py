import os
import random
from pathlib import Path
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix, f1_score


# =========================
# Config
# =========================
DATA_ROOT = Path('/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch')
TRAIN_DIR = DATA_ROOT / 'train'
VAL_DIR = DATA_ROOT / 'val'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Stage 1: representation learning (instance-balanced / natural sampling)
STAGE1_EPOCHS = 50
STAGE1_LR = 1e-3

# Stage 2: classifier retraining (freeze backbone, re-train fc only)
STAGE2_EPOCHS = 20
STAGE2_LR = 5e-3

SEED = 42

SAVE_DIR = Path('/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls')
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_SAVE = SAVE_DIR / 'patch_resnet18_stage1_ce_best.pth'
STAGE2_SAVE = SAVE_DIR / 'patch_resnet18_stage2_crt_balsoftmax_best.pth'


# =========================
# Utils
# =========================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def count_labels(dataset):
    labels = [sample[1] for sample in dataset.samples]
    class_count = Counter(labels)
    counts = [class_count[i] for i in range(len(dataset.classes))]
    return counts


# =========================
# Dataset
# =========================
def build_datasets():
    # Keep transforms close to your original baseline first.
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    train_dataset = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=val_tf)
    return train_dataset, val_dataset


# =========================
# Model
# =========================
class ResNet18WithFeat(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)
        self.feat_dim = feat_dim
        self.num_classes = num_classes

    def forward(self, x, return_feat=False):
        feat = self.backbone(x)
        logits = self.classifier(feat)
        if return_feat:
            return logits, feat
        return logits


# =========================
# Losses
# =========================
class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, samples_per_class):
        super().__init__()
        spc = torch.tensor(samples_per_class, dtype=torch.float32)
        self.register_buffer('log_prior', torch.log(spc.clamp(min=1.0)))

    def forward(self, logits, labels):
        balanced_logits = logits + self.log_prior.unsqueeze(0)
        return nn.functional.cross_entropy(balanced_logits, labels)


# =========================
# Train / Eval
# =========================
def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, class_names, print_detail=True):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_labels = []
    all_preds = []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average='macro')

    if print_detail:
        print('\nClassification Report:')
        print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
        print('Confusion Matrix:')
        print(confusion_matrix(all_labels, all_preds))

    return total_loss / total_num, total_correct / total_num, macro_f1


# =========================
# Stage 1: learn representation
# =========================
def run_stage1(train_dataset, val_dataset):
    print('\n===== Stage 1: train full model with natural sampling + CE =====')
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ResNet18WithFeat(num_classes=len(train_dataset.classes)).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=STAGE1_LR)

    best_f1 = -1.0

    for epoch in range(STAGE1_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, train_dataset.classes, print_detail=False)

        print(f'[Stage1][Epoch {epoch+1:03d}/{STAGE1_EPOCHS}] '
              f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | '
              f'Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val Macro-F1 {val_f1:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_macro_f1': best_f1,
                'classes': train_dataset.classes,
            }, STAGE1_SAVE)
            print(f'Saved Stage1 best to: {STAGE1_SAVE}')

    ckpt = torch.load(STAGE1_SAVE, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'\n[Stage1] Best Val Macro-F1: {ckpt["best_val_macro_f1"]:.4f}')

    print('\n[Stage1] Final detailed evaluation on val:')
    evaluate(model, val_loader, criterion, train_dataset.classes, print_detail=True)
    return model


# =========================
# Stage 2: freeze backbone, retrain classifier only
# =========================
def run_stage2(stage1_model, train_dataset, val_dataset):
    print('\n===== Stage 2: freeze backbone, retrain classifier with Balanced Softmax =====')

    train_counts = count_labels(train_dataset)
    print('Train class counts:', {train_dataset.classes[i]: c for i, c in enumerate(train_counts)})

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = stage1_model

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    # Re-init classifier from scratch for cRT style retraining
    model.classifier = nn.Linear(model.feat_dim, len(train_dataset.classes)).to(DEVICE)

    criterion = BalancedSoftmaxLoss(train_counts).to(DEVICE)
    optimizer = optim.AdamW(model.classifier.parameters(), lr=STAGE2_LR)

    best_f1 = -1.0

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc, val_f1 = evaluate(model, val_loader, criterion, train_dataset.classes, print_detail=False)

        print(f'[Stage2][Epoch {epoch+1:03d}/{STAGE2_EPOCHS}] '
              f'Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | '
              f'Val Loss {val_loss:.4f} | Val Acc {val_acc:.4f} | Val Macro-F1 {val_f1:.4f}')

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'best_val_macro_f1': best_f1,
                'classes': train_dataset.classes,
                'train_counts': train_counts,
            }, STAGE2_SAVE)
            print(f'Saved Stage2 best to: {STAGE2_SAVE}')

    ckpt = torch.load(STAGE2_SAVE, map_location='cpu')
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'\n[Stage2] Best Val Macro-F1: {ckpt["best_val_macro_f1"]:.4f}')

    print('\n[Stage2] Final detailed evaluation on val:')
    evaluate(model, val_loader, criterion, train_dataset.classes, print_detail=True)
    return model


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    train_dataset, val_dataset = build_datasets()
    print('Train classes:', train_dataset.classes)
    print('Val classes:', val_dataset.classes)

    if train_dataset.classes != val_dataset.classes:
        raise ValueError('Train/Val class names are inconsistent!')

    stage1_model = run_stage1(train_dataset, val_dataset)
    _ = run_stage2(stage1_model, train_dataset, val_dataset)

    print('\nDone.')
    print(f'Stage1 ckpt: {STAGE1_SAVE}')
    print(f'Stage2 ckpt: {STAGE2_SAVE}')


if __name__ == '__main__':
    main()

