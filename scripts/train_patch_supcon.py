import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# =========================
# 路径
# =========================
DATA_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
SAVE_PATH = "/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_supcon_resnet18_best.pth"

# =========================
# 超参数
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4
TEMPERATURE = 0.07
LAMBDA_CE = 0.7
LAMBDA_SUPCON = 0.3
PATIENCE = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class SupConDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, two_crop_transform):
        self.dataset = datasets.ImageFolder(root_dir)
        self.samples = self.dataset.samples
        self.classes = self.dataset.classes
        self.transform = two_crop_transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.dataset.loader(path)
        views = self.transform(img)
        return views[0], views[1], label


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


class SupConResNet18(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.projector = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))

        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        proj = F.normalize(proj, dim=1)
        logits = self.classifier(feat)
        return feat, proj, logits


class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """features: [bsz, n_views, dim] labels: [bsz].
        """
        device = features.device
        bsz = features.shape[0]
        n_views = features.shape[1]

        features = F.normalize(features, dim=2)
        features = features.view(bsz * n_views, -1)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = n_views
        contrast_feature = features
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        logits = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)

        logits_mask = torch.ones_like(mask)
        logits_mask = logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        loss = -mean_log_prob_pos
        loss = loss.view(anchor_count, bsz).mean()

        return loss


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )

    return train_tf, val_tf


def build_weighted_sampler(samples):
    labels = [label for _, label in samples]
    class_count = Counter(labels)

    print("Class count in train set:")
    for cls_idx, cnt in sorted(class_count.items()):
        print(f"class_idx={cls_idx}, count={cnt}")

    sample_weights = [1.0 / class_count[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def train_one_epoch(model, loader, ce_criterion, supcon_criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x1, x2, labels in loader:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        _feat1, proj1, logits1 = model(x1)
        _feat2, proj2, logits2 = model(x2)

        ce_loss = 0.5 * (ce_criterion(logits1, labels) + ce_criterion(logits2, labels))
        features = torch.stack([proj1, proj2], dim=1)
        supcon_loss = supcon_criterion(features, labels)

        loss = LAMBDA_CE * ce_loss + LAMBDA_SUPCON * supcon_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = logits1.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, ce_criterion, class_names):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_labels = []
    all_preds = []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        _, _, logits = model(imgs)
        loss = ce_criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return total_loss / total_num, total_correct / total_num


def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)

    train_tf, val_tf = build_transforms()

    train_dataset = SupConDataset(TRAIN_DIR, TwoCropTransform(train_tf))
    val_dataset = ValDataset(VAL_DIR, val_tf)

    print("Train classes:", train_dataset.classes)
    print("Val classes:", val_dataset.classes)

    sampler = build_weighted_sampler(train_dataset.samples)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = SupConResNet18(num_classes=len(train_dataset.classes)).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss()
    supcon_criterion = SupConLoss(temperature=TEMPERATURE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, ce_criterion, supcon_criterion, optimizer)
        val_loss, val_acc = evaluate(model, val_loader, ce_criterion, train_dataset.classes)

        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f}, Val   Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            no_improve = 0
            torch.save(model.state_dict(), SAVE_PATH)
            print(f"Saved best model to: {SAVE_PATH}")
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"Early stopping triggered. Best Val Acc: {best_val_acc:.4f}")
            break

    print(f"\nBest Val Acc: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
