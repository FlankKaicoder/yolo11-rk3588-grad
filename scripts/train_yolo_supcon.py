import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms

from ultralytics import YOLO  # 引入 YOLO 库

# =========================
# 路径配置 (和你之前完全一样)
# =========================
DATA_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
SAVE_PATH = "/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_supcon_yolo11n_best.pth"

# =========================
# 超参数 (使用之前跑出的防过拟合最佳配置)
# =========================
IMG_SIZE = 224
BATCH_SIZE = 64
STAGE1_EPOCHS = 150
STAGE2_EPOCHS = 40
LR_STAGE1 = 1e-3  # 保护预训练权重，文火慢炖
LR_STAGE2 = 1e-3
NUM_WORKERS = 8
TEMPERATURE = 0.07

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================
# 数据增强 (与 ResNet18 完全公平对比)
# =========================
class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


def build_transforms():
    train_tf_supcon = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
        ]
    )
    train_tf_cls = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
        ]
    )
    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return train_tf_supcon, train_tf_cls, val_tf


def build_weighted_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_count = Counter(labels)
    sample_weights = [1.0 / class_count[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)


# =========================
# 🌟 核心魔改：YOLO11 SupCon 模型包装器
# =========================
class YOLOSupCon(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()

        # 1. 加载官方预训练的 YOLO11n-cls 模型
        print("正在加载 YOLO11n-cls 作为 Backbone...")
        yolo = YOLO("yolo11n-cls.pt")
        self.encoder = yolo.model

        # 2. 获取分类头 (Ultralytics 的分类头在 model[-1])
        cls_head = self.encoder.model[-1]

        # 3. 剥离最后的全连接层，直接暴露出特征向量
        # YOLO11n-cls 的特征维度通常是 1024
        in_features = cls_head.linear.in_features
        cls_head.linear = nn.Identity()

        # 4. SupCon 的聚类投影头 (Stage 1 用)
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim)
        )

        # 5. 精细分类头，加入 Dropout 防过拟合 (Stage 2 用)
        self.classifier = nn.Sequential(nn.Dropout(p=0.5), nn.Linear(in_features, num_classes))

    def forward(self, x, mode="supcon"):
        # YOLO forward 会直接返回替换为 Identity 后的特征向量
        feat = self.encoder(x)
        if isinstance(feat, (tuple, list)):
            feat = feat[0]
        if mode == "supcon":
            proj = self.projector(feat)
            return F.normalize(proj, dim=1)
        elif mode == "cls":
            return self.classifier(feat)


# =========================
# SupCon 官方 Loss
# =========================
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
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
        logits_mask = torch.ones_like(mask).fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        loss = -mean_log_prob_pos.view(anchor_count, bsz).mean()
        return loss


# =========================
# 主训练循环 (与 ResNet18 逻辑完全一致，保证公平)
# =========================
def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    tf_supcon, tf_cls, tf_val = build_transforms()

    val_dataset = datasets.ImageFolder(VAL_DIR, transform=tf_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    class_names = val_dataset.classes

    model = YOLOSupCon(num_classes=len(class_names)).to(DEVICE)

    # === STAGE 1 ===
    print("\n" + "=" * 50)
    print("🚀 STAGE 1: YOLOv11 + SupCon Pre-training")
    print("=" * 50)

    dataset_stage1 = datasets.ImageFolder(TRAIN_DIR, transform=TwoCropTransform(tf_supcon))
    sampler_stage1 = build_weighted_sampler(dataset_stage1)
    loader_stage1 = DataLoader(
        dataset_stage1, batch_size=BATCH_SIZE, sampler=sampler_stage1, num_workers=NUM_WORKERS, drop_last=True
    )

    criterion_supcon = SupConLoss(temperature=TEMPERATURE)
    optimizer_stage1 = optim.SGD(
        list(model.encoder.parameters()) + list(model.projector.parameters()),
        lr=LR_STAGE1,
        momentum=0.9,
        weight_decay=1e-4,
    )
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=STAGE1_EPOCHS)

    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in loader_stage1:
            x1, x2 = images[0].to(DEVICE), images[1].to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer_stage1.zero_grad()
            proj1 = model(x1, mode="supcon")
            proj2 = model(x2, mode="supcon")
            features = torch.stack([proj1, proj2], dim=1)

            loss = criterion_supcon(features, labels)
            loss.backward()
            optimizer_stage1.step()
            total_loss += loss.item()

        scheduler_stage1.step()
        if (epoch + 1) % 10 == 0:
            print(f"Stage 1 - Epoch [{epoch + 1}/{STAGE1_EPOCHS}], Loss: {total_loss / len(loader_stage1):.4f}")

    # === STAGE 2 ===
    print("\n" + "=" * 50)
    print("🎯 STAGE 2: 全网微调 Fine-Tuning")
    print("=" * 50)

    dataset_stage2 = datasets.ImageFolder(TRAIN_DIR, transform=tf_cls)
    sampler_stage2 = build_weighted_sampler(dataset_stage2)
    loader_stage2 = DataLoader(dataset_stage2, batch_size=BATCH_SIZE, sampler=sampler_stage2, num_workers=NUM_WORKERS)

    criterion_ce = nn.CrossEntropyLoss()

    # 差分学习率：骨干网络微调极小，分类头正常
    optimizer_stage2 = optim.AdamW(
        [
            {"params": model.encoder.parameters(), "lr": 1e-5},
            {"params": model.classifier.parameters(), "lr": LR_STAGE2},
        ],
        weight_decay=1e-2,
    )

    best_val_acc = 0.0
    for epoch in range(STAGE2_EPOCHS):
        model.train()
        _train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in loader_stage2:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer_stage2.zero_grad()

            logits = model(imgs, mode="cls")
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizer_stage2.step()

            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += imgs.size(0)

        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs, mode="cls")
                val_correct += (logits.argmax(dim=1) == labels).sum().item()
                val_total += imgs.size(0)

        val_acc = val_correct / val_total
        print(
            f"Stage 2 - Epoch [{epoch + 1}/{STAGE2_EPOCHS}] | Train Acc: {train_correct / train_total:.4f} | Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print(f"\n🏆 最高验证集准确率: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()
