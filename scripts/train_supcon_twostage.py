import os
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix

# =========================
# 路径配置
# =========================
DATA_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
SAVE_PATH = "/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_supcon_resnet18_best.pth"

# =========================
# 超参数 (为 5090 量身定制)
# =========================
IMG_SIZE = 224
# 5090 显存极大，256 或 512 才是 SupCon 的正确打开方式！
BATCH_SIZE = 64 
STAGE1_EPOCHS = 150   # 阶段一：特征聚类
STAGE2_EPOCHS = 40   # 阶段二：线性探测 (Linear Probing)
LR_STAGE1 = 1e-3
LR_STAGE2 = 1e-3
NUM_WORKERS = 8      # 5090 配套的 CPU 通常也不错，拉高数据加载速度
TEMPERATURE = 0.07

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 1. 数据增强与 Dataset
# =========================
class TwoCropTransform:
    """生成两张不同增强视角的图片 (用于 Stage 1)"""
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]

def build_transforms():
    """修复了破坏性裁剪的数据增强"""
    # 对比学习专属强增强
    train_tf_supcon = transforms.Compose([
        transforms.RandomResizedCrop(IMG_SIZE, scale=(0.8,1.0)), # 绝对不要 RandomResizedCrop，会把边缘缺陷切没
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(15), # 增加一点旋转容忍度
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
    ])

    # 线性分类专属弱增强
    train_tf_cls = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    return train_tf_supcon, train_tf_cls, val_tf

# =========================
# 2. 模型定义 (分离 Forward 逻辑)
# =========================
class SupConResNet18(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        # Projection Head (仅在 Stage 1 使用)
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )

        # 线性分类头 (仅在 Stage 2 使用)
        #self.classifier = nn.Linear(in_features, num_classes)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),  # 每次随机断开 50% 的神经元
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x, mode='supcon'):
        """根据不同阶段切换 Forward 路线"""
        feat = self.encoder(x)
        if mode == 'supcon':
            # 阶段一：经过 Projector，输出归一化特征算对比损失
            proj = self.projector(feat)
            return F.normalize(proj, dim=1)
        elif mode == 'cls':
            # 阶段二：直接输入分类器算交叉熵
            return self.classifier(feat)

# =========================
# 3. 损失函数 (官方 SupCon)
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
# 4. 辅助函数
# =========================
def build_weighted_sampler(dataset):
    labels = [label for _, label in dataset.samples]
    class_count = Counter(labels)
    sample_weights = [1.0 / class_count[label] for label in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

# =========================
# 5. 训练主流程
# =========================
def main():
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    tf_supcon, tf_cls, tf_val = build_transforms()

    # 验证集 Dataset & Loader (全程不变)
    val_dataset = datasets.ImageFolder(VAL_DIR, transform=tf_val)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    class_names = val_dataset.classes

    model = SupConResNet18(num_classes=len(class_names)).to(DEVICE)
    
    # ---------------------------------------------------------
    # 🌟 STAGE 1: Supervised Contrastive Pre-training
    # 目标：拉近同类缺陷特征，推开异类，不关注具体分类
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🚀 开始 STAGE 1: Supervised Contrastive Learning")
    print("="*50)
    
    dataset_stage1 = datasets.ImageFolder(TRAIN_DIR, transform=TwoCropTransform(tf_supcon))
    sampler_stage1 = build_weighted_sampler(dataset_stage1)
    loader_stage1 = DataLoader(dataset_stage1, batch_size=BATCH_SIZE, sampler=sampler_stage1, num_workers=NUM_WORKERS, drop_last=True)
    
    criterion_supcon = SupConLoss(temperature=TEMPERATURE)
    # 仅优化 Encoder 和 Projector，忽略 Classifier
    #optimizer_stage1 = optim.AdamW(list(model.encoder.parameters()) + list(model.projector.parameters()), lr=LR_STAGE1)
    optimizer_stage1 = optim.SGD(list(model.encoder.parameters()) + list(model.projector.parameters()), lr=LR_STAGE1, momentum=0.9, weight_decay=1e-4)
    scheduler_stage1 = optim.lr_scheduler.CosineAnnealingLR(optimizer_stage1, T_max=STAGE1_EPOCHS)
    scheduler_stage1.step()

    for epoch in range(STAGE1_EPOCHS):
        model.train()
        total_loss = 0.0
        for images, labels in loader_stage1:
            x1,x2, = images[0],images[1]
            x1,x2,labels = x1.to(DEVICE),x2.to(DEVICE),labels.to(DEVICE)
            optimizer_stage1.zero_grad()
            
            proj1 = model(x1, mode='supcon')
            proj2 = model(x2, mode='supcon')
            features = torch.stack([proj1, proj2], dim=1)
            
            loss = criterion_supcon(features, labels)
            loss.backward()
            optimizer_stage1.step()
            total_loss += loss.item()
            
        print(f"Stage 1 - Epoch [{epoch+1}/{STAGE1_EPOCHS}], SupCon Loss: {total_loss/len(loader_stage1):.4f}")

    # ---------------------------------------------------------
    # 🌟 STAGE 2: Linear Probing (线性探测分类)
    # 目标：冻结训练好的骨干网络，只训练一层线性分类器
    # ---------------------------------------------------------
    print("\n" + "="*50)
    print("🎯 开始 STAGE 2: Linear Classifier Training")
    print("="*50)

    # 冻结 Encoder 和 Projector 的梯度
    #for param in model.encoder.parameters():
    #    param.requires_grad = False
    #for param in model.projector.parameters():
    #    param.requires_grad = False

    dataset_stage2 = datasets.ImageFolder(TRAIN_DIR, transform=tf_cls) # 恢复单图输入
    sampler_stage2 = build_weighted_sampler(dataset_stage2)
    loader_stage2 = DataLoader(dataset_stage2, batch_size=BATCH_SIZE, sampler=sampler_stage2, num_workers=NUM_WORKERS)
    
    criterion_ce = nn.CrossEntropyLoss()
    # 仅优化 Classifier
    #optimizer_stage2 = optim.AdamW(model.classifier.parameters(), lr=LR_STAGE2)
    optimizer_stage2 = optim.AdamW([
        {'params': model.encoder.parameters(), 'lr': 1e-5}, # 极小的学习率保护 Stage1 的成果
        {'params': model.classifier.parameters(), 'lr': 1e-3}
    ], weight_decay=1e-2)

    best_val_acc = 0.0

    for epoch in range(STAGE2_EPOCHS):
        model.train()
        model.encoder.eval() # 确保 BatchNorm 处于 eval 模式
        
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in loader_stage2:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer_stage2.zero_grad()
            
            logits = model(imgs, mode='cls')
            loss = criterion_ce(logits, labels)
            loss.backward()
            optimizer_stage2.step()
            
            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(dim=1) == labels).sum().item()
            train_total += imgs.size(0)

        # 验证评估
        model.eval()
        val_correct, val_total, all_labels, all_preds = 0, 0, [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
                logits = model(imgs, mode='cls')
                preds = logits.argmax(dim=1)
                
                val_correct += (preds == labels).sum().item()
                val_total += imgs.size(0)
                
                all_labels.extend(labels.cpu().numpy().tolist())
                all_preds.extend(preds.cpu().numpy().tolist())

        val_acc = val_correct / val_total
        print(f"Stage 2 - Epoch [{epoch+1}/{STAGE2_EPOCHS}] | Train Acc: {train_correct/train_total:.4f} | Val Acc: {val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)

    print("\n" + "="*50)
    print(f"🏆 训练完成！最高验证集准确率: {best_val_acc:.4f}")
    print("最佳模型验证集分类报告:")
    model.load_state_dict(torch.load(SAVE_PATH))
    model.eval()
    all_labels, all_preds = [], []
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs = imgs.to(DEVICE)
            logits = model(imgs, mode='cls')
            all_labels.extend(labels.numpy().tolist())
            all_preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))
    print("混淆矩阵:\n", confusion_matrix(all_labels, all_preds))

if __name__ == "__main__":
    main()