import os
from collections import Counter
from pathlib import Path

import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, models, transforms

# =========================
# 路径
# =========================
DATA_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"
SAVE_PATH = "/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_paco_resnet18_best.pth"

# =========================
# 超参数
# =========================
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3
NUM_WORKERS = 4
FEAT_DIM = 128
TEMPERATURE = 0.07

LAMBDA_CE = 0.8
LAMBDA_PACO = 0.2

PATIENCE = 10
BETA = 0.999

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class PaCoTrainDataset(torch.utils.data.Dataset):
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


class PaCoResNet18(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.projector = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))

        self.classifier = nn.Linear(in_features, num_classes)

        # PaCo风格：每类一个可学习proxy
        self.proxies = nn.Parameter(torch.randn(num_classes, feat_dim))
        nn.init.xavier_uniform_(self.proxies)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        proj = F.normalize(proj, dim=1)
        logits = self.classifier(feat)
        proxies = F.normalize(self.proxies, dim=1)
        return feat, proj, logits, proxies


def build_transforms():
    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(3),
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


def get_class_counts_from_samples(samples, num_classes):
    labels = [label for _, label in samples]
    counter = Counter(labels)
    counts = [counter[i] for i in range(num_classes)]
    return counts, counter


def build_weighted_sampler(samples):
    labels = [label for _, label in samples]
    class_count = Counter(labels)

    print("Class count in train set:")
    for cls_idx, cnt in sorted(class_count.items()):
        print(f"class_idx={cls_idx}, count={cnt}")

    sample_weights = [1.0 / class_count[label] for label in labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler


def get_class_balanced_weights(class_counts, beta=0.999):
    effective_num = 1.0 - torch.pow(
        torch.tensor(beta, dtype=torch.float32), torch.tensor(class_counts, dtype=torch.float32)
    )
    weights = (1.0 - beta) / (effective_num + 1e-12)
    weights = weights / weights.sum() * len(class_counts)
    return weights


class PaCoProxyLoss(nn.Module):
    """工程可落地的PaCo风格近似版： - 样本embedding与类proxy做对比 - 正类proxy拉近，负类proxy拉远 - 使用class-balanced权重提升尾部类贡献.
    """

    def __init__(self, class_counts, temperature=0.07, beta=0.999):
        super().__init__()
        self.temperature = temperature
        class_weights = get_class_balanced_weights(class_counts, beta=beta)
        self.register_buffer("class_weights", class_weights)

    def forward(self, features, labels, proxies):
        """features: [bsz, feat_dim] labels: [bsz] proxies: [num_classes, feat_dim].
        """
        features = F.normalize(features, dim=1)
        proxies = F.normalize(proxies, dim=1)

        logits = torch.matmul(features, proxies.T) / self.temperature
        log_prob = F.log_softmax(logits, dim=1)

        sample_weights = self.class_weights[labels]
        pos_log_prob = log_prob[torch.arange(labels.size(0), device=labels.device), labels]

        loss = -sample_weights * pos_log_prob
        return loss.mean()


def train_one_epoch(model, loader, ce_criterion, paco_criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for x1, x2, labels in loader:
        x1 = x1.to(DEVICE)
        x2 = x2.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        _, proj1, logits1, proxies = model(x1)
        _, proj2, logits2, proxies = model(x2)

        ce_loss = 0.5 * (ce_criterion(logits1, labels) + ce_criterion(logits2, labels))
        paco_loss = 0.5 * (paco_criterion(proj1, labels, proxies) + paco_criterion(proj2, labels, proxies))

        loss = LAMBDA_CE * ce_loss + LAMBDA_PACO * paco_loss
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

        _, _, logits, _ = model(imgs)
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
    train_dataset = PaCoTrainDataset(TRAIN_DIR, TwoCropTransform(train_tf))
    val_dataset = ValDataset(VAL_DIR, val_tf)

    print("Train classes:", train_dataset.classes)
    print("Val classes:", val_dataset.classes)

    num_classes = len(train_dataset.classes)
    class_counts, _counter = get_class_counts_from_samples(train_dataset.samples, num_classes)
    print("Train class counts:", class_counts)

    sampler = build_weighted_sampler(train_dataset.samples)

    class_weights = get_class_balanced_weights(class_counts, beta=BETA).to(DEVICE)
    print("Class-balanced CE weights:", class_weights)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, sampler=sampler, num_workers=NUM_WORKERS, drop_last=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

    model = PaCoResNet18(num_classes=num_classes, feat_dim=FEAT_DIM).to(DEVICE)
    ce_criterion = nn.CrossEntropyLoss(weight=class_weights)
    paco_criterion = PaCoProxyLoss(class_counts, temperature=TEMPERATURE, beta=BETA).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR)

    best_val_acc = 0.0
    no_improve = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, ce_criterion, paco_criterion, optimizer)
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
