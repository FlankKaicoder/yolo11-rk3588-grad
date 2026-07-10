from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import datasets, models, transforms

# =========================
# Config
# =========================
SEED = 42
DATA_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_patch")
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Stage 1: contrastive pretrain
STAGE1_EPOCHS = 40
STAGE1_LR = 1e-3
PROJ_DIM = 128
TEMPERATURE = 0.1

# Stage 2: linear probe
STAGE2_EPOCHS = 20
STAGE2_LR = 1e-3

# Stage 3: optional CE fine-tune
STAGE3_EPOCHS = 10
STAGE3_BACKBONE_LR = 1e-4
STAGE3_FC_LR = 1e-3

SAVE_DIR = Path("/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
STAGE1_PATH = SAVE_DIR / "patch_resnet18_stage1_dcl_last.pth"
STAGE2_PATH = SAVE_DIR / "patch_resnet18_stage2_linearprobe_best.pth"
STAGE3_PATH = SAVE_DIR / "patch_resnet18_stage3_finetune_best.pth"

USE_ALL_CLASSES_PER_BATCH = True  # current dataset has 4 classes; use all 4 in each batch


# =========================
# Utilities
# =========================


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# Dataset
# =========================
class ContrastiveImageFolder(Dataset):
    def __init__(self, root: Path, weak_transform, zoom_transform):
        self.base = datasets.ImageFolder(root)
        self.weak_transform = weak_transform
        self.zoom_transform = zoom_transform
        self.samples = self.base.samples
        self.classes = self.base.classes
        self.targets = [s[1] for s in self.samples]
        self.loader = self.base.loader

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self.loader(path)
        weak = self.weak_transform(img)
        zoom = self.zoom_transform(img)
        return weak, zoom, label


class BalancedBatchSampler(Sampler):
    """Build batches with n_classes_per_batch classes and n_samples_per_class samples. Uses replacement when a class has
    insufficient remaining samples.
    """

    def __init__(self, labels, batch_size: int, n_classes_per_batch: int | None = None):
        self.labels = np.array(labels)
        self.unique_classes = sorted(np.unique(self.labels).tolist())
        self.n_classes = len(self.unique_classes)
        self.batch_size = batch_size
        self.n_classes_per_batch = self.n_classes if n_classes_per_batch is None else n_classes_per_batch

        if self.batch_size % self.n_classes_per_batch != 0:
            raise ValueError(
                f"BATCH_SIZE={self.batch_size} must be divisible by n_classes_per_batch={self.n_classes_per_batch}"
            )
        self.n_samples_per_class = self.batch_size // self.n_classes_per_batch
        self.class_to_indices = {c: np.where(self.labels == c)[0].tolist() for c in self.unique_classes}
        self.num_batches = max(1, len(labels) // self.batch_size)

    def __iter__(self):
        for _ in range(self.num_batches):
            chosen_classes = random.sample(self.unique_classes, self.n_classes_per_batch)
            batch = []
            for c in chosen_classes:
                candidates = self.class_to_indices[c]
                if len(candidates) >= self.n_samples_per_class:
                    picked = random.sample(candidates, self.n_samples_per_class)
                else:
                    picked = random.choices(candidates, k=self.n_samples_per_class)
                batch.extend(picked)
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_batches


# =========================
# Transforms
# =========================


def build_transforms():
    # Keep weak view mild, because patches are already local
    weak_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02)], p=0.5
            ),
            transforms.ToTensor(),
        ]
    )

    # Zoom-positive view inspired by Learn From Zoom: slightly tighter crop around local region
    zoom_tf = transforms.Compose(
        [
            transforms.RandomResizedCrop(IMG_SIZE, scale=(0.70, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.05, hue=0.02)], p=0.5
            ),
            transforms.ToTensor(),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]
    )
    return weak_tf, zoom_tf, eval_tf


# =========================
# Model
# =========================
class ResNet18DCL(nn.Module):
    def __init__(self, num_classes: int, proj_dim: int = 128):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.feat_dim = feat_dim
        self.proj_head = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feat_dim, proj_dim),
        )
        self.classifier = nn.Linear(feat_dim, num_classes)

    def forward_features(self, x):
        return self.encoder(x)

    def forward_project(self, x):
        feat = self.forward_features(x)
        proj = self.proj_head(feat)
        proj = F.normalize(proj, dim=1)
        return proj

    def forward_classify(self, x):
        feat = self.forward_features(x)
        logits = self.classifier(feat)
        return logits, feat


# =========================
# Loss: Supervised DCL
# =========================
class SupervisedDCLLoss(nn.Module):
    """Simple supervised decoupled contrastive loss. For each anchor, all same-label samples (except self) are
    positives. Denominator contains only negatives, decoupled from positives.
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor):
        z = F.normalize(z, dim=1)
        sim = torch.matmul(z, z.T) / self.temperature  # [N, N]

        labels = labels.view(-1, 1)
        same = torch.eq(labels, labels.T).to(z.device)
        eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
        pos_mask = same & (~eye)
        neg_mask = ~same

        losses = []
        for i in range(z.size(0)):
            pos_logits = sim[i][pos_mask[i]]
            neg_logits = sim[i][neg_mask[i]]
            if pos_logits.numel() == 0 or neg_logits.numel() == 0:
                continue
            neg_lse = torch.logsumexp(neg_logits, dim=0)
            loss_i = (-pos_logits + neg_lse).mean()
            losses.append(loss_i)

        if len(losses) == 0:
            return sim.sum() * 0.0
        return torch.stack(losses).mean()


# =========================
# Stage 1: DCL Pretrain
# =========================
def train_stage1(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_num = 0

    for weak, zoom, labels in loader:
        weak = weak.to(DEVICE)
        zoom = zoom.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        z1 = model.forward_project(weak)
        z2 = model.forward_project(zoom)

        z = torch.cat([z1, z2], dim=0)
        y = torch.cat([labels, labels], dim=0)
        loss = criterion(z, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * weak.size(0)
        total_num += weak.size(0)

    return total_loss / total_num


# =========================
# Stage 2/3: Classification
# =========================


def train_one_epoch_ce(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()
        logits, _ = model.forward_classify(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate_ce(model, loader, criterion, class_names, title=""):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_labels = []
    all_preds = []

    for imgs, labels in loader:
        imgs = imgs.to(DEVICE)
        labels = labels.to(DEVICE)

        logits, _ = model.forward_classify(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())

    macro_f1 = f1_score(all_labels, all_preds, average="macro")

    if title:
        print(f"\n[{title}] Classification Report:")
    else:
        print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

    return {
        "loss": total_loss / total_num,
        "acc": total_correct / total_num,
        "macro_f1": macro_f1,
    }


# =========================
# Build loaders
# =========================
def build_datasets_and_loaders():
    weak_tf, zoom_tf, eval_tf = build_transforms()

    train_contrastive = ContrastiveImageFolder(TRAIN_DIR, weak_tf, zoom_tf)
    train_eval = datasets.ImageFolder(TRAIN_DIR, transform=eval_tf)
    val_eval = datasets.ImageFolder(VAL_DIR, transform=eval_tf)

    print("Train classes:", train_eval.classes)
    print("Val classes:", val_eval.classes)
    if train_eval.classes != val_eval.classes:
        raise ValueError("Train/Val class names are inconsistent!")

    labels = train_contrastive.targets
    class_count = Counter(labels)
    print("\nTrain class counts:")
    for cls_idx, cnt in sorted(class_count.items()):
        print(f"class_idx={cls_idx}, class_name={train_eval.classes[cls_idx]}, count={cnt}")

    n_classes_per_batch = len(train_eval.classes) if USE_ALL_CLASSES_PER_BATCH else None
    balanced_batch_sampler = BalancedBatchSampler(
        labels=labels,
        batch_size=BATCH_SIZE,
        n_classes_per_batch=n_classes_per_batch,
    )

    stage1_loader = DataLoader(
        train_contrastive,
        batch_sampler=balanced_batch_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    train_loader = DataLoader(
        train_eval,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_eval,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    return train_eval.classes, stage1_loader, train_loader, val_loader


# =========================
# Main
# =========================
def main():
    set_seed(SEED)

    class_names, stage1_loader, train_loader, val_loader = build_datasets_and_loaders()
    num_classes = len(class_names)

    model = ResNet18DCL(num_classes=num_classes, proj_dim=PROJ_DIM).to(DEVICE)
    dcl_criterion = SupervisedDCLLoss(temperature=TEMPERATURE)
    ce_criterion = nn.CrossEntropyLoss()

    print("\n===== Stage 1: DCL pretrain with zoom-positive =====")
    print(f"Device: {DEVICE}")
    print(f"BATCH_SIZE: {BATCH_SIZE}, PROJ_DIM: {PROJ_DIM}, TEMPERATURE: {TEMPERATURE}")

    optimizer_stage1 = optim.AdamW(
        list(model.encoder.parameters()) + list(model.proj_head.parameters()),
        lr=STAGE1_LR,
    )

    for epoch in range(STAGE1_EPOCHS):
        train_loss = train_stage1(model, stage1_loader, dcl_criterion, optimizer_stage1)
        print(f"[Stage1][Epoch {epoch + 1:03d}/{STAGE1_EPOCHS}] DCL Loss {train_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": class_names,
        },
        STAGE1_PATH,
    )
    print(f"Saved Stage1 checkpoint to: {STAGE1_PATH}")

    print("\n===== Stage 2: freeze encoder, linear probe with CE =====")
    # Freeze encoder and projection head
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.proj_head.parameters():
        p.requires_grad = False

    # Reset classifier
    model.classifier = nn.Linear(model.feat_dim, num_classes).to(DEVICE)

    optimizer_stage2 = optim.AdamW(model.classifier.parameters(), lr=STAGE2_LR)
    best_stage2_f1 = 0.0

    for epoch in range(STAGE2_EPOCHS):
        train_loss, train_acc = train_one_epoch_ce(model, train_loader, ce_criterion, optimizer_stage2)
        val_metrics = evaluate_ce(model, val_loader, ce_criterion, class_names, title=f"Stage2 Epoch {epoch + 1}")

        print(
            f"[Stage2][Epoch {epoch + 1:03d}/{STAGE2_EPOCHS}] "
            f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
            f"Val Loss {val_metrics['loss']:.4f} | Val Acc {val_metrics['acc']:.4f} | "
            f"Val Macro-F1 {val_metrics['macro_f1']:.4f}"
        )

        if val_metrics["macro_f1"] > best_stage2_f1:
            best_stage2_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "best_val_macro_f1": best_stage2_f1,
                    "classes": class_names,
                },
                STAGE2_PATH,
            )
            print(f"Saved Stage2 best to: {STAGE2_PATH}")

    print(f"\n[Stage2] Best Val Macro-F1: {best_stage2_f1:.4f}")

    # Load best Stage2
    ckpt = torch.load(STAGE2_PATH, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])

    if STAGE3_EPOCHS > 0:
        print("\n===== Stage 3: CE fine-tune last block + classifier =====")
        # Unfreeze layer4 and classifier only
        for p in model.encoder.parameters():
            p.requires_grad = False
        for p in model.encoder.layer4.parameters():
            p.requires_grad = True
        for p in model.classifier.parameters():
            p.requires_grad = True

        optimizer_stage3 = optim.AdamW(
            [
                {"params": model.encoder.layer4.parameters(), "lr": STAGE3_BACKBONE_LR},
                {"params": model.classifier.parameters(), "lr": STAGE3_FC_LR},
            ]
        )

        best_stage3_f1 = ckpt["best_val_macro_f1"]

        for epoch in range(STAGE3_EPOCHS):
            train_loss, train_acc = train_one_epoch_ce(model, train_loader, ce_criterion, optimizer_stage3)
            val_metrics = evaluate_ce(model, val_loader, ce_criterion, class_names, title=f"Stage3 Epoch {epoch + 1}")

            print(
                f"[Stage3][Epoch {epoch + 1:03d}/{STAGE3_EPOCHS}] "
                f"Train Loss {train_loss:.4f} | Train Acc {train_acc:.4f} | "
                f"Val Loss {val_metrics['loss']:.4f} | Val Acc {val_metrics['acc']:.4f} | "
                f"Val Macro-F1 {val_metrics['macro_f1']:.4f}"
            )

            if val_metrics["macro_f1"] > best_stage3_f1:
                best_stage3_f1 = val_metrics["macro_f1"]
                torch.save(
                    {
                        "epoch": epoch + 1,
                        "model_state_dict": model.state_dict(),
                        "best_val_macro_f1": best_stage3_f1,
                        "classes": class_names,
                    },
                    STAGE3_PATH,
                )
                print(f"Saved Stage3 best to: {STAGE3_PATH}")

        final_path = STAGE3_PATH if STAGE3_PATH.exists() else STAGE2_PATH
        final_best = best_stage3_f1
    else:
        final_path = STAGE2_PATH
        final_best = best_stage2_f1

    # Final evaluation on best model
    print(f"\nBest final Val Macro-F1: {final_best:.4f}")
    final_ckpt = torch.load(final_path, map_location=DEVICE)
    model.load_state_dict(final_ckpt["model_state_dict"])
    print(f"\nFinal detailed evaluation on val (from {final_path}):")
    _ = evaluate_ce(model, val_loader, ce_criterion, class_names, title="Final")

    print("\nDone.")
    print(f"Stage1 ckpt: {STAGE1_PATH}")
    print(f"Stage2 ckpt: {STAGE2_PATH}")
    if STAGE3_EPOCHS > 0:
        print(f"Stage3 ckpt: {STAGE3_PATH}")


if __name__ == "__main__":
    main()
