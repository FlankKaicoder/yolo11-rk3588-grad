import csv
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad")
DATA_ROOT = PROJECT_ROOT / "datasets" / "datasets_patch"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

EXP_NAME = "patch_ce_resnet18_fix_nosampler_ep150_v2"
RUN_DIR = PROJECT_ROOT / "runs" / "patch_cls" / EXP_NAME
CKPT_DIR = RUN_DIR / "checkpoints"
REPORT_DIR = RUN_DIR / "reports"

BEST_ACC_PATH = CKPT_DIR / "best_acc.pth"
BEST_F1_PATH = CKPT_DIR / "best_macro_f1.pth"
LAST_PATH = CKPT_DIR / "last.pth"
CSV_LOG_PATH = RUN_DIR / "metrics.csv"
TXT_LOG_PATH = RUN_DIR / "train.log"
CONFIG_PATH = RUN_DIR / "config.json"


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs():
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)


def build_logger():
    logger = logging.getLogger(EXP_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(TXT_LOG_PATH, mode="w", encoding="utf-8")
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def save_config():
    config = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "lr": LR,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "device": str(DEVICE),
        "exp_name": EXP_NAME,
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "note": "CE-only baseline, 150 epochs, no sampler, no class-balanced CE, same pipeline as other ep150 experiments.",
    }
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)


def build_transforms():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    return train_tf, val_tf


class TrainDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path, _ = self.samples[idx]
        return img, label, path


class ValDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform):
        self.dataset = datasets.ImageFolder(root_dir, transform=transform)
        self.classes = self.dataset.classes
        self.samples = self.dataset.samples

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        path, _ = self.samples[idx]
        return img, label, path


def build_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def get_class_counts_from_samples(samples, num_classes):
    labels = [label for _, label in samples]
    counter = Counter(labels)
    counts = [counter[i] for i in range(num_classes)]
    return counts, counter


def init_csv():
    with open(CSV_LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_loss",
                "val_acc",
                "macro_precision",
                "macro_recall",
                "macro_f1",
                "weighted_f1",
                "carbon_recall",
                "corrosion_recall",
                "missing_coating_recall",
                "missing_material_recall",
                "carbon_f1",
                "corrosion_f1",
                "missing_coating_f1",
                "missing_material_f1",
            ]
        )


def append_csv(row):
    with open(CSV_LOG_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_checkpoint(path, model, optimizer, epoch, metrics, class_names):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "class_names": class_names,
        "exp_name": EXP_NAME,
    }
    torch.save(ckpt, path)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for imgs, labels, _paths in loader:
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
        total_num += labels.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, criterion, class_names, epoch):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    all_labels = []
    all_preds = []
    all_probs = []
    all_paths = []

    for imgs, labels, paths in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)
        preds = logits.argmax(dim=1)

        total_loss += loss.item() * imgs.size(0)
        total_correct += (preds == labels).sum().item()
        total_num += imgs.size(0)

        all_labels.extend(labels.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_paths.extend(list(paths))

    report_dict = classification_report(
        all_labels, all_preds, target_names=class_names, digits=4, output_dict=True, zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds)

    report_save = {
        "epoch": epoch,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "samples": [
            {"path": p, "y_true": int(y), "y_pred": int(pred), "probs": [float(x) for x in prob]}
            for p, y, pred, prob in zip(all_paths, all_labels, all_preds, all_probs)
        ],
    }
    with open(REPORT_DIR / f"epoch_{epoch:03d}_report.json", "w", encoding="utf-8") as f:
        json.dump(report_save, f, indent=2, ensure_ascii=False)

    metrics = {
        "val_loss": total_loss / total_num,
        "val_acc": total_correct / total_num,
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
        "per_class_recall": [report_dict[name]["recall"] for name in class_names],
        "per_class_f1": [report_dict[name]["f1-score"] for name in class_names],
        "cm": cm,
        "report_dict": report_dict,
    }
    return metrics


def main():
    set_seed(SEED)
    ensure_dirs()
    save_config()
    init_csv()
    logger = build_logger()

    train_tf, val_tf = build_transforms()
    train_dataset = TrainDataset(TRAIN_DIR, train_tf)
    val_dataset = ValDataset(VAL_DIR, val_tf)

    logger.info(f"Experiment: {EXP_NAME}")
    logger.info(f"Run dir: {RUN_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Train classes: {train_dataset.classes}")
    logger.info(f"Val classes: {val_dataset.classes}")

    num_classes = len(train_dataset.classes)
    class_counts, counter = get_class_counts_from_samples(train_dataset.samples, num_classes)
    logger.info(f"Train class counts: {class_counts}")
    for cls_idx, cnt in sorted(counter.items()):
        logger.info(f"class_idx={cls_idx}, count={cnt}, class_name={train_dataset.classes[cls_idx]}")

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = build_model(num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    best_val_acc = 0.0
    best_macro_f1 = 0.0

    for epoch in range(1, EPOCHS + 1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        metrics = evaluate(model, val_loader, criterion, train_dataset.classes, epoch)
        scheduler.step()

        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        macro_precision = metrics["macro_precision"]
        macro_recall = metrics["macro_recall"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]
        c_recall, cor_recall, mc_recall, mm_recall = metrics["per_class_recall"]
        c_f1, cor_f1, mc_f1, mm_f1 = metrics["per_class_f1"]

        append_csv(
            [
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                macro_precision,
                macro_recall,
                macro_f1,
                weighted_f1,
                c_recall,
                cor_recall,
                mc_recall,
                mm_recall,
                c_f1,
                cor_f1,
                mc_f1,
                mm_f1,
            ]
        )

        logger.info(
            f"Epoch [{epoch}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
            f"Macro F1: {macro_f1:.4f}, Weighted F1: {weighted_f1:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {time.time() - start:.1f}s"
        )
        logger.info(
            f"Per-class Recall | carbon: {c_recall:.4f}, corrosion: {cor_recall:.4f}, "
            f"missing_coating: {mc_recall:.4f}, missing_material: {mm_recall:.4f}"
        )
        logger.info(
            f"Per-class F1     | carbon: {c_f1:.4f}, corrosion: {cor_f1:.4f}, "
            f"missing_coating: {mc_f1:.4f}, missing_material: {mm_f1:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(BEST_ACC_PATH, model, optimizer, epoch, metrics, train_dataset.classes)
            logger.info(f"Saved best_acc checkpoint to: {BEST_ACC_PATH}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_checkpoint(BEST_F1_PATH, model, optimizer, epoch, metrics, train_dataset.classes)
            logger.info(f"Saved best_macro_f1 checkpoint to: {BEST_F1_PATH}")

        save_checkpoint(LAST_PATH, model, optimizer, epoch, metrics, train_dataset.classes)

    logger.info("Training finished.")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Best Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
