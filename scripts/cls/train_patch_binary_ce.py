from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import ImageFile
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet18

ImageFile.LOAD_TRUNCATED_IMAGES = True


def parse_args():
    parser = argparse.ArgumentParser(description="Train binary patch classifier: defect vs background")
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="e.g. /root/autodl-tmp/yolo11-rk3588-grad/datasets/patch_binary_v1",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        required=True,
        help="e.g. /root/autodl-tmp/yolo11-rk3588-grad/runs/patch_binary/ce_resnet18_v1",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--pretrained", action="store_true", default=True)
    parser.add_argument("--early-stop", type=int, default=12, help="early stop patience on macro F1")
    return parser.parse_args()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def build_transforms(img_size: int):
    train_tf = transforms.Compose(
        [
            transforms.Resize((img_size + 32, img_size + 32)),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    val_tf = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return train_tf, val_tf


def build_dataloaders(data_root: Path, img_size: int, batch_size: int, num_workers: int):
    train_dir = data_root / "train"
    val_dir = data_root / "val"

    if not train_dir.exists():
        raise FileNotFoundError(f"train dir not found: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"val dir not found: {val_dir}")

    train_tf, val_tf = build_transforms(img_size)

    train_set = datasets.ImageFolder(train_dir, transform=train_tf)
    val_set = datasets.ImageFolder(val_dir, transform=val_tf)

    if train_set.class_to_idx != val_set.class_to_idx:
        raise RuntimeError(f"class_to_idx mismatch: train={train_set.class_to_idx}, val={val_set.class_to_idx}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )

    return train_set, val_set, train_loader, val_loader


def build_model(num_classes: int, pretrained: bool) -> nn.Module:
    # torchvision 新版 weights 参数比较烦，这里尽量稳一点
    model = None
    if pretrained:
        try:
            from torchvision.models import ResNet18_Weights

            model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            print("[Info] Loaded ImageNet pretrained ResNet18.")
        except Exception as e:
            print(f"[Warn] Failed to load pretrained weights, fallback to random init. Reason: {e}")
            model = resnet18(weights=None)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def compute_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    num_classes: int,
) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)  # rows=true, cols=pred
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics_from_cm(cm: np.ndarray, class_names: list[str]) -> dict:
    len(class_names)
    total = cm.sum()
    correct = np.trace(cm)
    acc = correct / total if total > 0 else 0.0

    per_class = {}
    f1_list = []

    for i, name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[name] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(cm[i, :].sum()),
        }
        f1_list.append(f1)

    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    return {
        "acc": float(acc),
        "macro_f1": macro_f1,
        "per_class": per_class,
    }


def save_confusion_matrix_figure(cm: np.ndarray, class_names: list[str], save_path: Path, title: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_history_plot(history: list[dict], save_dir: Path):
    ensure_dir(save_dir)

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_acc = [h["val_acc"] for h in history]
    val_macro_f1 = [h["val_macro_f1"] for h in history]

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, train_loss, label="train_loss")
    plt.plot(epochs, val_loss, label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "loss_curve.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    plt.plot(epochs, val_acc, label="val_acc")
    plt.plot(epochs, val_macro_f1, label="val_macro_f1")
    plt.xlabel("Epoch")
    plt.ylabel("Score")
    plt.title("Validation Metrics")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_dir / "val_metrics_curve.png", dpi=200)
    plt.close()


def save_history_csv(history: list[dict], save_path: Path):
    if not history:
        return
    ensure_dir(save_path.parent)
    keys = list(history[0].keys())
    with open(save_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(history)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    total = 0

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        total += bs

    return running_loss / max(1, total)


@torch.no_grad()
def evaluate(model, loader, criterion, device, class_names: list[str]):
    model.eval()
    running_loss = 0.0
    total = 0

    y_true, y_pred = [], []

    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(imgs)
        loss = criterion(logits, labels)

        preds = torch.argmax(logits, dim=1)

        bs = imgs.size(0)
        running_loss += loss.item() * bs
        total += bs

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    val_loss = running_loss / max(1, total)
    cm = compute_confusion_matrix(y_true, y_pred, num_classes=len(class_names))
    metrics = compute_metrics_from_cm(cm, class_names)

    return val_loss, cm, metrics


def save_checkpoint(
    save_path: Path,
    model: nn.Module,
    optimizer,
    scheduler,
    epoch: int,
    best_metric: float,
    class_to_idx: dict[str, int],
    history: list[dict],
    args,
):
    ensure_dir(save_path.parent)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "best_metric": best_metric,
            "class_to_idx": class_to_idx,
            "history": history,
            "args": vars(args),
        },
        save_path,
    )


def main():
    args = parse_args()
    set_seed(args.seed)

    data_root = Path(args.data_root)
    save_dir = Path(args.save_dir)
    ckpt_dir = save_dir / "checkpoints"
    plot_dir = save_dir / "plots"
    log_dir = save_dir / "logs"

    ensure_dir(save_dir)
    ensure_dir(ckpt_dir)
    ensure_dir(plot_dir)
    ensure_dir(log_dir)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Info] Using device: {device}")

    train_set, val_set, train_loader, val_loader = build_dataloaders(
        data_root=data_root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    class_to_idx = train_set.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]

    print(f"[Info] class_to_idx: {class_to_idx}")
    print(f"[Info] train samples: {len(train_set)}")
    print(f"[Info] val samples:   {len(val_set)}")

    # ImageFolder按字母排序，通常是 background=0, defect=1
    with open(log_dir / "class_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(class_to_idx, f, ensure_ascii=False, indent=2)

    model = build_model(num_classes=len(class_names), pretrained=args.pretrained)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    best_macro_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    no_improve_epochs = 0

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, cm, metrics = evaluate(model, val_loader, criterion, device, class_names)
        scheduler.step()

        val_acc = metrics["acc"]
        val_macro_f1 = metrics["macro_f1"]

        defect_recall = metrics["per_class"].get("defect", {}).get("recall", -1.0)
        background_precision = metrics["per_class"].get("background", {}).get("precision", -1.0)

        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_acc": round(val_acc, 6),
            "val_macro_f1": round(val_macro_f1, 6),
            "defect_recall": round(defect_recall, 6) if defect_recall >= 0 else "",
            "background_precision": round(background_precision, 6) if background_precision >= 0 else "",
            "lr": round(optimizer.param_groups[0]["lr"], 8),
        }
        history.append(row)

        print(
            f"Epoch [{epoch:03d}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.4f}  macro_f1={val_macro_f1:.4f}  "
            f"defect_recall={defect_recall:.4f}  bg_precision={background_precision:.4f}  "
            f"time={time.time() - epoch_start:.1f}s"
        )

        # 每轮保存 last
        save_checkpoint(
            ckpt_dir / "last.pth",
            model,
            optimizer,
            scheduler,
            epoch,
            best_macro_f1,
            class_to_idx,
            history,
            args,
        )

        # 最佳 macro F1
        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch
            no_improve_epochs = 0

            save_checkpoint(
                ckpt_dir / "best_macro_f1.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_macro_f1,
                class_to_idx,
                history,
                args,
            )

            save_confusion_matrix_figure(
                cm,
                class_names,
                plot_dir / "best_macro_f1_confusion_matrix.png",
                title=f"Best Macro F1 Confusion Matrix (epoch={epoch})",
            )

            with open(log_dir / "best_macro_f1_metrics.json", "w", encoding="utf-8") as f:
                json.dump(metrics, f, ensure_ascii=False, indent=2)
        else:
            no_improve_epochs += 1

        # 最佳 acc
        if val_acc > best_acc:
            best_acc = val_acc
            save_checkpoint(
                ckpt_dir / "best_acc.pth",
                model,
                optimizer,
                scheduler,
                epoch,
                best_acc,
                class_to_idx,
                history,
                args,
            )

        save_history_csv(history, log_dir / "history.csv")
        save_history_plot(history, plot_dir)

        if no_improve_epochs >= args.early_stop:
            print(f"[Info] Early stopping triggered at epoch {epoch}. Best epoch = {best_epoch}")
            break

    total_time = time.time() - start_time
    print(f"\nTraining done in {total_time / 60:.2f} minutes.")
    print(f"Best macro F1: {best_macro_f1:.4f} at epoch {best_epoch}")
    print(f"Best acc:      {best_acc:.4f}")
    print(f"Results saved to: {save_dir}")


if __name__ == "__main__":
    main()
