import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

def compute_metrics(y_true, y_pred, pos_idx):
    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        if t == pos_idx and p == pos_idx:
            tp += 1
        elif t != pos_idx and p == pos_idx:
            fp += 1
        elif t != pos_idx and p != pos_idx:
            tn += 1
        elif t == pos_idx and p != pos_idx:
            fn += 1

    acc = (tp + tn) / max(1, tp + tn + fp + fn)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    return {
        "acc": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }

def try_make_resnet18(pretrained):
    if pretrained:
        try:
            weights = models.ResNet18_Weights.DEFAULT
            model = models.resnet18(weights=weights)
            print("[INFO] using ImageNet pretrained ResNet18")
            return model
        except Exception as e:
            print("[WARN] pretrained failed, fallback to weights=None:", e)

    return models.resnet18(weights=None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--pretrained", action="store_true")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.10, hue=0.03),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_set = datasets.ImageFolder(str(Path(args.data_dir) / "train"), transform=train_tf)
    val_set = datasets.ImageFolder(str(Path(args.data_dir) / "val"), transform=val_tf)

    print("[INFO] class_to_idx:", train_set.class_to_idx)
    pos_idx = train_set.class_to_idx["missing_coating"]

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # class weight
    counts = [0, 0]
    for _, y in train_set.samples:
        counts[y] += 1
    total = sum(counts)
    weights = torch.tensor([total / max(1, c) for c in counts], dtype=torch.float32).to(device)
    print("[INFO] train counts:", counts, "weights:", weights.detach().cpu().tolist())

    model = try_make_resnet18(args.pretrained)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1 = -1
    best_path = out_dir / "best_f1.pth"
    last_path = out_dir / "last.pth"

    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        train_n = 0
        train_true, train_pred = [], []

        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * x.size(0)
            train_n += x.size(0)
            pred = logits.argmax(1)

            train_true.extend(y.detach().cpu().tolist())
            train_pred.extend(pred.detach().cpu().tolist())

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_n = 0
        val_true, val_pred = [], []

        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)

                logits = model(x)
                loss = criterion(logits, y)
                pred = logits.argmax(1)

                val_loss += loss.item() * x.size(0)
                val_n += x.size(0)
                val_true.extend(y.detach().cpu().tolist())
                val_pred.extend(pred.detach().cpu().tolist())

        train_metrics = compute_metrics(train_true, train_pred, pos_idx)
        val_metrics = compute_metrics(val_true, val_pred, pos_idx)

        row = {
            "epoch": epoch,
            "train_loss": train_loss / max(1, train_n),
            "val_loss": val_loss / max(1, val_n),
            "train": train_metrics,
            "val": val_metrics,
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)

        print(
            f"Epoch {epoch:03d}/{args.epochs} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_loss={row['val_loss']:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_p={val_metrics['precision']:.4f} "
            f"val_r={val_metrics['recall']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"tp={val_metrics['tp']} fp={val_metrics['fp']} "
            f"tn={val_metrics['tn']} fn={val_metrics['fn']}"
        )

        ckpt = {
            "model": model.state_dict(),
            "class_to_idx": train_set.class_to_idx,
            "pos_idx": pos_idx,
            "epoch": epoch,
            "val_metrics": val_metrics,
            "args": vars(args),
        }

        torch.save(ckpt, last_path)

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            torch.save(ckpt, best_path)
            print(f"[SAVE] best_f1={best_f1:.4f} -> {best_path}")

    with (out_dir / "history.json").open("w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    print("[DONE] best_f1:", best_f1)
    print("[DONE] best_path:", best_path)

if __name__ == "__main__":
    main()
