import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, Normalizer
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt


# =========================
# 基础配置
# =========================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROJECT_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad")
DATA_ROOT = PROJECT_ROOT / "datasets" / "datasets_patch"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

OUT_DIR = PROJECT_ROOT / "runs" / "repr_analysis_50_models_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# 模型路径配置
# 你只需要重点改这里
# =========================
MODEL_CONFIGS = [
    {
        "name": "ce50_bestf1",
        "type": "ce",
        "ckpt": PROJECT_ROOT / "runs" / "patch_cls" / "patch_ce_resnet18_fix_nosampler_v1" / "checkpoints" / "best_macro_f1.pth",
    },
    {
        "name": "bcl50_bestf1",
        "type": "bcl",
        "ckpt": PROJECT_ROOT / "runs" / "patch_cls" / "patch_bcl_resnet18_fix_nosampler_ce_v1" / "checkpoints" / "best_macro_f1.pth",
    },
    {
        # 这里请你自己填第一次两阶段（非经典冻结）的真实路径
        "name": "bcl2stage50_direct_bestf1",
        "type": "bcl",
        "ckpt": PROJECT_ROOT / "runs" / "patch_cls" / "patch_bcl_2stage_resnet18_fix_nosampler_v1" / "checkpoints" / "best_macro_f1.pth",
    },
    {
        "name": "bcl2stage50_freeze_bestf1",
        "type": "bcl",
        "ckpt": PROJECT_ROOT / "runs" / "patch_cls" / "patch_bcl_2stage_freeze_resnet18_fix_nosampler_ep50_v3" / "checkpoints" / "best_macro_f1.pth",
    },
]


def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_transform():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return tf


class PathImageFolder(torch.utils.data.Dataset):
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


class CEOnlyResNet18(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.model = models.resnet18(weights=None)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

    def extract_features(self, x):
        m = self.model
        x = m.conv1(x)
        x = m.bn1(x)
        x = m.relu(x)
        x = m.maxpool(x)

        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)

        x = m.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class BCLResNet18(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone
        self.projector = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, feat_dim)
        )
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        proj = F.normalize(proj, dim=1)
        logits = self.classifier(feat)
        return feat, proj, logits

    def extract_features(self, x):
        return self.encoder(x)


def build_model(model_type, num_classes):
    if model_type == "ce":
        return CEOnlyResNet18(num_classes)
    elif model_type == "bcl":
        return BCLResNet18(num_classes)
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise TypeError(f"Unsupported checkpoint format: {type(ckpt)}")

    # 关键修复：自动适配 CE wrapper 的键名前缀
    model_keys = list(model.state_dict().keys())
    ckpt_keys = list(state_dict.keys())

    model_has_prefix = len(model_keys) > 0 and model_keys[0].startswith("model.")
    ckpt_has_prefix = len(ckpt_keys) > 0 and ckpt_keys[0].startswith("model.")

    if model_has_prefix and not ckpt_has_prefix:
        state_dict = {f"model.{k}": v for k, v in state_dict.items()}
    elif (not model_has_prefix) and ckpt_has_prefix:
        state_dict = {k.replace("model.", "", 1): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return missing, unexpected


@torch.no_grad()
def extract_features(model, loader):
    model.eval()
    features = []
    labels = []
    paths = []

    for imgs, y, p in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        feat = model.extract_features(imgs)
        features.append(feat.cpu().numpy())
        labels.append(y.numpy())
        paths.extend(list(p))

    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)
    return features, labels, paths


def save_feature_npz(out_dir, split_name, feats, labels, paths, class_names):
    np.savez_compressed(
        out_dir / f"{split_name}_features.npz",
        features=feats,
        labels=labels,
        paths=np.array(paths),
        class_names=np.array(class_names),
    )


def run_linear_probe(train_X, train_y, val_X, val_y, class_names, out_dir):
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=5000,
            solver="lbfgs",
            random_state=SEED
        ))
    ])
    clf.fit(train_X, train_y)
    pred = clf.predict(val_X)

    acc = accuracy_score(val_y, pred)
    macro_f1 = f1_score(val_y, pred, average="macro")
    report = classification_report(val_y, pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(val_y, pred)

    with open(out_dir / "linear_probe_report.txt", "w", encoding="utf-8") as f:
        f.write(f"linear_probe_acc: {acc:.6f}\n")
        f.write(f"linear_probe_macro_f1: {macro_f1:.6f}\n\n")
        f.write(report)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / "linear_probe_confusion_matrix.csv", encoding="utf-8-sig"
    )

    return {"linear_probe_acc": acc, "linear_probe_macro_f1": macro_f1}


def run_knn(train_X, train_y, val_X, val_y, class_names, out_dir, k=5):
    clf = Pipeline([
        ("norm", Normalizer(norm="l2")),
        ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean"))
    ])
    clf.fit(train_X, train_y)
    pred = clf.predict(val_X)

    acc = accuracy_score(val_y, pred)
    macro_f1 = f1_score(val_y, pred, average="macro")
    report = classification_report(val_y, pred, target_names=class_names, digits=4, zero_division=0)
    cm = confusion_matrix(val_y, pred)

    with open(out_dir / "knn_report.txt", "w", encoding="utf-8") as f:
        f.write(f"knn_k: {k}\n")
        f.write(f"knn_acc: {acc:.6f}\n")
        f.write(f"knn_macro_f1: {macro_f1:.6f}\n\n")
        f.write(report)

    pd.DataFrame(cm, index=class_names, columns=class_names).to_csv(
        out_dir / "knn_confusion_matrix.csv", encoding="utf-8-sig"
    )

    return {"knn_acc": acc, "knn_macro_f1": macro_f1}


def compute_center_and_intra_stats(features, labels, class_names, out_dir, split_name):
    feats = features.astype(np.float64)
    unique_labels = np.unique(labels)

    centers = {}
    intra_rows = []

    for c in unique_labels:
        class_feats = feats[labels == c]
        center = class_feats.mean(axis=0)
        centers[c] = center

        dists = np.linalg.norm(class_feats - center, axis=1)
        intra_rows.append({
            "split": split_name,
            "class_idx": int(c),
            "class_name": class_names[c],
            "num_samples": len(class_feats),
            "mean_dist_to_center": float(dists.mean()),
            "std_dist_to_center": float(dists.std()),
            "variance_mean": float(np.var(class_feats, axis=0).mean()),
        })

    intra_df = pd.DataFrame(intra_rows)
    intra_df.to_csv(out_dir / f"{split_name}_intra_class_stats.csv", index=False, encoding="utf-8-sig")

    # 类间中心距离矩阵
    center_matrix = []
    for i in unique_labels:
        row = []
        for j in unique_labels:
            dist = np.linalg.norm(centers[i] - centers[j])
            row.append(float(dist))
        center_matrix.append(row)

    center_df = pd.DataFrame(center_matrix, index=class_names, columns=class_names)
    center_df.to_csv(out_dir / f"{split_name}_center_distance_matrix.csv", encoding="utf-8-sig")

    # 汇总指标
    off_diag = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j:
                off_diag.append(center_matrix[i][j])
    mean_inter = float(np.mean(off_diag))
    mean_intra = float(intra_df["mean_dist_to_center"].mean())

    return {
        f"{split_name}_mean_inter_class_center_dist": mean_inter,
        f"{split_name}_mean_intra_class_dist": mean_intra,
    }


def plot_tsne(features, labels, class_names, out_path, title):
    n = len(features)
    perplexity = min(30, max(5, n // 5))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        init="pca",
        learning_rate="auto",
        random_state=SEED
    )
    emb = tsne.fit_transform(features)

    plt.figure(figsize=(8, 6))
    for c, name in enumerate(class_names):
        idx = labels == c
        plt.scatter(emb[idx, 0], emb[idx, 1], s=18, label=name, alpha=0.8)
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=220)
    plt.close()


def analyze_one_model(cfg, train_dataset, val_dataset):
    model_name = cfg["name"]
    model_type = cfg["type"]
    ckpt_path = Path(cfg["ckpt"])

    if not ckpt_path.exists():
        print(f"[WARN] skip {model_name}, checkpoint not found: {ckpt_path}")
        return None

    model_out_dir = OUT_DIR / model_name
    model_out_dir.mkdir(parents=True, exist_ok=True)

    class_names = train_dataset.classes
    num_classes = len(class_names)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    model = build_model(model_type, num_classes).to(DEVICE)
    missing, unexpected = load_checkpoint(model, ckpt_path)
    print(f"[INFO] {model_name}: loaded.")
    if missing:
        print(f"[INFO] {model_name}: missing keys = {missing}")
    if unexpected:
        print(f"[INFO] {model_name}: unexpected keys = {unexpected}")
    # 如果 CE 模型还出现整片 missing/unexpected，直接报错，别继续分析假特征
    if model_type == "ce" and (len(missing) > 10 or len(unexpected) > 10):
        raise RuntimeError(
            f"{model_name} checkpoint did not load correctly. "
            f"Please check key prefix mapping before continuing."
        )

    train_X, train_y, train_paths = extract_features(model, train_loader)
    val_X, val_y, val_paths = extract_features(model, val_loader)

    save_feature_npz(model_out_dir, "train", train_X, train_y, train_paths, class_names)
    save_feature_npz(model_out_dir, "val", val_X, val_y, val_paths, class_names)

    summary = {
        "model_name": model_name,
        "model_type": model_type,
        "ckpt": str(ckpt_path),
    }

    summary.update(run_linear_probe(train_X, train_y, val_X, val_y, class_names, model_out_dir))
    summary.update(run_knn(train_X, train_y, val_X, val_y, class_names, model_out_dir, k=5))
    summary.update(compute_center_and_intra_stats(train_X, train_y, class_names, model_out_dir, "train"))
    summary.update(compute_center_and_intra_stats(val_X, val_y, class_names, model_out_dir, "val"))

    # 先只画 val 特征的 t-SNE，避免图太多
    plot_tsne(
        val_X, val_y, class_names,
        model_out_dir / "val_tsne.png",
        title=f"{model_name} - val t-SNE"
    )

    with open(model_out_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    return summary


def main():
    set_seed(SEED)

    tf = build_transform()
    train_dataset = PathImageFolder(TRAIN_DIR, tf)
    val_dataset = PathImageFolder(VAL_DIR, tf)

    summaries = []
    for cfg in MODEL_CONFIGS:
        s = analyze_one_model(cfg, train_dataset, val_dataset)
        if s is not None:
            summaries.append(s)

    if len(summaries) == 0:
        print("[ERROR] No valid model analyzed. Please check checkpoint paths.")
        return

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(OUT_DIR / "summary_all_models.csv", index=False, encoding="utf-8-sig")
    print(f"[DONE] summary saved to: {OUT_DIR / 'summary_all_models.csv'}")


if __name__ == "__main__":
    main()