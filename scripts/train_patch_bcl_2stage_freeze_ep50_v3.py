import csv
import json
import logging
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

# =========================
# 基础配置
# =========================
SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
TOTAL_EPOCHS = 50

STAGE1_EPOCHS = 20  # BCL预训练
PHASE1_EPOCHS = 5  # 只训 classifier
PHASE2_EPOCHS = 10  # 解冻 layer4 + classifier
PHASE3_EPOCHS = 15  # 全网络小学习率微调

assert STAGE1_EPOCHS + PHASE1_EPOCHS + PHASE2_EPOCHS + PHASE3_EPOCHS == TOTAL_EPOCHS

LR_STAGE1 = 1e-3
LR_PHASE1 = 1e-3
LR_PHASE2 = 3e-4
LR_PHASE3 = 1e-4
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4
TEMPERATURE = 0.07

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 路径与实验命名
# =========================
PROJECT_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad")
DATA_ROOT = PROJECT_ROOT / "datasets" / "datasets_patch"
TRAIN_DIR = DATA_ROOT / "train"
VAL_DIR = DATA_ROOT / "val"

EXP_NAME = "patch_bcl_2stage_freeze_resnet18_fix_nosampler_ep50_v3"
RUN_DIR = PROJECT_ROOT / "runs" / "patch_cls" / EXP_NAME
CKPT_DIR = RUN_DIR / "checkpoints"
REPORT_DIR = RUN_DIR / "reports"

STAGE1_BEST_PATH = CKPT_DIR / "stage1_best_bcl_loss.pth"
STAGE1_LAST_PATH = CKPT_DIR / "stage1_last.pth"
BEST_ACC_PATH = CKPT_DIR / "best_acc.pth"
BEST_F1_PATH = CKPT_DIR / "best_macro_f1.pth"
LAST_PATH = CKPT_DIR / "last.pth"

STAGE1_CSV_PATH = RUN_DIR / "stage1_pretrain_metrics.csv"
STAGE2_CSV_PATH = RUN_DIR / "stage2_finetune_metrics.csv"
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
        "total_epochs": TOTAL_EPOCHS,
        "stage1_epochs": STAGE1_EPOCHS,
        "phase1_epochs": PHASE1_EPOCHS,
        "phase2_epochs": PHASE2_EPOCHS,
        "phase3_epochs": PHASE3_EPOCHS,
        "lr_stage1": LR_STAGE1,
        "lr_phase1": LR_PHASE1,
        "lr_phase2": LR_PHASE2,
        "lr_phase3": LR_PHASE3,
        "weight_decay": WEIGHT_DECAY,
        "num_workers": NUM_WORKERS,
        "temperature": TEMPERATURE,
        "device": str(DEVICE),
        "exp_name": EXP_NAME,
        "train_dir": str(TRAIN_DIR),
        "val_dir": str(VAL_DIR),
        "note": (
            "Two-stage BCL, 50 epochs, classic freeze finetune. "
            "For fair comparison with 50-epoch CE-only, 50-epoch single-stage BCL, "
            "and the first 50-epoch non-classic two-stage run. "
            "No sampler, no class-balanced CE, same aug/normalize/backbone/optimizer family."
        ),
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


class TwoCropTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return [self.base_transform(x), self.base_transform(x)]


class BCLTrainDataset(torch.utils.data.Dataset):
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
        v1, v2 = self.transform(img)
        return v1, v2, label, path


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


class BCLResNet18(nn.Module):
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

    def reset_classifier(self):
        in_features = self.classifier.in_features
        out_features = self.classifier.out_features
        self.classifier = nn.Linear(in_features, out_features).to(next(self.parameters()).device)


class BalancedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        _bsz, n_views, _dim = features.shape

        features = F.normalize(features, dim=2)
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        mask = mask.repeat(n_views, n_views)

        logits = torch.matmul(contrast_feature, contrast_feature.T) / self.temperature
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        logits_mask = torch.ones_like(logits)
        logits_mask.fill_diagonal_(0)
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        mask_sum = mask.sum(dim=1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)

        mean_log_prob_pos = (mask * log_prob).sum(dim=1) / mask_sum
        return -mean_log_prob_pos.mean()


def get_class_counts_from_samples(samples, num_classes):
    labels = [label for _, label in samples]
    counter = Counter(labels)
    counts = [counter[i] for i in range(num_classes)]
    return counts, counter


def init_stage1_csv():
    with open(STAGE1_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(["epoch", "stage1_bcl_loss", "lr"])


def append_stage1_csv(row):
    with open(STAGE1_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def init_stage2_csv():
    with open(STAGE2_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(
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


def append_stage2_csv(row):
    with open(STAGE2_CSV_PATH, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def save_checkpoint(path, model, optimizer, epoch, metrics, class_names, stage_name):
    ckpt = {
        "stage": stage_name,
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "metrics": metrics,
        "class_names": class_names,
        "exp_name": EXP_NAME,
    }
    torch.save(ckpt, path)


def freeze_all_encoder(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = False


def unfreeze_layer4_and_classifier(model):
    for p in model.encoder.parameters():
        p.requires_grad = False
    for p in model.encoder.layer4.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = False


def unfreeze_all_for_finetune(model):
    for p in model.encoder.parameters():
        p.requires_grad = True
    for p in model.classifier.parameters():
        p.requires_grad = True
    for p in model.projector.parameters():
        p.requires_grad = False


def get_trainable_params(model):
    return [p for p in model.parameters() if p.requires_grad]


def stage1_train_one_epoch(model, loader, bcl_criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_num = 0

    for x1, x2, labels, _paths in loader:
        x1 = x1.to(DEVICE, non_blocking=True)
        x2 = x2.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        _, proj1, _ = model(x1)
        _, proj2, _ = model(x2)
        features = torch.stack([proj1, proj2], dim=1)
        bcl_loss = bcl_criterion(features, labels)
        bcl_loss.backward()
        optimizer.step()

        total_loss += bcl_loss.item() * labels.size(0)
        total_num += labels.size(0)

    return total_loss / total_num


def stage2_train_one_epoch(model, loader, ce_criterion, optimizer):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_num = 0

    for imgs, labels, _paths in loader:
        imgs = imgs.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        _, _, logits = model(imgs)
        loss = ce_criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * imgs.size(0)
        preds = logits.argmax(dim=1)
        total_correct += (preds == labels).sum().item()
        total_num += labels.size(0)

    return total_loss / total_num, total_correct / total_num


@torch.no_grad()
def evaluate(model, loader, ce_criterion, class_names, epoch):
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

        _, _, logits = model(imgs)
        loss = ce_criterion(logits, labels)
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
    init_stage1_csv()
    init_stage2_csv()
    logger = build_logger()

    train_tf, val_tf = build_transforms()

    stage1_dataset = BCLTrainDataset(TRAIN_DIR, TwoCropTransform(train_tf))
    stage2_train_dataset = TrainDataset(TRAIN_DIR, train_tf)
    val_dataset = ValDataset(VAL_DIR, val_tf)

    logger.info(f"Experiment: {EXP_NAME}")
    logger.info(f"Run dir: {RUN_DIR}")
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Train classes: {stage2_train_dataset.classes}")
    logger.info(f"Val classes: {val_dataset.classes}")

    num_classes = len(stage2_train_dataset.classes)
    class_counts, counter = get_class_counts_from_samples(stage2_train_dataset.samples, num_classes)
    logger.info(f"Train class counts: {class_counts}")
    for cls_idx, cnt in sorted(counter.items()):
        logger.info(f"class_idx={cls_idx}, count={cnt}, class_name={stage2_train_dataset.classes[cls_idx]}")

    stage1_loader = DataLoader(
        stage1_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True, drop_last=True
    )

    stage2_train_loader = DataLoader(
        stage2_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    model = BCLResNet18(num_classes).to(DEVICE)

    # =========================
    # Stage1: BCL pretraining
    # =========================
    logger.info("========== Stage1: BCL pretraining ==========")
    bcl_criterion = BalancedContrastiveLoss(TEMPERATURE).to(DEVICE)
    stage1_optimizer = optim.AdamW(
        list(model.encoder.parameters()) + list(model.projector.parameters()), lr=LR_STAGE1, weight_decay=WEIGHT_DECAY
    )
    stage1_scheduler = optim.lr_scheduler.CosineAnnealingLR(stage1_optimizer, T_max=STAGE1_EPOCHS)

    best_stage1_loss = float("inf")

    for epoch in range(1, STAGE1_EPOCHS + 1):
        start = time.time()
        stage1_loss = stage1_train_one_epoch(model, stage1_loader, bcl_criterion, stage1_optimizer)
        stage1_scheduler.step()

        append_stage1_csv([epoch, stage1_loss, stage1_optimizer.param_groups[0]["lr"]])

        logger.info(
            f"Stage1 Epoch [{epoch}/{STAGE1_EPOCHS}] | "
            f"BCL Loss: {stage1_loss:.4f} | "
            f"LR: {stage1_optimizer.param_groups[0]['lr']:.6f} | "
            f"Time: {time.time() - start:.1f}s"
        )

        stage1_metrics = {"stage1_bcl_loss": stage1_loss}

        if stage1_loss < best_stage1_loss:
            best_stage1_loss = stage1_loss
            save_checkpoint(
                STAGE1_BEST_PATH,
                model,
                stage1_optimizer,
                epoch,
                stage1_metrics,
                stage2_train_dataset.classes,
                "stage1_pretrain",
            )
            logger.info(f"Saved stage1 best checkpoint to: {STAGE1_BEST_PATH}")

        save_checkpoint(
            STAGE1_LAST_PATH,
            model,
            stage1_optimizer,
            epoch,
            stage1_metrics,
            stage2_train_dataset.classes,
            "stage1_pretrain",
        )

    logger.info(f"Stage1 finished. Best BCL Loss: {best_stage1_loss:.4f}")

    # =========================
    # Stage2: classic freeze finetune
    # =========================
    logger.info("========== Stage2: classic freeze finetune ==========")

    model.reset_classifier()
    ce_criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    best_macro_f1 = 0.0

    global_epoch = STAGE1_EPOCHS

    # -------- Phase1: classifier only --------
    logger.info("----- Phase1: classifier-only finetune -----")
    freeze_all_encoder(model)
    optimizer = optim.AdamW(get_trainable_params(model), lr=LR_PHASE1, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE1_EPOCHS)

    for local_epoch in range(1, PHASE1_EPOCHS + 1):
        global_epoch += 1
        start = time.time()
        train_loss, train_acc = stage2_train_one_epoch(model, stage2_train_loader, ce_criterion, optimizer)
        metrics = evaluate(model, val_loader, ce_criterion, stage2_train_dataset.classes, global_epoch)
        scheduler.step()

        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        macro_precision = metrics["macro_precision"]
        macro_recall = metrics["macro_recall"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]
        c_recall, cor_recall, mc_recall, mm_recall = metrics["per_class_recall"]
        c_f1, cor_f1, mc_f1, mm_f1 = metrics["per_class_f1"]

        append_stage2_csv(
            [
                global_epoch,
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
            f"Phase1 Epoch [{local_epoch}/{PHASE1_EPOCHS}] (Global {global_epoch}/{TOTAL_EPOCHS}) | "
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
            save_checkpoint(
                BEST_ACC_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_acc checkpoint to: {BEST_ACC_PATH}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_checkpoint(
                BEST_F1_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_macro_f1 checkpoint to: {BEST_F1_PATH}")

        save_checkpoint(LAST_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2")

    # -------- Phase2: layer4 + classifier --------
    logger.info("----- Phase2: layer4 + classifier finetune -----")
    unfreeze_layer4_and_classifier(model)
    optimizer = optim.AdamW(get_trainable_params(model), lr=LR_PHASE2, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE2_EPOCHS)

    for local_epoch in range(1, PHASE2_EPOCHS + 1):
        global_epoch += 1
        start = time.time()
        train_loss, train_acc = stage2_train_one_epoch(model, stage2_train_loader, ce_criterion, optimizer)
        metrics = evaluate(model, val_loader, ce_criterion, stage2_train_dataset.classes, global_epoch)
        scheduler.step()

        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        macro_precision = metrics["macro_precision"]
        macro_recall = metrics["macro_recall"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]
        c_recall, cor_recall, mc_recall, mm_recall = metrics["per_class_recall"]
        c_f1, cor_f1, mc_f1, mm_f1 = metrics["per_class_f1"]

        append_stage2_csv(
            [
                global_epoch,
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
            f"Phase2 Epoch [{local_epoch}/{PHASE2_EPOCHS}] (Global {global_epoch}/{TOTAL_EPOCHS}) | "
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
            save_checkpoint(
                BEST_ACC_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_acc checkpoint to: {BEST_ACC_PATH}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_checkpoint(
                BEST_F1_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_macro_f1 checkpoint to: {BEST_F1_PATH}")

        save_checkpoint(LAST_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2")

    # -------- Phase3: full-network low-lr --------
    logger.info("----- Phase3: full-network low-lr finetune -----")
    unfreeze_all_for_finetune(model)
    optimizer = optim.AdamW(get_trainable_params(model), lr=LR_PHASE3, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=PHASE3_EPOCHS)

    for local_epoch in range(1, PHASE3_EPOCHS + 1):
        global_epoch += 1
        start = time.time()
        train_loss, train_acc = stage2_train_one_epoch(model, stage2_train_loader, ce_criterion, optimizer)
        metrics = evaluate(model, val_loader, ce_criterion, stage2_train_dataset.classes, global_epoch)
        scheduler.step()

        val_loss = metrics["val_loss"]
        val_acc = metrics["val_acc"]
        macro_precision = metrics["macro_precision"]
        macro_recall = metrics["macro_recall"]
        macro_f1 = metrics["macro_f1"]
        weighted_f1 = metrics["weighted_f1"]
        c_recall, cor_recall, mc_recall, mm_recall = metrics["per_class_recall"]
        c_f1, cor_f1, mc_f1, mm_f1 = metrics["per_class_f1"]

        append_stage2_csv(
            [
                global_epoch,
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
            f"Phase3 Epoch [{local_epoch}/{PHASE3_EPOCHS}] (Global {global_epoch}/{TOTAL_EPOCHS}) | "
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
            save_checkpoint(
                BEST_ACC_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_acc checkpoint to: {BEST_ACC_PATH}")

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1
            save_checkpoint(
                BEST_F1_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2"
            )
            logger.info(f"Saved best_macro_f1 checkpoint to: {BEST_F1_PATH}")

        save_checkpoint(LAST_PATH, model, optimizer, global_epoch, metrics, stage2_train_dataset.classes, "stage2")

    logger.info("Training finished.")
    logger.info(f"Best Val Acc: {best_val_acc:.4f}")
    logger.info(f"Best Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()
