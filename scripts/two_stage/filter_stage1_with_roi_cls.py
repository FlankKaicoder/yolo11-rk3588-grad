import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn
from torchvision import models, transforms

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_pred_line(line):
    vals = list(map(float, line.strip().split()))
    if len(vals) < 7:
        return None

    cls = int(vals[0])

    # Ultralytics segment save_txt + save_conf:
    # cls x1 y1 x2 y2 ... conf
    if len(vals) % 2 == 0:
        conf = vals[-1]
        coords = vals[1:-1]
    else:
        conf = None
        coords = vals[1:]

    xs = coords[0::2]
    ys = coords[1::2]
    if len(xs) < 3 or len(ys) < 3:
        return None

    bbox = [min(xs), min(ys), max(xs), max(ys)]
    return {
        "cls": cls,
        "conf": conf,
        "coords": coords,
        "bbox": bbox,
        "line": line.strip(),
    }


def expand_to_square_pixel(bbox_n, w, h, expand=1.4):
    x1, y1, x2, y2 = bbox_n
    x1, x2 = x1 * w, x2 * w
    y1, y2 = y1 * h, y2 * h

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    side = max(bw, bh) * expand

    nx1 = round(cx - side / 2)
    ny1 = round(cy - side / 2)
    nx2 = round(cx + side / 2)
    ny2 = round(cy + side / 2)

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    if nx2 <= nx1 + 2 or ny2 <= ny1 + 2:
        return None

    return nx1, ny1, nx2, ny2


def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    class_to_idx = ckpt.get("class_to_idx", {"background": 0, "missing_coating": 1})
    pos_idx = ckpt.get("pos_idx", class_to_idx.get("missing_coating", 1))

    print("[INFO] class_to_idx:", class_to_idx)
    print("[INFO] pos_idx:", pos_idx)
    print("[INFO] epoch:", ckpt.get("epoch"))
    print("[INFO] val_metrics:", ckpt.get("val_metrics"))

    return model, pos_idx


def count_instances(label_path):
    if not label_path.exists():
        return 0
    txt = label_path.read_text(errors="ignore").strip()
    if txt == "":
        return 0
    return len([x for x in txt.splitlines() if x.strip()])


@torch.no_grad()
def score_crop(model, pos_idx, crop, tfm, device):
    x = tfm(crop).unsqueeze(0).to(device)
    logits = model(x)
    prob = F.softmax(logits, dim=1)[0, pos_idx].item()
    return prob


def filter_one_split(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    model, pos_idx = load_model(args.ckpt, device)

    tfm = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    img_dir = Path(args.img_dir)
    pred_dir = Path(args.pred_dir)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    thresholds = [float(x) for x in args.thresholds.split(",")]

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    stats = {th: {"in": 0, "keep": 0, "drop": 0, "files": 0} for th in thresholds}

    # 为每个 threshold 准备 labels 目录
    out_label_dirs = {}
    for th in thresholds:
        d = out_root / f"p{th:.2f}" / "labels"
        d.mkdir(parents=True, exist_ok=True)
        out_label_dirs[th] = d

    debug_rows = []

    for img_path in images:
        stem = img_path.stem
        pred_txt = pred_dir / f"{stem}.txt"

        if not pred_txt.exists():
            # 对每个阈值都不写 label，表示 0 预测
            continue

        lines = [x for x in pred_txt.read_text(errors="ignore").splitlines() if x.strip()]
        if not lines:
            continue

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        items = []
        for line in lines:
            item = parse_pred_line(line)
            if item is None:
                continue

            box = expand_to_square_pixel(item["bbox"], w, h, expand=args.expand)
            if box is None:
                continue

            crop = img.crop(box)
            p_mc = score_crop(model, pos_idx, crop, tfm, device)
            item["p_mc"] = p_mc
            items.append(item)

            debug_rows.append((img_path.name, item["conf"], p_mc, item["line"]))

        for th in thresholds:
            kept = [it["line"] for it in items if it["p_mc"] >= th]

            stats[th]["in"] += len(items)
            stats[th]["keep"] += len(kept)
            stats[th]["drop"] += len(items) - len(kept)

            if kept:
                out_txt = out_label_dirs[th] / f"{stem}.txt"
                out_txt.write_text("\n".join(kept) + "\n", encoding="utf-8")
                stats[th]["files"] += 1

    debug_path = out_root / "candidate_scores.tsv"
    with debug_path.open("w", encoding="utf-8") as f:
        f.write("image\tstage1_conf\tp_mc\tline\n")
        for image, conf, p_mc, line in debug_rows:
            f.write(f"{image}\t{conf}\t{p_mc:.6f}\t{line}\n")

    print("[OK] filtered outputs saved:", out_root)
    print("[OK] candidate scores:", debug_path)
    for th in thresholds:
        s = stats[th]
        print(f"threshold={th:.2f} in={s['in']} keep={s['keep']} drop={s['drop']} pred_files={s['files']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--expand", type=float, default=1.4)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    filter_one_split(args)


if __name__ == "__main__":
    main()
