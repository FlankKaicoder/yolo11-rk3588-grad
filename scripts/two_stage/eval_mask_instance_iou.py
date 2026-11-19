import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_label(path, has_conf):
    items = []
    if not path.exists():
        return items

    txt = path.read_text(errors="ignore").strip()
    if not txt:
        return items

    for line in txt.splitlines():
        vals = list(map(float, line.strip().split()))
        if len(vals) < 7:
            continue

        # YOLO segment label:
        # GT:   cls x1 y1 x2 y2 ...
        # Pred: cls x1 y1 x2 y2 ... conf
        if has_conf and len(vals) % 2 == 0:
            conf = vals[-1]
            coords = vals[1:-1]
        else:
            conf = 1.0
            coords = vals[1:]

        if len(coords) < 6:
            continue

        pts = list(zip(coords[0::2], coords[1::2]))
        items.append(
            {
                "conf": conf,
                "pts": pts,
                "line": line.strip(),
            }
        )

    return items


def raster_poly(pts_norm, canvas_w, canvas_h):
    mask = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(mask)

    pts = []
    for x, y in pts_norm:
        px = round(x * canvas_w)
        py = round(y * canvas_h)
        px = max(0, min(canvas_w - 1, px))
        py = max(0, min(canvas_h - 1, py))
        pts.append((px, py))

    if len(pts) >= 3:
        draw.polygon(pts, fill=1)

    return np.array(mask, dtype=bool)


def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)


def eval_split(img_dir, gt_dir, pred_dir, iou_th, raster_long=960):
    img_dir = Path(img_dir)
    gt_dir = Path(gt_dir)
    pred_dir = Path(pred_dir)

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_gt = 0
    total_pred = 0
    matched_ious = []
    detail_rows = []

    for img_path in images:
        stem = img_path.stem

        img = Image.open(img_path)
        w, h = img.size

        # 为了速度，不按原图 4K 栅格化，而是缩到最长边 raster_long
        if w >= h:
            cw = raster_long
            ch = max(1, round(h / w * raster_long))
        else:
            ch = raster_long
            cw = max(1, round(w / h * raster_long))

        gt_items = parse_label(gt_dir / f"{stem}.txt", has_conf=False)
        pred_items = parse_label(pred_dir / f"{stem}.txt", has_conf=True)

        gt_masks = [raster_poly(x["pts"], cw, ch) for x in gt_items]
        pred_masks = [raster_poly(x["pts"], cw, ch) for x in pred_items]

        # 预测按 conf 从高到低贪心匹配 GT
        order = sorted(range(len(pred_items)), key=lambda i: pred_items[i]["conf"], reverse=True)
        gt_used = set()

        img_tp = 0
        img_fp = 0
        img_match_ious = []

        for pi in order:
            best_iou = 0.0
            best_gi = -1

            for gi, gm in enumerate(gt_masks):
                if gi in gt_used:
                    continue

                iou = mask_iou(pred_masks[pi], gm)
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi

            if best_iou >= iou_th and best_gi >= 0:
                img_tp += 1
                gt_used.add(best_gi)
                img_match_ious.append(best_iou)
            else:
                img_fp += 1

        img_fn = len(gt_masks) - len(gt_used)

        total_tp += img_tp
        total_fp += img_fp
        total_fn += img_fn
        total_gt += len(gt_masks)
        total_pred += len(pred_masks)
        matched_ious.extend(img_match_ious)

        detail_rows.append(
            {
                "image": img_path.name,
                "gt": len(gt_masks),
                "pred": len(pred_masks),
                "tp": img_tp,
                "fp": img_fp,
                "fn": img_fn,
                "mean_matched_iou": float(np.mean(img_match_ious)) if img_match_ious else 0.0,
            }
        )

    precision = total_tp / max(1, total_tp + total_fp)
    recall = total_tp / max(1, total_tp + total_fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)

    summary = {
        "iou_th": iou_th,
        "tp": total_tp,
        "fp": total_fp,
        "fn": total_fn,
        "gt_total": total_gt,
        "pred_total": total_pred,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_matched_iou": float(np.mean(matched_ious)) if matched_ious else 0.0,
    }

    return summary, detail_rows


def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--stage1-confs", default="015,010,005")
    ap.add_argument("--stage2-thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--iou-ths", default="0.30,0.50,0.75")
    ap.add_argument("--raster-long", type=int, default=960)
    args = ap.parse_args()

    src = Path(args.src)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1_confs = [x.strip() for x in args.stage1_confs.split(",")]
    stage2_ps = [float(x) for x in args.stage2_thresholds.split(",")]
    splits = [x.strip() for x in args.splits.split(",")]
    iou_ths = [float(x) for x in args.iou_ths.split(",")]

    all_summary = []

    for split in splits:
        img_dir = src / "images" / split
        gt_dir = src / "labels" / split

        for conf_name in stage1_confs:
            # Stage1 原始输出
            stage1_dir = pred_root / f"stage1_{split}_conf{conf_name}_iou030" / "labels"

            if stage1_dir.exists():
                for iou_th in iou_ths:
                    summary, details = eval_split(
                        img_dir=img_dir,
                        gt_dir=gt_dir,
                        pred_dir=stage1_dir,
                        iou_th=iou_th,
                        raster_long=args.raster_long,
                    )

                    summary.update(
                        {
                            "split": split,
                            "method": "stage1",
                            "stage1_conf": conf_name,
                            "stage2_p": "none",
                        }
                    )
                    all_summary.append(summary)

                    detail_path = out_dir / f"details_{split}_stage1_conf{conf_name}_iou{iou_th:.2f}.csv"
                    write_csv(details, detail_path)
            else:
                print("[WARN] missing stage1 dir:", stage1_dir)

            # Stage2 v3 输出
            stage2_root = pred_root / f"stage2_{split}_conf{conf_name}_resnet18_v3_hybrid_balanced"

            if not stage2_root.exists():
                print("[WARN] missing stage2 root:", stage2_root)
                continue

            for p in stage2_ps:
                pred_dir = stage2_root / f"p{p:.2f}" / "labels"
                if not pred_dir.exists():
                    print("[WARN] missing pred dir:", pred_dir)
                    continue

                for iou_th in iou_ths:
                    summary, details = eval_split(
                        img_dir=img_dir,
                        gt_dir=gt_dir,
                        pred_dir=pred_dir,
                        iou_th=iou_th,
                        raster_long=args.raster_long,
                    )

                    summary.update(
                        {
                            "split": split,
                            "method": "stage2_v3",
                            "stage1_conf": conf_name,
                            "stage2_p": f"{p:.2f}",
                        }
                    )
                    all_summary.append(summary)

                    detail_path = out_dir / f"details_{split}_stage2v3_conf{conf_name}_p{p:.2f}_iou{iou_th:.2f}.csv"
                    write_csv(details, detail_path)

    summary_path = out_dir / "mask_instance_iou_summary.csv"
    write_csv(all_summary, summary_path)

    print("[OK] saved:", summary_path)
    print()
    print("===== TEST IoU=0.50 quick view =====")
    for r in all_summary:
        if r["split"] == "test" and abs(r["iou_th"] - 0.50) < 1e-9:
            print(
                f"method={r['method']:9s} "
                f"conf={r['stage1_conf']} "
                f"p={r['stage2_p']!s:>4s} "
                f"P={r['precision']:.3f} "
                f"R={r['recall']:.3f} "
                f"F1={r['f1']:.3f} "
                f"TP={r['tp']:3d} "
                f"FP={r['fp']:3d} "
                f"FN={r['fn']:3d} "
                f"pred={r['pred_total']:3d} "
                f"gt={r['gt_total']:3d} "
                f"mIoU={r['mean_matched_iou']:.3f}"
            )


if __name__ == "__main__":
    main()
