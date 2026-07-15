import argparse
import csv
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


def parse_seg_line(line, has_conf="auto"):
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    cls_id = int(float(parts[0]))
    vals = list(map(float, parts[1:]))

    conf = 1.0

    # YOLO seg:
    # GT:   cls x1 y1 x2 y2 ...
    # Pred: cls x1 y1 x2 y2 ... conf
    if has_conf == "true":
        if len(vals) >= 7 and len(vals) % 2 == 1:
            conf = vals[-1]
            coords = vals[:-1]
        else:
            coords = vals
    elif has_conf == "false":
        coords = vals
    else:
        # auto:
        # polygon coords 必须是偶数个，带 conf 时 vals 是奇数个
        if len(vals) % 2 == 1:
            conf = vals[-1]
            coords = vals[:-1]
        else:
            coords = vals

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    xs = coords[0::2]
    ys = coords[1::2]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "cls": cls_id,
        "conf": conf,
        "box": [x1, y1, x2, y2],
    }


def read_label_file(path, has_conf="auto"):
    items = []
    path = Path(path)
    if not path.exists():
        return items

    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = parse_seg_line(line, has_conf=has_conf)
            if item is not None:
                items.append(item)

    return items


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter
    if union <= 0:
        return 0.0

    return inter / union


def evaluate_one_image(gts, preds, iou_th):
    preds = sorted(preds, key=lambda x: x["conf"], reverse=True)

    matched_gt = set()
    tp = 0
    fp = 0

    detail_matches = []

    for pi, pred in enumerate(preds):
        best_iou = 0.0
        best_gi = -1

        for gi, gt in enumerate(gts):
            if gi in matched_gt:
                continue
            iou = box_iou(pred["box"], gt["box"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi

        if best_iou >= iou_th and best_gi >= 0:
            tp += 1
            matched_gt.add(best_gi)
            detail_matches.append((pi, best_gi, best_iou, "TP"))
        else:
            fp += 1
            detail_matches.append((pi, -1, best_iou, "FP"))

    fn = len(gts) - len(matched_gt)
    return tp, fp, fn, detail_matches


def find_image_stems(img_dir):
    img_dir = Path(img_dir)
    stems = []
    for ext in IMG_EXTS:
        stems.extend([p.stem for p in img_dir.glob(f"*{ext}")])
    return sorted(set(stems))


def eval_dir(src, split, pred_label_dir, iou_ths, method_name):
    src = Path(src)
    pred_label_dir = Path(pred_label_dir)

    img_dir = src / "images" / split
    gt_dir = src / "labels" / split

    stems = find_image_stems(img_dir)

    summary_rows = []
    detail_rows = []

    for iou_th in iou_ths:
        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_gt = 0
        total_pred = 0
        mean_best_ious = []

        for stem in stems:
            gt_path = gt_dir / f"{stem}.txt"
            pred_path = pred_label_dir / f"{stem}.txt"

            gts = read_label_file(gt_path, has_conf="false")
            preds = read_label_file(pred_path, has_conf="auto")

            total_gt += len(gts)
            total_pred += len(preds)

            tp, fp, fn, _matches = evaluate_one_image(gts, preds, iou_th)

            total_tp += tp
            total_fp += fp
            total_fn += fn

            # 每个 GT 的 best box IoU，用于定位质量观察
            for gi, gt in enumerate(gts):
                best = 0.0
                for pred in preds:
                    best = max(best, box_iou(pred["box"], gt["box"]))
                mean_best_ious.append(best)

            detail_rows.append(
                {
                    "method": method_name,
                    "split": split,
                    "iou_th": iou_th,
                    "image": stem,
                    "gt": len(gts),
                    "pred": len(preds),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                }
            )

        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        mean_best_iou = sum(mean_best_ious) / len(mean_best_ious) if mean_best_ious else 0.0

        summary_rows.append(
            {
                "method": method_name,
                "split": split,
                "iou_th": iou_th,
                "gt": total_gt,
                "pred": total_pred,
                "tp": total_tp,
                "fp": total_fp,
                "fn": total_fn,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "mean_best_box_iou": mean_best_iou,
            }
        )

    return summary_rows, detail_rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        return

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--splits", default="val,test")
    parser.add_argument("--iou-ths", default="0.30,0.50,0.75")

    parser.add_argument("--items", nargs="+", required=True, help="格式: method_name=pred_label_dir")

    args = parser.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [x.strip() for x in args.splits.split(",") if x.strip()]
    iou_ths = [float(x.strip()) for x in args.iou_ths.split(",") if x.strip()]

    all_summary = []
    all_details = []

    for item in args.items:
        if "=" not in item:
            raise ValueError(f"Bad item format: {item}, expected name=dir")

        method_name, pred_dir = item.split("=", 1)

        for split in splits:
            print(f"[EVAL] method={method_name} split={split} pred_dir={pred_dir}")
            summary_rows, detail_rows = eval_dir(
                src=src,
                split=split,
                pred_label_dir=pred_dir.replace("{split}", split),
                iou_ths=iou_ths,
                method_name=method_name,
            )
            all_summary.extend(summary_rows)
            all_details.extend(detail_rows)

    summary_path = out_dir / "box_from_seg_summary.csv"
    detail_path = out_dir / "box_from_seg_details.csv"

    write_csv(summary_path, all_summary)
    write_csv(detail_path, all_details)

    print(f"[OK] summary saved: {summary_path}")
    print(f"[OK] details saved: {detail_path}")

    print("\n===== TEST IoU=0.50 quick view =====")
    for r in all_summary:
        if r["split"] == "test" and abs(float(r["iou_th"]) - 0.50) < 1e-9:
            print(
                f"method={r['method']:<18} "
                f"P={r['precision']:.3f} R={r['recall']:.3f} F1={r['f1']:.3f} "
                f"TP={r['tp']:3d} FP={r['fp']:3d} FN={r['fn']:3d} "
                f"pred={r['pred']:3d} gt={r['gt']:3d} "
                f"mean_best_box_iou={r['mean_best_box_iou']:.3f}"
            )


if __name__ == "__main__":
    main()
