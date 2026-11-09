import argparse
import csv
import random
from pathlib import Path

from PIL import Image

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def parse_seg_label(path, has_conf=False):
    items = []
    if not path.exists():
        return items
    txt = path.read_text(errors="ignore").strip()
    if txt == "":
        return items

    for line in txt.splitlines():
        vals = list(map(float, line.strip().split()))
        if len(vals) < 7:
            continue

        if has_conf:
            conf = vals[-1]
            coords = vals[1:-1]
        else:
            conf = None
            coords = vals[1:]

        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) < 3:
            continue

        bbox = [min(xs), min(ys), max(xs), max(ys)]
        items.append({"bbox": bbox, "conf": conf, "line": line})
    return items


def iou_box(a, b):
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

    return inter / (area_a + area_b - inter + 1e-9)


def expand_to_square_pixel(bbox_n, w, h, expand=1.4):
    x1, y1, x2, y2 = bbox_n
    x1, x2 = x1 * w, x2 * w
    y1, y2 = y1 * h, y2 * h

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
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


def save_crop(img, box, path):
    crop = img.crop(box).resize((224, 224))
    crop.save(path, quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--pos-iou-min", type=float, default=0.20)
    ap.add_argument("--neg-iou-max", type=float, default=0.05)
    ap.add_argument("--expand", type=float, default=1.4)
    ap.add_argument("--include-gt-pos", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_dir)
    pred_dir = Path(args.pred_dir)
    out_dir = Path(args.out_dir)

    for split in ["train", "val"]:
        for cls in ["missing_coating", "background"]:
            (out_dir / split / cls).mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])
    names = [p.name for p in images]
    rng.shuffle(names)
    val_set = set(names[: max(1, int(len(names) * args.val_ratio))])

    rows = []
    cnt = {"cand_pos": 0, "cand_neg": 0, "ignore": 0, "gt_pos": 0}

    for img_path in images:
        split = "val" if img_path.name in val_set else "train"
        stem = img_path.stem

        img = Image.open(img_path).convert("RGB")
        w, h = img.size

        gt_items = parse_seg_label(gt_dir / f"{stem}.txt", has_conf=False)
        pred_items = parse_seg_label(pred_dir / f"{stem}.txt", has_conf=True)

        gt_boxes = [x["bbox"] for x in gt_items]

        # A. candidate-level samples
        for i, p in enumerate(pred_items):
            if gt_boxes:
                max_iou = max(iou_box(p["bbox"], g) for g in gt_boxes)
            else:
                max_iou = 0.0

            if max_iou >= args.pos_iou_min:
                cls_name = "missing_coating"
                source = "candidate_positive"
                cnt["cand_pos"] += 1
            elif max_iou <= args.neg_iou_max:
                cls_name = "background"
                source = "candidate_negative"
                cnt["cand_neg"] += 1
            else:
                cnt["ignore"] += 1
                continue

            box = expand_to_square_pixel(p["bbox"], w, h, expand=args.expand)
            if box is None:
                continue

            save_name = f"{stem}_{source}_{i:03d}_iou{max_iou:.3f}_c{p['conf']:.3f}.jpg"
            save_path = out_dir / split / cls_name / save_name
            save_crop(img, box, save_path)

            rows.append(
                {
                    "split": split,
                    "class": cls_name,
                    "source": source,
                    "image": img_path.name,
                    "crop": str(save_path),
                    "conf": p["conf"],
                    "max_iou_gt": max_iou,
                }
            )

        # B. optional GT positives, used as supplement only
        if args.include_gt_pos:
            for j, g in enumerate(gt_items):
                box = expand_to_square_pixel(g["bbox"], w, h, expand=args.expand)
                if box is None:
                    continue

                save_name = f"{stem}_gt_positive_{j:03d}.jpg"
                save_path = out_dir / split / "missing_coating" / save_name
                save_crop(img, box, save_path)

                rows.append(
                    {
                        "split": split,
                        "class": "missing_coating",
                        "source": "gt_positive_supplement",
                        "image": img_path.name,
                        "crop": str(save_path),
                        "conf": "",
                        "max_iou_gt": 1.0,
                    }
                )
                cnt["gt_pos"] += 1

    with (out_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class", "source", "image", "crop", "conf", "max_iou_gt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] saved:", out_dir)
    print("[COUNTS]", cnt)

    for split in ["train", "val"]:
        for cls in ["missing_coating", "background"]:
            n = len(list((out_dir / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}: {n}")


if __name__ == "__main__":
    main()
