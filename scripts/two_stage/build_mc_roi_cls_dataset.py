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
    if not txt:
        return items

    for line in txt.splitlines():
        vals = list(map(float, line.strip().split()))
        if len(vals) < 7:
            continue

        cls = int(vals[0])

        if has_conf:
            # Ultralytics segment save_txt + save_conf:
            # cls x1 y1 x2 y2 ... conf
            conf = vals[-1]
            coords = vals[1:-1]
        else:
            conf = None
            coords = vals[1:]

        xs = coords[0::2]
        ys = coords[1::2]
        if len(xs) < 3 or len(ys) < 3:
            continue

        x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)
        items.append({
            "cls": cls,
            "conf": conf,
            "bbox": [x1, y1, x2, y2],
            "line": line,
        })

    return items

def iou_box(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union

def expand_to_square_pixel(bbox_n, w, h, expand=1.3):
    x1, y1, x2, y2 = bbox_n
    x1, x2 = x1 * w, x2 * w
    y1, y2 = y1 * h, y2 * h

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    bw = max(2.0, x2 - x1)
    bh = max(2.0, y2 - y1)
    side = max(bw, bh) * expand

    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = int(round(cx + side / 2))
    ny2 = int(round(cy + side / 2))

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(w, nx2)
    ny2 = min(h, ny2)

    if nx2 <= nx1 + 2 or ny2 <= ny1 + 2:
        return None

    return nx1, ny1, nx2, ny2

def random_square_box(w, h, rng):
    short = min(w, h)
    side = rng.randint(max(48, int(short * 0.12)), max(64, int(short * 0.35)))
    side = min(side, w, h)
    x1 = rng.randint(0, max(0, w - side))
    y1 = rng.randint(0, max(0, h - side))
    return x1, y1, x1 + side, y1 + side

def pixel_to_norm_box(box, w, h):
    x1, y1, x2, y2 = box
    return [x1 / w, y1 / h, x2 / w, y2 / h]

def save_crop(img, box, save_path):
    crop = img.crop(box)
    crop = crop.resize((224, 224))
    crop.save(save_path, quality=95)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--val-ratio", type=float, default=0.2)
    ap.add_argument("--pos-expand", type=float, default=1.4)
    ap.add_argument("--neg-expand", type=float, default=1.4)
    ap.add_argument("--fp-conf-min", type=float, default=0.15)
    ap.add_argument("--neg-iou-max", type=float, default=0.05)
    ap.add_argument("--random-neg-per-empty", type=int, default=2)
    ap.add_argument("--random-neg-per-pos-img", type=int, default=1)
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
    image_names = [p.name for p in images]
    rng.shuffle(image_names)

    n_val = max(1, int(len(image_names) * args.val_ratio))
    val_set = set(image_names[:n_val])

    rows = []
    counters = {
        "pos": 0,
        "hard_neg": 0,
        "rand_neg_empty": 0,
        "rand_neg_posimg": 0,
    }

    for img_path in images:
        split = "val" if img_path.name in val_set else "train"

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print("[WARN] bad image:", img_path, e)
            continue

        w, h = img.size
        stem = img_path.stem

        gt_items = parse_seg_label(gt_dir / f"{stem}.txt", has_conf=False)
        pred_items = parse_seg_label(pred_dir / f"{stem}.txt", has_conf=True)
        gt_boxes = [x["bbox"] for x in gt_items]

        # 1) GT positive crops
        for i, item in enumerate(gt_items):
            box = expand_to_square_pixel(item["bbox"], w, h, expand=args.pos_expand)
            if box is None:
                continue

            save_name = f"{stem}_gtpos_{i:03d}.jpg"
            save_path = out_dir / split / "missing_coating" / save_name
            save_crop(img, box, save_path)

            rows.append({
                "split": split,
                "class": "missing_coating",
                "source": "gt_positive",
                "image": img_path.name,
                "crop": str(save_path),
                "conf": "",
                "max_iou_gt": 1.0,
            })
            counters["pos"] += 1

        # 2) YOLO hard false positive crops
        for i, item in enumerate(pred_items):
            conf = item["conf"]
            if conf is None or conf < args.fp_conf_min:
                continue

            max_iou = 0.0
            if gt_boxes:
                max_iou = max(iou_box(item["bbox"], g) for g in gt_boxes)

            if max_iou <= args.neg_iou_max:
                box = expand_to_square_pixel(item["bbox"], w, h, expand=args.neg_expand)
                if box is None:
                    continue

                save_name = f"{stem}_hardneg_{i:03d}_c{conf:.3f}.jpg"
                save_path = out_dir / split / "background" / save_name
                save_crop(img, box, save_path)

                rows.append({
                    "split": split,
                    "class": "background",
                    "source": "hard_false_positive",
                    "image": img_path.name,
                    "crop": str(save_path),
                    "conf": conf,
                    "max_iou_gt": max_iou,
                })
                counters["hard_neg"] += 1

        # 3) random negative crops
        if len(gt_items) == 0:
            n_rand = args.random_neg_per_empty
            src_name = "random_background_empty"
            counter_key = "rand_neg_empty"
        else:
            n_rand = args.random_neg_per_pos_img
            src_name = "random_background_posimg"
            counter_key = "rand_neg_posimg"

        tries = 0
        made = 0
        while made < n_rand and tries < n_rand * 50:
            tries += 1
            pix_box = random_square_box(w, h, rng)
            norm_box = pixel_to_norm_box(pix_box, w, h)

            max_iou = 0.0
            if gt_boxes:
                max_iou = max(iou_box(norm_box, g) for g in gt_boxes)

            if max_iou > args.neg_iou_max:
                continue

            save_name = f"{stem}_randneg_{made:03d}.jpg"
            save_path = out_dir / split / "background" / save_name
            save_crop(img, pix_box, save_path)

            rows.append({
                "split": split,
                "class": "background",
                "source": src_name,
                "image": img_path.name,
                "crop": str(save_path),
                "conf": "",
                "max_iou_gt": max_iou,
            })
            counters[counter_key] += 1
            made += 1

    index_csv = out_dir / "index.csv"
    with index_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["split", "class", "source", "image", "crop", "conf", "max_iou_gt"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("[OK] dataset saved:", out_dir)
    print("[OK] index:", index_csv)
    print("[COUNTERS]", counters)

    for split in ["train", "val"]:
        for cls in ["missing_coating", "background"]:
            n = len(list((out_dir / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}: {n}")

if __name__ == "__main__":
    main()
