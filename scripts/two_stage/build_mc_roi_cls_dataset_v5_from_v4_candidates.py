import argparse
import csv
import random
from pathlib import Path

from PIL import Image, ImageDraw


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"]


def parse_seg_line(line, has_conf=False):
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    cls_id = int(float(parts[0]))
    vals = list(map(float, parts[1:]))

    conf = 1.0
    if has_conf and len(vals) % 2 == 1:
        conf = vals[-1]
        vals = vals[:-1]

    if len(vals) < 6 or len(vals) % 2 != 0:
        return None

    xs = vals[0::2]
    ys = vals[1::2]

    if max(xs) <= min(xs) or max(ys) <= min(ys):
        return None

    return {
        "cls": cls_id,
        "poly": vals,
        "conf": conf,
        "bbox": [min(xs), min(ys), max(xs), max(ys)],
    }


def read_seg_file(path, has_conf=False):
    items = []
    if not path.exists():
        return items
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = parse_seg_line(line, has_conf=has_conf)
            if item is not None:
                items.append(item)
    return items


def find_image(img_dir, stem):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def expand_bbox(bbox, expand=1.20):
    x1, y1, x2, y2 = bbox
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * expand
    h = (y2 - y1) * expand
    nx1 = max(0.0, cx - w / 2.0)
    ny1 = max(0.0, cy - h / 2.0)
    nx2 = min(1.0, cx + w / 2.0)
    ny2 = min(1.0, cy + h / 2.0)
    return [nx1, ny1, nx2, ny2]


def crop_and_save(img, bbox_n, out_path, size=224):
    w, h = img.size
    x1, y1, x2, y2 = bbox_n
    box = (
        int(max(0, min(w - 1, x1 * w))),
        int(max(0, min(h - 1, y1 * h))),
        int(max(1, min(w, x2 * w))),
        int(max(1, min(h, y2 * h))),
    )

    if box[2] <= box[0] or box[3] <= box[1]:
        return False

    crop = img.crop(box).convert("RGB")
    crop = crop.resize((size, size), Image.BILINEAR)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    crop.save(out_path, quality=95)
    return True


def raster_poly(poly, img_w, img_h, raster_long=768):
    scale = raster_long / max(img_w, img_h)
    rw = max(1, int(round(img_w * scale)))
    rh = max(1, int(round(img_h * scale)))

    pts = []
    for x, y in zip(poly[0::2], poly[1::2]):
        pts.append((x * rw, y * rh))

    mask = Image.new("1", (rw, rh), 0)
    draw = ImageDraw.Draw(mask)
    draw.polygon(pts, outline=1, fill=1)
    return mask


def mask_iou(mask_a, mask_b):
    # PIL mode "1" -> bytes; but simple pixel loop is okay for this dataset size
    pa = mask_a.load()
    pb = mask_b.load()
    w, h = mask_a.size

    inter = 0
    union = 0
    for y in range(h):
        for x in range(w):
            a = 1 if pa[x, y] else 0
            b = 1 if pb[x, y] else 0
            if a or b:
                union += 1
                if a and b:
                    inter += 1

    if union == 0:
        return 0.0
    return inter / union


def best_iou(candidate, gts, img_w, img_h, raster_long):
    if not gts:
        return 0.0

    cmask = raster_poly(candidate["poly"], img_w, img_h, raster_long)
    best = 0.0
    for gt in gts:
        gmask = raster_poly(gt["poly"], img_w, img_h, raster_long)
        best = max(best, mask_iou(cmask, gmask))
    return best


def build_split(
    src,
    cand_label_dir,
    out_root,
    split,
    pos_iou,
    neg_iou,
    expand,
    crop_size,
    raster_long,
    max_neg_ratio,
    seed,
    include_gt_pos,
):
    random.seed(seed)

    img_dir = src / "images" / split
    gt_dir = src / "labels" / split
    cand_dir = Path(cand_label_dir)

    out_split = out_root / split
    pos_dir = out_split / "missing_coating"
    neg_dir = out_split / "background"
    pos_dir.mkdir(parents=True, exist_ok=True)
    neg_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    pos_items = []
    neg_items = []
    ambig_count = 0
    skipped_empty = 0

    image_paths = []
    for ext in IMG_EXTS:
        image_paths.extend(img_dir.glob(f"*{ext}"))
    image_paths = sorted(set(image_paths))

    for img_path in image_paths:
        stem = img_path.stem
        gt_path = gt_dir / f"{stem}.txt"
        cand_path = cand_dir / f"{stem}.txt"

        gts = read_seg_file(gt_path, has_conf=False)
        cands = read_seg_file(cand_path, has_conf=True)

        if not cands:
            skipped_empty += 1

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        # 1) candidate-based positives / hard negatives
        for ci, cand in enumerate(cands):
            biou = best_iou(cand, gts, img_w, img_h, raster_long)

            if biou >= pos_iou:
                label = "missing_coating"
                out_path = pos_dir / f"{stem}_cand{ci:03d}_iou{biou:.3f}_conf{cand['conf']:.3f}.jpg"
                item = (img, expand_bbox(cand["bbox"], expand), out_path, label, stem, ci, cand["conf"], biou, "candidate_pos")
                pos_items.append(item)

            elif biou < neg_iou:
                label = "background"
                out_path = neg_dir / f"{stem}_cand{ci:03d}_iou{biou:.3f}_conf{cand['conf']:.3f}.jpg"
                item = (img, expand_bbox(cand["bbox"], expand), out_path, label, stem, ci, cand["conf"], biou, "hard_negative")
                neg_items.append(item)

            else:
                ambig_count += 1

        # 2) add GT crops as clean positive anchors
        if include_gt_pos:
            for gi, gt in enumerate(gts):
                label = "missing_coating"
                out_path = pos_dir / f"{stem}_gt{gi:03d}.jpg"
                item = (img, expand_bbox(gt["bbox"], expand), out_path, label, stem, gi, 1.0, 1.0, "gt_pos")
                pos_items.append(item)

    # hard negative sampling: keep high-conf negatives first
    pos_count = len(pos_items)
    if max_neg_ratio > 0 and pos_count > 0:
        max_neg = int(pos_count * max_neg_ratio)
        neg_items = sorted(neg_items, key=lambda x: x[6], reverse=True)[:max_neg]

    all_items = pos_items + neg_items
    random.shuffle(all_items)

    saved = 0
    for img, bbox, out_path, label, stem, idx, conf, biou, source in all_items:
        ok = crop_and_save(img, bbox, out_path, crop_size)
        if not ok:
            continue
        rel = out_path.relative_to(out_root)
        rows.append({
            "split": split,
            "path": str(rel),
            "label": label,
            "image": stem,
            "candidate_idx": idx,
            "stage1_conf": conf,
            "best_iou": biou,
            "source": source,
        })
        saved += 1

    return {
        "split": split,
        "saved": saved,
        "positive": sum(1 for r in rows if r["label"] == "missing_coating"),
        "negative": sum(1 for r in rows if r["label"] == "background"),
        "ambiguous_skipped": ambig_count,
        "empty_pred_images": skipped_empty,
        "rows": rows,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--train-cand-dir", required=True)
    parser.add_argument("--val-cand-dir", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--pos-iou", type=float, default=0.50)
    parser.add_argument("--neg-iou", type=float, default=0.10)
    parser.add_argument("--expand", type=float, default=1.20)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--raster-long", type=int, default=768)
    parser.add_argument("--max-neg-ratio", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-gt-pos", action="store_true")
    args = parser.parse_args()

    src = Path(args.src)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    all_rows = []
    summaries = []

    specs = [
        ("train", args.train_cand_dir),
        ("val", args.val_cand_dir),
    ]

    for split, cand_dir in specs:
        summary = build_split(
            src=src,
            cand_label_dir=cand_dir,
            out_root=out_root,
            split=split,
            pos_iou=args.pos_iou,
            neg_iou=args.neg_iou,
            expand=args.expand,
            crop_size=args.crop_size,
            raster_long=args.raster_long,
            max_neg_ratio=args.max_neg_ratio,
            seed=args.seed,
            include_gt_pos=not args.no_gt_pos,
        )
        all_rows.extend(summary.pop("rows"))
        summaries.append(summary)

    index_path = out_root / "index.csv"
    with open(index_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "path", "label", "image", "candidate_idx",
            "stage1_conf", "best_iou", "source"
        ])
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = out_root / "summary.csv"
    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "split", "saved", "positive", "negative", "ambiguous_skipped", "empty_pred_images"
        ])
        writer.writeheader()
        writer.writerows(summaries)

    print(f"[DONE] out_root: {out_root}")
    print(f"[DONE] index: {index_path}")
    print(f"[DONE] summary: {summary_path}")
    for s in summaries:
        print(s)


if __name__ == "__main__":
    main()
