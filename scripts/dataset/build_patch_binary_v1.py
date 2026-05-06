import argparse
import csv
import math
import random
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from PIL import Image
import yaml


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build binary patch dataset (defect vs background) from YOLO-seg dataset."
    )
    parser.add_argument(
        "--src-root",
        type=str,
        required=True,
        help="Source segmentation dataset root, e.g. datasets/new_dataseg",
    )
    parser.add_argument(
        "--dst-root",
        type=str,
        required=True,
        help="Destination binary patch dataset root, e.g. datasets/patch_binary_v1",
    )
    parser.add_argument("--patch-size", type=int, default=224)
    parser.add_argument("--expand-ratio", type=float, default=0.15)
    parser.add_argument("--min-crop-size", type=int, default=48)
    parser.add_argument("--easy-bg-per-image", type=int, default=1)
    parser.add_argument("--near-bg-per-defect", type=int, default=1)
    parser.add_argument("--max-iou-bg-with-gt", type=float, default=0.05)
    parser.add_argument("--near-outer-scale", type=float, default=1.8)
    parser.add_argument("--max-sample-tries", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_class_names(src_root: Path) -> Dict[int, str]:
    yaml_path = src_root / "data.yaml"
    if not yaml_path.exists():
        return {}
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data.get("names", {})
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    if isinstance(names, dict):
        return {int(k): str(v) for k, v in names.items()}
    return {}


def list_images(image_dir: Path) -> List[Path]:
    return sorted([p for p in image_dir.iterdir() if p.suffix.lower() in IMG_EXTS])


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def xyxy_to_int_box(box: Tuple[float, float, float, float], w: int, h: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    x1 = int(math.floor(clamp(x1, 0, w - 1)))
    y1 = int(math.floor(clamp(y1, 0, h - 1)))
    x2 = int(math.ceil(clamp(x2, 1, w)))
    y2 = int(math.ceil(clamp(y2, 1, h)))
    if x2 <= x1:
        x2 = min(w, x1 + 1)
    if y2 <= y1:
        y2 = min(h, y1 + 1)
    return x1, y1, x2, y2


def box_iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return inter / union


def max_iou_with_boxes(candidate: Tuple[float, float, float, float], gt_boxes: List[Tuple[float, float, float, float]]) -> float:
    if not gt_boxes:
        return 0.0
    return max(box_iou_xyxy(candidate, gt) for gt in gt_boxes)


def expand_box(
    box: Tuple[float, float, float, float],
    img_w: int,
    img_h: int,
    expand_ratio: float,
    min_crop_size: int,
) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0

    bw = max(bw, float(min_crop_size))
    bh = max(bh, float(min_crop_size))

    bw = bw * (1.0 + 2.0 * expand_ratio)
    bh = bh * (1.0 + 2.0 * expand_ratio)

    nx1 = clamp(cx - bw / 2.0, 0.0, img_w - 1.0)
    ny1 = clamp(cy - bh / 2.0, 0.0, img_h - 1.0)
    nx2 = clamp(cx + bw / 2.0, 1.0, img_w * 1.0)
    ny2 = clamp(cy + bh / 2.0, 1.0, img_h * 1.0)

    if nx2 <= nx1:
        nx2 = min(float(img_w), nx1 + 1.0)
    if ny2 <= ny1:
        ny2 = min(float(img_h), ny1 + 1.0)

    return nx1, ny1, nx2, ny2


def crop_and_resize(img: Image.Image, box_xyxy: Tuple[float, float, float, float], patch_size: int) -> Image.Image:
    w, h = img.size
    x1, y1, x2, y2 = xyxy_to_int_box(box_xyxy, w, h)
    patch = img.crop((x1, y1, x2, y2))
    patch = patch.resize((patch_size, patch_size), Image.BILINEAR)
    return patch


def parse_yolo_seg_label(label_path: Path, img_w: int, img_h: int) -> List[Dict]:
    objs = []
    if not label_path.exists():
        return objs

    with open(label_path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]

    for idx, line in enumerate(lines):
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            class_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
        except ValueError:
            continue

        if len(coords) < 6 or len(coords) % 2 != 0:
            continue

        xs = coords[0::2]
        ys = coords[1::2]

        xs_px = [clamp(x * img_w, 0.0, img_w - 1.0) for x in xs]
        ys_px = [clamp(y * img_h, 0.0, img_h - 1.0) for y in ys]

        xmin = min(xs_px)
        ymin = min(ys_px)
        xmax = max(xs_px)
        ymax = max(ys_px)

        if xmax <= xmin or ymax <= ymin:
            continue

        objs.append(
            {
                "gt_id": idx,
                "class_id": class_id,
                "bbox_xyxy": (xmin, ymin, xmax, ymax),
            }
        )
    return objs


def safe_stem(path: Path) -> str:
    return path.stem.replace(" ", "_")


def save_patch(
    patch: Image.Image,
    out_dir: Path,
    filename: str,
):
    out_dir.mkdir(parents=True, exist_ok=True)
    patch.save(out_dir / filename)


def write_meta_csv(meta_path: Path, rows: List[Dict]):
    if not rows:
        return
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(meta_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_distribution_csv(out_path: Path, rows: List[Dict]):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    counts: Dict[Tuple[str, str, str], int] = {}
    for r in rows:
        key = (r["split"], r["binary_label"], r["patch_type"])
        counts[key] = counts.get(key, 0) + 1

    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "binary_label", "patch_type", "count"])
        for key in sorted(counts.keys()):
            writer.writerow([*key, counts[key]])


def sample_easy_background_boxes(
    img_w: int,
    img_h: int,
    gt_boxes: List[Tuple[float, float, float, float]],
    wh_pool: List[Tuple[int, int]],
    num_samples: int,
    max_iou_thresh: float,
    max_tries: int,
) -> List[Tuple[float, float, float, float]]:
    samples = []
    if not wh_pool:
        wh_pool = [(64, 64), (96, 96), (128, 128)]

    tries = 0
    while len(samples) < num_samples and tries < max_tries * max(1, num_samples):
        tries += 1
        bw, bh = random.choice(wh_pool)
        bw = min(max(48, int(bw)), img_w)
        bh = min(max(48, int(bh)), img_h)

        if img_w - bw <= 0 or img_h - bh <= 0:
            continue

        x1 = random.randint(0, img_w - bw)
        y1 = random.randint(0, img_h - bh)
        cand = (float(x1), float(y1), float(x1 + bw), float(y1 + bh))

        if max_iou_with_boxes(cand, gt_boxes) <= max_iou_thresh:
            samples.append(cand)

    return samples


def sample_near_background_boxes(
    gt_box,
    img_w,
    img_h,
    all_gt_boxes,
    num_samples,
    outer_scale,
    max_iou_thresh,
    max_tries,
):
    samples = []

    x1, y1, x2, y2 = gt_box
    bw = x2 - x1
    bh = y2 - y1

    # near patch 不要和 GT 一样大，缩小一点才更容易采到
    min_pw = max(48, int(round(0.5 * bw)))
    max_pw = max(min_pw, int(round(0.8 * bw)))
    min_ph = max(48, int(round(0.5 * bh)))
    max_ph = max(min_ph, int(round(0.8 * bh)))

    # 四个邻近带：左、右、上、下
    tries = 0
    while len(samples) < num_samples and tries < max_tries * max(1, num_samples):
        tries += 1

        pw = random.randint(min_pw, max_pw)
        ph = random.randint(min_ph, max_ph)

        direction = random.choice(["left", "right", "top", "bottom"])

        if direction == "left":
            rx1_min = max(0, int(x1 - pw))
            rx1_max = max(0, int(x1 - 1))
            ry1_min = max(0, int(y1 - 0.25 * bh))
            ry1_max = min(img_h - ph, int(y2 - ph + 0.25 * bh))
        elif direction == "right":
            rx1_min = min(img_w - pw, int(x2 + 1))
            rx1_max = min(img_w - pw, int(x2 + pw))
            ry1_min = max(0, int(y1 - 0.25 * bh))
            ry1_max = min(img_h - ph, int(y2 - ph + 0.25 * bh))
        elif direction == "top":
            rx1_min = max(0, int(x1 - 0.25 * bw))
            rx1_max = min(img_w - pw, int(x2 - pw + 0.25 * bw))
            ry1_min = max(0, int(y1 - ph))
            ry1_max = max(0, int(y1 - 1))
        else:  # bottom
            rx1_min = max(0, int(x1 - 0.25 * bw))
            rx1_max = min(img_w - pw, int(x2 - pw + 0.25 * bw))
            ry1_min = min(img_h - ph, int(y2 + 1))
            ry1_max = min(img_h - ph, int(y2 + ph))

        if rx1_min > rx1_max or ry1_min > ry1_max:
            continue

        rx1 = random.randint(rx1_min, rx1_max)
        ry1 = random.randint(ry1_min, ry1_max)
        cand = (float(rx1), float(ry1), float(rx1 + pw), float(ry1 + ph))

        if max_iou_with_boxes(cand, all_gt_boxes) <= max_iou_thresh:
            samples.append(cand)

    return samples


def build_wh_pool_for_split(src_root: Path, split: str) -> List[Tuple[int, int]]:
    image_dir = src_root / "images" / split
    label_dir = src_root / "labels" / split

    wh_pool = []
    for img_path in list_images(image_dir):
        label_path = label_dir / f"{img_path.stem}.txt"
        with Image.open(img_path) as img:
            w, h = img.size
        objs = parse_yolo_seg_label(label_path, w, h)
        for obj in objs:
            x1, y1, x2, y2 = obj["bbox_xyxy"]
            bw = max(16, int(round(x2 - x1)))
            bh = max(16, int(round(y2 - y1)))
            wh_pool.append((bw, bh))
    return wh_pool


def process_split(
    src_root: Path,
    dst_root: Path,
    split: str,
    class_names: Dict[int, str],
    patch_size: int,
    expand_ratio: float,
    min_crop_size: int,
    easy_bg_per_image: int,
    near_bg_per_defect: int,
    max_iou_bg_with_gt: float,
    near_outer_scale: float,
    max_sample_tries: int,
) -> List[Dict]:
    image_dir = src_root / "images" / split
    label_dir = src_root / "labels" / split

    out_defect_dir = dst_root / split / "defect"
    out_bg_dir = dst_root / split / "background"

    wh_pool = build_wh_pool_for_split(src_root, split)
    meta_rows = []

    images = list_images(image_dir)
    print(f"[{split}] found {len(images)} images")

    for img_idx, img_path in enumerate(images, start=1):
        label_path = label_dir / f"{img_path.stem}.txt"

        with Image.open(img_path) as img:
            img = img.convert("RGB")
            img_w, img_h = img.size

            objs = parse_yolo_seg_label(label_path, img_w, img_h)
            gt_boxes = [o["bbox_xyxy"] for o in objs]

            # 1) defect patches
            for obj in objs:
                gt_id = obj["gt_id"]
                class_id = obj["class_id"]
                orig_class = class_names.get(class_id, str(class_id))

                defect_box = expand_box(
                    obj["bbox_xyxy"],
                    img_w=img_w,
                    img_h=img_h,
                    expand_ratio=expand_ratio,
                    min_crop_size=min_crop_size,
                )

                patch = crop_and_resize(img, defect_box, patch_size)
                x1, y1, x2, y2 = xyxy_to_int_box(defect_box, img_w, img_h)

                fname = (
                    f"{safe_stem(img_path)}_defect_{orig_class}_gt{gt_id:03d}"
                    f"_x{x1}_y{y1}_x{x2}_y{y2}.png"
                )
                save_patch(patch, out_defect_dir, fname)

                meta_rows.append(
                    {
                        "patch_path": str((out_defect_dir / fname).resolve()),
                        "split": split,
                        "binary_label": "defect",
                        "patch_type": "defect",
                        "source_image": img_path.name,
                        "orig_class": orig_class,
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "img_w": img_w,
                        "img_h": img_h,
                        "gt_id": gt_id,
                        "iou_with_gt": 1.0,
                    }
                )

            # 2) easy background patches
            easy_boxes = sample_easy_background_boxes(
                img_w=img_w,
                img_h=img_h,
                gt_boxes=gt_boxes,
                wh_pool=wh_pool,
                num_samples=easy_bg_per_image,
                max_iou_thresh=max_iou_bg_with_gt,
                max_tries=max_sample_tries,
            )
            for bg_idx, bg_box in enumerate(easy_boxes):
                patch = crop_and_resize(img, bg_box, patch_size)
                x1, y1, x2, y2 = xyxy_to_int_box(bg_box, img_w, img_h)

                fname = (
                    f"{safe_stem(img_path)}_bg_easy_{bg_idx:03d}"
                    f"_x{x1}_y{y1}_x{x2}_y{y2}.png"
                )
                save_patch(patch, out_bg_dir, fname)

                meta_rows.append(
                    {
                        "patch_path": str((out_bg_dir / fname).resolve()),
                        "split": split,
                        "binary_label": "background",
                        "patch_type": "easy_bg",
                        "source_image": img_path.name,
                        "orig_class": "",
                        "x1": x1,
                        "y1": y1,
                        "x2": x2,
                        "y2": y2,
                        "img_w": img_w,
                        "img_h": img_h,
                        "gt_id": -1,
                        "iou_with_gt": round(max_iou_with_boxes(bg_box, gt_boxes), 6),
                    }
                )

            # 3) near background patches
            for obj in objs:
                gt_id = obj["gt_id"]
                near_boxes = sample_near_background_boxes(
                    gt_box=obj["bbox_xyxy"],
                    img_w=img_w,
                    img_h=img_h,
                    all_gt_boxes=gt_boxes,
                    num_samples=near_bg_per_defect,
                    outer_scale=near_outer_scale,
                    max_iou_thresh=max_iou_bg_with_gt,
                    max_tries=max_sample_tries,
                )

                for near_idx, bg_box in enumerate(near_boxes):
                    patch = crop_and_resize(img, bg_box, patch_size)
                    x1, y1, x2, y2 = xyxy_to_int_box(bg_box, img_w, img_h)

                    fname = (
                        f"{safe_stem(img_path)}_bg_near_fromgt{gt_id:03d}_{near_idx:03d}"
                        f"_x{x1}_y{y1}_x{x2}_y{y2}.png"
                    )
                    save_patch(patch, out_bg_dir, fname)

                    meta_rows.append(
                        {
                            "patch_path": str((out_bg_dir / fname).resolve()),
                            "split": split,
                            "binary_label": "background",
                            "patch_type": "near_bg",
                            "source_image": img_path.name,
                            "orig_class": "",
                            "x1": x1,
                            "y1": y1,
                            "x2": x2,
                            "y2": y2,
                            "img_w": img_w,
                            "img_h": img_h,
                            "gt_id": gt_id,
                            "iou_with_gt": round(max_iou_with_boxes(bg_box, gt_boxes), 6),
                        }
                    )

        if img_idx % 50 == 0 or img_idx == len(images):
            print(f"[{split}] processed {img_idx}/{len(images)} images")

    return meta_rows


def write_summary(summary_path: Path, all_rows: List[Dict]):
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    def count_rows(split: str, binary_label: Optional[str] = None, patch_type: Optional[str] = None) -> int:
        rows = [r for r in all_rows if r["split"] == split]
        if binary_label is not None:
            rows = [r for r in rows if r["binary_label"] == binary_label]
        if patch_type is not None:
            rows = [r for r in rows if r["patch_type"] == patch_type]
        return len(rows)

    lines = []
    for split in sorted(set(r["split"] for r in all_rows)):
        lines.append(f"[{split}]")
        lines.append(f"  defect:      {count_rows(split, 'defect')}")
        lines.append(f"  background:  {count_rows(split, 'background')}")
        lines.append(f"    easy_bg:   {count_rows(split, 'background', 'easy_bg')}")
        lines.append(f"    near_bg:   {count_rows(split, 'background', 'near_bg')}")
        lines.append("")

    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    random.seed(args.seed)

    src_root = Path(args.src_root)
    dst_root = Path(args.dst_root)

    class_names = load_class_names(src_root)

    for split in ["train", "val"]:
        if not (src_root / "images" / split).exists():
            raise FileNotFoundError(f"Missing directory: {src_root / 'images' / split}")
        if not (src_root / "labels" / split).exists():
            raise FileNotFoundError(f"Missing directory: {src_root / 'labels' / split}")

    all_rows = []
    for split in ["train", "val"]:
        rows = process_split(
            src_root=src_root,
            dst_root=dst_root,
            split=split,
            class_names=class_names,
            patch_size=args.patch_size,
            expand_ratio=args.expand_ratio,
            min_crop_size=args.min_crop_size,
            easy_bg_per_image=args.easy_bg_per_image,
            near_bg_per_defect=args.near_bg_per_defect,
            max_iou_bg_with_gt=args.max_iou_bg_with_gt,
            near_outer_scale=args.near_outer_scale,
            max_sample_tries=args.max_sample_tries,
        )
        all_rows.extend(rows)
        write_meta_csv(dst_root / "meta" / f"{split}_meta.csv", rows)

    write_distribution_csv(dst_root / "stats" / "class_distribution.csv", all_rows)
    write_summary(dst_root / "stats" / "dataset_summary.txt", all_rows)

    print("\nDone.")
    print(f"Output dataset: {dst_root}")
    print(f"Meta CSV saved to: {dst_root / 'meta'}")
    print(f"Stats saved to: {dst_root / 'stats'}")


if __name__ == "__main__":
    main()