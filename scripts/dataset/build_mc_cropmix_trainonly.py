import argparse
import random
import shutil
from pathlib import Path

import cv2
import yaml


IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def read_label(label_path: Path):
    if not label_path.exists() or label_path.stat().st_size == 0:
        return []

    lines = []
    for line in label_path.read_text().strip().splitlines():
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(float(parts[0]))
        coords = list(map(float, parts[1:]))
        if len(coords) % 2 != 0:
            continue
        lines.append((cls, coords))
    return lines


def polygon_bbox(coords, w, h):
    xs = [coords[i] * w for i in range(0, len(coords), 2)]
    ys = [coords[i] * h for i in range(1, len(coords), 2)]
    return min(xs), min(ys), max(xs), max(ys)


def polygon_inside_crop(coords, w, h, x1, y1, x2, y2):
    for i in range(0, len(coords), 2):
        x = coords[i] * w
        y = coords[i + 1] * h
        if not (x1 <= x <= x2 and y1 <= y <= y2):
            return False
    return True


def bbox_intersects(b1, b2):
    ax1, ay1, ax2, ay2 = b1
    bx1, by1, bx2, by2 = b2
    return not (ax2 <= bx1 or bx2 <= ax1 or ay2 <= by1 or by2 <= ay1)


def transform_polygon(coords, w, h, x1, y1, cw, ch):
    new_coords = []
    for i in range(0, len(coords), 2):
        x = coords[i] * w
        y = coords[i + 1] * h

        nx = (x - x1) / cw
        ny = (y - y1) / ch

        nx = min(max(nx, 0.0), 1.0)
        ny = min(max(ny, 0.0), 1.0)

        new_coords.extend([nx, ny])
    return new_coords


def make_square_crop(x1, y1, x2, y2, img_w, img_h, scale, min_size):
    bw = x2 - x1
    bh = y2 - y1
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    side = max(bw, bh) * scale
    side = max(side, min_size)
    side = min(side, img_w, img_h)

    nx1 = int(round(cx - side / 2))
    ny1 = int(round(cy - side / 2))
    nx2 = int(round(cx + side / 2))
    ny2 = int(round(cy + side / 2))

    if nx1 < 0:
        nx2 -= nx1
        nx1 = 0
    if ny1 < 0:
        ny2 -= ny1
        ny1 = 0
    if nx2 > img_w:
        shift = nx2 - img_w
        nx1 -= shift
        nx2 = img_w
    if ny2 > img_h:
        shift = ny2 - img_h
        ny1 -= shift
        ny2 = img_h

    nx1 = max(0, nx1)
    ny1 = max(0, ny1)
    nx2 = min(img_w, nx2)
    ny2 = min(img_h, ny2)

    return nx1, ny1, nx2, ny2


def write_label(path: Path, labels):
    with open(path, "w", encoding="utf-8") as f:
        for cls, coords in labels:
            coord_str = " ".join(f"{v:.6f}" for v in coords)
            f.write(f"{cls} {coord_str}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True)
    parser.add_argument("--dst", required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pos-crops-per-instance", type=int, default=2)
    parser.add_argument("--neg-crops", type=int, default=220)
    parser.add_argument("--crop-size", type=int, default=640)
    parser.add_argument("--min-crop", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)

    src = Path(args.src)
    dst = Path(args.dst)

    if args.overwrite and dst.exists():
        shutil.rmtree(dst)

    shutil.copytree(src, dst)

    src_img_train = src / "images" / "train"
    src_lbl_train = src / "labels" / "train"

    dst_img_train = dst / "images" / "train"
    dst_lbl_train = dst / "labels" / "train"

    pos_added = 0
    neg_added = 0
    skipped_partial = 0

    label_files = sorted(src_lbl_train.glob("*.txt"))

    # positive crops
    for lbl_path in label_files:
        labels = read_label(lbl_path)
        if not labels:
            continue

        img_path = find_image(src_img_train, lbl_path.stem)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        bboxes = [polygon_bbox(coords, w, h) for _, coords in labels]

        for idx, (cls, coords) in enumerate(labels):
            bx1, by1, bx2, by2 = bboxes[idx]

            for k in range(args.pos_crops_per_instance):
                scale = random.choice([1.6, 2.0, 2.4, 2.8])
                cx1, cy1, cx2, cy2 = make_square_crop(
                    bx1, by1, bx2, by2, w, h, scale, args.min_crop
                )

                crop_box = (cx1, cy1, cx2, cy2)

                crop_labels = []
                bad_partial = False

                for other_idx, (other_cls, other_coords) in enumerate(labels):
                    obox = bboxes[other_idx]
                    if not bbox_intersects(obox, crop_box):
                        continue

                    if not polygon_inside_crop(other_coords, w, h, cx1, cy1, cx2, cy2):
                        bad_partial = True
                        break

                    new_coords = transform_polygon(
                        other_coords, w, h, cx1, cy1, cx2 - cx1, cy2 - cy1
                    )
                    crop_labels.append((0, new_coords))

                if bad_partial or not crop_labels:
                    skipped_partial += 1
                    continue

                crop = img[cy1:cy2, cx1:cx2]
                crop = cv2.resize(crop, (args.crop_size, args.crop_size))

                out_stem = f"crop_pos_{lbl_path.stem}_{idx}_{k}"
                cv2.imwrite(str(dst_img_train / f"{out_stem}.jpg"), crop)
                write_label(dst_lbl_train / f"{out_stem}.txt", crop_labels)
                pos_added += 1

    # negative crops from empty-label train images
    empty_labels = [p for p in label_files if p.stat().st_size == 0]
    random.shuffle(empty_labels)

    for i in range(args.neg_crops):
        if not empty_labels:
            break

        lbl_path = empty_labels[i % len(empty_labels)]
        img_path = find_image(src_img_train, lbl_path.stem)
        if img_path is None:
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        h, w = img.shape[:2]
        side = min(args.crop_size, w, h)

        if w == side:
            x1 = 0
        else:
            x1 = random.randint(0, w - side)

        if h == side:
            y1 = 0
        else:
            y1 = random.randint(0, h - side)

        crop = img[y1:y1 + side, x1:x1 + side]
        crop = cv2.resize(crop, (args.crop_size, args.crop_size))

        out_stem = f"crop_neg_{lbl_path.stem}_{i}"
        cv2.imwrite(str(dst_img_train / f"{out_stem}.jpg"), crop)
        (dst_lbl_train / f"{out_stem}.txt").write_text("")
        neg_added += 1

    # rewrite data.yaml
    data = {
        "path": str(dst),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "missing_coating"},
    }

    with open(dst / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print("Done.")
    print(f"dst: {dst}")
    print(f"positive crops added: {pos_added}")
    print(f"negative crops added: {neg_added}")
    print(f"skipped partial crops: {skipped_partial}")
    print(f"train images: {len(list((dst / 'images' / 'train').glob('*')))}")
    print(f"train labels: {len(list((dst / 'labels' / 'train').glob('*.txt')))}")


if __name__ == "__main__":
    main()
