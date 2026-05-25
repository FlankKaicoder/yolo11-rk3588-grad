import shutil
from pathlib import Path

import yaml

SRC_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg")
DST_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg_clean_polygon")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

IOU_THR = 0.999
EPS = 1e-5


def get_images(img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def polygon_bbox(coords):
    xs = coords[0::2]
    ys = coords[1::2]
    return min(xs), min(ys), max(xs), max(ys)


def bbox_iou(a, b):
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


def is_same_bbox(a, b):
    return all(abs(x - y) <= EPS for x, y in zip(a, b)) or bbox_iou(a, b) >= IOU_THR


def parse_label(label_path: Path):
    lines = [x.strip() for x in label_path.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]

    polygons = []
    boxes_5col = []
    bad_lines = []

    for line in lines:
        parts = line.split()

        # YOLO-seg polygon:
        # class x1 y1 x2 y2 ...
        # 总列数 = 1 + 偶数坐标，所以应为奇数且 > 5
        if len(parts) > 5 and len(parts) % 2 == 1:
            try:
                cls = parts[0]
                coords = list(map(float, parts[1:]))
                polygons.append(
                    {
                        "cls": cls,
                        "coords": coords,
                        "line": line,
                        "bbox": polygon_bbox(coords),
                    }
                )
            except Exception:
                bad_lines.append(line)

        # 5列：class x1 y1 x2 y2
        elif len(parts) == 5:
            try:
                cls = parts[0]
                x1, y1, x2, y2 = map(float, parts[1:])
                boxes_5col.append(
                    {
                        "cls": cls,
                        "bbox": (x1, y1, x2, y2),
                        "line": line,
                    }
                )
            except Exception:
                bad_lines.append(line)

        else:
            bad_lines.append(line)

    kept_lines = [p["line"] for p in polygons]

    removed_dup_boxes = []
    suspicious_boxes = []

    for box in boxes_5col:
        matched = False
        for poly in polygons:
            if box["cls"] != poly["cls"]:
                continue
            if is_same_bbox(box["bbox"], poly["bbox"]):
                matched = True
                break

        if matched:
            removed_dup_boxes.append(box["line"])
        else:
            suspicious_boxes.append(box["line"])

    return kept_lines, removed_dup_boxes, suspicious_boxes, bad_lines


def convert_split(split):
    src_img_dir = SRC_ROOT / "images" / split
    src_label_dir = SRC_ROOT / "labels" / split

    dst_img_dir = DST_ROOT / "images" / split
    dst_label_dir = DST_ROOT / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    images = get_images(src_img_dir)

    total_images = 0
    total_polygon_kept = 0
    total_removed_boxes = 0
    total_suspicious_boxes = 0
    total_bad = 0
    missing_labels = 0
    empty_labels = 0

    suspicious_report = []

    for img in images:
        rel = img.relative_to(src_img_dir)

        dst_img = dst_img_dir / rel
        dst_lab = dst_label_dir / rel.with_suffix(".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, dst_img)
        total_images += 1

        src_lab = src_label_dir / rel.with_suffix(".txt")

        if not src_lab.exists():
            dst_lab.write_text("", encoding="utf-8")
            missing_labels += 1
            continue

        if not src_lab.read_text(encoding="utf-8", errors="ignore").strip():
            dst_lab.write_text("", encoding="utf-8")
            empty_labels += 1
            continue

        kept, removed, suspicious, bad = parse_label(src_lab)

        total_polygon_kept += len(kept)
        total_removed_boxes += len(removed)
        total_suspicious_boxes += len(suspicious)
        total_bad += len(bad)

        if suspicious:
            suspicious_report.append((src_lab, suspicious))

        dst_lab.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")

    print(
        f"[{split}] images={total_images}, "
        f"polygon_kept={total_polygon_kept}, "
        f"removed_dup_bbox={total_removed_boxes}, "
        f"suspicious_5col={total_suspicious_boxes}, "
        f"bad={total_bad}, "
        f"missing={missing_labels}, empty={empty_labels}"
    )

    if suspicious_report:
        report_path = DST_ROOT / f"suspicious_5col_{split}.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            for src_lab, lines in suspicious_report:
                f.write(f"\n### {src_lab}\n")
                for line in lines:
                    f.write(line + "\n")
        print(f"[WARN] suspicious 5-col boxes saved to: {report_path}")


def write_yaml():
    src_yaml = SRC_ROOT / "data.yaml"
    if src_yaml.exists():
        with open(src_yaml, encoding="utf-8") as f:
            data = yaml.safe_load(f)
    else:
        data = {}

    out = {
        "path": str(DST_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
    }

    if "names" in data:
        out["names"] = data["names"]
    else:
        out["names"] = {
            0: "missing_coating",
            1: "corrosion",
            2: "missing_material",
            3: "carbon",
        }

    with open(DST_ROOT / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] yaml saved: {DST_ROOT / 'data.yaml'}")


def main():
    if not SRC_ROOT.exists():
        raise FileNotFoundError(SRC_ROOT)

    if DST_ROOT.exists():
        print(f"[WARN] remove old cleaned dataset: {DST_ROOT}")
        shutil.rmtree(DST_ROOT)

    DST_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] SRC_ROOT={SRC_ROOT}")
    print(f"[INFO] DST_ROOT={DST_ROOT}")

    for split in SPLITS:
        convert_split(split)

    write_yaml()


if __name__ == "__main__":
    main()
