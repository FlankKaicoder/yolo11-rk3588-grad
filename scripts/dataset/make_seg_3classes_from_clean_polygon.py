from pathlib import Path
import argparse
import shutil
import yaml
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 原四分类：
# 0: missing_coating
# 1: corrosion
# 2: missing_material
# 3: carbon
#
# 新三分类：
# 0: missing = missing_coating + missing_material
# 1: corrosion
# 2: carbon
CLASS_MAP = {
    "0": "0",
    "2": "0",
    "1": "1",
    "3": "2",
}

NEW_NAMES = {
    0: "missing",
    1: "corrosion",
    2: "carbon",
}

def load_yaml(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def resolve_path(base: Path, p):
    p = Path(p)
    return p if p.is_absolute() else base / p

def image_to_label_path(img_path: Path):
    parts = list(img_path.parts)
    if "images" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")

def collect_images(img_dir: Path):
    if not img_dir.exists():
        return []
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])

def is_yolo_seg_polygon(parts):
    # YOLO-seg: class + 偶数个坐标
    # 所以总列数应为奇数，且 > 5
    return len(parts) > 5 and len(parts) % 2 == 1

def convert_split(split, src_yaml_data, src_root: Path, dst_root: Path):
    if split not in src_yaml_data:
        print(f"[WARN] yaml 中没有 {split}，跳过")
        return

    src_img_dir = resolve_path(src_root, src_yaml_data[split])
    src_lab_root = Path(str(src_img_dir).replace("/images/", "/labels/"))

    dst_img_dir = dst_root / "images" / split
    dst_lab_dir = dst_root / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(src_img_dir)

    image_count = 0
    missing_label_count = 0
    empty_label_count = 0
    object_count = 0
    bad_line_count = 0
    class_counter = Counter()

    for img in images:
        rel = img.relative_to(src_img_dir)

        dst_img = dst_img_dir / rel
        dst_lab = dst_lab_dir / rel.with_suffix(".txt")

        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, dst_img)
        image_count += 1

        src_lab = src_lab_root / rel.with_suffix(".txt")

        if not src_lab.exists():
            dst_lab.write_text("", encoding="utf-8")
            missing_label_count += 1
            continue

        lines = [
            x.strip()
            for x in src_lab.read_text(encoding="utf-8", errors="ignore").splitlines()
            if x.strip()
        ]

        if not lines:
            dst_lab.write_text("", encoding="utf-8")
            empty_label_count += 1
            continue

        out_lines = []

        for line in lines:
            parts = line.split()

            if not is_yolo_seg_polygon(parts):
                bad_line_count += 1
                print(f"[WARN] bad/non-polygon line skipped: {src_lab} -> {line}")
                continue

            old_cls = parts[0]
            if old_cls not in CLASS_MAP:
                bad_line_count += 1
                print(f"[WARN] unknown class skipped: {src_lab} -> {line}")
                continue

            new_cls = CLASS_MAP[old_cls]
            parts[0] = new_cls

            out_lines.append(" ".join(parts))
            class_counter[new_cls] += 1
            object_count += 1

        dst_lab.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), encoding="utf-8")

    print(f"\n[{split}]")
    print(f"images:          {image_count}")
    print(f"objects:         {object_count}")
    print(f"class_counter:   {dict(class_counter)}")
    print(f"missing labels:  {missing_label_count}")
    print(f"empty labels:    {empty_label_count}")
    print(f"bad lines:       {bad_line_count}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src-yaml",
        default="/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg_clean_polygon/data.yaml",
        help="清洗后的四分类 polygon 分割数据集 yaml",
    )
    parser.add_argument(
        "--dst-root",
        default="/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg_clean_polygon_3classes",
        help="输出三分类 clean 数据集目录",
    )
    args = parser.parse_args()

    src_yaml = Path(args.src_yaml)
    dst_root = Path(args.dst_root)

    data = load_yaml(src_yaml)

    src_root = Path(data.get("path", src_yaml.parent))
    if not src_root.is_absolute():
        src_root = src_yaml.parent / src_root

    if dst_root.exists():
        print(f"[WARN] 删除旧输出目录: {dst_root}")
        shutil.rmtree(dst_root)

    dst_root.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] src_yaml = {src_yaml}")
    print(f"[INFO] src_root = {src_root}")
    print(f"[INFO] dst_root = {dst_root}")

    for split in ["train", "val", "test"]:
        convert_split(split, data, src_root, dst_root)

    out_yaml = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": 3,
        "names": NEW_NAMES,
    }

    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"\n[OK] saved yaml: {dst_root / 'data.yaml'}")

if __name__ == "__main__":
    main()
