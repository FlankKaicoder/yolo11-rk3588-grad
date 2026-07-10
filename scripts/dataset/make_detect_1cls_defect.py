import shutil
from pathlib import Path

import yaml

# 原始四分类检测数据集
SRC_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect")

# 新生成的单类 defect 数据集，不会覆盖原数据集
DST_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect_1cls_defect")

SPLITS = ["train", "val", "test"]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def copy_images(split: str):
    src_img_dir = SRC_ROOT / "images" / split
    dst_img_dir = DST_ROOT / "images" / split

    if not src_img_dir.exists():
        print(f"[WARN] images/{split} 不存在，跳过")
        return

    dst_img_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in src_img_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        rel = img_path.relative_to(src_img_dir)
        out_path = dst_img_dir / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(img_path, out_path)
        count += 1

    print(f"[OK] copy images/{split}: {count} images")


def convert_labels(split: str):
    src_label_dir = SRC_ROOT / "labels" / split
    dst_label_dir = DST_ROOT / "labels" / split

    # 即使没有 label 目录，也创建目标目录
    dst_label_dir.mkdir(parents=True, exist_ok=True)

    if not src_label_dir.exists():
        print(f"[WARN] labels/{split} 不存在，跳过")
        return

    count_files = 0
    count_objs = 0

    for label_path in src_label_dir.rglob("*.txt"):
        rel = label_path.relative_to(src_label_dir)
        out_label = dst_label_dir / rel
        out_label.parent.mkdir(parents=True, exist_ok=True)

        new_lines = []

        with open(label_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()

                # YOLO detect 标签格式：
                # class x_center y_center w h
                #
                # 如果以后误拿 seg 标签也能处理：
                # class x1 y1 x2 y2 ...
                #
                # 核心就是第一列类别 id 全部改成 0
                parts[0] = "0"
                new_lines.append(" ".join(parts))
                count_objs += 1

        with open(out_label, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")
            else:
                f.write("")

        count_files += 1

    print(f"[OK] convert labels/{split}: {count_files} txt files, {count_objs} objects")


def create_empty_labels_for_images(split: str):
    """如果有图片没有对应 label，则创建空 txt。 对负样本图很重要。.
    """
    img_dir = DST_ROOT / "images" / split
    label_dir = DST_ROOT / "labels" / split

    if not img_dir.exists():
        return

    created = 0

    for img_path in img_dir.rglob("*"):
        if img_path.suffix.lower() not in IMG_EXTS:
            continue

        rel = img_path.relative_to(img_dir)
        label_path = label_dir / rel.with_suffix(".txt")

        if not label_path.exists():
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_path.write_text("", encoding="utf-8")
            created += 1

    print(f"[OK] create empty labels/{split}: {created} empty txt files")


def write_yaml():
    data = {
        "path": str(DST_ROOT),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {0: "defect"},
    }

    yaml_path = DST_ROOT / "data.yaml"
    DST_ROOT.mkdir(parents=True, exist_ok=True)

    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] write yaml: {yaml_path}")


def main():
    print(f"[INFO] src: {SRC_ROOT}")
    print(f"[INFO] dst: {DST_ROOT}")

    if not SRC_ROOT.exists():
        raise FileNotFoundError(f"原始数据集不存在: {SRC_ROOT}")

    if DST_ROOT.exists():
        print(f"[WARN] 目标数据集已存在，将覆盖其中同名文件，但不会影响原始数据集: {DST_ROOT}")

    for split in SPLITS:
        copy_images(split)
        convert_labels(split)
        create_empty_labels_for_images(split)

    write_yaml()

    print("\n[DONE] 单类 defect 检测数据集已生成")
    print(f"YAML: {DST_ROOT / 'data.yaml'}")


if __name__ == "__main__":
    main()
