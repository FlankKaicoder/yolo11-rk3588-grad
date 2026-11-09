import argparse
import shutil
from pathlib import Path

import yaml

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(base, p):
    p = Path(p)
    if p.is_absolute():
        return p
    return base / p


def image_to_label_path(img_path: Path):
    parts = list(img_path.parts)
    if "images" not in parts:
        return None
    idx = len(parts) - 1 - parts[::-1].index("images")
    parts[idx] = "labels"
    return Path(*parts).with_suffix(".txt")


def collect_images(img_dir: Path):
    return sorted([p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS])


def convert_split(split, src_yaml_data, src_root, dst_root):
    if split not in src_yaml_data:
        print(f"[WARN] yaml 中没有 {split}，跳过")
        return

    src_img_dir = resolve_path(src_root, src_yaml_data[split])
    if not src_img_dir.exists():
        print(f"[WARN] {split} 图片目录不存在: {src_img_dir}")
        return

    dst_img_dir = dst_root / "images" / split
    dst_lab_dir = dst_root / "labels" / split

    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lab_dir.mkdir(parents=True, exist_ok=True)

    images = collect_images(src_img_dir)

    img_count = 0
    label_count = 0
    non_empty_count = 0
    obj_count = 0
    box_like_count = 0
    missing_count = 0
    empty_count = 0

    for img in images:
        rel = img.relative_to(src_img_dir)

        dst_img = dst_img_dir / rel
        dst_lab = dst_lab_dir / rel.with_suffix(".txt")
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        dst_lab.parent.mkdir(parents=True, exist_ok=True)

        shutil.copy2(img, dst_img)
        img_count += 1

        src_lab = image_to_label_path(img)
        if src_lab is None or not src_lab.exists():
            dst_lab.write_text("", encoding="utf-8")
            missing_count += 1
            continue

        lines = [x.strip() for x in src_lab.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
        label_count += 1

        if not lines:
            dst_lab.write_text("", encoding="utf-8")
            empty_count += 1
            continue

        new_lines = []

        for line in lines:
            parts = line.split()

            # detect bbox: class x y w h，一共 5 列
            # segment polygon: class x1 y1 x2 y2 ...，一般大于 5 列
            if len(parts) == 5:
                box_like_count += 1
                print(f"[WARN] 疑似检测框标签，不是分割 polygon: {src_lab} -> {line}")
                continue

            if len(parts) < 7:
                print(f"[WARN] 分割标签点数太少，跳过: {src_lab} -> {line}")
                continue

            # class id 全部改成 0，polygon 坐标保持不变
            parts[0] = "0"
            new_lines.append(" ".join(parts))
            obj_count += 1

        if new_lines:
            non_empty_count += 1
            dst_lab.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        else:
            dst_lab.write_text("", encoding="utf-8")

    print(
        f"[{split}] images={img_count}, labels={label_count}, "
        f"non_empty={non_empty_count}, objects={obj_count}, "
        f"missing={missing_count}, empty={empty_count}, box_like_skipped={box_like_count}"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-yaml", required=True, help="原始多分类分割数据集 yaml")
    parser.add_argument("--dst-root", required=True, help="输出单类 defect 分割数据集目录")
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
        "nc": 1,
        "names": ["defect"],
    }

    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(out_yaml, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] saved yaml: {dst_root / 'data.yaml'}")


if __name__ == "__main__":
    main()
