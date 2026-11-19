import argparse
import shutil
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def load_names(data_yaml: Path):
    if not data_yaml.exists():
        return None

    if yaml is None:
        raise RuntimeError("需要安装 pyyaml：pip install pyyaml")

    with open(data_yaml, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    names = data.get("names", None)
    if names is None:
        return None

    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}

    if isinstance(names, list):
        return {i: name for i, name in enumerate(names)}

    raise RuntimeError(f"无法解析 names: {names}")


def find_target_id(src_root: Path, target_name: str, target_id):
    if target_id is not None:
        return int(target_id)

    data_yaml = src_root / "data.yaml"
    names = load_names(data_yaml)

    if names is None:
        raise RuntimeError("没有在 data.yaml 中找到 names，请手动指定 --target-id")

    for k, v in names.items():
        if str(v) == target_name:
            return int(k)

    raise RuntimeError(f"没有在 {data_yaml} 中找到类别名 {target_name}，当前 names={names}")


def get_image_dir(src_root: Path, split: str):
    candidates = [
        src_root / "images" / split,
        src_root / split / "images",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def get_label_dir(src_root: Path, split: str):
    candidates = [
        src_root / "labels" / split,
        src_root / split / "labels",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def convert_split(src_root: Path, dst_root: Path, split: str, target_id: int):
    src_img_dir = get_image_dir(src_root, split)
    src_lbl_dir = get_label_dir(src_root, split)

    if src_img_dir is None:
        print(f"[跳过] 没找到 {split} 图像目录")
        return None

    if src_lbl_dir is None:
        print(f"[警告] 没找到 {split} 标签目录，将全部视作负样本")

    dst_img_dir = dst_root / "images" / split
    dst_lbl_dir = dst_root / "labels" / split
    dst_img_dir.mkdir(parents=True, exist_ok=True)
    dst_lbl_dir.mkdir(parents=True, exist_ok=True)

    img_files = sorted([p for p in src_img_dir.iterdir() if p.suffix.lower() in IMG_EXTS])

    total_images = 0
    positive_images = 0
    negative_images = 0
    target_instances = 0

    for img_path in img_files:
        total_images += 1

        dst_img_path = dst_img_dir / img_path.name
        if not dst_img_path.exists():
            shutil.copy2(img_path, dst_img_path)

        src_lbl_path = None
        if src_lbl_dir is not None:
            candidate = src_lbl_dir / f"{img_path.stem}.txt"
            if candidate.exists():
                src_lbl_path = candidate

        new_lines = []

        if src_lbl_path is not None:
            with open(src_lbl_path, encoding="utf-8") as f:
                lines = f.readlines()

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 5:
                    continue

                old_cls = int(float(parts[0]))

                if old_cls == target_id:
                    parts[0] = "0"
                    new_lines.append(" ".join(parts))
                    target_instances += 1

        dst_lbl_path = dst_lbl_dir / f"{img_path.stem}.txt"
        with open(dst_lbl_path, "w", encoding="utf-8") as f:
            if new_lines:
                f.write("\n".join(new_lines) + "\n")

        if new_lines:
            positive_images += 1
        else:
            negative_images += 1

    return {
        "split": split,
        "total_images": total_images,
        "positive_images": positive_images,
        "negative_images": negative_images,
        "target_instances": target_instances,
    }


def write_data_yaml(dst_root: Path, target_name: str):
    data = {
        "path": str(dst_root),
        "train": "images/train",
        "val": "images/val",
        "names": {0: target_name},
    }

    if (dst_root / "images" / "test").exists():
        data["test"] = "images/test"

    with open(dst_root / "data.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, allow_unicode=True, sort_keys=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="原始四类 YOLO 分割数据集根目录")
    parser.add_argument("--dst", required=True, help="输出单类数据集根目录")
    parser.add_argument("--target", default="missing_coating", help="目标类别名")
    parser.add_argument("--target-id", default=None, help="如果类别名找不到，可手动指定类别 id")
    args = parser.parse_args()

    src_root = Path(args.src).resolve()
    dst_root = Path(args.dst).resolve()

    if not src_root.exists():
        raise RuntimeError(f"源数据集不存在: {src_root}")

    if yaml is None:
        raise RuntimeError("请先安装 pyyaml：pip install pyyaml")

    target_id = find_target_id(src_root, args.target, args.target_id)

    print("=" * 80)
    print(f"源数据集: {src_root}")
    print(f"输出数据集: {dst_root}")
    print(f"目标类别: {args.target}")
    print(f"目标原始 id: {target_id}")
    print("=" * 80)

    dst_root.mkdir(parents=True, exist_ok=True)

    stats = []
    for split in ["train", "val", "test"]:
        s = convert_split(src_root, dst_root, split, target_id)
        if s is not None:
            stats.append(s)

    write_data_yaml(dst_root, args.target)

    print("\n转换完成，统计如下：")
    for s in stats:
        print(
            f"{s['split']}: "
            f"total={s['total_images']}, "
            f"positive={s['positive_images']}, "
            f"negative={s['negative_images']}, "
            f"instances={s['target_instances']}"
        )

    print("\n生成 data.yaml:")
    print(dst_root / "data.yaml")


if __name__ == "__main__":
    main()
