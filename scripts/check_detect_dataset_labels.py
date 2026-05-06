from pathlib import Path

roots = [
    Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect"),
    Path("/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect_1cls_defect"),
]

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

for root in roots:
    print("\n" + "=" * 80)
    print("DATASET:", root)

    for split in ["train", "val", "test"]:
        img_dir = root / "images" / split
        label_dir = root / "labels" / split

        imgs = [p for p in img_dir.rglob("*") if p.suffix.lower() in IMG_EXTS] if img_dir.exists() else []
        labels = list(label_dir.rglob("*.txt")) if label_dir.exists() else []

        non_empty_labels = []
        total_objs = 0
        class_ids = {}

        for lp in labels:
            lines = [x.strip() for x in lp.read_text(encoding="utf-8", errors="ignore").splitlines() if x.strip()]
            if lines:
                non_empty_labels.append(lp)
            for line in lines:
                parts = line.split()
                if not parts:
                    continue
                cid = parts[0]
                class_ids[cid] = class_ids.get(cid, 0) + 1
                total_objs += 1

        missing = 0
        empty = 0

        for img in imgs:
            rel = img.relative_to(img_dir)
            lab = label_dir / rel.with_suffix(".txt")
            if not lab.exists():
                missing += 1
            elif not lab.read_text(encoding="utf-8", errors="ignore").strip():
                empty += 1

        print(f"\n[{split}]")
        print(f"images:            {len(imgs)}")
        print(f"label txt files:   {len(labels)}")
        print(f"non-empty labels:  {len(non_empty_labels)}")
        print(f"total objects:     {total_objs}")
        print(f"missing labels:    {missing}")
        print(f"empty labels:      {empty}")
        print(f"class ids:         {class_ids}")
