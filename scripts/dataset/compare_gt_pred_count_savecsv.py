import argparse
import csv
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}


def count_instances(label_path: Path) -> int:
    if not label_path.exists():
        return 0
    txt = label_path.read_text(errors="ignore").strip()
    if txt == "":
        return 0
    return len([line for line in txt.splitlines() if line.strip()])


def classify_case(gt, p15, p20):
    # 两个阈值都完全漏检
    if gt > 0 and p15 == 0 and p20 == 0:
        return "miss_both"

    # GT 没有，但两个阈值至少一个预测出来
    if gt == 0 and (p15 > 0 or p20 > 0):
        return "false_positive_candidate"

    # conf=0.15 下预测数量多于 GT，通常代表碎片化多检
    if gt > 0 and p15 > gt:
        return "possible_over_detect"

    # conf=0.15 下预测数量少于 GT，但不是完全漏检
    if gt > 0 and 0 < p15 < gt:
        return "possible_under_detect"

    # 0.15 和 0.20 结果不一致，说明该图对阈值敏感
    if p15 != p20:
        return "threshold_sensitive"

    return "ok_count"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-label-dir", required=True)
    ap.add_argument("--pred015", required=True)
    ap.add_argument("--pred020", required=True)
    ap.add_argument("--out-csv", required=True)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_label_dir)
    p15_dir = Path(args.pred015)
    p20_dir = Path(args.pred020)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    rows = []
    for img in images:
        stem = img.stem
        gt = count_instances(gt_dir / f"{stem}.txt")
        p15 = count_instances(p15_dir / f"{stem}.txt")
        p20 = count_instances(p20_dir / f"{stem}.txt")
        case = classify_case(gt, p15, p20)

        rows.append(
            {
                "image": img.name,
                "gt_count": gt,
                "pred015_count": p15,
                "pred020_count": p20,
                "case": case,
                "diff015_minus_gt": p15 - gt,
                "diff020_minus_gt": p20 - gt,
            }
        )

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "gt_count",
                "pred015_count",
                "pred020_count",
                "case",
                "diff015_minus_gt",
                "diff020_minus_gt",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] saved: {out_csv}")
    print(f"[INFO] images: {len(rows)}")

    from collections import Counter

    c = Counter(r["case"] for r in rows)
    print("\n[CASE COUNTS]")
    for k, v in c.most_common():
        print(f"{k}: {v}")

    print("\n[GT/PRED TOTAL]")
    print("gt total:", sum(r["gt_count"] for r in rows))
    print("pred015 total:", sum(r["pred015_count"] for r in rows))
    print("pred020 total:", sum(r["pred020_count"] for r in rows))
    print("gt positive images:", sum(1 for r in rows if r["gt_count"] > 0))
    print("pred015 positive images:", sum(1 for r in rows if r["pred015_count"] > 0))
    print("pred020 positive images:", sum(1 for r in rows if r["pred020_count"] > 0))


if __name__ == "__main__":
    main()
