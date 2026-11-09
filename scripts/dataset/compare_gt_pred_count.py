import argparse
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".PNG", ".JPEG"}


def count_lines(p: Path):
    if not p.exists():
        return 0
    txt = p.read_text(errors="ignore").strip()
    if not txt:
        return 0
    return len(txt.splitlines())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img-dir", required=True)
    parser.add_argument("--gt-label-dir", required=True)
    parser.add_argument("--pred015", required=True)
    parser.add_argument("--pred020", required=True)
    args = parser.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_label_dir)
    p15_dir = Path(args.pred015)
    p20_dir = Path(args.pred020)

    rows = []
    for img in sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS]):
        stem = img.stem
        gt = count_lines(gt_dir / f"{stem}.txt")
        p15 = count_lines(p15_dir / f"{stem}.txt")
        p20 = count_lines(p20_dir / f"{stem}.txt")
        rows.append((img.name, gt, p15, p20))

    print("image,gt,pred_conf015,pred_conf020,comment")
    for name, gt, p15, p20 in rows:
        comment = ""
        if gt > 0 and p15 == 0 and p20 == 0:
            comment = "miss_both"
        elif gt > 0 and p15 > 0 and p20 == 0:
            comment = "only_015_detects"
        elif gt == 0 and (p15 > 0 or p20 > 0):
            comment = "false_positive_candidate"
        elif gt > 0 and p15 > gt:
            comment = "possible_over_detect"
        elif gt > 0 and p15 < gt:
            comment = "possible_under_detect"
        print(f"{name},{gt},{p15},{p20},{comment}")


if __name__ == "__main__":
    main()
