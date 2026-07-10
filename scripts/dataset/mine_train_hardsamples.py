import argparse
from pathlib import Path

IMG_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"]


def count_lines(path: Path):
    if not path.exists():
        return 0
    text = path.read_text(errors="ignore").strip()
    if not text:
        return 0
    return len(text.splitlines())


def find_image(img_dir: Path, stem: str):
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p.name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-label-dir", required=True)
    ap.add_argument("--pred-label-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_label_dir)
    pred_dir = Path(args.pred_label_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    hard_pos = []
    hard_neg = []
    over = []
    under = []

    for img_path in sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS]):
        stem = img_path.stem
        gt = count_lines(gt_dir / f"{stem}.txt")
        pred = count_lines(pred_dir / f"{stem}.txt")

        tag = "ok"
        if gt > 0 and pred == 0:
            tag = "miss"
            hard_pos.append(img_path.name)
        elif gt > 0 and pred < gt:
            tag = "under"
            under.append(img_path.name)
            hard_pos.append(img_path.name)
        elif gt == 0 and pred > 0:
            tag = "false_positive"
            hard_neg.append(img_path.name)
        elif gt > 0 and pred > gt + 1:
            tag = "over"
            over.append(img_path.name)

        rows.append((img_path.name, gt, pred, tag))

    with open(out, "w", encoding="utf-8") as f:
        f.write("image,gt,pred,tag\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]},{r[3]}\n")

    list_dir = out.parent
    (list_dir / "hard_positive.txt").write_text("\n".join(hard_pos) + "\n")
    (list_dir / "hard_negative.txt").write_text("\n".join(hard_neg) + "\n")
    (list_dir / "over_detect.txt").write_text("\n".join(over) + "\n")
    (list_dir / "under_detect.txt").write_text("\n".join(under) + "\n")

    print(f"Saved: {out}")
    print(f"hard_positive: {len(hard_pos)}")
    print(f"hard_negative: {len(hard_neg)}")
    print(f"under: {len(under)}")
    print(f"over: {len(over)}")


if __name__ == "__main__":
    main()
