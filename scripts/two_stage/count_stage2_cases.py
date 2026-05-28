import argparse
from pathlib import Path
from collections import Counter

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

def count_txt(path):
    if not path.exists():
        return 0
    txt = path.read_text(errors="ignore").strip()
    if txt == "":
        return 0
    return len([x for x in txt.splitlines() if x.strip()])

def classify(gt, pred):
    if gt > 0 and pred == 0:
        return "miss"
    if gt == 0 and pred > 0:
        return "false_positive"
    if gt > 0 and pred > gt:
        return "over"
    if gt > 0 and 0 < pred < gt:
        return "under"
    return "ok"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--gt-dir", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    args = ap.parse_args()

    img_dir = Path(args.img_dir)
    gt_dir = Path(args.gt_dir)
    pred_root = Path(args.pred_root)
    thresholds = [float(x) for x in args.thresholds.split(",")]

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    for th in thresholds:
        pred_dir = pred_root / f"p{th:.2f}" / "labels"
        rows = []

        for img in images:
            stem = img.stem
            gt = count_txt(gt_dir / f"{stem}.txt")
            pred = count_txt(pred_dir / f"{stem}.txt")
            rows.append((img.name, gt, pred, classify(gt, pred)))

        c = Counter(r[3] for r in rows)
        gt_total = sum(r[1] for r in rows)
        pred_total = sum(r[2] for r in rows)
        gt_pos = sum(1 for r in rows if r[1] > 0)
        pred_pos = sum(1 for r in rows if r[2] > 0)

        print(f"\n===== Stage2 p_mc >= {th:.2f} =====")
        print("case:", dict(c))
        print("gt total:", gt_total, "pred total:", pred_total)
        print("gt pos:", gt_pos, "pred pos:", pred_pos)

        print("bad images:")
        for name, gt, pred, case in rows:
            if case != "ok":
                print(f"  {case:15s} gt={gt:<2d} pred={pred:<2d} {name}")

if __name__ == "__main__":
    main()
