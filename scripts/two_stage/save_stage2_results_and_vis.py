import argparse
import csv
import shutil
from pathlib import Path
from collections import Counter

from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

COLORS = {
    "gt": (0, 220, 0),
    "stage1": (255, 80, 0),
    "v1": (0, 120, 255),
    "v2": (180, 0, 255),
    "v3": (255, 0, 120),
}

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

def parse_label_line(line, has_conf):
    vals = list(map(float, line.strip().split()))
    if len(vals) < 7:
        return None, None

    if has_conf:
        # cls x1 y1 x2 y2 ... conf
        if len(vals) % 2 == 0:
            conf = vals[-1]
            coords = vals[1:-1]
        else:
            conf = None
            coords = vals[1:]
    else:
        conf = None
        coords = vals[1:]

    if len(coords) < 6:
        return None, None

    pts = list(zip(coords[0::2], coords[1::2]))
    return pts, conf

def draw_overlay(img_path, label_path, title, color, has_conf):
    img = Image.open(img_path).convert("RGB")
    w, h = img.size

    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    n = 0
    if label_path.exists():
        lines = [x for x in label_path.read_text(errors="ignore").splitlines() if x.strip()]
    else:
        lines = []

    for idx, line in enumerate(lines, 1):
        pts_norm, conf = parse_label_line(line, has_conf=has_conf)
        if pts_norm is None:
            continue

        pts = [(int(x * w), int(y * h)) for x, y in pts_norm]
        r, g, b = color

        draw.polygon(pts, fill=(r, g, b, 55), outline=(r, g, b, 255))
        if pts:
            tx, ty = pts[0]
            text = f"{idx}"
            if conf is not None:
                text += f":{conf:.2f}"
            draw.text((tx, ty), text, fill=(255, 255, 255, 255))

        n += 1

    out = Image.alpha_composite(base, overlay).convert("RGB")

    # resize for panel
    max_w = 360
    scale = min(1.0, max_w / out.size[0])
    new_size = (int(out.size[0] * scale), int(out.size[1] * scale))
    out = out.resize(new_size)

    # title bar
    bar_h = 34
    panel = Image.new("RGB", (out.size[0], out.size[1] + bar_h), (30, 30, 30))
    panel.paste(out, (0, bar_h))
    d = ImageDraw.Draw(panel)
    d.text((8, 8), f"{title} | n={n}", fill=(255, 255, 255))

    return panel

def hcat_panels(panels):
    max_h = max(p.height for p in panels)
    total_w = sum(p.width for p in panels)

    canvas = Image.new("RGB", (total_w, max_h), (20, 20, 20))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0))
        x += p.width
    return canvas

def copy_label_dir(src, dst):
    dst.mkdir(parents=True, exist_ok=True)
    if not src.exists():
        print("[WARN] missing label dir:", src)
        return
    for f in src.glob("*.txt"):
        shutil.copy2(f, dst / f.name)

def eval_one_method(images, gt_dir, pred_dir):
    rows = []
    for img_path in images:
        stem = img_path.stem
        gt = count_txt(gt_dir / f"{stem}.txt")
        pred = count_txt(pred_dir / f"{stem}.txt")
        case = classify(gt, pred)
        rows.append({
            "image": img_path.name,
            "gt_count": gt,
            "pred_count": pred,
            "case": case,
            "diff_pred_minus_gt": pred - gt,
        })
    return rows

def write_rows_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def summarize_rows(rows):
    c = Counter(r["case"] for r in rows)
    return {
        "ok": c.get("ok", 0),
        "miss": c.get("miss", 0),
        "false_positive": c.get("false_positive", 0),
        "over": c.get("over", 0),
        "under": c.get("under", 0),
        "gt_total": sum(r["gt_count"] for r in rows),
        "pred_total": sum(r["pred_count"] for r in rows),
        "gt_pos": sum(1 for r in rows if r["gt_count"] > 0),
        "pred_pos": sum(1 for r in rows if r["pred_count"] > 0),
    }

def save_summary_txt(summary_rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in summary_rows:
            f.write(
                f"{r['split']:4s} {r['method']:8s} {r['threshold']:>5s} "
                f"ok={r['ok']:3d} miss={r['miss']:3d} fp={r['false_positive']:3d} "
                f"over={r['over']:3d} under={r['under']:3d} "
                f"gt={r['gt_total']:3d} pred={r['pred_total']:3d} "
                f"gt_pos={r['gt_pos']:3d} pred_pos={r['pred_pos']:3d}\n"
            )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="dataset root, e.g. datasets/missing_coating_single_seg")
    ap.add_argument("--pred-root", required=True, help="two-stage prediction root")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--thresholds", default="0.20,0.30,0.40,0.50,0.60,0.70")
    ap.add_argument("--vis-thresholds", default="0.30,0.70")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--only-bad", action="store_true")
    args = ap.parse_args()

    src = Path(args.src)
    pred_root = Path(args.pred_root)
    out_root = Path(args.out_root)

    thresholds = [float(x) for x in args.thresholds.split(",")]
    vis_thresholds = [float(x) for x in args.vis_thresholds.split(",")]
    splits = [x.strip() for x in args.splits.split(",")]

    summary_rows = []

    for split in splits:
        print(f"\n================ {split} ================")

        img_dir = src / "images" / split
        gt_dir = src / "labels" / split
        images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

        # stage1
        stage1_dir = pred_root / f"stage1_{split}_conf015_iou030" / "labels"
        stage1_save_dir = out_root / "stage1_conf015_iou030" / split / "labels"
        copy_label_dir(stage1_dir, stage1_save_dir)

        rows = eval_one_method(images, gt_dir, stage1_dir)
        write_rows_csv(rows, out_root / "stage1_conf015_iou030" / split / "cases.csv")

        s = summarize_rows(rows)
        s.update({"split": split, "method": "stage1", "threshold": "none"})
        summary_rows.append(s)

        method_roots = {
            "v1": pred_root / f"stage2_{split}_resnet18_v1",
            "v2": pred_root / f"stage2_{split}_resnet18_v2_candidates",
            "v3": pred_root / f"stage2_{split}_resnet18_v3_hybrid_balanced",
        }

        # save labels + case csv
        for method, root in method_roots.items():
            for th in thresholds:
                th_name = f"p{th:.2f}"
                pred_dir = root / th_name / "labels"
                save_dir = out_root / method / split / th_name / "labels"

                copy_label_dir(pred_dir, save_dir)

                rows = eval_one_method(images, gt_dir, pred_dir)
                write_rows_csv(rows, out_root / method / split / th_name / "cases.csv")

                bad_rows = [r for r in rows if r["case"] != "ok"]
                write_rows_csv(bad_rows, out_root / method / split / th_name / "bad_cases.csv")

                s = summarize_rows(rows)
                s.update({"split": split, "method": method, "threshold": th_name})
                summary_rows.append(s)

        # visualization comparison
        for th in vis_thresholds:
            th_name = f"p{th:.2f}"
            vis_dir = out_root / "visual_compare" / split / th_name
            vis_dir.mkdir(parents=True, exist_ok=True)

            dirs_for_vis = {
                "GT": gt_dir,
                "Stage1": stage1_dir,
                "v1": method_roots["v1"] / th_name / "labels",
                "v2": method_roots["v2"] / th_name / "labels",
                "v3": method_roots["v3"] / th_name / "labels",
            }

            # if only_bad, use union of bad images among methods
            bad_set = set()
            if args.only_bad:
                for name, pdir in dirs_for_vis.items():
                    if name == "GT":
                        continue
                    rows = eval_one_method(images, gt_dir, pdir)
                    bad_set.update(r["image"] for r in rows if r["case"] != "ok")

            for img_path in images:
                if args.only_bad and img_path.name not in bad_set:
                    continue

                panels = [
                    draw_overlay(img_path, gt_dir / f"{img_path.stem}.txt", "GT", COLORS["gt"], has_conf=False),
                    draw_overlay(img_path, stage1_dir / f"{img_path.stem}.txt", "Stage1", COLORS["stage1"], has_conf=True),
                    draw_overlay(img_path, dirs_for_vis["v1"] / f"{img_path.stem}.txt", f"v1 {th_name}", COLORS["v1"], has_conf=True),
                    draw_overlay(img_path, dirs_for_vis["v2"] / f"{img_path.stem}.txt", f"v2 {th_name}", COLORS["v2"], has_conf=True),
                    draw_overlay(img_path, dirs_for_vis["v3"] / f"{img_path.stem}.txt", f"v3 {th_name}", COLORS["v3"], has_conf=True),
                ]

                canvas = hcat_panels(panels)
                save_path = vis_dir / f"{img_path.stem}.jpg"
                canvas.save(save_path, quality=92)

            print("[VIS]", split, th_name, "->", vis_dir)

    # save global summary
    summary_csv = out_root / "summary_all_methods.csv"
    write_rows_csv(summary_rows, summary_csv)
    save_summary_txt(summary_rows, out_root / "summary_all_methods.txt")

    print("\n[OK] all saved to:", out_root)
    print("[OK] summary csv:", summary_csv)
    print("[OK] summary txt:", out_root / "summary_all_methods.txt")

if __name__ == "__main__":
    main()
