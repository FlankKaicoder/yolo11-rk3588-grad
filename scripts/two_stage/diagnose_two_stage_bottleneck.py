import argparse
import csv
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG", ".BMP"}

def parse_label(path, has_conf):
    items = []
    if not path.exists():
        return items

    txt = path.read_text(errors="ignore").strip()
    if not txt:
        return items

    for line in txt.splitlines():
        vals = list(map(float, line.strip().split()))
        if len(vals) < 7:
            continue

        if has_conf and len(vals) % 2 == 0:
            conf = vals[-1]
            coords = vals[1:-1]
        else:
            conf = 1.0
            coords = vals[1:]

        pts = list(zip(coords[0::2], coords[1::2]))
        if len(pts) >= 3:
            items.append({
                "conf": conf,
                "pts": pts,
                "line": line.strip(),
            })

    return items

def raster_poly(pts_norm, cw, ch):
    mask = Image.new("L", (cw, ch), 0)
    draw = ImageDraw.Draw(mask)

    pts = []
    for x, y in pts_norm:
        px = int(round(x * cw))
        py = int(round(y * ch))
        px = max(0, min(cw - 1, px))
        py = max(0, min(ch - 1, py))
        pts.append((px, py))

    if len(pts) >= 3:
        draw.polygon(pts, fill=1)

    return np.array(mask, dtype=bool)

def mask_iou(a, b):
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union <= 0:
        return 0.0
    return float(inter / union)

def write_csv(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def canvas_size(img_path, raster_long):
    img = Image.open(img_path)
    w, h = img.size
    if w >= h:
        cw = raster_long
        ch = max(1, int(round(h / w * raster_long)))
    else:
        ch = raster_long
        cw = max(1, int(round(w / h * raster_long)))
    return cw, ch

def analyze_upper_bound(img_dir, gt_dir, pred_root, split, confs, out_dir, raster_long):
    upper_rows = []
    summary_rows = []

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    for conf in confs:
        pred_dir = pred_root / f"stage1_{split}_conf{conf}_iou030" / "labels"
        if not pred_dir.exists():
            print("[WARN] missing stage1:", pred_dir)
            continue

        for img_path in images:
            stem = img_path.stem
            cw, ch = canvas_size(img_path, raster_long)

            gt_items = parse_label(gt_dir / f"{stem}.txt", has_conf=False)
            pred_items = parse_label(pred_dir / f"{stem}.txt", has_conf=True)

            gt_masks = [raster_poly(x["pts"], cw, ch) for x in gt_items]
            pred_masks = [raster_poly(x["pts"], cw, ch) for x in pred_items]

            for gi, gm in enumerate(gt_masks):
                best_iou = 0.0
                best_conf = 0.0
                best_pi = -1

                for pi, pm in enumerate(pred_masks):
                    iou = mask_iou(gm, pm)
                    if iou > best_iou:
                        best_iou = iou
                        best_conf = pred_items[pi]["conf"]
                        best_pi = pi

                upper_rows.append({
                    "split": split,
                    "stage1_conf": conf,
                    "image": img_path.name,
                    "gt_index": gi,
                    "num_gt": len(gt_masks),
                    "num_pred": len(pred_masks),
                    "best_iou": best_iou,
                    "best_conf": best_conf,
                    "best_pred_index": best_pi,
                    "covered_iou030": int(best_iou >= 0.30),
                    "covered_iou050": int(best_iou >= 0.50),
                    "covered_iou075": int(best_iou >= 0.75),
                })

    # summary
    import pandas as pd
    df = pd.DataFrame(upper_rows)
    if len(df):
        for conf, g in df.groupby("stage1_conf"):
            summary_rows.append({
                "split": split,
                "stage1_conf": conf,
                "gt_instances": len(g),
                "candidate_recall_iou030": g["covered_iou030"].mean(),
                "candidate_recall_iou050": g["covered_iou050"].mean(),
                "candidate_recall_iou075": g["covered_iou075"].mean(),
                "mean_best_iou": g["best_iou"].mean(),
                "median_best_iou": g["best_iou"].median(),
                "num_uncovered_iou050": int((g["covered_iou050"] == 0).sum()),
            })

    write_csv(upper_rows, out_dir / f"{split}_stage1_candidate_upper_bound_details.csv")
    write_csv(summary_rows, out_dir / f"{split}_stage1_candidate_upper_bound_summary.csv")

    print(f"\n===== {split} Stage1 candidate upper bound =====")
    for r in summary_rows:
        print(
            f"conf={r['stage1_conf']} "
            f"GT={r['gt_instances']} "
            f"R@0.30={r['candidate_recall_iou030']:.3f} "
            f"R@0.50={r['candidate_recall_iou050']:.3f} "
            f"R@0.75={r['candidate_recall_iou075']:.3f} "
            f"mean_best_iou={r['mean_best_iou']:.3f} "
            f"uncovered@0.50={r['num_uncovered_iou050']}"
        )

def analyze_stage2_drop(img_dir, gt_dir, pred_root, split, confs, ps, out_dir, raster_long, iou_th):
    drop_rows = []
    summary_rows = []

    images = sorted([p for p in img_dir.iterdir() if p.suffix in IMG_EXTS])

    for conf in confs:
        stage1_dir = pred_root / f"stage1_{split}_conf{conf}_iou030" / "labels"
        if not stage1_dir.exists():
            print("[WARN] missing stage1:", stage1_dir)
            continue

        for p in ps:
            stage2_dir = pred_root / f"stage2_{split}_conf{conf}_resnet18_v3_hybrid_balanced" / f"p{p:.2f}" / "labels"
            if not stage2_dir.exists():
                print("[WARN] missing stage2:", stage2_dir)
                continue

            for img_path in images:
                stem = img_path.stem
                cw, ch = canvas_size(img_path, raster_long)

                gt_items = parse_label(gt_dir / f"{stem}.txt", has_conf=False)
                s1_items = parse_label(stage1_dir / f"{stem}.txt", has_conf=True)
                s2_items = parse_label(stage2_dir / f"{stem}.txt", has_conf=True)

                gt_masks = [raster_poly(x["pts"], cw, ch) for x in gt_items]
                s1_masks = [raster_poly(x["pts"], cw, ch) for x in s1_items]

                s2_lines = set(x["line"] for x in s2_items)

                for pi, item in enumerate(s1_items):
                    pm = s1_masks[pi]

                    best_iou = 0.0
                    best_gi = -1

                    for gi, gm in enumerate(gt_masks):
                        iou = mask_iou(pm, gm)
                        if iou > best_iou:
                            best_iou = iou
                            best_gi = gi

                    tp_like = best_iou >= iou_th
                    kept = item["line"] in s2_lines

                    drop_rows.append({
                        "split": split,
                        "stage1_conf": conf,
                        "stage2_p": f"{p:.2f}",
                        "image": img_path.name,
                        "stage1_pred_index": pi,
                        "stage1_conf_score": item["conf"],
                        "best_gt_index": best_gi,
                        "best_iou": best_iou,
                        "tp_like": int(tp_like),
                        "fp_like": int(not tp_like),
                        "kept_by_stage2": int(kept),
                        "dropped_by_stage2": int(not kept),
                        "dropped_tp_like": int(tp_like and not kept),
                        "dropped_fp_like": int((not tp_like) and not kept),
                    })

    import pandas as pd
    df = pd.DataFrame(drop_rows)
    if len(df):
        for (conf, p), g in df.groupby(["stage1_conf", "stage2_p"]):
            tp_like = int(g["tp_like"].sum())
            fp_like = int(g["fp_like"].sum())
            dropped_tp = int(g["dropped_tp_like"].sum())
            dropped_fp = int(g["dropped_fp_like"].sum())
            kept_tp = int(((g["tp_like"] == 1) & (g["kept_by_stage2"] == 1)).sum())
            kept_fp = int(((g["fp_like"] == 1) & (g["kept_by_stage2"] == 1)).sum())

            summary_rows.append({
                "split": split,
                "stage1_conf": conf,
                "stage2_p": p,
                "total_stage1_candidates": len(g),
                "tp_like_candidates": tp_like,
                "fp_like_candidates": fp_like,
                "kept_tp_like": kept_tp,
                "dropped_tp_like": dropped_tp,
                "kept_fp_like": kept_fp,
                "dropped_fp_like": dropped_fp,
                "tp_like_keep_rate": kept_tp / max(1, tp_like),
                "fp_like_drop_rate": dropped_fp / max(1, fp_like),
            })

    write_csv(drop_rows, out_dir / f"{split}_stage2_drop_details_iou{iou_th:.2f}.csv")
    write_csv(summary_rows, out_dir / f"{split}_stage2_drop_summary_iou{iou_th:.2f}.csv")

    print(f"\n===== {split} Stage2 drop analysis @IoU={iou_th:.2f} =====")
    for r in summary_rows:
        print(
            f"conf={r['stage1_conf']} p={r['stage2_p']} "
            f"total={r['total_stage1_candidates']} "
            f"TP-like={r['tp_like_candidates']} "
            f"FP-like={r['fp_like_candidates']} "
            f"kept_TP={r['kept_tp_like']} "
            f"drop_TP={r['dropped_tp_like']} "
            f"drop_FP={r['dropped_fp_like']} "
            f"TP_keep={r['tp_like_keep_rate']:.3f} "
            f"FP_drop={r['fp_like_drop_rate']:.3f}"
        )

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--pred-root", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--splits", default="test")
    ap.add_argument("--stage1-confs", default="015,010,005")
    ap.add_argument("--stage2-ps", default="0.30,0.60,0.70")
    ap.add_argument("--iou-th", type=float, default=0.50)
    ap.add_argument("--raster-long", type=int, default=960)
    args = ap.parse_args()

    src = Path(args.src)
    pred_root = Path(args.pred_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    splits = [x.strip() for x in args.splits.split(",")]
    confs = [x.strip() for x in args.stage1_confs.split(",")]
    ps = [float(x) for x in args.stage2_ps.split(",")]

    for split in splits:
        img_dir = src / "images" / split
        gt_dir = src / "labels" / split

        analyze_upper_bound(
            img_dir=img_dir,
            gt_dir=gt_dir,
            pred_root=pred_root,
            split=split,
            confs=confs,
            out_dir=out_dir,
            raster_long=args.raster_long,
        )

        analyze_stage2_drop(
            img_dir=img_dir,
            gt_dir=gt_dir,
            pred_root=pred_root,
            split=split,
            confs=confs,
            ps=ps,
            out_dir=out_dir,
            raster_long=args.raster_long,
            iou_th=args.iou_th,
        )

    print("\n[DONE] saved to:", out_dir)

if __name__ == "__main__":
    main()
