import argparse
from pathlib import Path


def parse_yolo_seg_line(line):
    parts = line.strip().split()
    if len(parts) < 7:
        return None

    cls_id = parts[0]
    vals = list(map(float, parts[1:]))

    # YOLO segment save_txt + save_conf:
    # cls x1 y1 x2 y2 ... conf
    # 如果没有 conf，则全部是 polygon 坐标
    if len(vals) % 2 == 1:
        conf = vals[-1]
        coords = vals[:-1]
    else:
        conf = 1.0
        coords = vals

    if len(coords) < 6 or len(coords) % 2 != 0:
        return None

    xs = coords[0::2]
    ys = coords[1::2]

    x1, y1 = min(xs), min(ys)
    x2, y2 = max(xs), max(ys)

    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))

    if x2 <= x1 or y2 <= y1:
        return None

    return {
        "cls": cls_id,
        "coords": coords,
        "conf": conf,
        "box": [x1, y1, x2, y2],
    }


def box_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter

    if union <= 0:
        return 0.0
    return inter / union


def nms(cands, iou_th):
    cands = sorted(cands, key=lambda x: x["conf"], reverse=True)
    keep = []

    for c in cands:
        duplicated = False
        for k in keep:
            if box_iou(c["box"], k["box"]) >= iou_th:
                duplicated = True
                break
        if not duplicated:
            keep.append(c)

    return keep


def format_candidate(c):
    coords_str = " ".join(f"{v:.6f}" for v in c["coords"])
    return f"{c['cls']} {coords_str} {c['conf']:.6f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-dirs", nargs="+", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--out-label-dir", required=True)
    parser.add_argument("--nms-iou", type=float, default=0.85)
    parser.add_argument("--copy-empty", action="store_true")
    args = parser.parse_args()

    image_dir = Path(args.image_dir)
    out_dir = Path(args.out_label_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image_stems = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.JPG", "*.JPEG", "*.PNG", "*.BMP"]:
        image_stems.extend([p.stem for p in image_dir.glob(ext)])

    pred_dirs = [Path(p) for p in args.pred_dirs]

    total_in = 0
    total_out = 0
    non_empty_imgs = 0

    for stem in sorted(set(image_stems)):
        all_cands = []

        for d in pred_dirs:
            label_file = d / "labels" / f"{stem}.txt"
            if not label_file.exists():
                continue

            with open(label_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    c = parse_yolo_seg_line(line)
                    if c is not None:
                        all_cands.append(c)

        total_in += len(all_cands)

        keep = nms(all_cands, args.nms_iou) if all_cands else []
        total_out += len(keep)

        out_file = out_dir / f"{stem}.txt"

        if keep:
            non_empty_imgs += 1
            with open(out_file, "w", encoding="utf-8") as f:
                for c in keep:
                    f.write(format_candidate(c) + "\n")
        else:
            if args.copy_empty:
                out_file.touch()

    print(f"[DONE] image_dir={image_dir}")
    print(f"[DONE] pred_dirs={len(pred_dirs)}")
    print(f"[DONE] total candidates before merge: {total_in}")
    print(f"[DONE] total candidates after  merge: {total_out}")
    print(f"[DONE] non-empty prediction images: {non_empty_imgs}")
    print(f"[DONE] out_label_dir={out_dir}")


if __name__ == "__main__":
    main()
