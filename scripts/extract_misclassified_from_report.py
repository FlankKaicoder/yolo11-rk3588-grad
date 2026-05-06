import json
import csv
import shutil
from pathlib import Path


# ========= 修改这里 =========
REPORT_JSON =Path("/root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_bcl_resnet18_fix_nosampler_ce_v1/reports/epoch_034_report.json")
OUT_DIR = Path("/root/autodl-tmp/yolo11-rk3588-grad/runs/error_analysis/singlebcl_epoch_034_miscls")
COPY_MODE = "copy"   # "copy" 或 "symlink"
# ==========================


def get_class_names(report_data):
    report = report_data["classification_report"]
    ignore_keys = {"accuracy", "macro avg", "weighted avg"}
    class_names = [k for k in report.keys() if k not in ignore_keys]
    return class_names


def safe_symlink(src: Path, dst: Path):
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    dst.symlink_to(src)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(REPORT_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    class_names = get_class_names(data)
    samples = data["samples"]

    csv_path = OUT_DIR / "misclassified.csv"
    summary_path = OUT_DIR / "summary.txt"

    misclassified = []

    for s in samples:
        y_true = s["y_true"]
        y_pred = s["y_pred"]
        if y_true == y_pred:
            continue

        true_name = class_names[y_true]
        pred_name = class_names[y_pred]
        img_path = Path(s["path"])
        probs = s["probs"]

        misclassified.append({
            "path": str(img_path),
            "filename": img_path.name,
            "true_idx": y_true,
            "pred_idx": y_pred,
            "true_name": true_name,
            "pred_name": pred_name,
            "pred_prob": probs[y_pred],
            "true_prob": probs[y_true],
            "all_probs": probs,
        })

    # 保存 csv
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow([
            "filename", "path",
            "true_idx", "true_name",
            "pred_idx", "pred_name",
            "pred_prob", "true_prob",
            "all_probs"
        ])
        for m in misclassified:
            writer.writerow([
                m["filename"], m["path"],
                m["true_idx"], m["true_name"],
                m["pred_idx"], m["pred_name"],
                f"{m['pred_prob']:.6f}",
                f"{m['true_prob']:.6f}",
                m["all_probs"]
            ])

    # 按 true->pred 分目录整理图像
    pair_count = {}
    for m in misclassified:
        pair_name = f"{m['true_name']}__TO__{m['pred_name']}"
        pair_dir = OUT_DIR / pair_name
        pair_dir.mkdir(parents=True, exist_ok=True)

        src = Path(m["path"])
        dst = pair_dir / src.name

        if src.exists():
            try:
                if COPY_MODE == "symlink":
                    safe_symlink(src, dst)
                else:
                    shutil.copy2(src, dst)
            except Exception as e:
                print(f"[WARN] failed to place {src}: {e}")
        else:
            print(f"[WARN] file not found: {src}")

        pair_count[pair_name] = pair_count.get(pair_name, 0) + 1

    # 保存 summary
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"report_json: {REPORT_JSON}\n")
        f.write(f"total_samples: {len(samples)}\n")
        f.write(f"misclassified: {len(misclassified)}\n\n")
        f.write("Misclassification pairs:\n")
        for k, v in sorted(pair_count.items(), key=lambda x: (-x[1], x[0])):
            f.write(f"{k}: {v}\n")

    print(f"Done.")
    print(f"Output dir: {OUT_DIR}")
    print(f"CSV saved to: {csv_path}")
    print(f"Summary saved to: {summary_path}")
    print(f"Misclassified samples: {len(misclassified)}")


if __name__ == "__main__":
    main()