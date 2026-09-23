from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_classifier_confmat(name, ckpt_path, out_dir):
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        print("[WARN] missing ckpt:", ckpt_path)
        return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    m = ckpt.get("val_metrics", {})

    tp = int(m.get("tp", 0))
    fp = int(m.get("fp", 0))
    tn = int(m.get("tn", 0))
    fn = int(m.get("fn", 0))

    # 行是真实类别，列是预测类别
    # background -> [TN, FP]
    # missing_coating -> [FN, TP]
    mat = np.array(
        [
            [tn, fp],
            [fn, tp],
        ]
    )

    labels = ["background", "missing_coating"]

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(mat)

    ax.set_title(f"{name} ROI Classifier Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticklabels(labels)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_png = Path(out_dir) / f"{name}_classifier_confusion_matrix.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    out_csv = Path(out_dir) / f"{name}_classifier_confusion_matrix.csv"
    pd.DataFrame(mat, index=labels, columns=labels).to_csv(out_csv)

    summary = {
        "name": name,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": m.get("precision", None),
        "recall": m.get("recall", None),
        "f1": m.get("f1", None),
        "acc": m.get("acc", None),
        "epoch": ckpt.get("epoch", None),
    }

    print("[OK]", out_png)
    return summary


def plot_classifier_summary(rows, out_dir):
    df = pd.DataFrame(rows)
    out_csv = Path(out_dir) / "classifier_v1_v2_v3_summary.csv"
    df.to_csv(out_csv, index=False)

    metrics = ["precision", "recall", "f1", "acc"]
    x = np.arange(len(df))

    fig, ax = plt.subplots(figsize=(9, 5))
    width = 0.18

    for k, metric in enumerate(metrics):
        vals = df[metric].astype(float).values
        ax.bar(x + (k - 1.5) * width, vals, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(df["name"].tolist())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("ROI Classifier Metrics: v1 vs v2 vs v3")
    ax.legend()
    fig.tight_layout()

    out_png = Path(out_dir) / "classifier_v1_v2_v3_metrics_bar.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print("[OK]", out_png)
    print("[OK]", out_csv)


def load_iou_summary(summary_csv):
    df = pd.read_csv(summary_csv)
    # 保证 stage1_conf 是字符串，方便画图
    df["stage1_conf"] = df["stage1_conf"].astype(str).str.zfill(3)
    df["stage2_p"] = df["stage2_p"].astype(str)
    return df


def plot_instance_confusion_bars(df, out_dir, split="test", iou_th=0.50):
    sub = df[(df["split"] == split) & (df["iou_th"] == iou_th)].copy()

    focus = []
    # Stage1 原始
    for conf in ["015", "010", "005"]:
        r = sub[(sub["method"] == "stage1") & (sub["stage1_conf"] == conf)]
        if len(r):
            rr = r.iloc[0].copy()
            rr["label"] = f"Stage1 c{conf}"
            focus.append(rr)

    # 常用 Stage2 组合
    combos = [
        ("015", "0.30"),
        ("015", "0.60"),
        ("015", "0.70"),
        ("010", "0.60"),
        ("005", "0.60"),
    ]

    for conf, p in combos:
        r = sub[
            (sub["method"] == "stage2_v3")
            & (sub["stage1_conf"] == conf)
            & (sub["stage2_p"].astype(str).isin([p, f"{float(p):.2f}"]))
        ]
        if len(r):
            rr = r.iloc[0].copy()
            rr["label"] = f"V3 c{conf} p{p}"
            focus.append(rr)

    fdf = pd.DataFrame(focus)
    out_csv = Path(out_dir) / f"{split}_iou{iou_th:.2f}_selected_instance_summary.csv"
    fdf.to_csv(out_csv, index=False)

    x = np.arange(len(fdf))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x - width, fdf["tp"], width, label="TP")
    ax.bar(x, fdf["fp"], width, label="FP")
    ax.bar(x + width, fdf["fn"], width, label="FN")

    ax.set_xticks(x)
    ax.set_xticklabels(fdf["label"].tolist(), rotation=25, ha="right")
    ax.set_ylabel("Instance Count")
    ax.set_title(f"{split} instance-level TP / FP / FN @ mask IoU={iou_th:.2f}")
    ax.legend()
    fig.tight_layout()

    out_png = Path(out_dir) / f"{split}_iou{iou_th:.2f}_tp_fp_fn_bar.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)

    print("[OK]", out_png)
    print("[OK]", out_csv)


def plot_prf_curves(df, out_dir, split="test", iou_th=0.50):
    sub = df[(df["split"] == split) & (df["iou_th"] == iou_th) & (df["method"] == "stage2_v3")].copy()

    # stage2_p 转数字
    sub["p_float"] = sub["stage2_p"].astype(float)

    for conf in ["015", "010", "005"]:
        cdf = sub[sub["stage1_conf"] == conf].sort_values("p_float")
        if len(cdf) == 0:
            continue

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(cdf["p_float"], cdf["precision"], marker="o", label="Precision")
        ax.plot(cdf["p_float"], cdf["recall"], marker="o", label="Recall")
        ax.plot(cdf["p_float"], cdf["f1"], marker="o", label="F1")

        ax.set_xlabel("Stage2 threshold p_mc")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.0)
        ax.set_title(f"{split} Stage2 V3 PR/F1 curve | Stage1 conf={conf} | IoU={iou_th:.2f}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()

        out_png = Path(out_dir) / f"{split}_v3_conf{conf}_prf_curve_iou{iou_th:.2f}.png"
        fig.savefig(out_png, dpi=180)
        plt.close(fig)
        print("[OK]", out_png)


def plot_precision_recall_scatter(df, out_dir, split="test", iou_th=0.50):
    sub = df[(df["split"] == split) & (df["iou_th"] == iou_th)].copy()

    fig, ax = plt.subplots(figsize=(7, 6))

    for method, marker in [("stage1", "x"), ("stage2_v3", "o")]:
        mdf = sub[sub["method"] == method]
        ax.scatter(mdf["recall"], mdf["precision"], label=method, marker=marker)

        for _, r in mdf.iterrows():
            if method == "stage1":
                label = f"c{r['stage1_conf']}"
            else:
                label = f"c{r['stage1_conf']}/p{r['stage2_p']}"
            ax.text(r["recall"] + 0.003, r["precision"] + 0.003, label, fontsize=7)

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 0.7)
    ax.set_ylim(0, 0.7)
    ax.set_title(f"{split} Precision-Recall scatter @ IoU={iou_th:.2f}")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    out_png = Path(out_dir) / f"{split}_precision_recall_scatter_iou{iou_th:.2f}.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print("[OK]", out_png)


def main():
    out_dir = Path("runs/segment/missing_coating_single_two_stage/report_figures")
    ensure_dir(out_dir)

    print("========== 1. ROI classifier confusion matrices ==========")
    ckpts = {
        "v1": "runs/two_stage/mc_roi_resnet18_v1/best_f1.pth",
        "v2": "runs/two_stage/mc_roi_resnet18_v2_candidates/best_f1.pth",
        "v3": "runs/two_stage/mc_roi_resnet18_v3_hybrid_balanced/best_f1.pth",
    }

    rows = []
    for name, ckpt in ckpts.items():
        row = save_classifier_confmat(name, ckpt, out_dir)
        if row is not None:
            rows.append(row)

    if rows:
        plot_classifier_summary(rows, out_dir)

    print("\n========== 2. System-level instance figures ==========")
    summary_csv = Path(
        "runs/segment/missing_coating_single_two_stage/two_stage_mask_instance_eval/mask_instance_iou_summary.csv"
    )

    if not summary_csv.exists():
        print("[ERROR] missing:", summary_csv)
        print("先运行 eval_mask_instance_iou.py 生成 mask_instance_iou_summary.csv")
        return

    df = load_iou_summary(summary_csv)

    plot_instance_confusion_bars(df, out_dir, split="test", iou_th=0.50)
    plot_prf_curves(df, out_dir, split="test", iou_th=0.50)
    plot_precision_recall_scatter(df, out_dir, split="test", iou_th=0.50)

    # val 也生成一份
    plot_instance_confusion_bars(df, out_dir, split="val", iou_th=0.50)
    plot_prf_curves(df, out_dir, split="val", iou_th=0.50)
    plot_precision_recall_scatter(df, out_dir, split="val", iou_th=0.50)

    print("\n[DONE] all figures saved to:", out_dir)


if __name__ == "__main__":
    main()
