from pathlib import Path
import csv
import numpy as np
import matplotlib.pyplot as plt


def save_csv(path, row_labels, col_labels, mat, na_mask=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["true\\pred"] + col_labels)
        for i, r in enumerate(row_labels):
            row = [r]
            for j, v in enumerate(mat[i]):
                if na_mask is not None and na_mask[i][j]:
                    row.append("N/A")
                else:
                    row.append(int(v))
            writer.writerow(row)


def plot_cm(path, title, row_labels, col_labels, mat, na_mask=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot_mat = np.array(mat, dtype=float)
    if na_mask is not None:
        for i in range(plot_mat.shape[0]):
            for j in range(plot_mat.shape[1]):
                if na_mask[i][j]:
                    plot_mat[i, j] = 0

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(plot_mat)

    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Ground Truth")

    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=20, ha="right")

    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)

    max_val = plot_mat.max() if plot_mat.size else 0

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            if na_mask is not None and na_mask[i][j]:
                text = "N/A"
            else:
                text = str(int(mat[i][j]))
            ax.text(
                j, i, text,
                ha="center",
                va="center",
                color="white" if plot_mat[i, j] > max_val * 0.5 else "black",
                fontsize=12
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def make_detection_cm(out_root, name, title, tp, fp, fn):
    row_labels = ["missing_coating", "background"]
    col_labels = ["missing_coating", "background"]

    mat = [
        [tp, fn],
        [fp, 0],
    ]

    na_mask = [
        [False, False],
        [False, True],
    ]

    save_csv(
        out_root / f"{name}.csv",
        row_labels,
        col_labels,
        mat,
        na_mask,
    )

    plot_cm(
        out_root / f"{name}.png",
        title,
        row_labels,
        col_labels,
        mat,
        na_mask,
    )


def main():
    out_root = Path(
        "/root/autodl-tmp/yolo11-rk3588-grad/"
        "runs/segment/missing_coating_single_two_stage/"
        "v4_ms_tta_union/confusion_matrices_v4_v5"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    # V4 main result:
    # test @ mask IoU=0.50, p=0.40
    # TP=29 FP=234 FN=26
    make_detection_cm(
        out_root=out_root,
        name="v4_detection_cm_p0.40_iou0.50",
        title="V4 Detection Confusion Matrix\nStage1 multi-scale + old Stage2 v3, p=0.40, IoU=0.50",
        tp=29,
        fp=234,
        fn=26,
    )

    # V4 stronger-filter comparison:
    # test @ mask IoU=0.50, p=0.60
    # TP=28 FP=215 FN=27
    make_detection_cm(
        out_root=out_root,
        name="v4_detection_cm_p0.60_iou0.50",
        title="V4 Detection Confusion Matrix\nStage1 multi-scale + old Stage2 v3, p=0.60, IoU=0.50",
        tp=28,
        fp=215,
        fn=27,
    )

    # V5 main result:
    # test @ mask IoU=0.50, p=0.30
    # TP=31 FP=203 FN=24
    make_detection_cm(
        out_root=out_root,
        name="v5_detection_cm_p0.30_iou0.50",
        title="V5 Detection Confusion Matrix\nStage1 multi-scale + hard-negative Stage2, p=0.30, IoU=0.50",
        tp=31,
        fp=203,
        fn=24,
    )

    # V5 stronger-filter comparison:
    # test @ mask IoU=0.50, p=0.50
    # TP=30 FP=191 FN=25
    make_detection_cm(
        out_root=out_root,
        name="v5_detection_cm_p0.50_iou0.50",
        title="V5 Detection Confusion Matrix\nStage1 multi-scale + hard-negative Stage2, p=0.50, IoU=0.50",
        tp=30,
        fp=191,
        fn=25,
    )

    print(f"[DONE] saved to: {out_root}")
    for p in sorted(out_root.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
