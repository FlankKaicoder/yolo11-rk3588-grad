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
                ha="center", va="center",
                color="white" if plot_mat[i, j] > max_val * 0.5 else "black",
                fontsize=12
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main():
    out_root = Path("/root/autodl-tmp/yolo11-rk3588-grad/runs/segment/missing_coating_single_two_stage/v4_ms_tta_union/final_v5/confusion_matrices")
    out_root.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Detection / segmentation confusion matrix
    # Rows: GT, Columns: Prediction
    #
    # For detection, TN is not well-defined because "background-background"
    # regions are not enumerated as instances. Therefore bottom-right is N/A.
    #
    # p=0.30 from v5 test @ mask IoU=0.50:
    # TP=31, FP=203, FN=24
    # ============================================================

    row_labels_det = ["missing_coating", "background"]
    col_labels_det = ["missing_coating", "background"]

    mat_p030 = [
        [31, 24],   # GT missing_coating: TP, FN
        [203, 0],   # GT background: FP, TN undefined
    ]
    na_det = [
        [False, False],
        [False, True],
    ]

    save_csv(
        out_root / "v5_detection_cm_p0.30_iou0.50.csv",
        row_labels_det,
        col_labels_det,
        mat_p030,
        na_det,
    )
    plot_cm(
        out_root / "v5_detection_cm_p0.30_iou0.50.png",
        "V5 Detection Confusion Matrix\np=0.30, mask IoU=0.50",
        row_labels_det,
        col_labels_det,
        mat_p030,
        na_det,
    )

    # p=0.50 from v5 test @ mask IoU=0.50:
    # TP=30, FP=191, FN=25
    mat_p050 = [
        [30, 25],
        [191, 0],
    ]

    save_csv(
        out_root / "v5_detection_cm_p0.50_iou0.50.csv",
        row_labels_det,
        col_labels_det,
        mat_p050,
        na_det,
    )
    plot_cm(
        out_root / "v5_detection_cm_p0.50_iou0.50.png",
        "V5 Detection Confusion Matrix\np=0.50, mask IoU=0.50",
        row_labels_det,
        col_labels_det,
        mat_p050,
        na_det,
    )

    # ============================================================
    # V5 ROI classifier validation confusion matrix
    #
    # class_to_idx:
    # background = 0
    # missing_coating = 1
    #
    # Best epoch:
    # TP=80, FP=48, TN=124, FN=24
    #
    # Rows: GT, Columns: Prediction
    # ============================================================

    row_labels_cls = ["background", "missing_coating"]
    col_labels_cls = ["background", "missing_coating"]

    mat_cls = [
        [124, 48],  # GT background: TN, FP
        [24, 80],   # GT missing_coating: FN, TP
    ]

    save_csv(
        out_root / "v5_roi_classifier_val_cm.csv",
        row_labels_cls,
        col_labels_cls,
        mat_cls,
    )
    plot_cm(
        out_root / "v5_roi_classifier_val_cm.png",
        "V5 ROI Classifier Confusion Matrix\nValidation Set",
        row_labels_cls,
        col_labels_cls,
        mat_cls,
    )

    print(f"[DONE] saved confusion matrices to: {out_root}")
    for p in sorted(out_root.glob("*")):
        print(p)


if __name__ == "__main__":
    main()
