from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

summary_csv = Path(
    "runs/segment/missing_coating_single_two_stage/two_stage_mask_instance_eval/mask_instance_iou_summary.csv"
)
out_dir = Path("runs/segment/missing_coating_single_two_stage/report_figures/detection_confusion_matrices")
out_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(summary_csv)
df["stage1_conf"] = df["stage1_conf"].astype(str).str.zfill(3)
df["stage2_p"] = df["stage2_p"].astype(str)

targets = [
    ("stage1", "015", "none", "Stage1_conf015"),
    ("stage2_v3", "015", "0.30", "V3_conf015_p030"),
    ("stage2_v3", "015", "0.60", "V3_conf015_p060"),
    ("stage2_v3", "015", "0.70", "V3_conf015_p070"),
    ("stage1", "005", "none", "Stage1_conf005"),
    ("stage2_v3", "005", "0.60", "V3_conf005_p060"),
]

for method, conf, p, name in targets:
    sub = df[(df["split"] == "test") & (df["iou_th"] == 0.50) & (df["method"] == method) & (df["stage1_conf"] == conf)]

    if p != "none":
        sub = sub[sub["stage2_p"].astype(float).round(2) == float(p)]
    else:
        sub = sub[sub["stage2_p"] == "none"]

    if len(sub) == 0:
        print("[WARN] missing", name)
        continue

    r = sub.iloc[0]
    tp = int(r["tp"])
    fp = int(r["fp"])
    fn = int(r["fn"])

    # 检测任务没有真正 TN，这里用 -1 表示 N/A
    mat = np.array(
        [
            [tp, fn],
            [fp, 0],
        ]
    )

    fig, ax = plt.subplots(figsize=(5.8, 4.8))
    im = ax.imshow(mat)

    ax.set_title(f"{name}\nDetection Confusion @ IoU=0.50")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred defect", "Pred none"])
    ax.set_yticklabels(["GT defect", "GT none"])

    texts = [
        [f"TP\n{tp}", f"FN\n{fn}"],
        [f"FP\n{fp}", "TN\nN/A"],
    ]

    for i in range(2):
        for j in range(2):
            ax.text(j, i, texts[i][j], ha="center", va="center", fontsize=12)

    fig.colorbar(im, ax=ax)
    fig.tight_layout()

    out_png = out_dir / f"{name}_detection_confusion.png"
    fig.savefig(out_png, dpi=180)
    plt.close(fig)
    print("[OK]", out_png)

print("[DONE] saved to:", out_dir)
