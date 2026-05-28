import argparse
import shutil
from pathlib import Path
import pandas as pd

def load_cases(path, prefix):
    df = pd.read_csv(path)
    df = df.rename(columns={
        "pred_count": f"{prefix}_pred",
        "case": f"{prefix}_case",
        "diff_pred_minus_gt": f"{prefix}_diff",
    })
    keep = ["image", "gt_count", f"{prefix}_pred", f"{prefix}_case", f"{prefix}_diff"]
    return df[keep]

def severity_rank(case):
    # 越大越差，用来判断是否改善
    return {
        "ok": 0,
        "under": 1,
        "over": 2,
        "false_positive": 3,
        "miss": 4,
    }.get(case, 9)

def tag_row(r):
    s1 = r["stage1_case"]
    v3 = r["v3_case"]
    s1_pred = r["stage1_pred"]
    v3_pred = r["v3_pred"]
    gt = r["gt_count"]

    # Stage1 假阳性被 v3 删掉
    if s1 == "false_positive" and v3 == "ok":
        return "01_v3_removed_false_positive"

    # Stage1 over，v3 变 ok
    if s1 == "over" and v3 == "ok":
        return "02_v3_fixed_over_to_ok"

    # Stage1 over，v3 还是 over 但数量下降
    if s1 == "over" and v3 == "over" and v3_pred < s1_pred:
        return "03_v3_reduced_over"

    # Stage1 over，v3 变 under，说明删多了
    if s1 == "over" and v3 == "under":
        return "04_v3_over_to_under_deleted_too_much"

    # Stage1 ok，v3 变 miss/under，明显变坏
    if s1 == "ok" and v3 in ["miss", "under"]:
        return "05_v3_hurt_ok_sample"

    # Stage1 有预测，v3 变 miss
    if s1 != "miss" and v3 == "miss":
        return "06_v3_created_or_kept_miss"

    # v3 仍然假阳性
    if v3 == "false_positive":
        return "07_v3_still_false_positive"

    # v3 仍然 over
    if v3 == "over":
        return "08_v3_still_over"

    # v3 仍然 under
    if v3 == "under":
        return "09_v3_still_under"

    # Stage1 miss，v3 也 miss，本质是一阶段没候选
    if s1 == "miss" and v3 == "miss":
        return "10_stage1_miss_stage2_cannot_fix"

    # v3 比 stage1 更好
    if severity_rank(v3) < severity_rank(s1):
        return "11_v3_improved_other"

    # v3 比 stage1 更差
    if severity_rank(v3) > severity_rank(s1):
        return "12_v3_worse_other"

    return "13_no_obvious_change"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--saved-root", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--threshold", default="p0.30")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    root = Path(args.saved_root)
    split = args.split
    th = args.threshold
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stage1 = load_cases(root / "stage1_conf015_iou030" / split / "cases.csv", "stage1")
    v1 = load_cases(root / "v1" / split / th / "cases.csv", "v1")
    v2 = load_cases(root / "v2" / split / th / "cases.csv", "v2")
    v3 = load_cases(root / "v3" / split / th / "cases.csv", "v3")

    df = stage1.merge(v1, on=["image", "gt_count"], how="left")
    df = df.merge(v2, on=["image", "gt_count"], how="left")
    df = df.merge(v3, on=["image", "gt_count"], how="left")

    df["v3_tag"] = df.apply(tag_row, axis=1)

    # 为了方便看，优先显示非 ok 或发生变化的
    def changed(r):
        cases = [r["stage1_case"], r["v1_case"], r["v2_case"], r["v3_case"]]
        return len(set(cases)) > 1 or any(c != "ok" for c in cases)

    df["changed_or_bad"] = df.apply(changed, axis=1)
    df2 = df[df["changed_or_bad"]].copy()

    csv_path = out_dir / f"case_compare_{split}_{th}.csv"
    df2.to_csv(csv_path, index=False)

    print("[OK] saved compare csv:", csv_path)
    print()
    print("===== v3_tag 统计 =====")
    print(df2["v3_tag"].value_counts())
    print()
    print("===== 前 50 条对比 =====")
    show_cols = [
        "image", "gt_count",
        "stage1_pred", "stage1_case",
        "v1_pred", "v1_case",
        "v2_pred", "v2_case",
        "v3_pred", "v3_case",
        "v3_tag",
    ]
    print(df2[show_cols].head(50).to_string(index=False))

    # 复制可视化图片到分类文件夹
    vis_src = root / "visual_compare" / split / th
    vis_out = out_dir / f"selected_visual_{split}_{th}"
    vis_out.mkdir(parents=True, exist_ok=True)

    copied = 0
    for _, r in df2.iterrows():
        stem = Path(r["image"]).stem
        src_img = vis_src / f"{stem}.jpg"
        if not src_img.exists():
            continue

        tag_dir = vis_out / r["v3_tag"]
        tag_dir.mkdir(parents=True, exist_ok=True)

        dst_img = tag_dir / f"{stem}.jpg"
        shutil.copy2(src_img, dst_img)
        copied += 1

    print()
    print("[OK] copied visual images:", copied)
    print("[OK] visual out:", vis_out)

if __name__ == "__main__":
    main()
