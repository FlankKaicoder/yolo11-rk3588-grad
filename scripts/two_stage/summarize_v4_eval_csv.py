from pathlib import Path
import pandas as pd
import re
import argparse


def infer_split(s):
    s = str(s)
    if "test" in s:
        return "test"
    if "val" in s:
        return "val"
    return ""


def infer_p(s):
    s = str(s)
    m = re.search(r"p([0-9]+\.[0-9]+)", s)
    if m:
        return float(m.group(1))
    m = re.search(r"threshold[_=-]?([0-9]+\.[0-9]+)", s)
    if m:
        return float(m.group(1))
    return None


def infer_stage(s):
    s = str(s)
    if "stage1" in s:
        return "stage1"
    if "stage2" in s:
        return "stage2"
    return ""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.csv_root)
    csvs = sorted(root.rglob("*.csv"))

    if not csvs:
        print(f"[ERR] no csv found under {root}")
        return

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
        except Exception as e:
            print(f"[WARN] failed to read {f}: {e}")
            continue

        df["source_file"] = str(f)
        df["source_name"] = f.name

        if "split" not in df.columns:
            df["split"] = df["source_file"].map(infer_split)

        if "p" not in df.columns and "threshold" not in df.columns:
            df["p"] = df["source_file"].map(infer_p)

        if "stage" not in df.columns:
            df["stage"] = df["source_file"].map(infer_stage)

        dfs.append(df)

    if not dfs:
        print("[ERR] all csv read failed")
        return

    all_df = pd.concat(dfs, ignore_index=True, sort=False)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    all_df.to_csv(out, index=False)

    print(f"[DONE] merged csv saved to: {out}")
    print(f"[INFO] shape: {all_df.shape}")
    print(f"[INFO] columns: {list(all_df.columns)}")

    print("\n===== preview =====")
    print(all_df.head(20).to_string(index=False))

    # 自动挑常见指标列
    prefer_cols = []
    for c in [
        "split", "stage", "p", "threshold", "iou_th", "iou", 
        "precision", "recall", "f1", "tp", "fp", "fn",
        "TP", "FP", "FN", "P", "R", "F1",
        "source_name"
    ]:
        if c in all_df.columns and c not in prefer_cols:
            prefer_cols.append(c)

    if prefer_cols:
        print("\n===== selected columns =====")
        print(all_df[prefer_cols].to_string(index=False))

    # 如果有 f1/recall/precision，就按 split + iou + p 排序输出
    score_col = None
    for c in ["f1", "F1", "mF1", "mask_f1"]:
        if c in all_df.columns:
            score_col = c
            break

    if score_col:
        print(f"\n===== best rows by {score_col} =====")
        sort_cols = []
        for c in ["split", "iou_th", "iou", "p", "threshold"]:
            if c in all_df.columns:
                sort_cols.append(c)
        show = all_df.sort_values(score_col, ascending=False)
        print(show[prefer_cols].head(30).to_string(index=False))


if __name__ == "__main__":
    main()
