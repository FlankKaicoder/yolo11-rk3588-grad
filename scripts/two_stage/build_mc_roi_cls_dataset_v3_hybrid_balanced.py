import argparse
import random
import shutil
from pathlib import Path

import pandas as pd


def safe_copy(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def sample_rows(df, n, rng, replace=False):
    if len(df) == 0 or n <= 0:
        return df.iloc[0:0].copy()
    if replace:
        idxs = [rng.randrange(len(df)) for _ in range(n)]
        return df.iloc[idxs].copy()
    n = min(n, len(df))
    idxs = rng.sample(range(len(df)), n)
    return df.iloc[idxs].copy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1-index", required=True)
    ap.add_argument("--v2-index", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--gt-supp-ratio", type=float, default=0.30)
    ap.add_argument("--neg-ratio", type=float, default=1.50)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    v1 = pd.read_csv(args.v1_index)
    v2 = pd.read_csv(args.v2_index)

    out_dir = Path(args.out_dir)
    if out_dir.exists():
        print("[WARN] out_dir exists, files may be overwritten:", out_dir)

    rows_out = []

    for split in ["train", "val"]:
        v1s = v1[v1["split"] == split].copy()
        v2s = v2[v2["split"] == split].copy()

        # 正样本：以 YOLO candidate positive 为主
        cand_pos = v2s[(v2s["class"] == "missing_coating") & (v2s["source"] == "candidate_positive")].copy()

        # 少量 GT positive 补充，避免只学 YOLO 框偏差
        gt_pos_pool = v1s[(v1s["class"] == "missing_coating") & (v1s["source"] == "gt_positive")].copy()
        n_gt_supp = int(len(cand_pos) * args.gt_supp_ratio)
        gt_pos = sample_rows(gt_pos_pool, n_gt_supp, rng, replace=False)

        pos = pd.concat([cand_pos, gt_pos], ignore_index=True)

        # 负样本：candidate negative + hard false positive + random bg
        cand_neg = v2s[(v2s["class"] == "background") & (v2s["source"] == "candidate_negative")].copy()
        hard_neg = v1s[(v1s["class"] == "background") & (v1s["source"] == "hard_false_positive")].copy()
        rand_neg_pool = v1s[
            (v1s["class"] == "background")
            & (v1s["source"].isin(["random_background_empty", "random_background_posimg"]))
        ].copy()

        base_neg = pd.concat([cand_neg, hard_neg], ignore_index=True)
        target_neg = int(max(len(base_neg), len(pos) * args.neg_ratio))
        need_rand = max(0, target_neg - len(base_neg))

        rand_neg = sample_rows(rand_neg_pool, need_rand, rng, replace=(need_rand > len(rand_neg_pool)))
        neg = pd.concat([base_neg, rand_neg], ignore_index=True)

        final = pd.concat([pos, neg], ignore_index=True)
        final = final.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        print(f"\n===== {split} =====")
        print("candidate_positive:", len(cand_pos))
        print("gt_positive_supp:", len(gt_pos))
        print("candidate_negative:", len(cand_neg))
        print("hard_false_positive:", len(hard_neg))
        print("random_negative:", len(rand_neg))
        print("final positive:", len(pos))
        print("final negative:", len(neg))

        for i, r in final.iterrows():
            cls = r["class"]
            src = Path(r["crop"])

            prefix = "v2" if "candidate_" in str(r["source"]) else "v1"
            dst_name = f"{prefix}_{r['source']}_{i:06d}_{src.name}"
            dst = out_dir / split / cls / dst_name

            safe_copy(src, dst)

            rr = r.to_dict()
            rr["new_crop"] = str(dst)
            rows_out.append(rr)

    out_index = out_dir / "index.csv"
    out_index.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows_out).to_csv(out_index, index=False)

    print("\n[OK] saved:", out_dir)
    print("[OK] index:", out_index)

    for split in ["train", "val"]:
        for cls in ["missing_coating", "background"]:
            n = len(list((out_dir / split / cls).glob("*.jpg")))
            print(f"{split}/{cls}: {n}")


if __name__ == "__main__":
    main()
