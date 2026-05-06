import pandas as pd
from pathlib import Path

BASE = Path("/root/autodl-tmp/yolo11-rk3588-grad/runs/repr_analysis_50_models_v1")

MODELS = {
    "ce50_bestf1": BASE / "ce50_bestf1" / "val_center_distance_matrix.csv",
    "bcl50_bestf1": BASE / "bcl50_bestf1" / "val_center_distance_matrix.csv",
    "bcl2stage50_direct_bestf1": BASE / "bcl2stage50_direct_bestf1" / "val_center_distance_matrix.csv",
    "bcl2stage50_freeze_bestf1": BASE / "bcl2stage50_freeze_bestf1" / "val_center_distance_matrix.csv",
}

FOCUS_PAIRS = [
    ("corrosion", "missing_material"),
    ("corrosion", "missing_coating"),
    ("missing_material", "missing_coating"),
    ("carbon", "corrosion"),
    ("carbon", "missing_material"),
]

rows = []
for model_name, csv_path in MODELS.items():
    df = pd.read_csv(csv_path, index_col=0)
    for a, b in FOCUS_PAIRS:
        rows.append({
            "model": model_name,
            "pair": f"{a} <-> {b}",
            "distance": float(df.loc[a, b]),
        })

out_df = pd.DataFrame(rows)
print(out_df)
out_df.to_csv(BASE / "focus_pairwise_distances.csv", index=False, encoding="utf-8-sig")
print(f"\nSaved to: {BASE / 'focus_pairwise_distances.csv'}")