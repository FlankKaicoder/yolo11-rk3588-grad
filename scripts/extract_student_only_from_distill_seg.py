from pathlib import Path
import sys
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path("/root/autodl-tmp/yolo11-rk3588-grad")
sys.path.insert(0, str(PROJECT_ROOT))

# 关键：保证自定义 DistillSegmentationModel 所在模块可被 import
import distill_seg_v2_fgbg.model  # noqa: F401

CLEAN_STUDENT_PT = PROJECT_ROOT / "runs/hier_2stage_seg/yolo11n_seg_stage1_defect_clean_imgsz640/weights/best.pt"

DISTILL_PT = PROJECT_ROOT / "runs/hier_2stage_seg/yolo11n_seg_stage1_defect_binary_teacher_distill_clean_lpos005_lneg002_h13_imgsz640/weights/best.pt"

OUT_PT = PROJECT_ROOT / "runs/hier_2stage_seg/yolo11n_seg_stage1_defect_binary_teacher_distill_clean_lpos005_lneg002_h13_imgsz640/weights/best_student_only.pt"

def count_params(model):
    return sum(p.numel() for p in model.parameters())

def main():
    print("[INFO] clean student:", CLEAN_STUDENT_PT)
    print("[INFO] distill ckpt:  ", DISTILL_PT)
    print("[INFO] output:        ", OUT_PT)

    # 1. 加载干净 YOLO student 结构
    clean_yolo = YOLO(str(CLEAN_STUDENT_PT))
    clean_model = clean_yolo.model.float()

    # 2. 加载蒸馏 checkpoint
    # PyTorch 2.6+ 默认 weights_only=True，会拒绝自定义类，所以这里显式关闭
    distill_ckpt = torch.load(DISTILL_PT, map_location="cpu", weights_only=False)
    distill_model = distill_ckpt["model"].float()

    print("[INFO] clean model class:  ", type(clean_model))
    print("[INFO] distill model class:", type(distill_model))
    print("[INFO] clean params:       ", count_params(clean_model))
    print("[INFO] distill params:     ", count_params(distill_model))

    clean_sd = clean_model.state_dict()
    distill_sd = distill_model.state_dict()

    matched = {}
    skipped = []

    for k, v in distill_sd.items():
        if k in clean_sd and clean_sd[k].shape == v.shape:
            matched[k] = v.detach().float()
        else:
            skipped.append(k)

    print(f"[INFO] clean state_dict keys:   {len(clean_sd)}")
    print(f"[INFO] distill state_dict keys: {len(distill_sd)}")
    print(f"[INFO] matched student keys:    {len(matched)}")
    print(f"[INFO] skipped distill keys:    {len(skipped)}")

    print("\n[INFO] first 20 skipped keys:")
    for k in skipped[:20]:
        print("  ", k)

    if len(matched) < len(clean_sd) * 0.8:
        print("\n[WARN] 匹配到的 student 参数太少，可能 key 前缀不一致。")
        print("[WARN] 先不要用输出权重，应该检查 distill_sd 的 key。")
        print("\n[INFO] first 30 distill keys:")
        for k in list(distill_sd.keys())[:30]:
            print("  ", k)
        return

    # 3. 把蒸馏模型中属于 YOLO student 的参数装进 clean YOLO student
    new_sd = clean_sd.copy()
    new_sd.update(matched)
    clean_model.load_state_dict(new_sd, strict=True)

    # 4. 用干净 checkpoint 的外壳保存，避免保存 DistillSegmentationModel
    clean_ckpt = torch.load(CLEAN_STUDENT_PT, map_location="cpu", weights_only=False)
    clean_ckpt["model"] = clean_model.float()
    clean_ckpt["ema"] = None
    clean_ckpt["optimizer"] = None

    if "train_args" not in clean_ckpt or clean_ckpt["train_args"] is None:
        clean_ckpt["train_args"] = {}

    clean_ckpt["train_args"]["name"] = "student_only_from_binary_teacher_distill"
    clean_ckpt["train_args"]["data"] = str(PROJECT_ROOT / "datasets/new_dataseg_clean_polygon_1cls_defect/data.yaml")

    OUT_PT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean_ckpt, OUT_PT)

    print("\n[OK] saved student-only checkpoint:", OUT_PT)

    # 5. 重新加载验证
    y = YOLO(str(OUT_PT))
    print("[CHECK] loaded class:", type(y.model))
    print("[CHECK] params:", count_params(y.model))
    print("[CHECK] layers:", len(y.model.model) if hasattr(y.model, "model") else "unknown")

if __name__ == "__main__":
    main()
