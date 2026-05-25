import argparse
import copy
import os
from datetime import datetime
from pathlib import Path

# 关键：必须先导入自定义蒸馏模型模块，torch.load 才能反序列化蒸馏版 checkpoint
import distill_seg.model  # noqa: F401
import torch

from ultralytics import YOLO


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--distill-ckpt",
        type=str,
        required=True,
        help="蒸馏训练得到的 best.pt / last.pt",
    )
    parser.add_argument(
        "--base-model",
        type=str,
        default="yolo11n-seg.pt",
        help="原始 student 模型壳子",
    )
    parser.add_argument(
        "--out",
        type=str,
        required=True,
        help="导出的 student-only 权重路径",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    distill_ckpt_path = Path(args.distill_ckpt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[1/5] Loading distill checkpoint: {distill_ckpt_path}")
    ckpt = torch.load(distill_ckpt_path, map_location="cpu", weights_only=False)

    # 优先取 ema，因为通常 best.pt 里 ema 更接近最终验证性能
    src_model = ckpt.get("ema", None)
    if src_model is None:
        src_model = ckpt.get("model", None)

    if src_model is None:
        raise RuntimeError("没有在 checkpoint 中找到 'ema' 或 'model'。")

    src_model = src_model.float().eval()

    print(f"[2/5] Building clean student model from: {args.base_model}")
    base_yolo = YOLO(args.base_model)
    student = base_yolo.model.float().eval()

    src_state = src_model.state_dict()
    dst_state = student.state_dict()

    print("[3/5] Filtering student-compatible parameters")
    filtered_state = {}
    skipped = []

    for k, v in src_state.items():
        # 显式跳过蒸馏专用模块
        if k.startswith("teacher.") or k.startswith("adapter."):
            skipped.append(k)
            continue

        if k in dst_state and dst_state[k].shape == v.shape:
            filtered_state[k] = v
        else:
            skipped.append(k)

    missing, unexpected = student.load_state_dict(filtered_state, strict=False)

    # 复制类别信息，避免 names 沿用 base-model 默认值
    if hasattr(src_model, "names"):
        student.names = copy.deepcopy(src_model.names)

    if hasattr(src_model, "nc"):
        student.nc = src_model.nc

    print(f"  loaded keys     : {len(filtered_state)}")
    print(f"  missing keys    : {len(missing)}")
    print(f"  unexpected keys : {len(unexpected)}")
    print(f"  skipped keys    : {len(skipped)}")

    if len(filtered_state) == 0:
        raise RuntimeError("没有成功加载任何 student 参数，请检查 base-model 是否匹配。")

    print(f"[4/5] Saving clean student-only checkpoint to: {out_path}")

    # 尽量模拟 Ultralytics 的保存风格，方便后续直接 YOLO(out_path) 使用
    save_model = copy.deepcopy(student).half().eval()

    export_ckpt = {
        "date": datetime.now().isoformat(),
        "model": save_model,
        "ema": None,
        "optimizer": None,
        "train_args": ckpt.get("train_args", {}),
        "epoch": ckpt.get("epoch", -1),
        "best_fitness": ckpt.get("best_fitness", None),
        "license": "AGPL-3.0",
        "docs": "https://docs.ultralytics.com",
    }

    torch.save(export_ckpt, out_path)

    size_mb = os.path.getsize(out_path) / (1024 * 1024)
    print(f"[5/5] Done. Exported file size: {size_mb:.2f} MB")

    print("\n建议下一步：")
    print(
        f"yolo segment val model={out_path} data=/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg/data.yaml imgsz=640 device=0"
    )


if __name__ == "__main__":
    main()
