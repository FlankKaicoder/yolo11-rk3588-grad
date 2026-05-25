import argparse

from distill_seg_v2_fgbg.model import DistillSegmentationModel
from ultralytics.models.yolo.segment.train import SegmentationTrainer


def unwrap_model(m):
    return m.module if hasattr(m, "module") else m


class DistillSegTrainer(SegmentationTrainer):
    def __init__(self, *args, distill_cfg=None, **kwargs):
        self.distill_cfg = distill_cfg
        super().__init__(*args, **kwargs)

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = DistillSegmentationModel(
            cfg=cfg,
            ch=3,
            nc=self.data["nc"],
            verbose=verbose,
            distill_cfg=self.distill_cfg,
        )

        if weights is not None:
            model.load(weights)

        return model

    def save_model(self):
        raw_model = unwrap_model(self.model)

        ema_model = None
        if getattr(self, "ema", None) is not None and hasattr(self.ema, "ema"):
            ema_model = unwrap_model(self.ema.ema)

        if hasattr(raw_model, "_remove_distill_hook"):
            raw_model._remove_distill_hook()

        if ema_model is not None and hasattr(ema_model, "_remove_distill_hook"):
            ema_model._remove_distill_hook()

        try:
            super().save_model()
        finally:
            if hasattr(raw_model, "_register_distill_hook"):
                raw_model._register_distill_hook()


def parse_args():
    parser = argparse.ArgumentParser()

    # 基础训练配置
    parser.add_argument("--seg-model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--data", type=str, required=True)

    # FG/BG teacher
    parser.add_argument("--teacher-ckpt", type=str, required=True)
    parser.add_argument("--hook-idx", type=int, default=13)

    # 新版蒸馏权重
    parser.add_argument("--lambda-pos", type=float, default=0.1)
    parser.add_argument("--lambda-neg", type=float, default=0.1)

    # ROI 采样参数
    parser.add_argument("--easy-bg-per-image", type=int, default=0)
    parser.add_argument("--near-bg-per-defect", type=int, default=1)
    parser.add_argument("--min-crop-size", type=int, default=48)
    parser.add_argument("--max-iou-bg-with-gt", type=float, default=0.05)

    # 训练参数
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--workers", type=int, default=8)

    # 输出目录
    parser.add_argument(
        "--project",
        type=str,
        default="/root/autodl-tmp/yolo11-rk3588-grad/runs/segment/distill_fgbg",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="yolo11n_seg_fgbg_distill_v1",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    distill_cfg = {
        "teacher_ckpt": args.teacher_ckpt,
        "teacher_input_size": 224,
        "teacher_mean": (0.485, 0.456, 0.406),
        "teacher_std": (0.229, 0.224, 0.225),
        "teacher_out_dim": 512,
        "student_feat_dim": 128,
        "hook_idx": args.hook_idx,
        "lambda_pos": args.lambda_pos,
        "lambda_neg": args.lambda_neg,
        "easy_bg_per_image": args.easy_bg_per_image,
        "near_bg_per_defect": args.near_bg_per_defect,
        "min_crop_size": args.min_crop_size,
        "max_iou_bg_with_gt": args.max_iou_bg_with_gt,
    }

    overrides = {
        "model": args.seg_model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
        "exist_ok": True,
        "optimizer": "auto",
        # 第一版先关掉重增强
        "mosaic": 0.0,
        "mixup": 0.0,
        "copy_paste": 0.0,
    }

    print("[FG/BG Distill Config]")
    for k, v in distill_cfg.items():
        print(f"  {k}: {v}")

    trainer = DistillSegTrainer(
        overrides=overrides,
        distill_cfg=distill_cfg,
    )
    trainer.train()


if __name__ == "__main__":
    main()
