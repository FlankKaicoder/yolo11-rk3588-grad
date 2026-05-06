import os

def main():
    cmd = r"""
python train_yolo_seg_distill.py \
  --seg-model yolo11n-seg.pt \
  --data /root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg/data.yaml \
  --teacher-ckpt /root/autodl-tmp/yolo11-rk3588-grad/runs/patch_cls/patch_bcl_resnet18_fix_nosampler_ce_v1/checkpoints/best_macro_f1.pth \
  --hook-idx 13 \
  --lambda-dist 0.2 \
  --imgsz 640 \
  --epochs 200 \
  --batch 32 \
  --device 0 \
  --workers 8 \
  --project /root/autodl-tmp/yolo11-rk3588-grad/runs/segment/distill \
  --name yolo11n_seg_bcl_distill_v1
"""
    os.system(cmd)

if __name__ == "__main__":
    main()