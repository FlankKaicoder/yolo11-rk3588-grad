#!/bin/bash
set -e

DATA=/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect_4cls_final/datasets_detect_4cls_final.yaml
PROJECT=/root/autodl-tmp/yolo11-rk3588-grad/runs/hole4cls_final

P2_SIMAM_MODEL=/root/autodl-tmp/yolo11-rk3588-grad/ultralytics/cfg/models/11/yolo11_p2_simam.yaml
P2_CBAM_MODEL=/root/autodl-tmp/yolo11-rk3588-grad/ultralytics/cfg/models/11/yolo11_p2_cbam.yaml

EPOCHS=200
IMGSZ=640
BATCH=16
DEVICE=0
WORKERS=8
SEED=42

echo "===================="
echo "开始训练：P2 + SimAM"
echo "===================="
yolo detect train \
  model=${P2_SIMAM_MODEL} \
  data=${DATA} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  device=${DEVICE} \
  workers=${WORKERS} \
  seed=${SEED} \
  pretrained=yolo11n.pt \
  project=${PROJECT} \
  name=hole4cls_yolo11n_p2_simam

echo "===================="
echo "开始训练：P2 + CBAM"
echo "===================="
yolo detect train \
  model=${P2_CBAM_MODEL} \
  data=${DATA} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  device=${DEVICE} \
  workers=${WORKERS} \
  seed=${SEED} \
  pretrained=yolo11n.pt \
  project=${PROJECT} \
  name=hole4cls_yolo11n_p2_cbam

echo "===================="
echo "SimAM 和 CBAM 两个实验训练完成"
echo "===================="
