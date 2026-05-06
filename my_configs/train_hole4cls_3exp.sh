#!/bin/bash
set -e

DATA=/root/autodl-tmp/yolo11-rk3588-grad/datasets/datasets_detect_4cls_final/datasets_detect_4cls_final.yaml
PROJECT=/root/autodl-tmp/yolo11-rk3588-grad/runs/hole4cls_final

BASELINE_MODEL=yolo11n.pt
P2_MODEL=/root/autodl-tmp/yolo11-rk3588-grad/ultralytics/cfg/models/11/yolo11_p2.yaml
P2_ECA_MODEL=/root/autodl-tmp/yolo11-rk3588-grad/ultralytics/cfg/models/11/yolo11_p2_eca.yaml

EPOCHS=200
IMGSZ=640
BATCH=16
DEVICE=0
WORKERS=8
SEED=42

echo "===================="
echo "开始训练：baseline"
echo "===================="
yolo detect train \
  model=${BASELINE_MODEL} \
  data=${DATA} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  device=${DEVICE} \
  workers=${WORKERS} \
  seed=${SEED} \
  project=${PROJECT} \
  name=hole4cls_yolo11n_baseline

echo "===================="
echo "开始训练：P2"
echo "===================="
yolo detect train \
  model=${P2_MODEL} \
  data=${DATA} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  device=${DEVICE} \
  workers=${WORKERS} \
  seed=${SEED} \
  pretrained=yolo11n.pt \
  project=${PROJECT} \
  name=hole4cls_yolo11n_p2

echo "===================="
echo "开始训练：P2 + ECA"
echo "===================="
yolo detect train \
  model=${P2_ECA_MODEL} \
  data=${DATA} \
  epochs=${EPOCHS} \
  imgsz=${IMGSZ} \
  batch=${BATCH} \
  device=${DEVICE} \
  workers=${WORKERS} \
  seed=${SEED} \
  pretrained=yolo11n.pt \
  project=${PROJECT} \
  name=hole4cls_yolo11n_p2_eca

echo "===================="
echo "三个实验训练完成"
echo "===================="
