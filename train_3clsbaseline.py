from ultralytics import YOLO


def main():
    # 1. 加载官方预训练的 YOLOv11n 分割模型 (作为起点)
    model = YOLO("yolo11n-seg.pt")

    # 2. 开启火力全开的训练
    model.train(
        # 数据集路径 (指向你刚刚生成的那个全新的 yaml)
        data="/root/autodl-tmp/yolo11-rk3588-grad/datasets/new_dataseg_3classes/data.yaml",
        # 训练轮数与早停机制 (数据少，轮数要拉满，让它慢慢学)
        epochs=200,
        patience=50,
        # 硬件与图像设置
        batch=32,  # 你的 4090 显存很大，32 毫无压力
        imgsz=640,
        device=0,
        workers=8,
        # 项目保存路径 (换个新名字，别和之前的混了)
        project="/root/autodl-tmp/yolo11-rk3588-grad/runs/segment",
        name="yolo11_baseline_3classes",
        # 🌟 关键数据增强 (对付小样本必须火力全开)
        # mosaic=1.0,        # 必须开启 100% 马赛克增强
        # mixup=0.1,         # 开启 10% 图像混合
        # hsv_h=0.015,       # 颜色色调增强
        # hsv_s=0.7,         # 饱和度增强
        # hsv_v=0.4,         # 亮度增强
        # degrees=10.0,      # 允许 10度 以内的轻微旋转 (工业孔探探头经常旋转)
        # fliplr=0.5         # 50% 概率水平翻转
    )


if __name__ == "__main__":
    main()
