from ultralytics import YOLO
import os

def main():
    model = YOLO("yolo11n-seg.pt")  

    # ---------------------------------------------------------
    # 修改点：定义你的绝对路径
    # ---------------------------------------------------------
    # 建议将结果保存在数据盘（如 /root/autodl-tmp/ 或 /mnt/data/）
    # 而不是系统盘，防止空间不足。
    absolute_save_dir = "/root/autodl-tmp/yolo11-rk3588-grad/runs/segment/lastdataseg"
    
    # 确保目录存在，不存在则创建
    if not os.path.exists(absolute_save_dir):
        os.makedirs(absolute_save_dir)
        print(f"创建目录: {absolute_save_dir}")

    yaml_path = "/root/autodl-tmp/yolo11-rk3588-grad/datasets/olddataseg/data.yaml" 
    
    results = model.train(
        data=yaml_path,
        epochs=200,
        imgsz=640,
        batch=32,
        device="0",
        
        # --- 核心路径配置 ---
        # project 传入绝对路径，YOLO 会把这里作为结果的总入口
        project=absolute_save_dir, 
        
        # name 是本次训练的子文件夹名称
        # 最终保存路径为：{project}/{name}
        name="yolo11_baseline_oldsegdata", 
        
        exist_ok=True, # 如果文件夹已存在，直接在里面更新而不创建 _v2, _v3
        
        # 其他参数保持不变...
        workers=8,
        optimizer='auto',
    )
    
    print(f"🚀 训练结果已实时保存至: {results.save_dir}")

if __name__ == '__main__':
    # main()ndows 下是必须的，在 Linux 服务器上也可以防止多进程加载数据时出现僵尸进程
    main()