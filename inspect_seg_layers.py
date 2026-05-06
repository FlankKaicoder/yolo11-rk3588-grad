import argparse
import torch
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="yolo11n-seg.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    yolo = YOLO(args.model)
    model = yolo.model  # SegmentationModel
    modules = model.model  # ModuleList

    handles = []

    def make_hook(i):
        def hook(module, inputs, outputs):
            if torch.is_tensor(outputs):
                print(f"[{i:02d}] {module.__class__.__name__:<30} -> {tuple(outputs.shape)}")
            elif isinstance(outputs, (list, tuple)):
                shapes = []
                for o in outputs:
                    if torch.is_tensor(o):
                        shapes.append(tuple(o.shape))
                    else:
                        shapes.append(type(o).__name__)
                print(f"[{i:02d}] {module.__class__.__name__:<30} -> {shapes}")
            else:
                print(f"[{i:02d}] {module.__class__.__name__:<30} -> {type(outputs).__name__}")
        return hook

    for i, m in enumerate(modules):
        handles.append(m.register_forward_hook(make_hook(i)))

    x = torch.zeros(1, 3, args.imgsz, args.imgsz)
    model.eval()
    with torch.no_grad():
        _ = model(x)

    for h in handles:
        h.remove()


if __name__ == "__main__":
    main()