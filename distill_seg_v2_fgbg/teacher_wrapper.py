import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


class BinaryTeacher(nn.Module):
    def __init__(
        self,
        ckpt_path,
        input_size=224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.input_size = input_size

        ckpt = torch.load(ckpt_path, map_location="cpu")
        state_dict = ckpt["model_state_dict"]

        if "fc.weight" not in state_dict:
            raise KeyError(f"'fc.weight' not found in checkpoint: {ckpt_path}")
        num_classes = state_dict["fc.weight"].shape[0]

        self.model = resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)
        self.model.load_state_dict(state_dict, strict=True)

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

        mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def _forward_feat(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.avgpool(x)
        feat = torch.flatten(x, 1)   # [B, 512]
        return feat

    @torch.no_grad()
    def forward(self, x):
        """
        x: [B,3,H,W], 值域默认 0~1
        返回 teacher feat: [B, 512]
        """
        p = next(self.model.parameters())
        dev = p.device
        dtype = p.dtype

        x = x.to(device=dev, dtype=dtype, non_blocking=True)
        x = torch.clamp(x, 0.0, 1.0)
        x = F.interpolate(
            x,
            size=(self.input_size, self.input_size),
            mode="bilinear",
            align_corners=False,
        )

        mean = self.mean.to(device=dev, dtype=dtype)
        std = self.std.to(device=dev, dtype=dtype)
        x = (x - mean) / std

        feat = self._forward_feat(x)
        return feat
