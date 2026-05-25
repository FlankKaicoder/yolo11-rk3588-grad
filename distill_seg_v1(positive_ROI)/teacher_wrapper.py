import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class BCLResNet18(nn.Module):
    def __init__(self, num_classes, feat_dim=128):
        super().__init__()
        backbone = models.resnet18(weights=None)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.encoder = backbone

        self.projector = nn.Sequential(nn.Linear(in_features, 512), nn.ReLU(inplace=True), nn.Linear(512, feat_dim))
        self.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        feat = self.encoder(x)
        proj = self.projector(feat)
        proj = F.normalize(proj, dim=1)
        logits = self.classifier(feat)
        return feat, proj, logits


class BCLTeacher(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        input_size: int = 224,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        default_num_classes: int = 4,
        feat_dim: int = 128,
    ):
        super().__init__()
        self.input_size = input_size
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1), persistent=False)

        # 关键修复：PyTorch 2.6+ 需要显式关闭 weights_only
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        if isinstance(ckpt, dict) and "class_names" in ckpt:
            num_classes = len(ckpt["class_names"])
        else:
            num_classes = default_num_classes

        self.model = BCLResNet18(num_classes=num_classes, feat_dim=feat_dim)

        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
        elif isinstance(ckpt, dict):
            state_dict = ckpt
        else:
            raise RuntimeError("teacher checkpoint 格式不对，至少应该是 dict。")

        new_state = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            new_state[nk] = v

        missing, unexpected = self.model.load_state_dict(new_state, strict=False)
        print(f"[Teacher] missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")

        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x):
        x = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        x = (x - self.mean) / self.std
        feat, _, _ = self.model(x)
        return feat
