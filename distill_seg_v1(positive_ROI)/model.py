import torch
from ultralytics.nn.tasks import SegmentationModel

from distill_seg.adapter import StudentAdapter
from distill_seg.losses import cosine_distill_loss
from distill_seg.roi_pool import crop_image_tensor, pool_feature_from_box
from distill_seg.teacher_wrapper import BCLTeacher


class DistillSegCriterion:
    def __init__(self, model, base_criterion):
        self.model = model
        self.base_criterion = base_criterion
        self.lambda_dist = float(model.distill_cfg["lambda_dist"])

    def _compute_distill_loss(self, batch):
        feat = getattr(self.model, "_distill_feat", None)
        if feat is None:
            return batch["img"].new_tensor(0.0)

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        imgs = batch["img"]               # [B, 3, H, W]
        bboxes = batch["bboxes"]          # [N, 4] normalized xywh
        batch_idx = batch["batch_idx"]    # [N]
        img_h, img_w = imgs.shape[-2:]

        teacher_feats = []
        student_feats = []

        for i in range(len(bboxes)):
            bi = int(batch_idx[i].item())

            crop = crop_image_tensor(imgs[bi:bi + 1], bboxes[i])
            if crop is None:
                continue

            roi_feat = pool_feature_from_box(feat[bi:bi + 1], bboxes[i], img_h, img_w)
            if roi_feat is None:
                continue

            with torch.no_grad():
                t = self.model.teacher(crop)   # [1, 512]

            s = self.model.adapter(roi_feat)   # [1, 512]

            teacher_feats.append(t)
            student_feats.append(s)

        if len(teacher_feats) == 0:
            return imgs.new_tensor(0.0)

        teacher_feats = torch.cat(teacher_feats, dim=0)
        student_feats = torch.cat(student_feats, dim=0)

        loss_dist = cosine_distill_loss(student_feats, teacher_feats)
        return loss_dist

    def __call__(self, preds, batch):
        seg_loss, seg_items = self.base_criterion(preds, batch)
        dist_loss = self._compute_distill_loss(batch)

        bs = batch["img"].shape[0]
        total_loss = seg_loss + self.lambda_dist * dist_loss * bs

        self.model._last_distill_loss = float(dist_loss.detach().item())
        return total_loss, seg_items


class DistillSegmentationModel(SegmentationModel):
    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True, distill_cfg=None):
        self.distill_cfg = distill_cfg or {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.teacher = BCLTeacher(
            ckpt_path=self.distill_cfg["teacher_ckpt"],
            input_size=self.distill_cfg.get("teacher_input_size", 224),
            mean=self.distill_cfg.get("teacher_mean", (0.485, 0.456, 0.406)),
            std=self.distill_cfg.get("teacher_std", (0.229, 0.224, 0.225)),
            default_num_classes=self.distill_cfg.get("teacher_num_classes", 4),
            feat_dim=self.distill_cfg.get("teacher_feat_dim", 128),
        )

        self.adapter = StudentAdapter(
            in_dim=self.distill_cfg.get("student_feat_dim", 128),
            out_dim=self.distill_cfg.get("teacher_out_dim", 512),
        )

        self._distill_feat = None
        self._last_distill_loss = 0.0
        self._hook_handle = None

        self._register_distill_hook()

        for p in self.teacher.parameters():
            p.requires_grad = False
        self.teacher.eval()

    def _get_hook_idx(self):
        hook_idx = int(self.distill_cfg["hook_idx"])
        if hook_idx < 0:
            hook_idx = len(self.model) + hook_idx
        return hook_idx

    def _distill_hook_fn(self, module, inputs, outputs):
        self._distill_feat = outputs

    def _register_distill_hook(self):
        hook_idx = self._get_hook_idx()
        target_module = self.model[hook_idx]

        # 防止重复注册
        self._remove_distill_hook()

        self._hook_handle = target_module.register_forward_hook(self._distill_hook_fn)
        print(f"[Distill] hook layer index = {hook_idx}, module = {target_module.__class__.__name__}")

    def _remove_distill_hook(self):
        hook_idx = self._get_hook_idx()
        target_module = self.model[hook_idx]

        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
        except Exception:
            pass

        # 双保险，防止 deepcopy/ema 里残留 forward_hooks
        try:
            target_module._forward_hooks.clear()
        except Exception:
            pass

        self._hook_handle = None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()  # 防止 trainer 调 train() 时把 teacher 带成 train
        return self

    def init_criterion(self):
        base_criterion = super().init_criterion()
        return DistillSegCriterion(self, base_criterion)