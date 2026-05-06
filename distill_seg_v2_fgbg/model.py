import torch
from ultralytics.nn.tasks import SegmentationModel

from distill_seg_v2_fgbg.adapter import StudentAdapter
from distill_seg_v2_fgbg.losses import cosine_distill_loss
from distill_seg_v2_fgbg.roi_pool import crop_image_tensor, pool_feature_from_box
from distill_seg_v2_fgbg.teacher_wrapper import BinaryTeacher
from distill_seg_v2_fgbg.fgbg_sampler import collect_pos_neg_boxes


class DistillSegCriterion:
    def __init__(self, model, base_criterion):
        self.model = model
        self.base_criterion = base_criterion
        self.lambda_pos = float(model.distill_cfg["lambda_pos"])
        self.lambda_neg = float(model.distill_cfg["lambda_neg"])

    def _compute_one_branch_loss(self, feat, imgs, sampled_boxes):
        teacher_feats = []
        student_feats = []

        img_h, img_w = imgs.shape[-2:]

        for bi, box_xyxy in sampled_boxes:
            crop = crop_image_tensor(
                imgs[bi:bi + 1],
                box_xyxy,
                box_format="xyxy",
            )
            if crop is None:
                continue

            roi_feat = pool_feature_from_box(
                feat[bi:bi + 1],
                box_xyxy,
                img_h,
                img_w,
                box_format="xyxy",
            )
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

        return cosine_distill_loss(student_feats, teacher_feats)

    def _compute_distill_loss(self, batch):
        feat = getattr(self.model, "_distill_feat", None)
        if feat is None:
            zero = batch["img"].new_tensor(0.0)
            return zero, zero, zero

        if isinstance(feat, (list, tuple)):
            feat = feat[0]

        imgs = batch["img"]

        pos_boxes, neg_boxes = collect_pos_neg_boxes(
            batch=batch,
            imgs=imgs,
            easy_bg_per_image=self.model.distill_cfg.get("easy_bg_per_image", 0),
            near_bg_per_defect=self.model.distill_cfg.get("near_bg_per_defect", 1),
            min_crop_size=self.model.distill_cfg.get("min_crop_size", 48),
            max_iou_bg_with_gt=self.model.distill_cfg.get("max_iou_bg_with_gt", 0.05),
        )

        pos_loss = self._compute_one_branch_loss(feat, imgs, pos_boxes)
        neg_loss = self._compute_one_branch_loss(feat, imgs, neg_boxes)

        total_dist = self.lambda_pos * pos_loss + self.lambda_neg * neg_loss
        return total_dist, pos_loss, neg_loss

    def __call__(self, preds, batch):
        seg_loss, seg_items = self.base_criterion(preds, batch)
        dist_loss, pos_loss, neg_loss = self._compute_distill_loss(batch)

        bs = batch["img"].shape[0]
        total_loss = seg_loss + dist_loss * bs

        self.model._last_distill_loss = float(dist_loss.detach().item())
        self.model._last_pos_loss = float(pos_loss.detach().item())
        self.model._last_neg_loss = float(neg_loss.detach().item())

        return total_loss, seg_items


class DistillSegmentationModel(SegmentationModel):
    def __init__(self, cfg="yolo11n-seg.yaml", ch=3, nc=None, verbose=True, distill_cfg=None):
        self.distill_cfg = distill_cfg or {}
        super().__init__(cfg=cfg, ch=ch, nc=nc, verbose=verbose)

        self.teacher = BinaryTeacher(
            ckpt_path=self.distill_cfg["teacher_ckpt"],
            input_size=self.distill_cfg.get("teacher_input_size", 224),
            mean=self.distill_cfg.get("teacher_mean", (0.485, 0.456, 0.406)),
            std=self.distill_cfg.get("teacher_std", (0.229, 0.224, 0.225)),
        )

        self.adapter = StudentAdapter(
            in_dim=self.distill_cfg.get("student_feat_dim", 128),
            out_dim=self.distill_cfg.get("teacher_out_dim", 512),
        )

        self._distill_feat = None
        self._last_distill_loss = 0.0
        self._last_pos_loss = 0.0
        self._last_neg_loss = 0.0
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

        self._remove_distill_hook()
        self._hook_handle = target_module.register_forward_hook(self._distill_hook_fn)
        print(f"[Distill-FGBG] hook layer index = {hook_idx}, module = {target_module.__class__.__name__}")

    def _remove_distill_hook(self):
        hook_idx = self._get_hook_idx()
        target_module = self.model[hook_idx]

        try:
            if self._hook_handle is not None:
                self._hook_handle.remove()
        except Exception:
            pass

        try:
            target_module._forward_hooks.clear()
        except Exception:
            pass

        self._hook_handle = None

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        return self

    def init_criterion(self):
        base_criterion = super().init_criterion()
        return DistillSegCriterion(self, base_criterion)
