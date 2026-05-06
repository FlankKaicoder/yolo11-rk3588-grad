import torch
import torch.nn.functional as F


def xywhn_to_xyxy_pixels(box_xywhn, img_h, img_w):
    x, y, w, h = box_xywhn.tolist()
    x1 = (x - w / 2.0) * img_w
    y1 = (y - h / 2.0) * img_h
    x2 = (x + w / 2.0) * img_w
    y2 = (y + h / 2.0) * img_h
    return x1, y1, x2, y2


def clamp_box(x1, y1, x2, y2, h, w):
    x1 = max(0, min(w - 1, int(round(x1))))
    y1 = max(0, min(h - 1, int(round(y1))))
    x2 = max(x1 + 1, min(w, int(round(x2))))
    y2 = max(y1 + 1, min(h, int(round(y2))))
    return x1, y1, x2, y2


def crop_image_tensor(img_bchw, box_xywhn):
    _, _, img_h, img_w = img_bchw.shape
    x1, y1, x2, y2 = xywhn_to_xyxy_pixels(box_xywhn, img_h, img_w)
    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, img_h, img_w)

    crop = img_bchw[:, :, y1:y2, x1:x2]
    if crop.numel() == 0 or crop.shape[-1] < 2 or crop.shape[-2] < 2:
        return None
    return crop


def pool_feature_from_box(feat_bchw, box_xywhn, img_h, img_w):
    _, _, feat_h, feat_w = feat_bchw.shape
    x1, y1, x2, y2 = xywhn_to_xyxy_pixels(box_xywhn, img_h, img_w)

    fx1 = x1 / img_w * feat_w
    fy1 = y1 / img_h * feat_h
    fx2 = x2 / img_w * feat_w
    fy2 = y2 / img_h * feat_h

    fx1, fy1, fx2, fy2 = clamp_box(fx1, fy1, fx2, fy2, feat_h, feat_w)
    roi = feat_bchw[:, :, fy1:fy2, fx1:fx2]

    if roi.numel() == 0 or roi.shape[-1] < 1 or roi.shape[-2] < 1:
        return None

    pooled = F.adaptive_avg_pool2d(roi, (1, 1)).flatten(1)  # [1, C]
    return pooled