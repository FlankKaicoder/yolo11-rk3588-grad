import random


def xywhn_to_xyxy_px(box_xywhn, img_w, img_h):
    cx, cy, w, h = box_xywhn.tolist()
    x1 = (cx - w / 2.0) * img_w
    y1 = (cy - h / 2.0) * img_h
    x2 = (cx + w / 2.0) * img_w
    y2 = (cy + h / 2.0) * img_h
    return x1, y1, x2, y2


def box_iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-12)


def max_iou_with_gt(box, gt_boxes):
    if len(gt_boxes) == 0:
        return 0.0
    return max(box_iou_xyxy(box, gt) for gt in gt_boxes)


def sample_near_bg_for_one_gt(gt_box, img_w, img_h, all_gt_boxes,
                              min_crop_size=48, max_iou_bg_with_gt=0.05, max_tries=80):
    x1, y1, x2, y2 = gt_box
    bw = x2 - x1
    bh = y2 - y1

    min_pw = max(min_crop_size, int(round(0.5 * bw)))
    max_pw = max(min_pw, int(round(0.8 * bw)))
    min_ph = max(min_crop_size, int(round(0.5 * bh)))
    max_ph = max(min_ph, int(round(0.8 * bh)))

    for _ in range(max_tries):
        pw = random.randint(min_pw, max_pw)
        ph = random.randint(min_ph, max_ph)
        direction = random.choice(["left", "right", "top", "bottom"])

        if direction == "left":
            rx1_min = max(0, int(x1 - pw))
            rx1_max = max(0, int(x1 - 1))
            ry1_min = max(0, int(y1 - 0.25 * bh))
            ry1_max = min(img_h - ph, int(y2 - ph + 0.25 * bh))
        elif direction == "right":
            rx1_min = min(img_w - pw, int(x2 + 1))
            rx1_max = min(img_w - pw, int(x2 + pw))
            ry1_min = max(0, int(y1 - 0.25 * bh))
            ry1_max = min(img_h - ph, int(y2 - ph + 0.25 * bh))
        elif direction == "top":
            rx1_min = max(0, int(x1 - 0.25 * bw))
            rx1_max = min(img_w - pw, int(x2 - pw + 0.25 * bw))
            ry1_min = max(0, int(y1 - ph))
            ry1_max = max(0, int(y1 - 1))
        else:
            rx1_min = max(0, int(x1 - 0.25 * bw))
            rx1_max = min(img_w - pw, int(x2 - pw + 0.25 * bw))
            ry1_min = min(img_h - ph, int(y2 + 1))
            ry1_max = min(img_h - ph, int(y2 + ph))

        if rx1_min > rx1_max or ry1_min > ry1_max:
            continue

        rx1 = random.randint(rx1_min, rx1_max)
        ry1 = random.randint(ry1_min, ry1_max)
        cand = (float(rx1), float(ry1), float(rx1 + pw), float(ry1 + ph))

        if max_iou_with_gt(cand, all_gt_boxes) <= max_iou_bg_with_gt:
            return cand

    return None


def collect_pos_neg_boxes(batch, imgs, easy_bg_per_image=1, near_bg_per_defect=1,
                          min_crop_size=48, max_iou_bg_with_gt=0.05):
    _, _, img_h, img_w = imgs.shape
    batch_idx = batch["batch_idx"].view(-1).cpu()
    bboxes = batch["bboxes"].cpu()

    num_imgs = imgs.shape[0]
    gt_boxes_by_img = [[] for _ in range(num_imgs)]

    for i in range(len(bboxes)):
        bi = int(batch_idx[i].item())
        gt_boxes_by_img[bi].append(xywhn_to_xyxy_px(bboxes[i], img_w, img_h))

    pos_boxes = []
    neg_boxes = []

    for bi in range(num_imgs):
        gt_boxes = gt_boxes_by_img[bi]

        for gt in gt_boxes:
            pos_boxes.append((bi, gt))

        for gt in gt_boxes:
            for _ in range(near_bg_per_defect):
                bg = sample_near_bg_for_one_gt(
                    gt, img_w, img_h, gt_boxes,
                    min_crop_size=min_crop_size,
                    max_iou_bg_with_gt=max_iou_bg_with_gt,
                )
                if bg is not None:
                    neg_boxes.append((bi, bg))

        # v1 先不复杂化 easy bg，后面再补
    return pos_boxes, neg_boxes