import torch.nn.functional as F


def cosine_distill_loss(student_feat, teacher_feat):
    student_feat = F.normalize(student_feat, dim=-1)
    teacher_feat = F.normalize(teacher_feat, dim=-1)
    return 1.0 - (student_feat * teacher_feat).sum(dim=-1).mean()