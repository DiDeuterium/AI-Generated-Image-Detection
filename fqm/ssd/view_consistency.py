"""Matched views of one crop and optional prediction consistency."""
import random

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T

from .data import EnsureMinSize
from .domain_shift import apply_degradation


class PairedDegradation:
    """Re-encode both labels under the same distribution of processing."""

    def __init__(self, size, mean, std, hflip):
        self.size = size
        self.crop_region = T.Compose([
            EnsureMinSize(size + 8),
            T.RandomCrop(size + 8),
            T.RandomHorizontalFlip(hflip),
        ])
        ops = [T.ToTensor()]
        if mean is not None and std is not None:
            ops.append(T.Normalize(mean, std))
        self.to_tensor = T.Compose(ops)

    def __call__(self, image):
        region = self.crop_region(image)
        weak = apply_degradation(region, 1., random.randint(90, 95), 0.)
        scale = random.uniform(.75, 1.)
        reduced = apply_degradation(region, scale, None, 0.)
        restored = reduced.resize(region.size, Image.Resampling.BICUBIC)
        degraded = apply_degradation(restored, 1., random.randint(70, 85), 0.)
        x, y = random.randrange(9), random.randrange(9)
        box = (x, y, x + self.size, y + self.size)
        return torch.stack([self.to_tensor(view.crop(box)) for view in (weak, degraded)])


def paired_view_loss(logits, labels, criterion, consistency_weight):
    """Supervise both views and optionally penalize symmetric KL divergence."""
    supervised = criterion(logits.flatten(0, 1), labels.repeat_interleave(2))
    if not consistency_weight:
        return supervised

    weak, degraded = F.log_softmax(logits.float(), dim=-1).unbind(dim=1)
    symmetric_kl = .5 * ((weak.exp() - degraded.exp()) * (weak - degraded)).sum(dim=-1)
    return supervised + consistency_weight * symmetric_kl.mean()
