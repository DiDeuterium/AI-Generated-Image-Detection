"""多视图测试时增强（TTA）：五裁剪 + 水平翻转，对 softmax 概率取平均。

只在验证集上使用，不参与训练和选模；用于衡量 TTA 能否稳定提升同一个 checkpoint 的指标。
"""

from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from .data import EnsureMinSize


class TTATransform:
    """返回 (V, 3, H, W)：五裁剪、可选水平翻转，顺序固定。"""

    def __init__(self, input_size: int, resize_size: int, mean, std,
                 mode: str = 'protocol', flip: bool = True, five_crop: bool = True):
        self.size = int(input_size)
        self.mode = mode
        self.flip = flip
        self.five_crop = five_crop
        self.mean = torch.tensor(mean).view(3, 1, 1)
        self.std = torch.tensor(std).view(3, 1, 1)
        if mode == 'native':
            self.geo = EnsureMinSize(input_size)
        else:
            self.geo = T.Resize(resize_size, interpolation=T.InterpolationMode.BILINEAR,
                                antialias=True)
        self.to_tensor = T.ToTensor()
        self.five = T.FiveCrop(self.size)

    def __call__(self, img: torch.Tensor):
        img = self.geo(img)
        crops = list(self.five(img)) if self.five_crop else [T.CenterCrop(self.size)(img)]
        tensors = [(self.to_tensor(crop) - self.mean) / self.std for crop in crops]
        if self.flip:
            tensors += [torch.flip(t, dims=[2]) for t in tensors]
        return torch.stack(tensors)


class TTADataset(Dataset):
    def __init__(self, rows, transform):
        from PIL import Image
        self.rows = rows
        self.transform = transform
        self._open = Image.open

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with self._open(row['path']) as im:
            views = self.transform(im.convert('RGB'))
        return views, row['label'], index


@torch.no_grad()
def evaluate_tta(model, loader, device: torch.device, amp: bool, amp_dtype) -> Dict:
    """对每个样本的多视图概率取平均，返回标签与概率。"""
    import numpy as np
    model.eval()
    probs_all, labels_all = [], []
    for views, labels, _ in loader:
        batch, num_views = views.shape[0], views.shape[1]
        flat = views.view(batch * num_views, *views.shape[2:]).to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=amp):
            logits = model(flat)
        probs = torch.softmax(logits.float(), dim=1).view(batch, num_views, -1).mean(1)
        probs_all.append(probs.cpu().numpy())
        labels_all.append(labels.numpy())
    return {'labels': np.concatenate(labels_all), 'probs': np.concatenate(probs_all)}
