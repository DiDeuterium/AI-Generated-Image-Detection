"""数据读取、统一预处理与验证集 JPEG 重编码。

协议见仓库根目录 README「统一实验协议」与 ddc/实验设置.md：
按 splits/train_val.csv 的 split 列读取，短边缩放到 256、训练随机裁 224 + 水平翻转，
验证中心裁 224；像素归一化统计只由训练子集估计。
"""

import csv
import io
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .domain_shift import RandomDomainShift


# --------------------------------------------------------------------------------------
# 划分清单
# --------------------------------------------------------------------------------------
def read_splits(csv_path: Path, image_dir: Path) -> Tuple[List[Dict], List[Dict]]:
    """读取固定划分清单，返回 (train_rows, val_rows)。

    每行是 dict(image_id, label, path)；`path` 由 image_dir 与清单中的 image_path 拼接。
    路径只用于加载图像，不作为模型输入。
    """
    train_rows, val_rows = [], []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            item = {
                'image_id': row['image_id'],
                'label': int(row['label']),
                'path': image_dir / row['image_path'],
            }
            if row['split'] == 'train':
                train_rows.append(item)
            elif row['split'] == 'val':
                val_rows.append(item)
            else:
                raise ValueError(f'未知 split: {row["split"]}')
    if not train_rows or not val_rows:
        raise RuntimeError(f'{csv_path} 中缺少 train/val 行')
    return train_rows, val_rows


def read_test_ids(csv_path: Path, test_dir: Path) -> List[Dict]:
    """按提交模板顺序读取测试图（只含 image_id 与路径）。"""
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            rows.append({
                'image_id': row['image_id'],
                'label': -1,
                'path': test_dir / row['image_id'],
            })
    return rows


# --------------------------------------------------------------------------------------
# 预处理
# --------------------------------------------------------------------------------------
class EnsureMinSize:
    """只在短边小于目标尺寸时上采样，保留原图的压缩痕迹尺度（native 模式使用）。"""

    def __init__(self, size: int):
        self.size = int(size)

    def __call__(self, img: Image.Image) -> Image.Image:
        width, height = img.size
        if min(width, height) >= self.size:
            return img
        scale = self.size / min(width, height)
        return img.resize((int(round(width * scale)), int(round(height * scale))),
                          Image.BILINEAR)


class JpegAugment:
    """以概率 p 对图像做一次随机质量的 JPEG 重编码（两类等概率施加）。"""

    def __init__(self, prob: float = 0.0, quality_min: int = 30, quality_max: int = 90):
        self.prob = float(prob)
        self.quality_min = int(quality_min)
        self.quality_max = int(quality_max)

    def __call__(self, img: Image.Image) -> Image.Image:
        if self.prob <= 0.0 or random.random() >= self.prob:
            return img
        quality = random.randint(self.quality_min, self.quality_max)
        buffer = io.BytesIO()
        img.save(buffer, format='JPEG', quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as reencoded:
            return reencoded.convert('RGB')


def _geometry_ops(input_size: int, resize_size: Optional[int], train: bool,
                  aug: Optional[Dict]) -> List:
    """按模式拼出几何变换：protocol（缩放到短边再裁）/ rrc（随机尺寸裁剪）/ native（原尺度裁剪）。"""
    aug = aug or {}
    mode = aug.get('mode', 'protocol')
    interpolation = T.InterpolationMode.BILINEAR

    if mode == 'native':
        ops = [EnsureMinSize(input_size)]
    else:
        ops = [T.Resize(resize_size, interpolation=interpolation, antialias=True)]

    domain_shift = aug.get('domain_shift') if train else None
    if domain_shift is not None:
        if (aug.get('jpeg_aug') or {}).get('prob', 0.):
            raise ValueError('Specify domain_shift or jpeg_aug, not both')
        ops.insert(0, RandomDomainShift(**domain_shift))

    if train:
        if mode == 'rrc':
            scale = tuple(aug.get('rrc_scale', (0.6, 1.0)))
            ratio = tuple(aug.get('rrc_ratio', (0.75, 1.3333)))
            ops.append(T.RandomResizedCrop(input_size, scale=scale, ratio=ratio,
                                           interpolation=interpolation, antialias=True))
        else:
            ops.append(T.RandomCrop(input_size))
        flip = float(aug.get('hflip', 0.5))
        if flip > 0:
            ops.append(T.RandomHorizontalFlip(p=flip))
    else:
        ops.append(T.CenterCrop(input_size))

    jitter = aug.get('color_jitter') if train else None
    if jitter:
        ops.append(T.ColorJitter(brightness=float(jitter.get('brightness', 0.0)),
                                 contrast=float(jitter.get('contrast', 0.0)),
                                 saturation=float(jitter.get('saturation', 0.0)),
                                 hue=0.0))
    jpeg = aug.get('jpeg_aug') if train else None
    if jpeg and float(jpeg.get('prob', 0.0)) > 0:
        ops.append(JpegAugment(float(jpeg['prob']), int(jpeg.get('quality_min', 30)),
                               int(jpeg.get('quality_max', 90))))
    return ops


def build_transform(input_size: int, resize_size: Optional[int], mean: Optional[Sequence[float]],
                    std: Optional[Sequence[float]], train: bool,
                    aug: Optional[Dict] = None) -> T.Compose:
    """统一预处理；mean/std 为 None 时只做几何变换并把像素缩放到 [0,1]。"""
    ops = _geometry_ops(input_size, resize_size, train, aug)
    ops.append(T.ToTensor())
    if mean is not None and std is not None:
        ops.append(T.Normalize(mean=mean, std=std))
    return T.Compose(ops)


class ImageDataset(Dataset):
    """按行读取图像并转 RGB，返回 (图像张量, 标签, 行号)。"""

    def __init__(self, rows: List[Dict], transform, domain_shift=None):
        self.rows = rows
        self.transform = transform
        self.domain_shift = domain_shift

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        with Image.open(row['path']) as im:
            image = im.convert('RGB')
        if self.domain_shift is not None:
            image = self.domain_shift(image, row['image_id'])
        tensor = self.transform(image)
        return tensor, row['label'], index


# --------------------------------------------------------------------------------------
# 归一化统计（仅训练子集，用验证阶段的确定性几何变换）
# --------------------------------------------------------------------------------------
def load_norm_stats(path: Path) -> Optional[Dict[str, object]]:
    if path.exists():
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    return None


def compute_norm_stats(train_rows: List[Dict], input_size: int, resize_size: int,
                       num_workers: int, batch_size: int = 64) -> Dict[str, object]:
    """在训练子集上按验证几何变换累计每通道均值与总体标准差（像素先缩放至 [0,1]）。"""
    transform = build_transform(input_size, resize_size, None, None, train=False)
    loader = DataLoader(ImageDataset(train_rows, transform), batch_size=batch_size,
                        shuffle=False, num_workers=num_workers, pin_memory=True)
    total = torch.zeros(3, dtype=torch.float64)
    total_sq = torch.zeros(3, dtype=torch.float64)
    count = 0
    for images, _, _ in loader:
        images = images.double()
        total += images.sum(dim=(0, 2, 3))
        total_sq += (images ** 2).sum(dim=(0, 2, 3))
        count += images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / count
    var = total_sq / count - mean ** 2
    std = var.clamp_min(1e-12).sqrt()
    return {
        'mean': [round(float(v), 6) for v in mean],
        'std': [round(float(v), 6) for v in std],
        'num_images': len(train_rows),
        'num_pixels_per_channel': count,
    }


def save_norm_stats(stats: Dict[str, object], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write('\n')


# --------------------------------------------------------------------------------------
# 验证集统一 JPEG 重编码
# --------------------------------------------------------------------------------------
def ensure_jpeg_cache(rows: List[Dict], cache_dir: Path, quality: int = 75,
                      subsampling: int = 2, num_threads: int = 8) -> List[Dict]:
    """把行中的图像按固定质量重编码（Pillow，quality=75、4:2:0），返回替换路径后的行。

    结果缓存到 cache_dir，已存在的文件直接复用，保证原图与重编码版按 image_id 配对。
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    num_threads = max(1, int(num_threads))

    def _encode(row: Dict) -> Dict:
        dst = cache_dir / (Path(row['image_id']).stem + '.jpg')
        if not dst.exists():
            with Image.open(row['path']) as im:
                im.convert('RGB').save(dst, format='JPEG', quality=quality,
                                       subsampling=subsampling)
        return {**row, 'path': dst}

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        return list(pool.map(_encode, rows))



# --------------------------------------------------------------------------------------
# DataLoader
# --------------------------------------------------------------------------------------
def make_worker_init_fn(base_seed: int):
    def _init(worker_id: int) -> None:
        seed = base_seed + worker_id
        np.random.seed(seed % (2 ** 32))
        torch.manual_seed(seed)

    return _init


def build_loader(rows: List[Dict], transform, batch_size: int, shuffle: bool,
                 num_workers: int, seed: int, drop_last: bool = False,
                 persistent: bool = False) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        ImageDataset(rows, transform),
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent and num_workers > 0,
        prefetch_factor=4 if num_workers > 0 else None,
        worker_init_fn=make_worker_init_fn(seed),
        generator=generator,
    )
