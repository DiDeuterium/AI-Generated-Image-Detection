"""压缩/合成伪迹先验模块（Compression & synthesis Artifact Prior, CAP）。

动机：本任务里真实图经过 JPEG 编码、生成图是 PNG，两类在「压缩伪迹」和「高频残差统计」上
存在系统性差异。让卷积骨干从零学出这些物理量既慢又不稳，尤其在随机裁剪后 JPEG 网格相位未知。

本模块用固定算子显式估计两类统计量，作为先验通道与 RGB 拼接送入骨干：
1. 网格相位搜索的块效应（纵向/横向各一组：强度、相位 sin、相位 cos）——对裁剪偏移不敏感；
2. 高频残差能量：|Laplacian| 与「像素减四邻域均值」（NPR 式残差），取 log1p 稳定动态范围。

模块本身没有卷积参数（只有逐通道可学仿射），可挂在任意骨干前，便于做消融对照。
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['GridBlockinessPrior', 'HighFrequencyPrior', 'CompressionPrior',
           'PriorAugmentedBackbone']


class GridBlockinessPrior(nn.Module):
    """网格相位搜索的块效应先验：输出 6 通道（h/v × 强度、相位 sin、相位 cos）。"""

    def __init__(self, block: int = 8, smooth: int = 8, eps: float = 1e-3):
        super().__init__()
        self.block = int(block)
        self.smooth = max(1, int(smooth))
        self.eps = float(eps)
        self.register_buffer('phase_angles',
                             torch.arange(self.block) * (2.0 * torch.pi / self.block),
                             persistent=False)

    @property
    def out_channels(self) -> int:
        return 6

    def _pad(self, diff: torch.Tensor, dim: int) -> torch.Tensor:
        """把两个空间维都补到 block 的整数倍，保证纵/横分支分辨率一致。"""
        block = self.block
        h, w = diff.shape[2], diff.shape[3]
        pad_h = (block - h % block) % block
        pad_w = (block - w % block) % block
        if pad_h == 0 and pad_w == 0:
            return diff
        return F.pad(diff, (0, pad_w, 0, pad_h))

    def _phase_maps(self, diff: torch.Tensor,
                    dim: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """返回 (N, P, Hb, Wb) 的块边界能量与块内能量；P = block，Hb/Wb 为块单元分辨率。"""
        diff = self._pad(diff, dim)
        block = self.block
        n = diff.shape[0]
        if dim == 2:      # 纵向差分：按行分块
            h, w = diff.shape[2], diff.shape[3]
            rolled = torch.stack([torch.roll(diff, shifts=-p, dims=2)
                                  for p in range(block)], dim=1)      # (N,P,H,W)
            blocks = rolled.reshape(n, block, h // block, block, w)
            boundary = blocks[..., block - 1, :]                       # (N,P,Hb,W)
            interior = blocks[..., :block - 1, :].mean(dim=3)          # (N,P,Hb,W)
            kernel = (1, block)
        else:             # 横向差分：按列分块
            h, w = diff.shape[2], diff.shape[3]
            rolled = torch.stack([torch.roll(diff, shifts=-p, dims=3)
                                  for p in range(block)], dim=1)
            blocks = rolled.reshape(n, block, h, w // block, block)
            boundary = blocks[..., block - 1]                          # (N,P,H,Wb)
            interior = blocks[..., :block - 1].mean(dim=4)             # (N,P,H,Wb)
            kernel = (block, 1)
        boundary = F.avg_pool2d(boundary, kernel)
        interior = F.avg_pool2d(interior, kernel)
        if self.smooth > 1:
            pad = self.smooth // 2
            boundary = F.avg_pool2d(boundary, self.smooth, stride=1, padding=pad)
            interior = F.avg_pool2d(interior, self.smooth, stride=1, padding=pad)
        return boundary, interior

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        dv = torch.zeros_like(gray)
        dh = torch.zeros_like(gray)
        dv[:, :, 1:, :] = (gray[:, :, 1:, :] - gray[:, :, :-1, :]).abs()
        dh[:, :, :, 1:] = (gray[:, :, :, 1:] - gray[:, :, :, :-1]).abs()
        features = []
        for diff, dim in ((dv, 2), (dh, 3)):
            boundary, interior = self._phase_maps(diff, dim)
            ratio = torch.log((boundary + self.eps) / (interior + self.eps))
            weights = torch.softmax(ratio.flatten(1), dim=1).view_as(ratio)
            strength = (weights * ratio).sum(dim=1, keepdim=True)
            angles = self.phase_angles.view(1, -1, 1, 1)
            phase_sin = (weights * torch.sin(angles)).sum(dim=1, keepdim=True)
            phase_cos = (weights * torch.cos(angles)).sum(dim=1, keepdim=True)
            features.extend([strength, phase_sin, phase_cos])
        return torch.cat(features, dim=1)



class HighFrequencyPrior(nn.Module):
    """高频残差先验：|Laplacian| 与 NPR 式「像素 - 四邻域均值」，各取 log1p。"""

    def __init__(self, scale: float = 10.0):
        super().__init__()
        self.scale = float(scale)
        laplace = torch.tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]])
        nbr = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]) / 4.0
        self.register_buffer('laplace_kernel', laplace.view(1, 1, 3, 3), persistent=False)
        self.register_buffer('nbr_kernel', nbr.view(1, 1, 3, 3), persistent=False)

    @property
    def out_channels(self) -> int:
        return 2

    def forward(self, gray: torch.Tensor) -> torch.Tensor:
        pad = F.pad(gray, (1, 1, 1, 1), mode='reflect')
        laplace = F.conv2d(pad, self.laplace_kernel).abs()
        residual = (gray - F.conv2d(pad, self.nbr_kernel)).abs()
        return torch.cat([torch.log1p(self.scale * laplace),
                          torch.log1p(self.scale * residual)], dim=1)


class CompressionPrior(nn.Module):
    """组合先验：块效应（可选）+ 高频残差（可选），带可学习的逐通道仿射。

    Args:
        parts: 取自 {'block', 'highfreq'} 的子集。
        block: JPEG 块大小；消融时用 7 等错误值作为「非 8 网格」对照。
        smooth: 块级统计的局部平滑窗（块单元数）。
        detach_feature: 若为 True，先验不向图像反传梯度。
    """

    def __init__(self, parts: Tuple[str, ...] = ('block', 'highfreq'), block: int = 8,
                 smooth: int = 8, scale: float = 10.0, detach_feature: bool = False):
        super().__init__()
        self.block_prior = (GridBlockinessPrior(block=block, smooth=smooth)
                            if 'block' in parts else None)
        self.hf_prior = HighFrequencyPrior(scale=scale) if 'highfreq' in parts else None
        channels = ((self.block_prior.out_channels if self.block_prior else 0)
                    + (self.hf_prior.out_channels if self.hf_prior else 0))
        if channels == 0:
            raise ValueError('CompressionPrior 至少需要一个部分')
        self.channels = channels
        self.detach_feature = bool(detach_feature)
        self.affine_scale = nn.Parameter(torch.ones(channels))
        self.affine_bias = nn.Parameter(torch.zeros(channels))

    @property
    def out_channels(self) -> int:
        return self.channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.detach_feature):
            gray = x.mean(dim=1, keepdim=True).float()
            feats = []
            if self.block_prior is not None:
                block_map = self.block_prior(gray)
                feats.append(F.interpolate(block_map, size=x.shape[-2:], mode='bilinear',
                                           align_corners=False))
            if self.hf_prior is not None:
                feats.append(self.hf_prior(gray))
            prior = torch.cat(feats, dim=1)
        prior = prior * self.affine_scale.view(1, -1, 1, 1) + self.affine_bias.view(1, -1, 1, 1)
        return prior.to(dtype=x.dtype)


class PriorAugmentedBackbone(nn.Module):
    """把先验与 RGB 拼接后送入骨干，保持骨干原始接口。"""

    def __init__(self, backbone: nn.Module, prior: CompressionPrior, in_chans: int = 3):
        super().__init__()
        self.prior = prior
        self.backbone = backbone
        self.in_chans = in_chans

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prior = self.prior(x[:, :self.in_chans])
        return self.backbone(torch.cat([x, prior], dim=1))

    def forward_features(self, x: torch.Tensor):
        prior = self.prior(x[:, :self.in_chans])
        return self.backbone.forward_features(torch.cat([x, prior], dim=1))

    @property
    def num_features(self):
        return self.backbone.num_features
