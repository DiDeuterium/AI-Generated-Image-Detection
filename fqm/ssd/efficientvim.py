"""EfficientViM 的纯 PyTorch 实现（无预训练权重，可直接从随机初始化训练）。

论文：EfficientViM: Efficient Vision Mamba with Hidden State Mixer based State Space
Duality, CVPR 2025 (https://arxiv.org/abs/2411.15241)，官方代码 https://github.com/mlvlab/EfficientViM

本文件是 timm `timm/models/efficientvim.py`（Apache-2.0，Copyright 2025 Ross Wightman，
由官方 MIT 代码移植）的结构等价精简版：去掉 timm 的注册/特征提取/融合/蒸馏分支，
只保留 HSM-SSD 骨干与 4 路 softmax 加权分类头，并把 timm.layers 的工具函数用 torch 重写。
模块命名与原实现一致（patch_embed / stages / norm / heads / weights），便于对照论文。
"""

from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _ckpt

__all__ = ['EfficientViM', 'efficientvim_m1']


# --------------------------------------------------------------------------------------
# 基础算子
# --------------------------------------------------------------------------------------
def trunc_normal_(tensor: torch.Tensor, mean: float = 0.0, std: float = 1.0,
                  a: float = -2.0, b: float = 2.0) -> torch.Tensor:
    return nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


def calculate_drop_path_rates(drop_path_rate: float, depth: int,
                              start_rate: float = 0.0) -> List[float]:
    """随机深度按块线性递增，与原实现一致。"""
    if drop_path_rate <= 0.0:
        return [0.0] * depth
    return [start_rate + (drop_path_rate - start_rate) * i / max(depth - 1, 1)
            for i in range(depth)]


class DropPath(nn.Module):
    """Stochastic depth（每个样本独立丢弃残差分支）。"""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        mask = keep + torch.rand(shape, dtype=x.dtype, device=x.device)
        return x.div(keep) * mask.floor_()

    def extra_repr(self) -> str:
        return f'drop_prob={self.drop_prob:.3f}'


class LayerNorm2d(nn.Module):
    """通道维 LayerNorm（NCHW），对应原实现的 LayerNorm2d。"""

    def __init__(self, num_channels: int, eps: float = 1e-6, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_channels, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.zeros(num_channels, device=device, dtype=dtype))

    def reset_parameters(self) -> None:
        nn.init.ones_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]
class SqueezeExcite(nn.Module):
    """SE 通道注意力（1x1 降维 -> 激活 -> 1x1 升维 -> sigmoid 门控）。"""

    def __init__(self, channels: int, rd_ratio: float = 0.25,
                 act_layer=nn.SiLU, device=None, dtype=None):
        super().__init__()
        rd_channels = max(1, int(channels * rd_ratio))
        dd = {'device': device, 'dtype': dtype}
        self.conv_reduce = nn.Conv2d(channels, rd_channels, 1, bias=True, **dd)
        self.act = act_layer()
        self.conv_expand = nn.Conv2d(rd_channels, channels, 1, bias=True, **dd)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvLayer1d(nn.Module):
    """Conv1d + 可选归一化 + 可选激活（原实现的 ConvLayer1d）。"""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 norm: Optional[type] = nn.BatchNorm1d, act_layer: Optional[type] = nn.ReLU,
                 bn_weight_init: float = 1.0, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv1d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False, **dd)
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x


class ConvLayer2d(nn.Module):
    """Conv2d + 可选归一化 + 可选激活（原实现的 ConvLayer2d）。"""

    def __init__(self, in_dim: int, out_dim: int, kernel_size: int = 3, stride: int = 1,
                 padding: int = 0, dilation: int = 1, groups: int = 1,
                 norm: Optional[type] = nn.BatchNorm2d, act_layer: Optional[type] = nn.ReLU,
                 bn_weight_init: float = 1.0, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=False, **dd)
        self.norm = norm(out_dim, **dd) if norm else None
        self.act = act_layer() if act_layer else None
        if isinstance(self.norm, nn.modules.batchnorm._BatchNorm):
            nn.init.constant_(self.norm.weight, bn_weight_init)
            nn.init.constant_(self.norm.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.act is not None:
            x = self.act(x)
        return x

# --------------------------------------------------------------------------------------
# 骨干组件
# --------------------------------------------------------------------------------------
class Stem(nn.Module):
    """4 层 stride-2 卷积 stem，总下采样 16 倍。"""

    def __init__(self, in_dim: int, dim: int, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, dim // 8, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 8, dim // 4, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 4, dim // 2, kernel_size=3, stride=2, padding=1, **dd),
            ConvLayer2d(dim // 2, dim, kernel_size=3, stride=2, padding=1,
                        act_layer=None, **dd),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class FFN(nn.Module):
    """两层 1x1 卷积构成的前馈网络。"""

    def __init__(self, in_dim: int, dim: int, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.fc1 = ConvLayer2d(in_dim, dim, 1, **dd)
        self.fc2 = ConvLayer2d(dim, in_dim, 1, act_layer=None, bn_weight_init=0, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.fc1(x))


class PatchMerging(nn.Module):
    """MBConv 式 2 倍下采样（含 SE）与两条深度可分离残差支路。"""

    def __init__(self, in_dim: int, out_dim: int, ratio: float = 4.0,
                 device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        hidden_dim = int(out_dim * ratio)
        self.conv = nn.Sequential(
            ConvLayer2d(in_dim, hidden_dim, kernel_size=1, **dd),
            ConvLayer2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1,
                        groups=hidden_dim, **dd),
            SqueezeExcite(hidden_dim, 0.25, **dd),
            ConvLayer2d(hidden_dim, out_dim, kernel_size=1, act_layer=None, **dd),
        )
        self.dwconv1 = ConvLayer2d(in_dim, in_dim, 3, padding=1, groups=in_dim,
                                   act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(out_dim, out_dim, 3, padding=1, groups=out_dim,
                                   act_layer=None, **dd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dwconv1(x)
        x = self.conv(x)
        x = x + self.dwconv2(x)
        return x


class HSMSSD(nn.Module):
    """Hidden State Mixer based State Space Duality。

    Mamba 式的门控状态递推写成 chunk 对偶形式：由输入相关的 `dt`/`B` 得到转移矩阵并作用在
    展平后的空间 token 上，再用 `C` 读出隐藏状态，最后经压缩隐藏状态的通道混合
    （hz_proj + SiLU 门控）。全部由 conv / softmax / matmul 组成，不依赖 mamba_ssm。
    """

    def __init__(self, dim: int, ssd_expand: float = 1.0, state_dim: int = 64,
                 a_init_range: Tuple[float, float] = (1.0, 16.0), device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.ssd_expand = ssd_expand
        self.d_inner = int(self.ssd_expand * dim)
        self.state_dim = state_dim
        self.a_init_range = a_init_range

        self.BCdt_proj = ConvLayer1d(dim, 3 * state_dim, 1, norm=None, act_layer=None, **dd)
        conv_dim = self.state_dim * 3
        self.dw = ConvLayer2d(conv_dim, conv_dim, 3, 1, 1, groups=conv_dim, norm=None,
                              act_layer=None, bn_weight_init=0, **dd)
        self.hz_proj = ConvLayer1d(dim, 2 * self.d_inner, 1, norm=None, act_layer=None, **dd)
        self.out_proj = ConvLayer1d(self.d_inner, dim, 1, norm=None, act_layer=None,
                                    bn_weight_init=0, **dd)

        self.A = nn.Parameter(torch.empty(self.state_dim, **dd))
        self.D = nn.Parameter(torch.ones(1, device=device, dtype=dtype))
        self.act = nn.SiLU()
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.uniform_(self.A, *self.a_init_range)
        nn.init.ones_(self.D)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch, _, height, width = x.shape
        x = x.flatten(2)

        BCdt = self.dw(self.BCdt_proj(x).reshape(batch, -1, height, width)).flatten(2)
        b, c, dt = torch.split(BCdt, [self.state_dim, self.state_dim, self.state_dim], dim=1)
        a = (dt + self.A.view(1, -1, 1)).softmax(-1)

        h = x @ (a * b).transpose(-2, -1)                     # 压缩隐藏状态 (B, C, N)
        h, z = torch.split(self.hz_proj(h), [self.d_inner, self.d_inner], dim=1)
        h = self.out_proj(h * self.act(z) + h * self.D)       # 混合后的隐藏状态 (B, C, N)
        y = h @ c                                             # 读出 (B, C, L)
        return y.reshape(batch, -1, height, width).contiguous(), h

class EfficientViMBlock(nn.Module):
    """dwconv -> HSM-SSD -> dwconv -> FFN，每段带 sigmoid 门控的残差。"""

    def __init__(self, dim: int, mlp_ratio: float = 4.0, ssd_expand: float = 1.0,
                 state_dim: int = 64, drop_path: float = 0.0, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        self.mixer = HSMSSD(dim, ssd_expand=ssd_expand, state_dim=state_dim, **dd)
        self.norm = LayerNorm2d(dim, eps=1e-5, **dd)
        self.dwconv1 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0,
                                   act_layer=None, **dd)
        self.dwconv2 = ConvLayer2d(dim, dim, 3, padding=1, groups=dim, bn_weight_init=0,
                                   act_layer=None, **dd)
        self.ffn = FFN(dim, int(dim * mlp_ratio), **dd)
        self.alpha = nn.Parameter(1e-4 * torch.ones(4, dim, device=device, dtype=dtype))
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def reset_parameters(self) -> None:
        nn.init.constant_(self.alpha, 1e-4)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = torch.sigmoid(self.alpha).view(4, -1, 1, 1)

        x = x + self.drop_path(alpha[0] * (self.dwconv1(x) - x))

        x_prev = x
        x, h = self.mixer(self.norm(x))
        x = x_prev + self.drop_path(alpha[1] * (x - x_prev))

        x = x + self.drop_path(alpha[2] * (self.dwconv2(x) - x))
        x = x + self.drop_path(alpha[3] * (self.ffn(x) - x))
        return x, h


class EfficientViMStage(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, depth: int, mlp_ratio: float = 4.0,
                 downsample: Optional[type] = None, ssd_expand: float = 1.0,
                 state_dim: int = 64, drop_path: Optional[List[float]] = None,
                 device=None, dtype=None):
        super().__init__()
        assert depth > 0
        self.depth = depth
        self.grad_checkpointing = False
        drop_path = drop_path or [0.0] * depth
        self.blocks = nn.ModuleList([
            EfficientViMBlock(dim=in_dim, mlp_ratio=mlp_ratio, ssd_expand=ssd_expand,
                              state_dim=state_dim, drop_path=drop_path[i],
                              device=device, dtype=dtype)
            for i in range(depth)
        ])
        self.downsample = (downsample(in_dim=in_dim, out_dim=out_dim, device=device,
                                      dtype=dtype) if downsample is not None else None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回（本 stage 特征图, 下一 stage 输入, 最后一个 block 的隐藏状态）。"""
        h = None
        for blk in self.blocks:
            if self.grad_checkpointing and self.training:
                x, h = _ckpt(blk, x, use_reentrant=False)
            else:
                x, h = blk(x)
        x_out = x
        if self.downsample is not None:
            x = self.downsample(x)
        return x_out, x, h

class EfficientViM(nn.Module):
    """EfficientViM 骨干 + 4 路 softmax 加权分类头（二分类时 num_classes=2）。

    Args:
        in_chans: 输入通道数（RGB 为 3）。
        num_classes: 分类数；本任务为 2。
        embed_dim: 三个 stage 的通道宽度。
        depths: 三个 stage 的 block 数。
        mlp_ratio: FFN 扩张比。
        ssd_expand: HSM-SSD 隐藏状态扩张比。
        state_dim: 每个 stage 的 HSM-SSD 状态维度。
        global_pool: 'avg' 表示对每个分支做全局平均池化。
        drop_rate: 分类头 dropout。
        drop_path_rate: 随机深度终值（按块线性递增）。
    """

    def __init__(self, in_chans: int = 3, num_classes: int = 1000,
                 embed_dim: Tuple[int, ...] = (128, 192, 320),
                 depths: Tuple[int, ...] = (2, 2, 2), mlp_ratio: float = 4.0,
                 ssd_expand: float = 1.0, state_dim: Tuple[int, ...] = (49, 25, 9),
                 global_pool: str = 'avg', drop_rate: float = 0.0,
                 drop_path_rate: float = 0.0, device=None, dtype=None):
        super().__init__()
        dd = {'device': device, 'dtype': dtype}
        assert global_pool in ('avg', '')
        assert global_pool or num_classes == 0
        assert len(depths) == len(embed_dim) == len(state_dim)

        self.in_chans = in_chans
        self.global_pool = global_pool
        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = tuple(embed_dim)
        self.grad_checkpointing = False
        self.feature_info: List[Dict] = []

        self.patch_embed = Stem(in_dim=in_chans, dim=embed_dim[0], **dd)
        dpr = calculate_drop_path_rates(drop_path_rate, sum(depths))

        stages = []
        block_idx = 0
        reduction = 16
        for i_layer in range(self.num_layers):
            stage_dpr = dpr[block_idx: block_idx + depths[i_layer]]
            block_idx += depths[i_layer]
            stages.append(EfficientViMStage(
                in_dim=int(embed_dim[i_layer]),
                out_dim=int(embed_dim[i_layer + 1]) if (i_layer < self.num_layers - 1) else 0,
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio,
                downsample=PatchMerging if (i_layer < self.num_layers - 1) else None,
                ssd_expand=ssd_expand,
                state_dim=state_dim[i_layer],
                drop_path=stage_dpr,
                **dd,
            ))
            self.feature_info.append(dict(num_chs=embed_dim[i_layer], reduction=reduction,
                                          module=f'stages.{i_layer}'))
            if i_layer < self.num_layers - 1:
                reduction *= 2
        self.stages = nn.ModuleList(stages)

        self.num_features = self.head_hidden_size = sum(embed_dim) + embed_dim[-1]
        self.weights = nn.Parameter(torch.ones(self.num_layers + 1, **dd),
                                    requires_grad=num_classes > 0)
        self.norm = nn.ModuleList([
            *[nn.LayerNorm(dim, eps=1e-5, **dd) for dim in embed_dim],
            LayerNorm2d(embed_dim[-1], eps=1e-5, **dd),
        ])
        self.head_drop = nn.Dropout(drop_rate)
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes, **dd) if num_classes > 0 else nn.Identity()
            for dim in (*self.embed_dim, self.embed_dim[-1])
        ])
        self.init_weights(needs_reset=False)

    # ------------------------------------------------------------------ 初始化
    @torch.jit.ignore
    def init_weights(self, needs_reset: bool = True) -> None:
        nn.init.ones_(self.weights)
        self.apply(partial(self._init_weights, needs_reset=needs_reset))

    def _init_weights(self, m: nn.Module, needs_reset: bool = True) -> None:
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.BatchNorm1d, LayerNorm2d)):
            m.reset_parameters()
        elif needs_reset and hasattr(m, 'reset_parameters'):
            m.reset_parameters()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {name for name, _ in self.named_parameters() if name.endswith('.D')}

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True) -> None:
        self.grad_checkpointing = enable
        for stage in self.stages:
            stage.grad_checkpointing = enable

    def reset_classifier(self, num_classes: int, global_pool: Optional[str] = None) -> None:
        if global_pool is not None:
            assert global_pool in ('avg', '')
            self.global_pool = global_pool
            assert self.global_pool or num_classes == 0
        self.num_classes = num_classes
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
            for dim in (*self.embed_dim, self.embed_dim[-1])
        ])
        self.weights.requires_grad_(num_classes > 0)

    # ------------------------------------------------------------------ 前向
    def forward_features(self, x: torch.Tensor) -> List[torch.Tensor]:
        """返回三个 stage 的归一化隐藏状态与最终特征图（均为 NCL）。"""
        x = self.patch_embed(x)
        features = []
        for stage, norm in zip(self.stages, self.norm):
            _, x, h = stage(x)
            features.append(norm(h.transpose(1, 2)).transpose(1, 2))
        features.append(self.norm[-1](x).flatten(2))
        return features

    def forward_head(self, x: List[torch.Tensor],
                     pre_logits: bool = False) -> Union[torch.Tensor, List[torch.Tensor]]:
        if not self.global_pool:
            return x
        x = [self.head_drop(t.mean(-1)) for t in x]
        if pre_logits or self.num_classes == 0:
            return torch.cat(x, dim=1)
        weights = self.weights.softmax(-1)
        logits = [weights[i] * head(x[i]) for i, head in enumerate(self.heads)]
        return torch.stack(logits).sum(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(x))


def efficientvim_m1(num_classes: int = 2, in_chans: int = 3, **kwargs) -> EfficientViM:
    """EfficientViM-M1：6.7M 参数（ImageNet 配置），分类头替换为 `num_classes`。"""
    model_args = dict(
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=(128, 192, 320),
        depths=(2, 2, 2),
        state_dim=(49, 25, 9),
    )
    model_args.update(kwargs)
    return EfficientViM(**model_args)
