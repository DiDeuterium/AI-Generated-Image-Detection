"""Train one native-patch texture model from random weights."""
import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
import time

from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from detection.data import jpeg_roundtrip
from detection.training import save_checkpoint, seed_all
from lqy.canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, canonical_views
from lqy.codec_patch import labeled_data, tables_by_frequency
from lqy.codec_scale import contrast_components
from lqy.source_holdout import ResidualBlock, block
from lqy_resize_retrain import training_split

MODEL = "native_texture_base"
RESAMPLING_MODEL = "resampling_texture_base"
WEIGHTED_MODEL = "resize_jpeg_weighted_texture_base"
RANKED_MODEL = "resize_jpeg_ranked_texture_base"
WIDE_MODEL = "wide_native_texture_base"
DEEP_MODEL = "deep_native_texture_base"
RESIZE_JPEG_WEIGHT = 2.0
RANK_LOSS_WEIGHT = .1
HARD_REAL_COUNT = 2
BASE_CHANNELS = (24, 32, 48, 64)
WIDE_CHANNELS = (24, 48, 72, 96)
BASE_DEPTHS = (1, 1, 1, 1)
DEEP_DEPTHS = (1, 1, 3, 3)
SEED = 42
BATCH = 24
KERNELS = (Image.Resampling.BICUBIC, Image.Resampling.LANCZOS, Image.Resampling.BILINEAR)


class NativeTextureBase(CorrelationCNN):
    def __init__(self, *, channels=BASE_CHANNELS, depths=BASE_DEPTHS):
        super().__init__(CORRELATION_VARIANT)
        # Extract local detail before reducing the spatial grid a second time.
        layers = []
        input_channels = 24
        for output_channels, depth in zip(channels, depths, strict=True):
            layers.append(block(input_channels, output_channels, 3, 2))
            layers.extend(ResidualBlock(output_channels) for _ in range(depth))
            input_channels = output_channels
        self.features = nn.Sequential(*layers)
        final = channels[-1]
        pooled_size = 2 * final + self.pairs.shape[1]
        if final != BASE_CHANNELS[-1]:
            self.projection = nn.Conv2d(final, self.projection.out_channels, 1, bias=False)
            self.classifier[0] = nn.LayerNorm(pooled_size)
            self.classifier[1] = nn.Linear(pooled_size, self.classifier[1].out_features)
        self.attention = nn.Linear(pooled_size, 1)
        nn.init.zeros_(self.attention.weight)
        nn.init.zeros_(self.attention.bias)

    def pool_patches(self, views):
        patches = self.patch_embeddings(views).reshape(len(views), views.shape[1], -1)
        return self.pool_embeddings(patches)

    def pool_embeddings(self, patches):
        weights = self.attention(patches).float().softmax(dim=1)
        mean = patches.mean(dim=1)
        focused = (patches * weights).sum(dim=1)
        return .5 * (mean + focused)

    def forward(self, views):
        return self.classifier(self.pool_patches(views))


def resampling_residuals(rgb):
    """Probe luminance with a 3/4-scale bilinear round trip along each axis."""
    luma = 0.299 * rgb[:, :1] + 0.587 * rgb[:, 1:2] + 0.114 * rgb[:, 2:3]
    height, width = luma.shape[-2:]

    def round_trip(pixels, size):
        reduced = F.interpolate(pixels, size=size, mode="bilinear", align_corners=False, antialias=True)
        return F.interpolate(reduced, size=(height, width), mode="bilinear", align_corners=False)

    horizontal = round_trip(luma, (height, 3 * width // 4))
    vertical_size = (3 * height // 4, width)
    vertical = round_trip(luma, vertical_size)
    both = round_trip(horizontal, vertical_size)
    # The mixed difference isolates detail removed by filtering along both axes.
    return luma - horizontal, luma - vertical, luma - horizontal - vertical + both


class ResamplingTexture(NativeTextureBase):
    def patch_features(self, views):
        rgb, residue, change, scale = contrast_components(views)
        kernels = self.high_pass - self.high_pass.mean(dim=(2, 3), keepdim=True)
        learned = F.conv2d(F.pad(rgb, (2, 2, 2, 2), mode="reflect"), kernels, groups=3)
        resampling = tuple(6 * response for response in resampling_residuals(rgb))
        return torch.cat((*resampling, 6 * residue, 6 * change,
                          (residue / scale).tanh(), (change / scale).tanh(), 6 * learned), dim=1)


def acquisition_scale(size, assignment):
    fraction = assignment[1] / 255
    return .62 + .34 * fraction if min(size) >= 160 else 1.08 + .30 * fraction


def acquisition(image, assignment, *, scale=None):
    condition = assignment[0] % 4
    if condition in (2, 3):
        if scale is None:
            scale = acquisition_scale(image.size, assignment)
        size = tuple(max(1, round(axis * scale)) for axis in image.size)
        image = image.resize(size, KERNELS[assignment[2] % len(KERNELS)])
    if condition in (1, 3):
        image = jpeg_roundtrip(image, quality=66 + assignment[3] % 29, subsampling=assignment[4] % 3)
    return image


def extra_acquisition(image, assignment):
    independent = sha256(b"native-texture-extra-views-v1:" + assignment).digest()
    return acquisition(image, independent)


def sample_view_pair(views, generator):
    if len(views) == 2:
        return views
    first, second = torch.randperm(len(views), generator=generator)[:2].tolist()
    return views[first], views[second]


def resize_jpeg_weights(samples):
    # canonical_views renders the first pair from assignment[5:] and the second from assignment.
    assignments = [sha256(f"canonical-augmentation:{row.image_id}".encode()).digest() for row in samples]
    return tuple(torch.tensor([RESIZE_JPEG_WEIGHT if assignment[offset] % 4 == 3 else 1.0
                               for assignment in assignments], dtype=torch.float32) for offset in (5, 0))


def paired_loss(model, first, second, labels):
    # Concatenate views so both see the same batch statistics.
    a, b = model(torch.cat((first, second))).float().chunk(2)
    return consistency_loss(a, b, labels)


def weighted_paired_loss(model, first, second, labels):
    pixels_a, weights_a = first
    pixels_b, weights_b = second
    a, b = model(torch.cat((pixels_a, pixels_b))).float().chunk(2)
    return consistency_loss(a, b, labels, (weights_a, weights_b))


def hard_pair_ranking(logits, labels, resize_jpeg):
    margins = logits[:, 1] - logits[:, 0]
    real = resize_jpeg & (labels == 0)
    fake = resize_jpeg & (labels == 1)
    # Train against the highest-scored resized and recompressed photographs in the batch.
    real_indices = margins.masked_fill(~real, -torch.inf).topk(min(HARD_REAL_COUNT, len(margins))).indices
    pair_mask = real[real_indices, None] & fake[None, :]
    violations = margins[real_indices, None] - margins[None, :]
    losses = F.softplus(violations.masked_fill(~pair_mask, 0)).masked_fill(~pair_mask, 0)
    return losses.sum() / pair_mask.sum().clamp_min(1)


def ranked_paired_loss(model, first, second, labels):
    pixels_a, weights_a = first
    pixels_b, weights_b = second
    logits = model(torch.cat((pixels_a, pixels_b))).float()
    a, b = logits.chunk(2)
    resize_jpeg = torch.cat((weights_a, weights_b)) > 1
    ranking = hard_pair_ranking(logits, labels.repeat(2), resize_jpeg)
    return consistency_loss(a, b, labels) + RANK_LOSS_WEIGHT * ranking


def consistency_loss(a, b, labels, weights=None):
    if weights is None:
        supervised = .5 * (F.cross_entropy(a, labels) + F.cross_entropy(b, labels))
    else:
        weights_a, weights_b = weights
        loss_a = F.cross_entropy(a, labels, reduction="none")
        loss_b = F.cross_entropy(b, labels, reduction="none")
        supervised = (loss_a * weights_a + loss_b * weights_b).sum() / (weights_a + weights_b).sum()
    difference = a.softmax(dim=1) - b.softmax(dim=1)
    return supervised + .1 * difference.square().mean()


def batch_view(view, ids, device):
    return view[ids].to(device)


def weighted_batch_view(view, ids, device):
    pixels, weights = view
    return batch_view(pixels, ids, device), batch_view(weights, ids, device)


def fit(model, views, labels, output, device, epochs, deadline, *,
        variant=MODEL, get_batch=batch_view, loss_fn=paired_loss, sample_count=None, backbone_lr_scale=1.0):
    count = len(labels) if labels is not None else sample_count
    if not count:
        raise ValueError("Expected a positive training sample count")
    learning_rate, minimum_lr = .001, .00003
    if backbone_lr_scale == 1.0:
        parameters = (param for param in model.parameters() if param.requires_grad)
        optimizer = torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=.02)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=minimum_lr)
    else:
        head = [param for param in model.classifier.parameters() if param.requires_grad]
        head_ids = {id(param) for param in head}
        backbone = [param for param in model.parameters() if param.requires_grad and id(param) not in head_ids]
        optimizer = torch.optim.AdamW([
            {"params": head}, {"params": backbone, "lr": learning_rate * backbone_lr_scale},
        ], lr=learning_rate, weight_decay=.02)
        floor = minimum_lr / learning_rate

        def cosine(epoch):
            return floor + (1 - floor) * .5 * (1 + math.cos(math.pi * epoch / epochs))

        schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine)
    scaler = torch.amp.GradScaler(device.type, init_scale=2048)
    order = torch.Generator().manual_seed(SEED)
    view_order = torch.Generator().manual_seed(SEED + 1)
    log = []
    for epoch in range(1, epochs + 1):
        model.train()
        started = time.monotonic()
        running = torch.zeros((), device=device)
        for ids in torch.randperm(count, generator=order).split(BATCH):
            if time.monotonic() > deadline:
                raise TimeoutError("Slurm training budget elapsed")
            pair = sample_view_pair(views, view_order)
            first, second = (get_batch(source, ids, device) for source in pair)
            targets = labels[ids].to(device) if labels is not None else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16):
                loss = loss_fn(model, first, second, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.detach() * len(ids)
        loss = running.item() / count
        if not torch.isfinite(running).item():
            raise ValueError(f"Non-finite loss in epoch {epoch}")
        log.append({"epoch": epoch, "loss": loss, "lr": schedule.get_last_lr()[0],
                    "seconds": time.monotonic() - started})
        if backbone_lr_scale != 1.0:
            log[-1]["backbone_lr"] = schedule.get_last_lr()[1]
        schedule.step()
        state = {"variant": variant, "seed": SEED, "epoch": epoch, "weights": model.state_dict()}
        save_checkpoint(output / "last.pt", state)
        save_checkpoint(output / f"epoch-{epoch}.pt", state)
        (output / "epochs.json").write_text(json.dumps(log, indent=2) + "\n")
        print(f"Epoch {epoch}/{epochs} loss={loss:.4f} seconds={log[-1]['seconds']:.1f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--epochs", default=24, type=int)
    parser.add_argument("--view-pairs", default=1, type=int, choices=(1, 2))
    experiments = parser.add_mutually_exclusive_group()
    experiments.add_argument("--resampling-response", action="store_true")
    experiments.add_argument("--weight-resize-jpeg", action="store_true")
    experiments.add_argument("--rank-resize-jpeg", action="store_true")
    experiments.add_argument("--wide", action="store_true", help="widen the last three CNN stages")
    experiments.add_argument("--deep", action="store_true", help="add residual blocks to the last two CNN stages")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs must be positive")
    if (args.weight_resize_jpeg or args.rank_resize_jpeg) and args.view_pairs != 1:
        parser.error("resize-JPEG loss experiments use the baseline's two rendered views")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    device = torch.device("cuda")
    deadline = time.monotonic() + 85 * 60
    split, details = labeled_data()
    train, calibration = training_split(split["train"])
    tables, _ = tables_by_frequency(split["train"])
    seed_all(SEED, device)
    variant, model_class = (RESAMPLING_MODEL, ResamplingTexture) if args.resampling_response else (MODEL, NativeTextureBase)
    if args.weight_resize_jpeg:
        variant = WEIGHTED_MODEL
    if args.rank_resize_jpeg:
        variant = RANKED_MODEL
    channels = WIDE_CHANNELS if args.wide else BASE_CHANNELS
    depths = DEEP_DEPTHS if args.deep else BASE_DEPTHS
    if args.wide:
        variant = WIDE_MODEL
    if args.deep:
        variant = DEEP_MODEL
    model = model_class(channels=channels, depths=depths).to(device, memory_format=torch.channels_last)
    report = {"variant": variant, "gpu": torch.cuda.get_device_name(device), "seed": SEED, "epochs": args.epochs,
              "split_sha256": details["split_sha256"], "train_count": len(train), "calibration_count": len(calibration),
              "parameters": sum(param.numel() for param in model.parameters()),
              "stage_channels": channels,
              "stage_residual_blocks": depths,
              "training_views": 2 * args.view_pairs, "stage": "training"}
    weights = resize_jpeg_weights(train) if args.weight_resize_jpeg or args.rank_resize_jpeg else None
    if weights is not None:
        report["resize_jpeg_counts_by_view"] = [int((view > 1).sum()) for view in weights]
    if args.weight_resize_jpeg:
        report["resize_jpeg_weight"] = RESIZE_JPEG_WEIGHT
    if args.rank_resize_jpeg:
        report.update(ranking_loss_weight=RANK_LOSS_WEIGHT, hard_real_count=HARD_REAL_COUNT)
    summary = args.output / "summary.json"
    summary.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    views = canonical_views(train, tables[0], deadline, paired=True, alter_both=True, acquisition=acquisition)
    if args.view_pairs == 2:
        print("Preparing two independently rendered additional training views", flush=True)
        views += canonical_views(train, tables[0], deadline, paired=True, alter_both=True,
                                 acquisition=extra_acquisition)
    fit_options = {}
    if weights is not None:
        views = tuple(zip(views, weights, strict=True))
        objective = ranked_paired_loss if args.rank_resize_jpeg else weighted_paired_loss
        fit_options = {"get_batch": weighted_batch_view, "loss_fn": objective}
    fit(model, views, torch.tensor([row.label for row in train]), args.output, device, args.epochs, deadline,
        variant=variant, **fit_options)
    report["stage"] = "training_complete"
    summary.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
