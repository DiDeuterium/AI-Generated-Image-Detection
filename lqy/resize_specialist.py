"""Train a CNN specifically on the resizing regime where the released model loses recall."""

import argparse
from functools import partial
from pathlib import Path
import time

from PIL import Image

from detection.data import jpeg_roundtrip

from . import codec_canonical
from .canonical_augmentation import CORRELATION_VARIANT, CorrelationCNN, canonical_views
from .cnn import device_for, run_lock


def near_resize(image, assignment):
    position = assignment[0] / 255
    factor = (0.70 + 0.21 * position) if min(image.size) >= 160 else (1.10 + 0.24 * position)
    size = tuple(max(1, round(extent * factor)) for extent in image.size)
    kernels = (Image.Resampling.BICUBIC, Image.Resampling.LANCZOS, Image.Resampling.BILINEAR)
    altered = image.resize(size, kernels[assignment[1] % len(kernels)])
    if assignment[2] % 2:
        return altered
    return jpeg_roundtrip(altered, quality=72 + assignment[3] % 24, subsampling=assignment[4] % 3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=("xpu", "cuda", "cpu"), default="xpu")
    parser.add_argument("--calibration-reference", type=Path,
                        default=Path("lqy/runs/source-holdout/canonical-robust-calibration.json"))
    parser.add_argument("--narrow-reference", type=Path, default=Path("lqy/runs/codec-final/upstream-trained"))
    parser.add_argument("--near-reference", type=Path, default=Path("lqy/runs/codec-final/near-upstream"))
    parser.add_argument("--night-reference", type=Path, default=Path("lqy/runs/night-train/2026-11-21"))
    parser.add_argument("--spectrum-reference", type=Path, default=Path("lqy/runs/spectrum-patch/round-01"))
    args = parser.parse_args()
    training = partial(canonical_views, paired=True, alter_both=True, acquisition=near_resize)
    with run_lock(args.output):
        codec_canonical.run(args, device_for(args.device, 6), time.monotonic() + 3600,
                            training_loader=training, model_variant=CORRELATION_VARIANT, model_factory=CorrelationCNN,
                            augmentation_check={"resize_factor_large": [0.70, 0.91],
                                                "resize_factor_small": [1.10, 1.34], "jpeg_quality": [72, 95]},
                            screen_weights=(0.25, *codec_canonical.WEIGHTS),
                            confirm_profiles=("jpeg76", "alternate_resize_jpeg"))


if __name__ == "__main__":
    main()
