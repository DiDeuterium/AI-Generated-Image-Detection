"""Image-level degradation before geometry and deterministic proxy parameters."""
import hashlib
import io
import random

import numpy as np
from PIL import Image, ImageFilter


PROXY_CASES = ("scale75_q75", "moderate", "strong")


def apply_degradation(image, scale, quality, blur):
    if scale < 1:
        size = tuple(max(8, round(side * scale)) for side in image.size)
        image = image.resize(size, Image.Resampling.BICUBIC)
    if blur:
        image = image.filter(ImageFilter.GaussianBlur(blur))
    if quality is None:
        return image
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling=2)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")


class RandomDomainShift:
    """Sample the same distribution for both classes, always re-encoding."""
    def __init__(self, native_prob, blur_prob):
        if not 0 <= native_prob <= 1 or not 0 <= blur_prob <= 1:
            raise ValueError("Domain probabilities must be between 0 and 1")
        self.native_prob = native_prob
        self.blur_prob = blur_prob

    def __call__(self, image):
        scale = 1. if random.random() < self.native_prob else random.uniform(.5, 1.)
        blur = random.uniform(.35, .8) if random.random() < self.blur_prob else 0.
        return apply_degradation(image, scale, random.randint(60, 95), blur)


class FixedDomainShift:
    """Deterministic variants keyed by image ID, independent of class and workers."""
    def __init__(self, case):
        if case not in PROXY_CASES:
            raise ValueError(f"Unknown proxy: {case}")
        self.case = case

    def __call__(self, image, image_id):
        digest = hashlib.sha256(f"training-proxy:{image_id}".encode()).digest()
        rng = random.Random(int.from_bytes(digest[:8], "big"))
        if self.case == "scale75_q75":
            return apply_degradation(image, .75, 75, 0.)
        if self.case == "moderate":
            return apply_degradation(image, rng.uniform(.75, 1.), rng.randint(73, 95), 0.)
        return apply_degradation(image, rng.uniform(.5, .75), rng.randint(60, 80),
                                 rng.choice((0., .45, .8)))


def legacy_degrade(image, image_id, case):
    if case == "orig":
        return image
    seed = hashlib.sha256(f"20261121:{image_id}".encode()).digest()
    rng = np.random.default_rng(int.from_bytes(seed[:8], "big"))
    scale, quality, blur = 1., None, 0.
    if case == "q75":
        quality = 75
    elif case in ("scale75", "scale75_q75"):
        scale = .75
        quality = 75 if case == "scale75_q75" else None
    elif case == "moderate":
        scale, quality = rng.uniform(.75, 1), int(rng.integers(73, 96))
    elif case == "strong":
        scale, quality = rng.uniform(.5, .75), int(rng.integers(60, 81))
        blur = float(rng.choice([0., .45, .8]))
    else:
        raise ValueError(case)
    return apply_degradation(image, scale, quality, blur)
