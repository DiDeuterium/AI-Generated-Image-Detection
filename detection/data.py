import csv
import hashlib
import io
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    image_id: str
    path: Path
    label: int = -1


def read_csv(path, required):
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not set(required).issubset(reader.fieldnames or ()):
            raise ValueError(f"{path} must contain the columns {', '.join(required)}")
        rows = list(reader)
    ids = [row["image_id"] for row in rows]
    if not ids or any(not image_id or image_id.strip() != image_id for image_id in ids):
        raise ValueError(f"{path} contains missing or invalid image IDs")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path} contains duplicate image IDs")
    return rows


def fingerprint(path):
    # Make provenance independent of Git checkout line endings and an optional BOM.
    return hashlib.sha256(path.read_text(encoding="utf-8-sig").encode("utf-8")).hexdigest()


def read_split(manifest, image_root, expected_counts=None):
    groups = {"train": [], "val": []}
    paths = set()
    for row in read_csv(manifest, ("image_id", "image_path", "label", "split")):
        if row["split"] not in groups or row["label"] not in {"0", "1"}:
            raise ValueError(f"Invalid split or label for {row['image_id']}")
        if not row["image_path"]:
            raise ValueError(f"Missing image path for {row['image_id']}")
        relative = Path(row["image_path"])
        if relative.anchor or ".." in relative.parts:
            raise ValueError(f"Expected a relative image path, got {relative}")
        path = image_root / relative
        if path in paths:
            raise ValueError(f"Image path appears more than once: {path}")
        paths.add(path)
        groups[row["split"]].append(Sample(row["image_id"], path, int(row["label"])))

    for name, samples in groups.items():
        counts = Counter(sample.label for sample in samples)
        if set(counts) != {0, 1}:
            raise ValueError(f"{name} must contain both classes")
        if expected_counts and counts != {0: expected_counts[name], 1: expected_counts[name]}:
            raise ValueError(f"Unexpected {name} class counts: {dict(counts)}")
    missing = [path for path in sorted(paths) if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} images missing under {image_root}. First: {missing[0]}")
    return groups


def read_test(template, image_dir):
    rows = read_csv(template, ("image_id", "label"))
    files = {path.name: path for path in image_dir.iterdir()
             if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS}
    ids = [row["image_id"] for row in rows]
    missing, extra = set(ids) - files.keys(), files.keys() - set(ids)
    if missing or extra:
        raise ValueError(f"Test filenames must match template IDs. "
                         f"Missing: {sorted(missing)[:3]}; extra: {sorted(extra)[:3]}")
    return [Sample(image_id, files[image_id]) for image_id in ids]


def jpeg_roundtrip(image, quality, subsampling):
    with io.BytesIO() as buffer:
        image.save(buffer, format="JPEG", quality=quality, subsampling=subsampling)
        buffer.seek(0)
        with Image.open(buffer) as decoded:
            return decoded.convert("RGB")


class ImageDataset(Dataset):
    def __init__(
        self, samples, short_side, crop_size, *, norm=None, train=False,
        flip_probability=0.5, jpeg_probability=0.0, quality=75, subsampling=2,
    ):
        self.samples = samples
        self.jpeg_probability = jpeg_probability
        self.quality = quality
        self.subsampling = subsampling
        self.train = train
        self.crop_size = crop_size
        self.flip_probability = flip_probability
        self.resize = transforms.Resize(short_side, interpolation=transforms.InterpolationMode.BILINEAR,
                                        antialias=True)
        self.center_crop = transforms.CenterCrop(crop_size)
        ops = [transforms.ToTensor()]
        if norm is not None:
            ops.append(transforms.Normalize(norm["mean"], norm["std"]))
        self.transform = transforms.Compose(ops)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        rng = random
        if isinstance(index, tuple):
            epoch_seed, index = index
            rng = random.Random((epoch_seed << 32) + index)
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as source:
                image = source.convert("RGB")
        except (UnidentifiedImageError, OSError) as exc:
            raise ValueError(f"Cannot decode {sample.image_id} at {sample.path}: {exc}") from exc
        recode = self.jpeg_probability == 1
        if self.train or 0 < self.jpeg_probability < 1:
            recode = rng.random() < self.jpeg_probability
        if recode:
            image = jpeg_roundtrip(image, self.quality, self.subsampling)
        image = self.resize(image)
        if self.train:
            width, height = image.size
            top = rng.randrange(height - self.crop_size + 1)
            left = rng.randrange(width - self.crop_size + 1)
            image = transforms.functional.crop(image, top, left, self.crop_size, self.crop_size)
            if rng.random() < self.flip_probability:
                image = transforms.functional.hflip(image)
        else:
            image = self.center_crop(image)
        return self.transform(image), sample.label, sample.image_id


class EpochSampler(Sampler):
    """Send the epoch seed with each index to keep augmentation independent of worker lifetime."""

    def __init__(self, dataset):
        self.dataset = dataset
        self.seed = 0

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        seed = self.seed
        generator = torch.Generator().manual_seed(seed)
        return ((seed, index) for index in torch.randperm(len(self), generator=generator).tolist())


def seed_worker(_worker_id):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(dataset, batch_size, workers, *, shuffle=False, seed=0, pin_memory=False,
                sampler=None, persistent=False):
    return DataLoader(
        dataset, batch_size=batch_size, num_workers=workers, shuffle=shuffle,
        pin_memory=pin_memory, worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(seed), sampler=sampler,
        persistent_workers=persistent and workers > 0,
    )


def estimate_normalization(loader):
    total = torch.zeros(3, dtype=torch.float64)
    squares = torch.zeros_like(total)
    pixels = 0
    for images, _, _ in loader:
        images = images.to(torch.float64)
        total += images.sum(dim=(0, 2, 3))
        squares += images.square().sum(dim=(0, 2, 3))
        pixels += images.shape[0] * images.shape[2] * images.shape[3]
    mean = total / pixels
    std = (squares / pixels - mean.square()).clamp_min(0).sqrt()
    if not torch.isfinite(std).all() or (std <= 0).any():
        raise ValueError("Training images must have positive variance in every channel")
    return {"mean": mean.tolist(), "std": std.tolist(), "pixel_count_per_channel": pixels}
