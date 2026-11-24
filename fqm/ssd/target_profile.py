"""Match the empirical joint distribution of image geometry and encoding."""
import hashlib
import io
import random
from collections import Counter
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, JpegImagePlugin


def stable_key(namespace, value):
    return hashlib.sha256(f"{namespace}:{value}".encode()).digest()


def collect_profiles(directory):
    profiles = Counter()
    for path in sorted(Path(directory).iterdir()):
        if not path.is_file():
            continue
        with Image.open(path) as image:
            encoding = image.format
            tables, sampling = (), None
            if encoding == "JPEG":
                tables = tuple(tuple(table) for _, table in sorted(image.quantization.items()))
                sampling = JpegImagePlugin.get_sampling(image)
            profiles[(image.width, image.height, encoding, sampling, tables)] += 1
    if not profiles:
        raise ValueError("No image profiles found")
    return [[width, height, encoding, sampling, tables, count]
            for (width, height, encoding, sampling, tables), count in sorted(profiles.items())]


def encode(image, profile):
    _, _, encoding, sampling, tables, _ = profile
    if encoding == "PNG":
        return image
    if encoding == "WEBP":
        options = {"quality": 80}
    elif encoding == "JPEG" and sampling in (-1, 0, 1, 2) and len(tables) == 2:
        options = {"qtables": [list(table) for table in tables], "subsampling": 2 if sampling == -1 else sampling}
    elif encoding == "JPEG" and sampling == -1 and len(tables) == 1:
        image = image.convert("L")
        options = {"qtables": [list(tables[0])]}
    else:
        raise ValueError(f"Unsupported profile: {encoding} sampling={sampling} tables={len(tables)}")
    with io.BytesIO() as buffer:
        image.save(buffer, format=encoding, **options)
        buffer.seek(0)
        with Image.open(buffer) as encoded:
            return encoded.convert("RGB")


def process(image, profile, mode, center, *, other_blur_radius=0.):
    if mode in ("geometry", "both"):
        image = ImageOps.fit(image, tuple(profile[:2]), method=Image.Resampling.BICUBIC,
                             centering=center)
    if mode == "both" and profile[0] != 500 and other_blur_radius:
        image = image.filter(ImageFilter.GaussianBlur(other_blur_radius))
    if mode in ("encoding", "both"):
        image = encode(image, profile)
    return image


def validation_groups(rows):
    """Split each class into a checkpoint-selection half and a held-out half."""
    halves = [[], []]
    for label in (0, 1):
        group = sorted((r for r in rows if r["label"] == label),
                       key=lambda row: stable_key("target-profile-holdout-v1", row["image_id"]))
        mid = len(group) // 2
        halves[0].extend(group[:mid])
        halves[1].extend(group[mid:])
    return tuple(halves)


class FixedTargetMatch:
    """Give the two labels identical profile and crop-center assignments per half."""
    def __init__(self, rows, profiles, mode, seed, *, other_blur_radius=0.):
        self.mode = mode
        self.other_blur_radius = other_blur_radius
        self.assignments = {}
        pool = [profile for profile in profiles for _ in range(profile[-1])]
        rng = random.Random(seed)
        for half in validation_groups(rows):
            groups = [[row for row in half if row["label"] == label] for label in (0, 1)]
            if len(groups[0]) != len(groups[1]):
                raise ValueError("Expected balanced validation classes")
            count = len(groups[0])
            chosen = [pool[int((i + .5) * len(pool) / count)] for i in range(count)]
            rng.shuffle(chosen)
            for left, right, profile in zip(*groups, chosen):
                center = (rng.uniform(.15, .85), rng.uniform(.15, .85))
                for row in (left, right):
                    self.assignments[row["image_id"]] = (profile, center)

    def __call__(self, image, image_id):
        profile, center = self.assignments[image_id]
        return process(image, profile, self.mode, center, other_blur_radius=self.other_blur_radius)

    def cohort(self, image_id):
        profile, _ = self.assignments[image_id]
        return "width500" if profile[0] == 500 else "other"


class RandomTargetMatch:
    """Draw a joint geometry/encoding profile independently of the image label."""
    def __init__(self, profiles):
        self.pool = [profile for profile in profiles for _ in range(profile[-1])]

    def __call__(self, image):
        profile = random.choice(self.pool)
        return process(image, profile, "both", (random.random(), random.random()))
