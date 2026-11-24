"""Match resize and JPEG treatments between real and generated training images."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha256

from PIL import Image

from lqy.night_train import CROP
from native_texture_base import acquisition


def short_side(sample):
    with Image.open(sample.path) as image:
        return min(image.size)


def shared_scale(assignment, smaller_short_side):
    if assignment[0] % 4 < 2:
        return 1.0
    fraction = assignment[1] / 255
    # Restrict downsampling to pairs for which both images can still supply native 128px patches.
    lower = max(.62, CROP / smaller_short_side)
    if lower < .96 and assignment[5] % 2 == 0:
        return lower + (.96 - lower) * fraction
    return 1.08 + .30 * fraction


class BalancedAcquisition:
    def __init__(self, samples):
        with ThreadPoolExecutor(max_workers=4) as workers:
            sizes = dict(zip((row.image_id for row in samples), workers.map(short_side, samples), strict=True))
        classes = [sorted((row for row in samples if row.label == label), key=lambda row: sizes[row.image_id])
                   for label in (0, 1)]
        self.plans = {}
        self.counts = Counter()
        for real, generated in zip(*classes, strict=True):
            key = sha256(f"balanced-acquisition-v1:{real.image_id}:{generated.image_id}".encode()).digest()
            min_side = min(sizes[real.image_id], sizes[generated.image_id])
            plan = tuple((assignment, shared_scale(assignment, min_side)) for assignment in (key[5:], key))
            self.plans[real.image_id] = self.plans[generated.image_id] = plan
            for assignment, scale in plan:
                direction = "down" if scale < 1 else "up" if scale > 1 else "native"
                codec = "jpeg" if assignment[0] % 2 else "no_jpeg"
                self.counts[f"{direction}_{codec}"] += 1
        if len(self.plans) != len(samples):
            raise ValueError("Expected a unique treatment plan for every training image")

    def __call__(self, image, sample):
        for assignment, scale in self.plans[sample.image_id]:
            yield acquisition(image, assignment, scale=scale), assignment[0] % 4 == 3
