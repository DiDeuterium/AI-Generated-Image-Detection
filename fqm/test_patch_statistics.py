"""Properties of local image descriptors used in the CPU baseline."""
import unittest

import numpy as np
from PIL import Image

from patch_statistics import local_features


class PatchStatisticsTests(unittest.TestCase):
    def test_features_have_fixed_shape_and_are_finite_for_small_flat_images(self):
        small = Image.new("RGB", (37, 60), (70, 70, 70))
        large = Image.new("RGB", (200, 120), (70, 70, 70))
        first, second = local_features(small), local_features(large)
        self.assertTrue(np.isfinite(first).all())
        np.testing.assert_array_equal(first, second)

    def test_local_patterns_change_features_at_constant_size_and_brightness(self):
        rng = np.random.default_rng(42)
        values = rng.integers(50, 180, (128, 128), dtype=np.uint8)
        scrambled = rng.permutation(values.ravel()).reshape(values.shape)
        smooth = np.sort(values, axis=1)
        self.assertGreater(np.linalg.norm(local_features(Image.fromarray(smooth))
                                          - local_features(Image.fromarray(scrambled))), .1)


if __name__ == "__main__":
    unittest.main()
