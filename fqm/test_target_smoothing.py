"""Check that mild blur is confined to its assigned post-processing stage."""
import unittest

import numpy as np
from PIL import Image, ImageFilter

from ssd.target_profile import process


class TargetSmoothingTests(unittest.TestCase):
    def test_other_width_blurs_after_resize_and_before_encoding(self):
        noise = np.random.default_rng(42).integers(0, 256, size=(80, 80, 3), dtype=np.uint8)
        image = Image.fromarray(noise)
        profile = [56, 56, "PNG", None, [], 1]
        plain = process(image, profile, "both", (.5, .5))
        blurred = process(image, profile, "both", (.5, .5), other_blur_radius=.35)
        self.assertFalse(np.array_equal(plain, blurred))
        np.testing.assert_array_equal(blurred, plain.filter(ImageFilter.GaussianBlur(.35)))

    def test_width500_and_ablations_are_bit_identical(self):
        noise = np.random.default_rng(42).integers(0, 256, size=(512, 512, 3), dtype=np.uint8)
        image = Image.fromarray(noise)
        for width, mode in ((500, "both"), (400, "geometry"), (400, "encoding")):
            profile = [width, 400, "PNG", None, [], 1]
            plain = process(image, profile, mode, (.5, .5))
            candidate = process(image, profile, mode, (.5, .5), other_blur_radius=.35)
            np.testing.assert_array_equal(plain, candidate)


if __name__ == "__main__":
    unittest.main()
