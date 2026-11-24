"""Regression checks for the single-model image classifier."""
import unittest

from PIL import Image
import torch

from native_texture_base import NativeTextureBase, acquisition, sample_view_pair


class NativeTextureTests(unittest.TestCase):
    def test_patch_order_does_not_change_the_image_prediction(self):
        torch.manual_seed(12)
        model = NativeTextureBase().eval()
        patches = torch.randint(0, 256, (2, 3, 6, 48, 48), dtype=torch.uint8)
        with torch.inference_mode():
            original = model(patches)
            reordered = model(patches[:, [2, 0, 1]])
        self.assertEqual(original.shape, (2, 2))
        self.assertTrue(torch.isfinite(original).all().item())
        torch.testing.assert_close(original, reordered)

    def test_four_views_are_sampled_without_replacement(self):
        first = torch.Generator().manual_seed(3)
        second = torch.Generator().manual_seed(3)
        views = (0, 1, 2, 3)
        pairs = [sample_view_pair(views, first) for _ in range(24)]
        self.assertEqual(pairs, [sample_view_pair(views, second) for _ in range(24)])
        self.assertEqual({view for pair in pairs for view in pair}, set(views))
        self.assertTrue(all(a != b for a, b in pairs))
        self.assertEqual(sample_view_pair(views[:2], first), views[:2])

    def test_native_and_codec_only_cases_preserve_dimensions(self):
        image = Image.new("RGB", (120, 140), color=(30, 80, 120))
        for case in (0, 1):
            self.assertEqual(acquisition(image, bytes([case, 20, 2, 3, 4])).size, image.size)
        self.assertNotEqual(acquisition(image, bytes([3, 20, 2, 3, 4])).size, image.size)


if __name__ == "__main__":
    unittest.main()
