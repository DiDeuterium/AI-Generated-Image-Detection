"""Smoke checks for the single-checkpoint, two-scale image classifier."""
import unittest

import torch
from PIL import Image

from multiscale_texture_base import MultiscaleTexture, native_scales, paired_loss


class MultiscaleTests(unittest.TestCase):
    def test_native_patches_keep_both_requested_sizes(self):
        image = Image.new("RGB", (128, 128), color=(50, 100, 135))
        qtable = [[5] * 64, [8] * 64]
        small, large = native_scales(image, qtable)
        self.assertEqual(small.shape, (5, 6, 96, 96))
        self.assertEqual(large.shape, (5, 6, 160, 160))

    def test_both_scales_influence_one_prediction_and_backward(self):
        torch.manual_seed(42)
        model = MultiscaleTexture()
        shape = (2, 2, 6)
        small = torch.randint(0, 256, (*shape, 48, 48), dtype=torch.uint8)
        large = torch.randint(0, 256, (*shape, 64, 64), dtype=torch.uint8)
        logits = model(small, large)
        self.assertEqual(logits.shape, (2, 2))
        self.assertFalse(torch.equal(logits, model(small, large.roll(1, dims=0))))
        loss = paired_loss(model, (small, large), (small.flip(-1), large.flip(-1)), torch.tensor([0, 1]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsNotNone(model.scale_fusion.weight.grad)
        self.assertIsNotNone(model.high_pass.grad)


if __name__ == "__main__":
    unittest.main()
