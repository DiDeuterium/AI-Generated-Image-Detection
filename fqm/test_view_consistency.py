"""Focused checks for the matched paired-view experiment."""
import unittest

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from ssd.view_consistency import PairedDegradation, paired_view_loss


class PairedViewTests(unittest.TestCase):
    def test_both_views_contain_the_same_image_region(self):
        image = np.zeros((72, 80, 3), dtype=np.uint8)
        image[..., 0] = np.arange(80) * 3
        image[..., 1] = (np.arange(72) * 3)[:, None]
        transform = PairedDegradation(32, None, None, hflip=.5)
        weak, degraded = transform(Image.fromarray(image))
        self.assertEqual(weak.shape, (3, 32, 32))
        self.assertLess((weak - degraded).abs().mean().item(), .04)

    def test_control_supervises_both_views(self):
        scores = torch.tensor([[[4., -1.], [-1., 4.]], [[-1., 4.], [-2., 2.]]])
        labels = torch.tensor([0, 1])
        expected = F.cross_entropy(scores.flatten(0, 1), labels.repeat_interleave(2))
        actual = paired_view_loss(scores, labels, F.cross_entropy, 0.)
        torch.testing.assert_close(actual, expected)

    def test_only_different_predictions_incur_a_penalty(self):
        scores = torch.tensor([[[4., -1.], [-1., 4.]]], requires_grad=True)
        labels = torch.tensor([0])
        control = paired_view_loss(scores, labels, F.cross_entropy, 0.)
        consistent = paired_view_loss(scores, labels, F.cross_entropy, .2)
        self.assertGreater(consistent.item(), control.item())
        (consistent - control).backward()
        self.assertTrue((scores.grad.abs().sum(dim=-1) > 0).all().item())
        same = scores.detach()[:, :1].expand_as(scores)
        torch.testing.assert_close(paired_view_loss(same, labels, F.cross_entropy, .2),
                                   paired_view_loss(same, labels, F.cross_entropy, 0.))


if __name__ == "__main__":
    unittest.main()
