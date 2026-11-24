"""Test fixed candidate selection and independent confirmation."""
import unittest


from lqy_resize_retrain import confirmation_passes, select_weight


class SelectionTests(unittest.TestCase):
    def test_weight_preserves_target_accuracy_and_improves_worst_case(self):
        previous = {"target": .70, "width500": .72, "other": .68, "strong": .67}
        eligible = {"target": .70, "width500": .72, "other": .68, "strong": .70}
        worse_target = {"target": .68, "width500": .71, "other": .70, "strong": .74}
        self.assertEqual(select_weight({0.: previous, .25: eligible, .5: worse_target}), .25)

    def test_confirmation_rejects_a_small_inconsistent_gain(self):
        base = {"target": .70, "width500": .72, "other": .68, "strong": .69}
        candidate = {"target": .701, "width500": .72, "other": .682, "strong": .70}
        self.assertFalse(confirmation_passes(base, candidate))
        candidate["other"] = .695
        self.assertTrue(confirmation_passes(base, candidate))


if __name__ == "__main__":
    unittest.main()
