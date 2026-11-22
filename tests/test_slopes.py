"""The slopes that backpropagation computes must agree with the slopes found by nudging every knob."""
import unittest

import numpy as np

import backprop_tutor
from office import X1, X2, Y
from tinynet import Net


class BackpropMatchesNudging(unittest.TestCase):
    def test_tinynet(self):
        for X, inputs, hidden in ((X1, 1, 2), (X2, 2, 4)):
            for seed in range(3):
                with self.subTest(inputs=inputs, hidden=hidden, seed=seed):
                    net = Net.random(inputs, hidden, seed)
                    np.testing.assert_allclose(
                        net.backprop(X, Y).knobs(), net.nudge_slopes(X, Y), rtol=1e-5, atol=1e-8,
                        err_msg=f"Net.backprop and nudging disagree: {inputs} input(s), {hidden} hidden, seed {seed}")

    def test_deep_networks(self):
        worst = backprop_tutor.deep_check()
        self.assertLess(worst, 1e-6, f"deep_slopes in backprop_tutor.py and nudging differ by up to {worst:.1e}")


if __name__ == "__main__":
    unittest.main()
