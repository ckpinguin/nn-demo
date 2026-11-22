"""nn.py and the courses must use the same afternoons.

nn.py makes its own copy of the data on purpose, so that it can be read on its own. This test runs it and compares.
"""
import contextlib
import io
import runpy
import unittest
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import tinynet

DEMO = Path(__file__).resolve().parent.parent / "nn.py"


class SameAfternoons(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*non-interactive", category=UserWarning)  # plt.show() on Agg
            cls.demo = runpy.run_path(str(DEMO))
        plt.close("all")

    def test_same_afternoons(self):
        for name in ("X_train", "X_test", "y_train", "y_test"):
            with self.subTest(array=name):
                self.assertTrue(np.array_equal(self.demo[name], getattr(tinynet, name)),
                                f"{name} of nn.py differs from {name} of tinynet.py")

    def test_same_rescaling(self):
        self.assertTrue(np.allclose(self.demo["rescale"](self.demo["X_train"]), tinynet.rescale(tinynet.X_train)),
                        "rescale() of nn.py and of tinynet.py give different numbers")


if __name__ == "__main__":
    unittest.main()
