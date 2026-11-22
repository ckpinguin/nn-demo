"""Every page of both courses must open without an error, and its text must fit the panel on the right."""
import importlib
import unittest

import matplotlib.pyplot as plt

from tutorkit import Tutor


class EveryPageOpens(unittest.TestCase):
    def check_course(self, module_name):
        course = importlib.import_module(module_name).course
        tutor = Tutor(course)
        self.addCleanup(plt.close, tutor.fig)
        for i, (_, title, *_) in enumerate(course.steps):
            with self.subTest(course=module_name, step=i + 1, title=title):
                tutor.show(i)
                tutor.fig.canvas.draw()
                self.assertEqual(tutor.layout_problems, [], "the text of this page does not fit")

    def test_nn_tutor(self):
        self.check_course("nn_tutor")

    def test_backprop_tutor(self):
        self.check_course("backprop_tutor")


if __name__ == "__main__":
    unittest.main()
