"""The labels beside the code listings of backprop_tutor.py must find the lines they describe.

A label names a piece of its line, so it moves along when lines are added above it. If its line is rewritten,
the label must fail loudly instead of sitting beside the wrong line.
"""
import unittest

from backprop_tutor import BACKPROP_MARKS, DEEP_MARKS, FORWARD_MARKS, code_of, deep_slopes, mark_lines
from tinynet import Net


class CodeLabels(unittest.TestCase):
    def test_every_label_finds_one_line(self):
        for name, function, marks in (("Net.forward", Net.forward, FORWARD_MARKS),
                                      ("Net.backprop", Net.backprop, BACKPROP_MARKS),
                                      ("deep_slopes", deep_slopes, DEEP_MARKS)):
            with self.subTest(listing=name):
                lines = mark_lines(code_of(function), marks)
                self.assertEqual(sorted(lines.values()), sorted(marks.values()), "two labels landed on one line")

    def test_text_in_no_line(self):
        with self.assertRaises(ValueError) as caught:
            mark_lines("a = 1\nb = 2", {"c = 3": "start"})
        self.assertIn("'start'", str(caught.exception))
        self.assertIn("'c = 3'", str(caught.exception))

    def test_text_in_two_lines(self):
        with self.assertRaises(ValueError) as caught:
            mark_lines("a = 1\na = 1 + 1", {"a = 1": "start"})
        self.assertIn("'start'", str(caught.exception))
        self.assertIn("2 lines", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
