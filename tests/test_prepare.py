"""A page with a slow computation shows a notice first in a window, but never in an export or in these tests."""
import unittest

import matplotlib.pyplot as plt

from tutorkit import Course, Tutor


class Preparing(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.course = Course("Preparing", [], {}, "", name="prep_course")
        self.course.part("the only part")
        self.course.step("quick")(lambda t: self.calls.append("page 1"))
        self.course.step("slow", prepare=lambda: self.calls.append("prepare"))(lambda t: self.calls.append("page 2"))

    def tutor(self, **options):
        tutor = Tutor(self.course, **options)
        self.addCleanup(plt.close, tutor.fig)
        return tutor

    @staticmethod
    def notice_shown(tutor):
        return any("Preparing" in text.get_text() and text.get_visible() for text in tutor.fig.texts)

    def test_without_window_prepare_runs_first(self):
        tutor = self.tutor()
        tutor.show(1)
        self.assertEqual(self.calls, ["prepare", "page 2"])
        self.assertFalse(self.notice_shown(tutor))
        self.assertIn(1, tutor.prepared)
        tutor.show(1)
        self.assertEqual(self.calls, ["prepare", "page 2", "page 2"])

    def test_window_shows_the_notice_first(self):
        tutor = self.tutor(interactive=True)
        tutor.show(1)
        self.assertEqual(self.calls, [])
        self.assertTrue(self.notice_shown(tutor))
        tutor.build(1)                   # what the timer does a moment later
        self.assertEqual(self.calls, ["prepare", "page 2"])
        self.assertFalse(self.notice_shown(tutor))
        tutor.show(1)                    # coming back: no notice, no second preparation
        self.assertEqual(self.calls, ["prepare", "page 2", "page 2"])
        self.assertFalse(self.notice_shown(tutor))

    def test_leaving_before_the_page_is_ready(self):
        tutor = self.tutor(interactive=True)
        tutor.show(1)
        tutor.show(0)
        self.assertEqual(self.calls, ["page 1"])
        self.assertNotIn(1, tutor.prepared)
        self.assertFalse(self.notice_shown(tutor))

    def test_quick_page_has_no_notice(self):
        tutor = self.tutor(interactive=True)
        tutor.show(0)
        self.assertEqual(self.calls, ["page 1"])
        self.assertFalse(self.notice_shown(tutor))


if __name__ == "__main__":
    unittest.main()
