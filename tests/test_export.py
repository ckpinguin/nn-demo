"""Exported pages carry the course name, so two courses can share a folder, and old leftover pages are removed."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

from tutorkit import Course


def tiny_course(name, steps):
    course = Course(f"Course {name}", [], {}, "", name=name)
    course.part("the only part")
    for k in range(steps):
        course.step(f"page {k + 1}")(lambda t, k=k: t.write(f"Page {k + 1}."))
    return course


class Export(unittest.TestCase):
    def setUp(self):
        self.folder = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def export(self, course):
        """Export into the test folder, returning the saved files and what the export printed."""
        printed = io.StringIO()
        with contextlib.redirect_stdout(printed):
            saved = course.export(self.folder)
        return saved, printed.getvalue()

    def pages(self, name, count):
        return [self.folder / f"{name}_step_{k:02d}.png" for k in range(1, count + 1)]

    def test_two_courses_share_a_folder(self):
        (self.folder / "notes.txt").write_text("keep me")
        demo, _ = self.export(tiny_course("demo_course", 3))
        other, _ = self.export(tiny_course("other_course", 2))
        self.assertEqual(demo, self.pages("demo_course", 3))
        self.assertEqual(other, self.pages("other_course", 2))
        expected = self.pages("demo_course", 3) + self.pages("other_course", 2) + [self.folder / "notes.txt"]
        self.assertEqual(sorted(self.folder.iterdir()), sorted(expected))
        self.assertEqual((self.folder / "notes.txt").read_text(), "keep me")

    def test_leftover_pages_are_removed(self):
        self.export(tiny_course("other_course", 2))
        other_before = {path.name: path.read_bytes() for path in self.pages("other_course", 2)}
        (self.folder / "demo_course_step_07.png").write_bytes(b"from a longer version")
        _, printed = self.export(tiny_course("demo_course", 3))
        self.assertFalse((self.folder / "demo_course_step_07.png").exists())
        self.assertIn("removed leftover demo_course_step_07.png", printed)
        self.assertTrue(all(path.exists() for path in self.pages("demo_course", 3)))
        self.assertEqual({path.name: path.read_bytes() for path in self.pages("other_course", 2)}, other_before)

    def test_name_is_a_plain_word(self):
        with self.assertRaises(ValueError):
            Course("A course", [], {}, "", name="Bad-Name")


if __name__ == "__main__":
    unittest.main()
