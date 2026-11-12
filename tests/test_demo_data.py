"""The demo generator is local, repeatable and never overwrites user data."""

import tempfile
import unittest
from pathlib import Path

from scripts.create_demo_data import create_demo_data


class DemoDataTests(unittest.TestCase):
    def test_seed_reproduces_complete_tree(self):
        with tempfile.TemporaryDirectory() as temp:
            first = create_demo_data(Path(temp) / "first", groups=3, images_per_group=2)
            second = create_demo_data(Path(temp) / "second", groups=3, images_per_group=2)
            files = sorted(first.rglob("*.png"))
            self.assertEqual(len(files), 6)
            self.assertEqual(len(list(first.iterdir())), 3)
            for path in files:
                self.assertEqual(path.read_bytes(), (second / path.relative_to(first)).read_bytes())

    def test_existing_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            marker = Path(temp) / "keep.txt"
            marker.write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                create_demo_data(temp)
            self.assertEqual(marker.read_text(), "keep")

    def test_invalid_sizes_do_not_create_output(self):
        with tempfile.TemporaryDirectory() as temp:
            for kwargs in ({"groups": 2}, {"images_per_group": 0}, {"seed": True}):
                output = Path(temp) / "invalid"
                with self.assertRaises(ValueError):
                    create_demo_data(output, **kwargs)
                self.assertFalse(output.exists())
