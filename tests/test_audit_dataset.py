"""Audit regressions use local synthetic images and explicit COCO labels."""

import io
import json
import shutil
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import numpy as np
from PIL import Image

from app.audit_dataset import audit_dataset as legacy_audit_dataset
from app.dataset.audit import audit_dataset, main


class DatasetAuditTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.documents = {}
        self.manifest = {"seed": 1, "groups": {}, "images": []}
        for number, split in enumerate(("train", "valid", "test"), start=1):
            folder = self.root / split
            folder.mkdir()
            pixels = np.random.default_rng(number).integers(0, 256, (12, 16, 3), dtype=np.uint8)
            Image.fromarray(pixels).save(folder / "frame.png")
            group = f"source-{number}"
            self.manifest["groups"][split] = [group]
            self.manifest["images"].append({"split": split, "group_id": group, "output": f"{split}/frame.png"})
            self.documents[split] = {
                "categories": [{"id": 1, "name": "fixture-class"}],
                "images": [{"id": number, "file_name": "frame.png", "width": 16, "height": 12}],
                "annotations": [{"id": number, "image_id": number, "category_id": 1, "bbox": [0, 0, 4, 3]}],
            }
        self.save()

    def save(self):
        (self.root / "split_manifest.json").write_text(json.dumps(self.manifest))
        for split, document in self.documents.items():
            (self.root / split / "_annotations.coco.json").write_text(json.dumps(document))

    def audit(self, **kwargs):
        return audit_dataset(self.root, distance=0, rare_min=1, **kwargs)

    def test_valid_labels_count_objects_and_leave_source_files_unchanged(self):
        before = {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}
        report = self.audit()
        self.assertIs(legacy_audit_dataset, audit_dataset)
        self.assertEqual(report["status"], "AUTOMATIC_CHECKS_PASSED")
        for split in ("train", "valid", "test"):
            self.assertEqual(report["labels"]["counts"][split]["classes"]["fixture-class"],
                             {"objects": 1, "images_with_class": 1})
        self.assertEqual(before, {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()})

    def test_missing_labels_remain_pending_and_reviewed_negatives_are_visible(self):
        (self.root / "test" / "_annotations.coco.json").unlink()
        report = self.audit()
        self.assertEqual(report["status"], "PENDING_LABELS")
        self.assertIsNone(report["labels"]["counts"]["test"]["classes"])
        self.documents["valid"]["annotations"] = []
        self.save()
        report = self.audit()
        self.assertEqual(report["status"], "NEEDS_REVIEW")
        self.assertEqual(report["labels"]["counts"]["valid"]["empty_annotation_images"], 1)

    def test_invalid_geometry_is_blocking_and_not_counted(self):
        invalid = ([-1, 0, 4, 3], [0, 0, 0, 3], [15, 0, 4, 3],
                   [0, 0, float("nan"), 3], [False, 0, 4, 3], [0, 0, "4", 3], [0, 0, 4])
        for box in invalid:
            with self.subTest(box=box):
                self.documents["train"]["annotations"][0]["bbox"] = box
                self.save()
                report = self.audit()
                self.assertEqual(report["status"], "BLOCKED")
                self.assertTrue(any("bbox" in error for error in report["errors"]))
                self.assertEqual(report["labels"]["counts"]["train"]["classes"]["fixture-class"]["objects"], 0)

    def test_declared_dimensions_must_match_image_pixels(self):
        self.documents["valid"]["images"][0]["width"] = 17
        self.save()
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("dimensions" in error for error in report["errors"]))

    def test_exact_duplicate_content_and_shared_groups_are_blocking(self):
        shutil.copyfile(self.root / "train" / "frame.png", self.root / "test" / "frame.png")
        self.manifest["images"][2]["group_id"] = "source-1"
        self.save()
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertEqual(len(report["exact_cross_split_duplicates"]), 1)
        self.assertIn("source-1", report["group_overlaps"])

    def test_duplicate_annotation_and_unknown_class_are_rejected(self):
        duplicate = dict(self.documents["train"]["annotations"][0])
        self.documents["train"]["annotations"].append(duplicate)
        self.documents["valid"]["annotations"][0]["category_id"] = 99
        self.save()
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("repeated annotation" in error for error in report["errors"]))
        self.assertTrue(any("unknown image/class" in error for error in report["errors"]))
        self.assertEqual(report["labels"]["counts"]["train"]["classes"]["fixture-class"]["objects"], 1)

    def test_unlisted_image_blocks_audit(self):
        shutil.copyfile(self.root / "train" / "frame.png", self.root / "valid" / "unlisted.png")
        report = self.audit()
        self.assertEqual(report["status"], "BLOCKED")
        self.assertTrue(any("absent from the manifest" in error for error in report["errors"]))

    def test_truncated_similarity_scan_cannot_pass(self):
        # Different file bytes, identical perceptual fingerprints.
        for value, split in enumerate(("train", "valid", "test"), start=1):
            Image.new("RGB", (16, 12), (value, value, value)).save(self.root / split / "frame.png")
        report = self.audit(max_pairs=1)
        self.assertEqual(report["status"], "BLOCKED")
        self.assertFalse(report["similarity_scan_complete"])
        self.assertTrue(any("Candidate limit" in error for error in report["errors"]))

    def test_unsafe_manifest_and_coco_paths_are_rejected(self):
        for path in ("../outside.png", "C:\\outside.png", "/outside.png"):
            with self.subTest(path=path):
                self.documents["train"]["images"][0]["file_name"] = path
                self.save()
                self.assertEqual(self.audit()["status"], "BLOCKED")
        self.manifest["images"][0]["output"] = "valid/frame.png"
        self.save()
        with self.assertRaisesRegex(ValueError, "outside its recorded partition"):
            self.audit()

    def test_malformed_documents_and_invalid_settings_raise_clear_errors(self):
        (self.root / "split_manifest.json").write_text('{"images": [null]}')
        with self.assertRaisesRegex(ValueError, "Each image"):
            self.audit()
        self.save()
        (self.root / "train" / "_annotations.coco.json").write_text('{"images": {}}')
        with self.assertRaisesRegex(ValueError, "COCO needs"):
            self.audit()
        with self.assertRaisesRegex(ValueError, "positive integers"):
            audit_dataset(self.root, distance=65)

    def test_command_saves_report_and_returns_failure_for_bad_input(self):
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main([str(self.root), "--distance", "0", "--rare-min-images", "1"]), 0)
        self.assertEqual(json.loads((self.root / "audit_report.json").read_text())["status"],
                         "AUTOMATIC_CHECKS_PASSED")
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            main([str(self.root), "--max-pairs", "0"])
        self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
