"""Local result files preserve predictions and reject stale or invalid data."""

import csv
import dataclasses
import json
import os
import shutil
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from app.inference import Detection, InferenceResult
from app.results import (
    ResultFileError, create_analysis, export_csv, load_analysis, save_analysis, source_image_sha256,
)


class LocalResultTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.image = self.root / "images" / "frame.png"
        self.image.parent.mkdir()
        Image.new("RGB", (20, 10), "gray").save(self.image)
        self.result = InferenceResult(20, 10, .25, (Detection(17, .9, (1.5, 2, 18, 9)),))
        self.classes = {17: "Almond, whole", 21: "Foreign material"}
        self.record = create_analysis(self.image, self.result, self.classes,
                                      model_label="candidate-1", created_at=datetime(2026, 1, 2, tzinfo=timezone.utc))
        self.path = self.root / "results" / "analysis.json"

    def save(self):
        save_analysis(self.path, self.record)
        return json.loads(self.path.read_text())

    def mutate(self, change):
        data = self.save()
        change(data)
        self.path.write_text(json.dumps(data))

    def test_json_roundtrip_preserves_metadata_predictions_and_relative_source(self):
        data = self.save()
        self.assertEqual(data["schema_version"], 1)
        self.assertEqual(data["image"]["path"], "../images/frame.png")
        self.assertEqual(data["prediction"]["detections"][0]["bbox_xyxy"], [1.5, 2., 18., 9.])
        loaded = load_analysis(self.path)
        self.assertEqual(loaded, self.record)
        self.assertEqual(loaded.result.confidence_floor, .25)
        self.assertEqual(loaded.model_label, "candidate-1")

    def test_relative_result_and_source_can_move_together(self):
        self.save()
        moved = self.root / "moved"
        moved.mkdir()
        shutil.move(self.image.parent, moved / "images")
        shutil.move(self.path.parent, moved / "results")
        loaded = load_analysis(moved / "results/analysis.json")
        self.assertEqual(loaded.image_path, moved / "images/frame.png")
        self.assertEqual(loaded.result, self.record.result)

    def test_saved_results_open_through_directory_and_file_symlinks(self):
        self.path.parent.mkdir()
        alias_folder = self.root / "nested" / "more"
        alias_folder.mkdir(parents=True)
        directory_alias = alias_folder / "results"
        try:
            directory_alias.symlink_to(self.path.parent, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("Symlink creation unavailable")
        save_analysis(directory_alias / "analysis.json", self.record)
        self.assertEqual(load_analysis(self.path), self.record)
        file_alias = alias_folder / "saved.json"
        file_alias.symlink_to(self.path)
        self.assertEqual(load_analysis(file_alias), self.record)

    def test_csv_matches_detection_values_and_quotes_class_names(self):
        target = export_csv(self.root / "analysis.csv", self.record)
        with target.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["class_name"], "Almond, whole")
        self.assertEqual(row["category_id"], "17")
        self.assertEqual(float(row["confidence"]), .9)
        self.assertEqual([float(row[key]) for key in ("x_min", "y_min", "x_max", "y_max")], [1.5, 2, 18, 9])
        self.assertEqual(row["image_sha256"], self.record.image_sha256)
        self.assertEqual(row["model_label"], "candidate-1")

    def test_csv_formula_leading_text_is_literal_while_json_keeps_original_text(self):
        for prefix in ("=", "+", "-", "@", " \t=", "\r\n+"):
            name = prefix + "1+1"
            record = create_analysis(self.image, self.result, {17: name}, model_label=name)
            csv_path = export_csv(self.root / "formula.csv", record, overwrite=True)
            with csv_path.open(newline="") as handle:
                row, = list(csv.DictReader(handle))
            self.assertEqual(row["class_name"], "'" + name)
            self.assertEqual(row["model_label"], "'" + name)
            save_analysis(self.path, record, overwrite=True)
            loaded = load_analysis(self.path)
            self.assertEqual(loaded.class_names[17], name)
            self.assertEqual(loaded.model_label, name)

    def test_successful_empty_analysis_roundtrips_and_csv_has_headers_only(self):
        record = create_analysis(self.image, InferenceResult(20, 10, .25), self.classes)
        save_analysis(self.path, record)
        self.assertEqual(load_analysis(self.path).result.detections, ())
        self.assertIsNone(load_analysis(self.path).model_label)
        path = export_csv(self.root / "empty.csv", record)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            self.assertIn("category_id", reader.fieldnames)
            self.assertEqual(list(reader), [])

    def test_existing_output_is_preserved_unless_replacement_is_explicit(self):
        self.path.parent.mkdir()
        self.path.write_text("existing")
        with self.assertRaisesRegex(ResultFileError, "already exists"):
            save_analysis(self.path, self.record)
        self.assertEqual(self.path.read_text(), "existing")
        save_analysis(self.path, self.record, overwrite=True)
        self.assertEqual(load_analysis(self.path), self.record)

    def test_failed_atomic_replacement_preserves_previous_file_and_cleans_staging(self):
        self.path.parent.mkdir()
        self.path.write_text("existing")
        with patch("app.results.os.replace", side_effect=OSError("disk failure")):
            with self.assertRaisesRegex(ResultFileError, "disk failure"):
                save_analysis(self.path, self.record, overwrite=True)
        self.assertEqual(self.path.read_text(), "existing")
        self.assertEqual(list(self.path.parent.glob(".dinos-result-*.tmp")), [])

    def test_source_cannot_be_an_output_even_with_replacement_or_a_symlink(self):
        before = self.image.read_bytes()
        targets = [self.image]
        alias = self.root / "alias.png"
        try:
            alias.symlink_to(self.image)
            targets.append(alias)
        except (OSError, NotImplementedError):
            pass
        for function in (save_analysis, export_csv):
            for target in targets:
                with self.subTest(function=function.__name__, target=target):
                    with self.assertRaisesRegex(ResultFileError, "different from the source"):
                        function(target, self.record, overwrite=True)
        self.assertEqual(self.image.read_bytes(), before)

    def test_changed_source_blocks_loading_saving_and_exporting(self):
        self.save()
        Image.new("RGB", (20, 10), "blue").save(self.image)
        for action in (lambda: load_analysis(self.path),
                       lambda: save_analysis(self.root / "new.json", self.record),
                       lambda: export_csv(self.root / "new.csv", self.record)):
            with self.assertRaisesRegex(ResultFileError, "SHA-256"):
                action()
        self.assertFalse((self.root / "new.json").exists())
        self.assertFalse((self.root / "new.csv").exists())

    def test_pre_prediction_digest_rejects_same_size_image_changed_during_inference(self):
        before = source_image_sha256(self.image)
        self.assertEqual(before, self.record.image_sha256)
        Image.new("RGB", (20, 10), "blue").save(self.image)
        with self.assertRaisesRegex(ResultFileError, "changed during analysis"):
            create_analysis(self.image, self.result, self.classes, expected_image_sha256=before)
        with self.assertRaisesRegex(ResultFileError, "Expected image SHA-256"):
            create_analysis(self.image, self.result, self.classes, expected_image_sha256="invalid")

    def test_missing_source_is_an_explicit_load_failure(self):
        self.save()
        self.image.unlink()
        with self.assertRaisesRegex(ResultFileError, "source image"):
            load_analysis(self.path)

    def test_unsupported_and_nonregular_sources_are_rejected_before_hashing(self):
        candidates = [self.root, self.root / "missing.png", Path("/dev/zero")]
        fifo = self.root / "pipe.png"
        if hasattr(os, "mkfifo"):
            os.mkfifo(fifo)
            candidates.append(fifo)
        with patch("app.results._sha256") as hasher:
            for path in candidates:
                with self.subTest(path=path), self.assertRaises(ResultFileError):
                    source_image_sha256(path)
                with self.subTest(path=path), self.assertRaises(ResultFileError):
                    create_analysis(path, self.result, self.classes)
            hasher.assert_not_called()

    def test_saved_json_nonregular_source_is_rejected_before_hashing(self):
        if not hasattr(os, "mkfifo"):
            self.skipTest("FIFO creation unavailable")
        payload = self.save()
        fifo = self.root / "pipe.png"
        os.mkfifo(fifo)
        payload["image"]["path"] = str(fifo)
        self.path.write_text(json.dumps(payload))
        with patch("app.results._sha256") as hasher:
            with self.assertRaisesRegex(ResultFileError, "regular file"):
                load_analysis(self.path)
            hasher.assert_not_called()

    def test_snapshot_is_immutable_and_does_not_follow_class_map_mutations(self):
        self.classes[17] = "changed"
        self.assertEqual(self.record.class_names[17], "Almond, whole")
        with self.assertRaises(TypeError):
            self.record.class_names[17] = "changed"
        with self.assertRaises(dataclasses.FrozenInstanceError):
            self.record.image_sha256 = "0" * 64

    def test_create_analysis_rejects_wrong_dimensions_or_naive_timestamp(self):
        with self.assertRaisesRegex(ResultFileError, "dimensions"):
            create_analysis(self.image, InferenceResult(10, 20, .25), self.classes)
        with self.assertRaisesRegex(ResultFileError, "timezone"):
            create_analysis(self.image, self.result, self.classes, created_at=datetime(2026, 1, 1))

    def test_malformed_json_and_duplicate_fields_are_rejected(self):
        self.path.parent.mkdir()
        for text in ("{", '{"schema_version":1,"schema_version":1}'):
            self.path.write_text(text)
            with self.assertRaises(ResultFileError):
                load_analysis(self.path)

    def test_load_rejects_unsupported_schema_unknown_classes_and_invalid_numbers(self):
        base = self.save()
        cases = [
            lambda p: p.update(schema_version=True),
            lambda p: p.update(schema_version=2),
            lambda p: p.update(created_at="2026-01-02"),
            lambda p: p.update(model_label=" "),
            lambda p: p["image"].update(sha256="invalid"),
            lambda p: p["image"].update(width=True),
            lambda p: p["classes"][0].update(category_id="17"),
            lambda p: p["classes"][0].update(name=""),
            lambda p: p["classes"].append(dict(p["classes"][0])),
            lambda p: p["prediction"]["detections"][0].update(category_id=99),
            lambda p: p["prediction"]["detections"][0].update(confidence=float("nan")),
            lambda p: p["prediction"]["detections"][0].update(bbox_xyxy=[0, 0, 21, 10]),
        ]
        for change in cases:
            payload = json.loads(json.dumps(base))
            change(payload)
            self.path.write_text(json.dumps(payload))
            with self.subTest(payload=payload), self.assertRaises(ResultFileError):
                load_analysis(self.path)


if __name__ == "__main__":
    unittest.main()
