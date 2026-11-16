"""Controlled, CPU-only acceptance coverage for SCRUM-65 and SCRUM-66."""

import copy
import json
import shutil
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

from app.checkpoint import sha256_file
from app.dataset.frozen_test_set import freeze_test_set, save_test_lock
from app.dataset.importer import import_coco_dataset
from app.dataset.manifest import build_manifest
from app.dataset.partitions import build_partition_manifest
from app.dataset.records import GroundTruthObject, SourcePolicy
from app.dataset.rfdetr_export import SplitAssignment, export_rfdetr_dataset, load_file_inputs
from app.error_summary import summarize_validation_errors
from app.evaluate import (
    MATCHING_RULE, EvaluationError, _labels_for, _match,
    evaluate_validation, extract_validation_errors,
)
from app.inference import Detection, InferenceResult
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class ValidationDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        dataset = load_dataset_fixture(self.root / "dataset")
        labels = (
            GroundTruthObject(101, 1, 1, (0, 0, 10, 10)),
            GroundTruthObject(102, 1, 1, (20, 0, 10, 10)),
            GroundTruthObject(103, 1, 1, (40, 0, 10, 10)),
            GroundTruthObject(104, 1, 2, (60, 0, 10, 10)),
        )
        parsed = dataset.annotations[0]
        objects = labels + tuple(obj for obj in parsed.objects if obj.coco_image_id != 1)
        self.dataset = replace(
            dataset,
            annotations=(replace(parsed, objects=objects),),
            accepted=tuple(replace(image, lighting=" bright ", almond_type="Nonpareil")
                           if image.image_id == "fix-001" else image for image in dataset.accepted),
        )
        annotation_path = self.dataset.resolve(parsed.path)
        document = json.loads(annotation_path.read_text())
        document["annotations"] = [
            {"id": obj.annotation_id, "image_id": obj.coco_image_id,
             "category_id": obj.category_id, "bbox": list(obj.bbox_xywh)}
            for obj in objects
        ]
        annotation_path.write_text(json.dumps(document))
        self.predictions = (
            Detection(1, 0.99, (0, 0, 10, 10)),       # TP
            Detection(1, 0.95, (0, 0, 10, 10)),       # duplicate FP
            Detection(2, 0.90, (20, 0, 30, 10)),     # confusion
            Detection(1, 0.85, (44, 0, 54, 10)),     # localization: IoU 3/7
            Detection(1, 0.80, (0, 30, 10, 40)),     # background FP
        )

    def report(self, predictions=None):
        predictions = self.predictions if predictions is None else predictions
        predictions = tuple(sorted(predictions, key=lambda item: -item.confidence))
        manifest = build_manifest(self.dataset, "diagnostic-v1")
        image = self.dataset.accepted[0]
        labels = _labels_for(image, self.dataset, {1: "A", 2: "B"})
        matched = _match(predictions, labels, 0.5)
        report = {
            "schema_version": 1,
            "evaluation_id": "controlled-evaluation",
            "checkpoint": {"sha256": "controlled-checkpoint"},
            "dataset": {"version": manifest.dataset_version, "content_hash": manifest.content_hash,
                        "validation_image_ids": [image.image_id]},
            "classes": [{"class_id": 1, "class_name": "A"}, {"class_id": 2, "class_name": "B"}],
            "settings": {"split": "valid", "matching_rule": MATCHING_RULE,
                         "confidence_floor": 0.5, "iou_threshold": 0.5},
            "summary": {
                "true_positives": len(matched["matches"]),
                "false_positives": len(matched["false_positive_prediction_indices"]),
                "false_negatives": len(matched["false_negative_label_indices"]),
            },
            "records": [{
                "image_id": image.image_id, "image_path": image.image_path.as_posix(),
                "image_sha256": sha256_file(self.dataset.resolve(image.image_path)),
                "labels": labels,
                "predictions": [{"category_id": item.category_id, "confidence": item.confidence,
                                 "bbox_xyxy": list(item.bbox_xyxy)} for item in predictions],
                **matched,
            }],
        }
        return report, manifest

    def export(self, predictions=None):
        report, manifest = self.report(predictions)
        exported = extract_validation_errors(report)
        exported["evaluation_sha256"] = "controlled-evaluation-digest"
        return exported, manifest

    def test_fixture_produces_all_categories_without_mutating_metrics(self):
        report, _ = self.report()
        original = copy.deepcopy(report)
        exported = extract_validation_errors(report)
        self.assertEqual(exported["summary"], {
            "error_count": 5, "matched_detection_count": 1,
            "by_error_type": {"false_positive": 2, "miss": 1,
                              "class_confusion": 1, "localization": 1},
        })
        self.assertEqual(report, original)
        self.assertEqual(report["summary"], {"true_positives": 1, "false_positives": 4,
                                            "false_negatives": 3})
        for error in exported["errors"]:
            self.assertEqual(error["image_id"], "fix-001")
            self.assertEqual(error["matching_rule"], MATCHING_RULE)
            self.assertEqual(error["iou_threshold"], 0.5)
            self.assertTrue(self.dataset.resolve(error["image_path"]).is_file())
            if error["error_type"] == "false_positive":
                self.assertIsNone(error["label"])
            elif error["error_type"] == "miss":
                self.assertIsNone(error["prediction"])
            if error["label"] is not None:
                self.assertIsNone(error["label"]["confidence"])

    def test_no_label_or_prediction_is_reused(self):
        report, _ = self.report()
        exported = extract_validation_errors(report)
        labels = [item["label_index"] for item in report["records"][0]["matches"]]
        predictions = [item["prediction_index"] for item in report["records"][0]["matches"]]
        labels += [error["label"]["label_index"] for error in exported["errors"]
                   if error["label"] is not None]
        predictions += [error["prediction"]["prediction_index"] for error in exported["errors"]
                        if error["prediction"] is not None]
        self.assertEqual(sorted(labels), list(range(4)))
        self.assertEqual(sorted(predictions), list(range(5)))
        self.assertEqual(sum(error["reason"] == "duplicate_detection" for error in exported["errors"]), 1)

    def test_exact_iou_boundary_and_deterministic_label_tie(self):
        exported, _ = self.export((Detection(1, 0.9, (0, 0, 5, 10)),))
        self.assertEqual(exported["summary"]["matched_detection_count"], 1)
        self.assertEqual(exported["summary"]["by_error_type"]["localization"], 0)
        labels = [{"category_id": 1, "bbox_xyxy": [0, 0, 10, 10]} for _ in range(2)]
        prediction = Detection(1, 0.9, (0, 0, 10, 10))
        first = _match((prediction, prediction), labels, 0.5)
        self.assertEqual([pair["label_index"] for pair in first["matches"]], [0, 1])

    def test_correct_match_takes_priority_over_higher_confidence_wrong_class(self):
        exported, _ = self.export((Detection(2, 0.99, (0, 0, 10, 10)),
                                   Detection(1, 0.9, (0, 0, 10, 10))))
        self.assertEqual(exported["summary"]["matched_detection_count"], 1)
        self.assertEqual(exported["summary"]["by_error_type"]["class_confusion"], 0)

    def test_saved_matching_and_validation_membership_are_checked(self):
        for mutate in (
            lambda report: report["settings"].update(split="test"),
            lambda report: report["settings"].update(matching_rule="unsupported"),
            lambda report: report["records"][0].update(matches=[]),
            lambda report: report["dataset"].update(validation_image_ids=["another-image"]),
        ):
            report, _ = self.report()
            mutate(report)
            with self.assertRaises(EvaluationError):
                extract_validation_errors(report)

    def test_all_dimensions_reconcile_and_examples_resolve(self):
        exported, manifest = self.export()
        summary = summarize_validation_errors(exported, manifest, self.dataset.root)
        errors = {error["error_id"]: error for error in exported["errors"]}
        for groups in summary["groups"].values():
            self.assertEqual(sum(group["error_count"] for group in groups), 5)
            for group in groups:
                self.assertEqual(sum(group["by_error_type"].values()), group["error_count"])
                for example in group["examples"]:
                    self.assertIn(example["error_id"], errors)
                    self.assertTrue(self.dataset.resolve(example["image_path"]).is_file())
        classes = {group["group"]: group for group in summary["groups"]["class"]}
        self.assertEqual(classes["1"]["error_count"], 4)
        self.assertEqual(classes["2"]["error_count"], 1)
        self.assertEqual(summary["grouping_rules"]["source"], "source_lot")

    def test_missing_metadata_is_unknown_without_losing_errors(self):
        self.dataset = replace(self.dataset, accepted=tuple(
            replace(image, source_lot=None, lighting=" ", almond_type=None)
            if image.image_id == "fix-001" else image for image in self.dataset.accepted
        ))
        exported, manifest = self.export()
        summary = summarize_validation_errors(exported, manifest, self.dataset.root)
        for dimension in ("source", "lighting", "almond_type"):
            unknown = next(group for group in summary["groups"][dimension] if group["group"] == "unknown")
            self.assertEqual(unknown["error_count"], 5)

    def test_empty_export_has_explicit_empty_groups(self):
        predictions = tuple(Detection(obj.category_id, 0.9, (x, y, x + w, y + h))
                            for obj in self.dataset.annotations[0].objects if obj.coco_image_id == 1
                            for x, y, w, h in [obj.bbox_xywh])
        exported, manifest = self.export(predictions)
        summary = summarize_validation_errors(exported, manifest, self.dataset.root)
        self.assertEqual(summary["error_count"], 0)
        for groups in summary["groups"].values():
            self.assertTrue(groups)
            self.assertTrue(all(group["error_count"] == 0 and group["examples"] == []
                                and group["examples_status"] == "empty" for group in groups))

    def test_missing_or_changed_images_keep_counts_without_broken_example_links(self):
        exported, manifest = self.export()
        self.dataset.resolve(exported["errors"][0]["image_path"]).write_bytes(b"replaced")
        for missing in (False, True):
            if missing:
                self.dataset.resolve(exported["errors"][0]["image_path"]).unlink()
            summary = summarize_validation_errors(exported, manifest, self.dataset.root)
            for groups in summary["groups"].values():
                self.assertEqual(sum(group["error_count"] for group in groups), 5)
                for group in groups:
                    if group["error_count"]:
                        self.assertEqual(group["examples"], [])
                        self.assertEqual(group["examples_status"], "unavailable")

    def test_duplicate_errors_invalid_totals_and_foreign_images_are_rejected(self):
        for mutate in (
            lambda exported: exported["errors"].append(exported["errors"][0]),
            lambda exported: exported["summary"].update(error_count=999),
            lambda exported: exported["errors"][0].update(image_id="fix-003"),
            lambda exported: exported["dataset"].update(content_hash="different"),
        ):
            exported, manifest = self.export()
            mutate(exported)
            with self.assertRaises(ValueError):
                summarize_validation_errors(exported, manifest, self.dataset.root)

    def file_inputs(self, dataset=None):
        dataset = self.dataset if dataset is None else dataset
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(build_manifest(dataset, "diagnostic-v1").to_dict()))
        splits_path = self.root / "splits.json"
        splits_path.write_text(json.dumps({"images": [
            {"image_id": image.image_id, "split": "valid" if index == 0 else "train",
             "group_id": f"group-{index}"}
            for index, image in enumerate(dataset.accepted)
        ]}))
        return manifest_path, splits_path

    def test_optional_metadata_round_trips_and_changes_dataset_hash(self):
        manifest, splits = self.file_inputs()
        dataset, loaded, _ = load_file_inputs(self.dataset.root, manifest, splits)
        self.assertEqual(dataset.accepted[0].lighting, "bright")
        self.assertEqual(loaded.images[0].almond_type, "Nonpareil")
        without = replace(self.dataset, accepted=tuple(replace(image, lighting=None, almond_type=None)
                                                      for image in self.dataset.accepted))
        old_manifest = build_manifest(without, "diagnostic-v1")
        self.assertNotEqual(loaded.content_hash, old_manifest.content_hash)
        self.assertNotIn("lighting", old_manifest.images[0].to_dict())
        manifest, splits = self.file_inputs(without)
        _, legacy_loaded, _ = load_file_inputs(without.root, manifest, splits)
        self.assertIsNone(legacy_loaded.images[0].lighting)
        self.assertEqual(legacy_loaded.to_dict(), old_manifest.to_dict())

    def test_importer_uses_existing_provenance_mapping(self):
        dataset = import_coco_dataset(
            self.dataset.root, annotation_file="annotations/instances.json",
            class_mapping={1: "A", 2: "B"}, dataset_namespace="diagnostics",
            image_directory="images", allow_negative_images=True,
            provenance_by_image={"images/two_a.png": {"lighting": " dim ", "almond_type": "Carmel"}},
        )
        image = next(image for image in dataset.accepted if image.image_path.name == "two_a.png")
        self.assertEqual((image.lighting, image.almond_type), ("dim", "Carmel"))

    def test_metadata_survives_partition_and_rfdetr_exports(self):
        dataset = replace(self.dataset, source_policy=SourcePolicy(False, True))
        partition = build_partition_manifest(dataset, "diagnostic-v1", {
            "fix-001": "valid", "fix-002": "valid", "fix-003": "train",
        })
        row = next(row for row in partition.to_dict()["images"] if row["image_id"] == "fix-001")
        self.assertEqual((row["lighting"], row["almond_type"]), ("bright", "Nonpareil"))
        export_root = self.root / "rfdetr-export"
        export_rfdetr_dataset(
            dataset, build_manifest(dataset, "diagnostic-v1"),
            [SplitAssignment(row.image.image_id, row.split, row.group_id) for row in partition.images],
            {1: "A", 2: "B"}, export_root,
        )
        coco = json.loads((export_root / "valid/_annotations.coco.json").read_text())
        provenance = json.loads((export_root / "dataset_provenance.json").read_text())
        for rows, key in ((coco["images"], "stable_image_id"), (provenance["images"], "image_id")):
            row = next(row for row in rows if row[key] == "fix-001")
            self.assertEqual((row["lighting"], row["almond_type"]), ("bright", "Nonpareil"))

    def evaluate(self, **changes):
        manifest, splits = self.file_inputs()
        lock_root = self.root / "frozen"
        (lock_root / "test").mkdir(parents=True)
        shutil.copyfile(self.dataset.resolve("images/negative.png"), lock_root / "test/locked.png")
        (lock_root / "split_manifest.json").write_text(json.dumps({"images": [{
            "split": "test", "group_id": "frozen-group", "output": "test/locked.png",
            "image_id": "locked-image",
        }]}))
        lock = save_test_lock(freeze_test_set(lock_root, "test-v1"), self.root / "locks")
        classes = self.root / "classes.json"
        classes.write_text(json.dumps({"1": "A", "2": "B"}))
        checkpoint = self.root / "checkpoint.pth"
        checkpoint.write_bytes(b"fixture checkpoint")
        detections = self.predictions

        class FakeDetector:
            calls = 0

            def predict(self, image, *, confidence_floor):
                self.calls += 1
                return InferenceResult(image.width, image.height, confidence_floor, detections)

        detector = FakeDetector()
        options = dict(
            variant="nano", confidence_floor=0.5, iou_threshold=0.5,
            detector_loader=lambda *args, **kwargs: detector,
            errors_output=self.root / "errors.json", error_summary_output=self.root / "summary.json",
        )
        options.update(changes)
        report = evaluate_validation(
            self.dataset.root, manifest, splits, lock, checkpoint, classes,
            self.root / "evaluation.json", **options,
        )
        return report, detector

    def test_evaluation_writes_linked_exports_using_only_validation_images(self):
        report, detector = self.evaluate()
        exported = json.loads((self.root / "errors.json").read_text())
        summary = json.loads((self.root / "summary.json").read_text())
        self.assertEqual(detector.calls, 1)
        self.assertEqual(report["dataset"]["validation_image_ids"], ["fix-001"])
        self.assertEqual(exported["evaluation_id"], report["evaluation_id"])
        self.assertEqual(exported["evaluation_sha256"], sha256_file(self.root / "evaluation.json"))
        self.assertEqual(summary["error_export_sha256"], sha256_file(self.root / "errors.json"))
        self.assertEqual(summary["error_count"], 5)

    def test_output_collision_is_rejected_before_inference(self):
        def forbidden(*args, **kwargs):
            self.fail("Detector must not be loaded for colliding outputs")

        with self.assertRaisesRegex(EvaluationError, "distinct"):
            self.evaluate(errors_output=self.root / "evaluation.json", detector_loader=forbidden)

    def test_combined_benchmark_and_markdown_report_still_work(self):
        report, detector = self.evaluate(
            benchmark_output=self.root / "benchmark.json", report_output=self.root / "report.md",
            warmup_runs=0,
        )
        benchmark = json.loads((self.root / "benchmark.json").read_text())
        self.assertEqual(benchmark["evaluation_sha256"], sha256_file(self.root / "evaluation.json"))
        self.assertEqual(benchmark["evaluation_id"], report["evaluation_id"])
        self.assertEqual(detector.calls, 2)
        self.assertIn("# Validation evaluation report", (self.root / "report.md").read_text())

    def test_evaluation_without_diagnostic_flags_retains_existing_record_shape(self):
        report, detector = self.evaluate(errors_output=None, error_summary_output=None)
        self.assertEqual(detector.calls, 1)
        self.assertNotIn("image_sha256", report["records"][0])
        self.assertFalse((self.root / "errors.json").exists())

    def test_summary_requires_error_export(self):
        with self.assertRaisesRegex(EvaluationError, "requires errors_output"):
            self.evaluate(errors_output=None)

    def test_changed_pixels_during_prediction_prevent_export(self):
        image_path = self.dataset.resolve("images/two_a.png")

        class ChangingDetector:
            def predict(inner, image, *, confidence_floor):
                image_path.write_bytes(image_path.read_bytes() + b"changed")
                return InferenceResult(image.width, image.height, confidence_floor, ())

        with self.assertRaisesRegex(EvaluationError, "changed during inference"):
            self.evaluate(detector_loader=lambda *args, **kwargs: ChangingDetector())
        self.assertFalse((self.root / "errors.json").exists())

    def test_frozen_validation_image_is_rejected_before_inference(self):
        shutil.copyfile(self.dataset.resolve("images/negative.png"),
                        self.dataset.resolve("images/two_a.png"))

        def forbidden(*args, **kwargs):
            self.fail("Frozen test content must be rejected before loading the detector")

        with self.assertRaisesRegex(EvaluationError, "Frozen test membership"):
            self.evaluate(detector_loader=forbidden)


if __name__ == "__main__":
    unittest.main()
