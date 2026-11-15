import json
import shutil
from pathlib import Path

import pytest

from app.dataset.frozen_test_set import freeze_test_set, save_test_lock
from app.dataset.manifest import build_manifest
from app.evaluate import EvaluationError, evaluate_validation
from app.inference import Detection, InferenceResult
from app.dataset.rfdetr_export import SplitAssignment
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


class FakeDetector:
    model_label = "fake-rfdetr"

    def __init__(self, detections=None):
        self.calls = []
        self.detections = detections if detections is not None else (
            Detection(1, 0.9, (10, 10, 30, 25)),
        )

    def predict(self, image, *, confidence_floor):
        self.calls.append((image.width, image.height, confidence_floor))
        return InferenceResult(image.width, image.height, confidence_floor, self.detections)


def inputs(tmp_path: Path, *, frozen_image_id="locked-image"):
    dataset = load_dataset_fixture(tmp_path / "dataset")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(build_manifest(dataset, "dataset-v1").to_dict()))
    split_path = tmp_path / "splits.json"
    assignments = (
        SplitAssignment("fix-001", "valid", "validation-group"),
        SplitAssignment("fix-002", "train", "training-group"),
        SplitAssignment("fix-003", "train", "other-training-group"),
    )
    split_path.write_text(json.dumps({"images": [
        {"image_id": item.image_id, "split": item.split, "group_id": item.source_group_id}
        for item in assignments
    ]}))

    lock_dataset = tmp_path / "lock-dataset"
    (lock_dataset / "test").mkdir(parents=True)
    image = lock_dataset / "test" / "locked.png"
    shutil.copyfile(dataset.resolve("images/negative.png"), image)
    (lock_dataset / "split_manifest.json").write_text(json.dumps({"images": [{
        "split": "test", "group_id": "frozen-group", "output": "test/locked.png",
        "image_id": frozen_image_id,
    }]}))
    lock = freeze_test_set(lock_dataset, "test-v1")
    lock_path = save_test_lock(lock, tmp_path / "locks")

    class_path = tmp_path / "classes.json"
    class_path.write_text(json.dumps({"1": "A", "2": "B"}))
    checkpoint = tmp_path / "checkpoint.pth"
    checkpoint.write_bytes(b"checkpoint identity")
    return dataset, manifest_path, split_path, lock_path, class_path, checkpoint


def run(tmp_path, *, detector=None, **changes):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    detector = detector or FakeDetector()
    options = dict(
        dataset_root=dataset.root,
        manifest_path=manifest,
        split_manifest_path=splits,
        frozen_test_lock_path=lock,
        checkpoint_path=checkpoint,
        class_map_path=classes,
        output_path=tmp_path / "evaluation.json",
        variant="nano",
        confidence_floor=0.5,
        iou_threshold=0.5,
        detector_loader=lambda *args, **kwargs: detector,
    )
    options.update(changes)
    return evaluate_validation(**options), detector


def test_evaluates_exact_validation_ids_and_records_predictions_labels_and_rules(tmp_path):
    report, detector = run(tmp_path)

    assert report["dataset"]["validation_image_ids"] == ["fix-001"]
    assert [row["image_id"] for row in report["records"]] == ["fix-001"]
    assert report["records"][0]["labels"]
    assert report["records"][0]["predictions"][0]["category_id"] == 1
    assert report["summary"]["true_positives"] == 1
    assert report["settings"] == {
        "split": "valid",
        "confidence_floor": 0.5,
        "matching_rule": "greedy_one_to_one_class_aware_descending_confidence",
        "iou_threshold": 0.5,
        "average_precision_rule": "101_point_interpolated_precision_recall",
        "average_precision_iou_thresholds": [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        "matching_tie_break_rule": "image_id_then_prediction_index; label_index_for_equal_iou",
    }
    assert report["checkpoint"]["sha256"]
    assert report["dataset"]["version"] == "dataset-v1"
    assert report["metrics"]["metric_names"]["map_50"] == "mAP@0.50"
    assert report["metrics"]["metric_names"]["map_50_95"] == "mAP@0.50:0.95"
    assert report["metrics"]["task"] == "bounding_box_detection"
    assert report["evaluation_id"]
    assert len(detector.calls) == 1
    assert json.loads((tmp_path / "evaluation.json").read_text()) == report


def test_benchmark_links_saved_evaluation_without_schema_changes(tmp_path):
    from app.checkpoint import sha256_file
    report, detector = run(tmp_path, benchmark_output=tmp_path / "benchmark.json", warmup_runs=2)
    saved = json.loads((tmp_path / "benchmark.json").read_text())
    assert saved["evaluation_id"] == report["evaluation_id"]
    assert saved["evaluation_sha256"] == sha256_file(tmp_path / "evaluation.json")
    for key in ("checkpoint", "dataset", "settings"):
        assert saved[key] == report[key]
    assert "benchmark" not in report
    assert saved["benchmark"]["sample_count"] == 1
    assert len(detector.calls) == 4


def test_evaluation_benchmark_and_report_command(tmp_path, monkeypatch):
    from app import evaluate
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    monkeypatch.setattr(evaluate, "load_detector", lambda *args, **kwargs: FakeDetector())
    assert evaluate.main([str(dataset.root), str(manifest), str(splits), str(lock),
                          str(checkpoint), str(classes), str(tmp_path / "evaluation.json"),
                          "--variant", "nano", "--confidence-floor", "0.5", "--iou-threshold", "0.5",
                          "--benchmark-output", str(tmp_path / "benchmark.json"),
                          "--report-output", str(tmp_path / "report.md")]) == 0
    assert "# Validation evaluation report" in (tmp_path / "report.md").read_text()


def test_report_requires_benchmark(tmp_path):
    with pytest.raises(EvaluationError, match="report_output requires"):
        run(tmp_path, report_output=tmp_path / "report.md")


def test_report_path_collision_precedes_inference(tmp_path):
    detector = FakeDetector()
    with pytest.raises(EvaluationError, match="distinct"):
        run(tmp_path, detector=detector, benchmark_output=tmp_path / "benchmark.json",
            report_output=tmp_path / "evaluation.json")
    assert detector.calls == []


def test_reporting_failure_retains_json_only(tmp_path, monkeypatch):
    from app.evaluation_report import ReportError
    def fail(*args):
        raise ReportError("fixture report failure")
    monkeypatch.setattr("app.evaluation_report.generate_report", fail)
    with pytest.raises(EvaluationError, match="JSON retained"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json", report_output=tmp_path / "report.md")
    assert (tmp_path / "evaluation.json").exists() and (tmp_path / "benchmark.json").exists()
    assert not (tmp_path / "report.md").exists()


def test_benchmark_failure_retains_only_valid_evaluation(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("fixture benchmark failure")
    monkeypatch.setattr("app.benchmark.benchmark_validation", fail)
    with pytest.raises(EvaluationError, match="valid evaluation retained"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json")
    assert json.loads((tmp_path / "evaluation.json").read_text())["metrics"]
    assert not (tmp_path / "benchmark.json").exists()


@pytest.mark.parametrize("name", ["evaluation.json", "checkpoint.pth", "classes.json"])
def test_benchmark_rejects_output_collisions_before_loading(tmp_path, name):
    detector = FakeDetector()
    with pytest.raises(EvaluationError, match="distinct"):
        run(tmp_path, detector=detector, benchmark_output=tmp_path / name)
    assert detector.calls == []


def test_benchmark_refuses_existing_output(tmp_path):
    target = tmp_path / "benchmark.json"
    target.write_text("existing result")
    with pytest.raises(EvaluationError, match="new output"):
        run(tmp_path, benchmark_output=target)
    assert target.read_text() == "existing result"


def test_benchmark_write_failure_does_not_publish_partial_file(tmp_path, monkeypatch):
    from app import evaluate
    original = evaluate.write_json_atomic
    def write(path, document, **kwargs):
        if kwargs.get("operation") == "write benchmark":
            raise OSError("fixture disk full")
        return original(path, document, **kwargs)
    monkeypatch.setattr(evaluate, "write_json_atomic", write)
    with pytest.raises(EvaluationError, match="disk full"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json")
    assert (tmp_path / "evaluation.json").exists()
    assert not (tmp_path / "benchmark.json").exists()


@pytest.mark.parametrize("filename", ["evaluation.json", "benchmark.json"])
def test_combined_workflow_preserves_concurrently_created_json(tmp_path, monkeypatch, filename):
    from app import run_artifacts
    original = run_artifacts.os.link
    concurrent = tmp_path / filename
    marker = b'{"owner": "another process"}'

    def race(source, destination):
        if Path(destination) == concurrent:
            concurrent.write_bytes(marker)
        original(source, destination)

    monkeypatch.setattr(run_artifacts.os, "link", race)
    with pytest.raises(EvaluationError):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json", report_output=tmp_path / "report.md")
    assert concurrent.read_bytes() == marker
    assert not (tmp_path / "report.md").exists()
    assert not list(tmp_path.glob(".*.tmp"))
    if filename == "evaluation.json":
        assert not (tmp_path / "benchmark.json").exists()
    else:
        assert json.loads((tmp_path / "evaluation.json").read_text())["metrics"]


def test_evaluation_only_still_overwrites_existing_result(tmp_path):
    target = tmp_path / "evaluation.json"
    target.write_text("old evaluation")
    report, _ = run(tmp_path)
    assert json.loads(target.read_text()) == report


def test_no_link_support_fails_without_publishing_json(tmp_path, monkeypatch):
    from app import run_artifacts
    original = run_artifacts.os.link
    def unsupported(source, destination):
        if Path(destination) == tmp_path / "evaluation.json":
            raise OSError("hard links unsupported")
        return original(source, destination)
    monkeypatch.setattr("app.run_artifacts.os.link", unsupported)
    with pytest.raises(EvaluationError, match="hard links unsupported"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json", report_output=tmp_path / "report.md")
    assert not (tmp_path / "evaluation.json").exists()
    assert not (tmp_path / "benchmark.json").exists()
    assert not (tmp_path / "report.md").exists()
    assert not list(tmp_path.glob(".*.tmp"))


@pytest.mark.parametrize("warmup", [-1, True, 1.5])
def test_invalid_benchmark_warmup(tmp_path, warmup):
    with pytest.raises(EvaluationError, match="nonnegative integer"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json", warmup_runs=warmup)


def test_warmup_requires_benchmark(tmp_path):
    with pytest.raises(EvaluationError, match="requires"):
        run(tmp_path, warmup_runs=5)


def test_benchmark_rechecks_frozen_content(tmp_path, monkeypatch):
    from app import evaluate
    original = evaluate._select_validation
    calls = []
    def select(*args):
        calls.append(True)
        if len(calls) == 2:
            raise EvaluationError("Frozen test membership conflict")
        return original(*args)
    monkeypatch.setattr(evaluate, "_select_validation", select)
    with pytest.raises(EvaluationError, match="Frozen test membership"):
        run(tmp_path, benchmark_output=tmp_path / "benchmark.json")
    assert len(calls) == 2
    assert not (tmp_path / "benchmark.json").exists()


def test_known_fixture_predictions_produce_ap_operating_counts_and_unavailable_empty_class(tmp_path):
    detector = FakeDetector((
        Detection(1, 0.9, (10, 10, 30, 25)),
        Detection(2, 0.85, (55, 40, 65, 50)),
        Detection(1, 0.8, (0, 0, 5, 5)),
        Detection(1, 0.7, (44, 30, 64, 45)),
    ))
    report, _ = run(tmp_path, detector=detector)

    metrics = report["metrics"]
    assert metrics["average_precision"]["map_50"] == pytest.approx(253 / 303)
    assert metrics["average_precision"]["map_50_95"] == pytest.approx(193 / 303)
    assert metrics["average_precision"]["included_class_ids"] == [1]
    class_a, class_b = metrics["per_class"]
    assert class_a["class_id"] == 1
    assert class_a["ground_truth_count"] == 2
    assert class_a["prediction_count"] == 3
    assert class_a["average_precision"]["ap_50"] == pytest.approx(253 / 303)
    assert class_a["average_precision"]["ap_50_95"] == pytest.approx(193 / 303)
    assert class_a["operating_precision"] == pytest.approx(2 / 3)
    assert class_a["operating_recall"] == 1.0
    assert class_b["class_id"] == 2
    assert class_b["ground_truth_count"] == 0
    assert class_b["prediction_count"] == 1
    assert class_b["average_precision"]["ap_50"] is None
    assert class_b["average_precision"]["ap_50_95"] is None
    assert class_b["operating_recall"] is None
    assert class_b["operating_precision"] == 0.0

    operating = metrics["operating_point"]
    assert operating["confidence_threshold"] == report["settings"]["confidence_floor"] == 0.5
    assert operating["iou_threshold"] == report["settings"]["iou_threshold"] == 0.5
    assert operating["matching_rule"] == report["settings"]["matching_rule"]
    assert operating["labeled_object_count"] == 2
    assert operating["prediction_count"] == 4
    assert operating["true_positive_count"] == 2
    assert operating["false_positive_count"] == 2
    assert operating["false_negative_count"] == 0
    assert operating["precision"] == 0.5
    assert operating["recall"] == 1.0


def test_frozen_validation_id_is_rejected_before_detector_or_image_loading(tmp_path, monkeypatch):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path, frozen_image_id="fix-001")
    from app import evaluate

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before detector or image loading")

    monkeypatch.setattr(evaluate, "_read_rgb", forbidden)
    with pytest.raises(EvaluationError, match=r"Frozen test membership conflict: validation image_id fix-001"):
        evaluate.evaluate_validation(
            dataset.root, manifest, splits, lock, checkpoint, classes, tmp_path / "bad.json",
            variant="nano", confidence_floor=0.5, iou_threshold=0.5,
            detector_loader=forbidden,
        )
    assert not (tmp_path / "bad.json").exists()


def test_frozen_validation_group_is_rejected_before_detector_loading(tmp_path):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    split_document = json.loads(splits.read_text())
    next(row for row in split_document["images"] if row["image_id"] == "fix-001")["group_id"] = "frozen-group"
    splits.write_text(json.dumps(split_document))

    def forbidden(*args, **kwargs):
        raise AssertionError("must reject before detector loading")

    with pytest.raises(EvaluationError, match=r"Frozen test membership conflict: validation group_id frozen-group"):
        evaluate_validation(
            dataset.root, manifest, splits, lock, checkpoint, classes, tmp_path / "bad-group.json",
            variant="nano", confidence_floor=0.5, iou_threshold=0.5,
            detector_loader=forbidden,
        )
    assert not (tmp_path / "bad-group.json").exists()


def test_source_group_cannot_cross_training_and_validation(tmp_path):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    split_document = json.loads(splits.read_text())
    next(row for row in split_document["images"] if row["image_id"] == "fix-002")["group_id"] = "validation-group"
    splits.write_text(json.dumps(split_document))

    with pytest.raises(EvaluationError, match=r"Source group validation-group appears in multiple splits"):
        evaluate_validation(
            dataset.root, manifest, splits, lock, checkpoint, classes, tmp_path / "bad-group.json",
            variant="nano", confidence_floor=0.5, iou_threshold=0.5,
            detector_loader=lambda *args, **kwargs: FakeDetector(),
        )


def test_missing_manifest_labels_or_checkpoint_have_specific_failures(tmp_path):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    arguments = [dataset.root, manifest, splits, lock, checkpoint, classes, tmp_path / "result.json"]
    options = dict(variant="nano", confidence_floor=0.5, iou_threshold=0.5,
                   detector_loader=lambda *args, **kwargs: FakeDetector())

    with pytest.raises(EvaluationError, match="Validation manifest or label failure"):
        evaluate_validation(dataset.root, tmp_path / "missing.json", splits, lock, checkpoint,
                            classes, tmp_path / "missing-manifest.json", **options)

    annotation = dataset.resolve(dataset.accepted[0].annotation_path)
    original_annotations = annotation.read_bytes()
    annotation.unlink()
    with pytest.raises(EvaluationError, match="Validation manifest or label failure"):
        evaluate_validation(*arguments, **options)
    annotation.write_bytes(original_annotations)

    checkpoint.unlink()
    with pytest.raises(EvaluationError, match="Checkpoint file is missing"):
        evaluate_validation(*arguments, **options)
    assert not (tmp_path / "result.json").exists()


def test_missing_selected_image_fails_without_publishing_a_partial_report(tmp_path):
    dataset, manifest, splits, lock, classes, checkpoint = inputs(tmp_path)
    dataset.resolve("images/two_a.png").unlink()

    with pytest.raises(EvaluationError, match=r"Validation image is missing for image_id fix-001"):
        evaluate_validation(
            dataset.root, manifest, splits, lock, checkpoint, classes, tmp_path / "partial.json",
            variant="nano", confidence_floor=0.5, iou_threshold=0.5,
            detector_loader=lambda *args, **kwargs: FakeDetector(),
        )
    assert not (tmp_path / "partial.json").exists()


@pytest.mark.parametrize("floor,threshold", [(-0.1, 0.5), (0.5, 1.1), (float("nan"), 0.5)])
def test_invalid_evaluation_settings_are_rejected(tmp_path, floor, threshold):
    with pytest.raises(ValueError):
        run(tmp_path, confidence_floor=floor, iou_threshold=threshold)
