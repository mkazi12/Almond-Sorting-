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