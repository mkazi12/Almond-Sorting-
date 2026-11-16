import json
import shutil

import pytest

from app.checkpoint import sha256_file
from app.inference import Detection
from app.threshold_sweep import SweepError, main, run_sweep, sweep_evaluation, threshold_range
from tests.test_evaluate import FakeDetector, run


# Validation image fix-001 has two class-A labels: [10,10,30,25] and [40,30,60,45].
# Class B has no validation ground truth. Scores sit exactly on several thresholds.
KNOWN_SCORES = (
    Detection(1, 0.95, (10, 10, 30, 25)),  # TP for the first label
    Detection(1, 0.8, (40, 30, 60, 45)),   # TP for the second label
    Detection(1, 0.7, (0, 40, 10, 50)),    # FP: no overlapping label
    Detection(2, 0.6, (40, 30, 60, 45)),   # FP: class B has no ground truth
    Detection(1, 0.5, (10, 10, 30, 25)),   # FP: duplicate of an already matched label
)

# threshold -> (predictions, TP, FP, FN, precision, recall, class A, class B)
EXPECTED = {
    0.5: (5, 2, 3, 0, 0.4, 1.0, (4, 2, 0.5, 1.0), (1, 0, 0.0, None)),
    0.6: (4, 2, 2, 0, 0.5, 1.0, (3, 2, 2 / 3, 1.0), (1, 0, 0.0, None)),
    0.7: (3, 2, 1, 0, 2 / 3, 1.0, (3, 2, 2 / 3, 1.0), (0, 0, None, None)),
    0.8: (2, 2, 0, 0, 1.0, 1.0, (2, 2, 1.0, 1.0), (0, 0, None, None)),
    0.9: (1, 1, 0, 1, 1.0, 0.5, (1, 1, 1.0, 0.5), (0, 0, None, None)),
}


@pytest.fixture
def evaluation(tmp_path):
    report, _ = run(tmp_path, detector=FakeDetector(KNOWN_SCORES))
    return report, tmp_path / "evaluation.json"


def test_known_scores_give_expected_counts_and_metrics_at_each_threshold(evaluation, tmp_path):
    _, path = evaluation
    sweep = run_sweep(path, tmp_path / "sweep.json", minimum="0.5", maximum="0.9", step="0.1")

    assert [row["confidence_threshold"] for row in sweep["results"]] == list(EXPECTED)
    for row in sweep["results"]:
        predictions, tp, fp, fn, precision, recall, class_a, class_b = EXPECTED[row["confidence_threshold"]]
        overall = row["overall"]
        assert overall["image_count"] == 1
        assert overall["ground_truth_count"] == 2
        assert (overall["prediction_count"], overall["true_positive_count"],
                overall["false_positive_count"], overall["false_negative_count"]) == (predictions, tp, fp, fn)
        assert overall["precision"] == pytest.approx(precision)
        assert overall["recall"] == pytest.approx(recall)

        a, b = row["per_class"]
        assert (a["class_name"], a["ground_truth_count"]) == ("A", 2)
        assert (a["prediction_count"], a["true_positive_count"]) == class_a[:2]
        assert a["precision"] == pytest.approx(class_a[2]) and a["recall"] == pytest.approx(class_a[3])
        assert (b["class_name"], b["ground_truth_count"]) == ("B", 0)
        assert (b["prediction_count"], b["true_positive_count"], b["precision"]) == class_b[:3]
        assert b["recall"] is None  # unsupported class stays unavailable, never 0
    assert json.loads((tmp_path / "sweep.json").read_text()) == sweep


def test_equality_boundary_keeps_scores_equal_to_threshold():
    # 0.5 + 3 * 0.1 is 0.8000000000000002 in binary floats; the range must be exact.
    assert threshold_range("0.5", "0.9", "0.1") == (0.5, 0.6, 0.7, 0.8, 0.9)
    assert threshold_range(0.35, 0.35, 0) == (0.35,)


def test_threshold_at_floor_reproduces_evaluation_operating_point(evaluation):
    report, _ = evaluation
    row = sweep_evaluation(report, [0.5])["results"][0]
    operating = report["metrics"]["operating_point"]
    assert row["overall"]["prediction_count"] == operating["prediction_count"]
    assert row["overall"]["true_positive_count"] == operating["true_positive_count"]
    assert row["overall"]["precision"] == operating["precision"]
    assert row["overall"]["recall"] == operating["recall"]
    for swept, evaluated in zip(row["per_class"], report["metrics"]["per_class"]):
        assert swept["precision"] == evaluated["operating_precision"]
        assert swept["recall"] == evaluated["operating_recall"]
        assert swept["ground_truth_count"] == evaluated["ground_truth_count"]


def test_output_records_checkpoint_dataset_range_iou_and_matching_rules(evaluation, tmp_path):
    report, path = evaluation
    sweep = run_sweep(path, tmp_path / "sweep.json", minimum=0.5, maximum=0.7, step=0.1)
    assert sweep["evaluation"] == {
        "evaluation_id": report["evaluation_id"],
        "path": str(path.resolve()), "sha256": sha256_file(path),
    }
    assert sweep["checkpoint"] == report["checkpoint"] and sweep["checkpoint"]["sha256"]
    assert sweep["dataset"] == report["dataset"]
    assert sweep["dataset"]["validation_image_ids"] == ["fix-001"]
    assert sweep["settings"] == {
        "split": "valid",
        "prediction_confidence_floor": 0.5,
        "confidence_thresholds": [0.5, 0.6, 0.7],
        "threshold_rule": "keep prediction when confidence >= threshold",
        "iou_threshold": 0.5,
        "matching_rule": "greedy_one_to_one_class_aware_descending_confidence",
        "matching_tie_break_rule": "image_id_then_prediction_index; label_index_for_equal_iou",
    }


def test_threshold_below_prediction_floor_is_rejected_clearly(evaluation, tmp_path):
    _, path = evaluation
    with pytest.raises(SweepError, match=r"0.3 is below .* floor 0.5.*--confidence-floor 0.3"):
        run_sweep(path, tmp_path / "sweep.json", minimum="0.3", maximum="0.9", step="0.1")
    assert not (tmp_path / "sweep.json").exists()


def test_lower_floor_evaluation_supports_lower_thresholds(tmp_path):
    report, _ = run(tmp_path, detector=FakeDetector(KNOWN_SCORES), confidence_floor=0.3)
    rows = sweep_evaluation(report, threshold_range("0.3", "0.5", "0.1"))["results"]
    assert [row["overall"]["prediction_count"] for row in rows] == [5, 5, 5]


def test_sweep_reads_no_images(evaluation, tmp_path):
    _, path = evaluation
    # Remove every dataset and frozen-test image; only the saved JSON remains.
    shutil.rmtree(tmp_path / "dataset")
    shutil.rmtree(tmp_path / "lock-dataset")
    sweep = run_sweep(path, tmp_path / "sweep.json", minimum="0.5", maximum="0.9", step="0.1")
    assert len(sweep["results"]) == 5


@pytest.mark.parametrize("minimum,maximum,step,message", [
    ("0.9", "0.5", "0.1", "minimum <= maximum"),
    ("-0.1", "0.5", "0.1", "0 <= minimum"),
    ("0.5", "1.1", "0.1", "maximum <= 1"),
    ("0.5", "0.9", "0", "greater than 0"),
    ("0.5", "0.9", "0.3", "whole number"),
    ("nan", "0.9", "0.1", "finite"),
])
def test_invalid_ranges_are_rejected(minimum, maximum, step, message):
    with pytest.raises(SweepError, match=message):
        threshold_range(minimum, maximum, step)


def test_non_validation_evaluation_is_rejected(evaluation):
    report, _ = evaluation
    report["settings"]["split"] = "test"
    with pytest.raises(SweepError, match="only accept validation"):
        sweep_evaluation(report, [0.5])


def test_tampered_matches_are_rejected(evaluation):
    report, _ = evaluation
    report["records"][0]["matches"] = []
    with pytest.raises(SweepError, match="disagree with the matching rule"):
        sweep_evaluation(report, [0.5])


def test_prediction_below_saved_floor_is_rejected(evaluation):
    report, _ = evaluation
    report["records"][0]["predictions"][-1]["confidence"] = 0.4
    with pytest.raises(SweepError, match="below the confidence floor"):
        sweep_evaluation(report, [0.5])


def test_existing_output_is_never_overwritten(evaluation, tmp_path):
    _, path = evaluation
    target = tmp_path / "sweep.json"
    target.write_text("previous sweep")
    with pytest.raises(SweepError, match="must be a new file"):
        run_sweep(path, target, minimum="0.5", maximum="0.9", step="0.1")
    with pytest.raises(SweepError, match="must be a new file"):
        run_sweep(path, path, minimum="0.5", maximum="0.9", step="0.1")
    assert target.read_text() == "previous sweep"


def test_command_line(evaluation, tmp_path, capsys):
    _, path = evaluation
    assert main([str(path), str(tmp_path / "out" / "sweep.json"),
                 "--min-confidence", "0.5", "--max-confidence", "0.9", "--step", "0.1"]) == 0
    assert "Swept 5 thresholds" in capsys.readouterr().out
    assert (tmp_path / "out" / "sweep.json").is_file()
    with pytest.raises(SystemExit) as stopped:
        main([str(path), str(tmp_path / "low.json"), "--min-confidence", "0.1", "--max-confidence", "0.9"])
    assert stopped.value.code == 2
    assert "below the evaluation's prediction confidence floor" in capsys.readouterr().err
