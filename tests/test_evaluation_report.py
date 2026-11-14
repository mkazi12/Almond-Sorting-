"""Saved JSON fixtures produced by the existing fake-detector evaluation workflow."""
import hashlib
import json

import pytest

from app.evaluation_report import ReportError, generate_report, main
from tests.test_evaluate import run


@pytest.fixture
def artifacts(tmp_path):
    evaluation = tmp_path / "evaluation.json"
    benchmark = tmp_path / "benchmark.json"
    run(tmp_path, benchmark_output=benchmark, warmup_runs=0)
    return evaluation, benchmark, tmp_path / "report.md"


def rewrite(path, change):
    document = json.loads(path.read_text())
    change(document)
    path.write_text(json.dumps(document))


def test_renders_saved_values_without_recalculation(artifacts, monkeypatch):
    evaluation, benchmark, output = artifacts
    def forbidden(*args, **kwargs):
        raise AssertionError("Must only render saved metrics")
    monkeypatch.setattr("app.evaluate._calculate_metrics", forbidden)
    monkeypatch.setattr("app.benchmark.percentile", forbidden)
    rewrite(benchmark, lambda doc: doc["benchmark"].update(p50_latency_ms=123.456, p95_latency_ms=789.012))
    generate_report(evaluation, benchmark, output)
    text = output.read_text(encoding="utf-8")
    saved = json.loads(evaluation.read_text())
    assert saved["evaluation_id"] in text
    assert saved["checkpoint"]["sha256"] in text
    assert "dataset-v1" in text
    assert "123.456" in text and "789.012" in text
    assert "| Confidence threshold | 0.5 |" in text
    assert "| Box mAP@0.50 | " + str(saved["metrics"]["average_precision"]["map_50"]) + " |" in text
    assert "N/A" in text
    assert "Hardware and runtime" in text
    assert "Package torch" in text


@pytest.mark.parametrize("field", ["evaluation_id", "evaluation_created_at", "checkpoint", "dataset", "settings", "evaluation_sha256"])
def test_mismatch_never_publishes_report(artifacts, field):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: doc.update({field: "wrong"}))
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


def test_hash_checks_exact_evaluation_bytes(artifacts):
    evaluation, benchmark, output = artifacts
    evaluation.write_bytes(evaluation.read_bytes() + b"\n")
    with pytest.raises(ReportError, match="hash mismatch"):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    lambda doc: doc.update(schema_version=2),
    lambda doc: doc["benchmark"]["runtime"].update(device="cuda"),
    lambda doc: doc["benchmark"].update(sample_count=3),
    lambda doc: doc["benchmark"]["samples"][0].update(image_id="test-image"),
    lambda doc: doc["benchmark"]["samples"][0].update(width=999),
    lambda doc: doc["benchmark"].update(p50_latency_ms=float("nan")),
    lambda doc: doc["benchmark"].pop("p95_latency_ms"),
])
def test_invalid_json_fields_fail(artifacts, change):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, change)
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


def test_escapes_class_text(artifacts):
    evaluation, benchmark, output = artifacts
    def change(doc):
        doc["classes"][0]["class_name"] = "A|<script>\nB"
        doc["metrics"]["per_class"][0]["class_name"] = "A|<script>\nB"
    rewrite(evaluation, change)
    rewrite(benchmark, lambda doc: doc.update(evaluation_sha256=hashlib.sha256(evaluation.read_bytes()).hexdigest()))
    generate_report(evaluation, benchmark, output)
    assert "A&#124;&lt;script&gt; B" in output.read_text(encoding="utf-8")


def test_output_failure_cleans_temporary_file(artifacts, monkeypatch):
    evaluation, benchmark, output = artifacts
    def fail(*args):
        raise OSError("fixture publication failure")
    monkeypatch.setattr("app.evaluation_report.os.link", fail)
    with pytest.raises(ReportError, match="publication failure"):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()
    assert not list(output.parent.glob(".report.md-*"))


def test_existing_output_preserved(artifacts):
    evaluation, benchmark, output = artifacts
    output.write_text("prior report")
    with pytest.raises(ReportError, match="new file"):
        generate_report(evaluation, benchmark, output)
    assert output.read_text() == "prior report"


def test_standalone_command(artifacts):
    assert main([str(path) for path in artifacts]) == 0


def test_malformed_json(artifacts):
    evaluation, benchmark, output = artifacts
    benchmark.write_text('{"schema_version": 1, "schema_version": 1}')
    with pytest.raises(ReportError, match="Duplicate"):
        generate_report(evaluation, benchmark, output)


def test_concurrently_created_report_is_not_overwritten(artifacts, monkeypatch):
    import os
    evaluation, benchmark, output = artifacts
    original = os.link
    def race(source, target):
        output.write_text("another report")
        original(source, target)
    monkeypatch.setattr("app.evaluation_report.os.link", race)
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert output.read_text() == "another report"
    assert not list(output.parent.glob(".report.md-*"))


@pytest.mark.parametrize("value", [-64, 0, 64.5, 64.0, True, "64", [], float("inf")])
def test_invalid_model_resolution_is_rejected(artifacts, value):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: doc["benchmark"].update(model_input_resolution=value))
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["width", "height", "count"])
@pytest.mark.parametrize("value", [-1, 0, 1.5, True, "1"])
def test_invalid_resolution_summary_fields_are_rejected(artifacts, field, value):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: doc["benchmark"]["resolutions"][0].update({field: value}))
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


@pytest.mark.parametrize("change", [
    lambda rows: rows.clear(),
    lambda rows: rows.append(dict(rows[0])),
    lambda rows: rows[0].update(count=2),
    lambda rows: rows[0].update(width=999),
    lambda rows: rows.append(dict(width=1, height=1, count=1)),
])
def test_resolution_summary_must_match_samples(artifacts, change):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: change(doc["benchmark"]["resolutions"]))
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


@pytest.mark.parametrize("field", ["width", "height"])
@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_sample_dimensions_are_rejected(artifacts, field, value):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: doc["benchmark"]["samples"][0].update({field: value}))
    with pytest.raises(ReportError):
        generate_report(evaluation, benchmark, output)
    assert not output.exists()


@pytest.mark.parametrize("resolution", [None, 640])
def test_unknown_or_positive_model_resolution_is_supported(artifacts, resolution):
    evaluation, benchmark, output = artifacts
    rewrite(benchmark, lambda doc: doc["benchmark"].update(model_input_resolution=resolution))
    generate_report(evaluation, benchmark, output)
    assert output.exists()
