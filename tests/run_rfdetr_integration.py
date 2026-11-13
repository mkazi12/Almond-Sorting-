"""Opt-in real RF-DETR one-epoch smoke check (not a model-quality test).

Run with the isolated training interpreter from the repository root:
python -m tests.run_rfdetr_integration
"""

import json
from pathlib import Path
import sys
from dataclasses import replace
from uuid import uuid4

from app.dataset.manifest import build_manifest
from app.dataset.rfdetr_export import SplitAssignment, export_rfdetr_dataset
from app.train_rfdetr import launch
from tests.fixtures.load_dataset_fixture import load_dataset_fixture


def main():
    import torch
    torch.set_num_threads(2)
    root = Path(".local-test/rfdetr-integration").resolve()
    root.mkdir(parents=True, exist_ok=True)
    attempt = root / f"attempt-{uuid4().hex[:8]}"
    attempt.mkdir()
    print(f"Integration evidence: {attempt}", flush=True)
    dataset = load_dataset_fixture(attempt / "source")
    # Explicit independent synthetic groups; these placeholders are not captured data.
    dataset = replace(dataset, accepted=tuple(
        replace(image, source_lot=f"lot-{index}", source_session=f"session-{index}",
                near_duplicate_group_id=f"dup-{index}")
        for index, image in enumerate(dataset.accepted)
    ))
    output = attempt / "dataset"
    export_rfdetr_dataset(
        dataset, build_manifest(dataset, "integration-synthetic-v1"),
        [SplitAssignment("fix-001", "train", "synthetic-1"),
         SplitAssignment("fix-002", "valid", "synthetic-2"),
         SplitAssignment("fix-003", "train", "synthetic-3")],
        {1: "A", 2: "B"}, output,
    )
    # A deliberately unusable test dataset: any attempt to read it fails.
    test = output / "test"
    test.mkdir()
    (test / "_annotations.coco.json").write_text("FROZEN TEST MUST NOT BE OPENED")
    forbidden = str(test).casefold()

    def forbid_test_reads(event, args):
        if event == "open" and isinstance(args[0], (str, bytes)):
            candidate = str(Path(args[0]).resolve()).casefold()
            if candidate.startswith(forbidden):
                raise AssertionError(f"Frozen test access: {args[0]}")

    sys.addaudithook(forbid_test_reads)
    checkpoint = root / "rf-detr-nano.pth"
    if not checkpoint.is_file():
        raise RuntimeError(f"Download the official nano checkpoint first: {checkpoint}")
    config = attempt / "config.json"
    config.write_text(json.dumps(dict(
        variant="nano", dataset_dir=str(output), initial_checkpoint=str(checkpoint),
        device="cpu", seed=42, output_dir=str(attempt / "run"),
        training=dict(epochs=1, batch_size=1, grad_accum_steps=1, num_workers=0,
                      checkpoint_interval=1, multi_scale=False, expanded_scales=False),
    )))
    resolved = launch(config)
    run = attempt / "run"
    for name in ("checkpoint.pth", "checkpoint_best_total.pth", "resolved_config.json", "log.txt",
                 "console.log", "run.json"):
        assert (run / name).is_file(), name
    assert resolved["train_kwargs"]["run_test"] is False
    record = json.loads((run / "run.json").read_text())
    assert record["run_id"] == resolved["run_id"] and record["status"] == "completed", record
    assert record["checkpoints"]["latest"]["epoch"] == 0
    assert record["checkpoints"]["best"]["selected_by"]["split"] == "valid"
    print("INTEGRATION PASSED: real one-epoch CPU training, negative image, verified run record, no test access")


if __name__ == "__main__":
    main()
