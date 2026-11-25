"""Opt-in real RF-DETR package/load/restore smoke test; no quality claims.

python -m tests.run_model_workflow_integration PATH_TO_NANO_CHECKPOINT
"""

import json
from pathlib import Path
import sys
import tempfile
import zipfile

from app.model_workflow import ModelWorkflow, sha256


def main():
    import torch
    torch.set_num_threads(2)
    checkpoint = Path(sys.argv[1]).resolve()
    root = Path(".local-test/model-workflow")
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=root) as temporary:
        work = Path(temporary)
        # The official nano checkpoint has 91 outputs. Only the synthetic smoke
        # label is exposed here; no production class/provenance claims are made.
        from argparse import Namespace
        with torch.serialization.safe_globals([Namespace]):
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        count = state["model"]["class_embed.bias"].shape[0]
        del state
        record = {
            "schema_version": 1, "format": "rfdetr-state-dict-v1", "rfdetr_version": "1.3.0",
            "model_id": "smoke-nano", "variant": "nano", "model_version": "smoke-v1",
            "class_map": {"1": "Smoke-only label"}, "output_class_map": [None, 1] + [None] * (count - 2),
            "dataset_version": "smoke-only-no-dataset", "training_provenance": {"purpose": "loader smoke test"},
            "validation_summary": None, "checkpoint": "weights.pth", "checkpoint_sha256": sha256(checkpoint),
        }
        package = work / "input.zip"
        with zipfile.ZipFile(package, "w") as archive:
            archive.writestr("model.json", json.dumps(record))
            archive.write(checkpoint, "weights.pth")
        workflow = ModelWorkflow(work / "registry")
        imported = workflow.import_package(package)
        print("Real checkpoint imported and forward pass verified", flush=True)
        workflow.activate(imported.model_id)
        restored = ModelWorkflow(workflow.root).restore()
        assert restored.model_id == imported.model_id
        exported = workflow.export_package(imported.model_id, work / "export.zip")
        other = ModelWorkflow(work / "second").import_package(exported)
        assert sha256(other.checkpoint) == record["checkpoint_sha256"]
        assert json.loads(other.metadata.read_text()) == record
        print("PASS: real load, activation, restart restoration and export/reimport preserve checkpoint and metadata", flush=True)


if __name__ == "__main__":
    main()
