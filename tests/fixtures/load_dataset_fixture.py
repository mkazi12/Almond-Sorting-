"""Load the authored development dataset fixture.

This module builds ``DatasetInput`` from checked-in JSON and writes
placeholder images under a caller-supplied root. It is not an importer
or a validator. Accepted and rejected records are authored, including
the simulated rejection and the image that is missing a lot.
"""

import json
import shutil
import struct
import zlib
from pathlib import Path

from app.dataset.records import (
    AcceptedImage,
    ClassSpec,
    CocoImage,
    DatasetInput,
    GroundTruthObject,
    ParsedAnnotations,
    RejectionReason,
    RejectedImage,
    SourcePolicy,
    resolve_dataset_path,
)


FIXTURE_DIR = Path(__file__).resolve().parent / "dataset"

# Solid placeholder colors so each accepted image is a real local file.
# These pixels are not captured almond data.
_PLACEHOLDER_COLORS = {
    "fix-001": (186, 140, 90),
    "fix-002": (90, 150, 170),
    "fix-003": (230, 230, 230),
}


def load_dataset_fixture(root: Path) -> DatasetInput:
    """Materialize the fixture under ``root`` and return its records."""
    root.mkdir(parents=True, exist_ok=True)
    acceptance = _read_json(FIXTURE_DIR / "acceptance.json")
    shutil.copyfile(
        FIXTURE_DIR / "acceptance.json",
        resolve_dataset_path(root, "acceptance.json"),
    )

    annotation_paths: list[str] = []
    for entry in acceptance["accepted"]:
        annotation_path = entry["annotation_path"]
        if annotation_path not in annotation_paths:
            annotation_paths.append(annotation_path)
        _write_placeholder(
            root,
            entry["image_path"],
            entry["width"],
            entry["height"],
            _PLACEHOLDER_COLORS.get(entry["image_id"], (200, 200, 200)),
        )

    parsed = tuple(_copy_annotations(root, path) for path in annotation_paths)
    policy = acceptance["source_policy"]
    return DatasetInput(
        root=root,
        source_policy=SourcePolicy(
            require_lot=policy["require_lot"],
            require_session=policy["require_session"],
        ),
        accepted=tuple(_accepted(entry) for entry in acceptance["accepted"]),
        rejected=tuple(_rejected(entry) for entry in acceptance["rejected"]),
        annotations=parsed,
    )


def _copy_annotations(root: Path, annotation_path: str) -> ParsedAnnotations:
    source = FIXTURE_DIR / annotation_path
    destination = resolve_dataset_path(root, annotation_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)
    document = _read_json(source)
    return ParsedAnnotations(
        path=annotation_path,
        classes=tuple(
            ClassSpec(category_id=item["id"], name=item["name"])
            for item in document["categories"]
        ),
        images=tuple(
            CocoImage(
                coco_image_id=item["id"],
                file_name=item["file_name"],
                width=item["width"],
                height=item["height"],
            )
            for item in document["images"]
        ),
        objects=tuple(
            GroundTruthObject(
                annotation_id=item["id"],
                coco_image_id=item["image_id"],
                category_id=item["category_id"],
                bbox_xywh=tuple(item["bbox"]),
            )
            for item in document["annotations"]
        ),
    )


def _accepted(entry: dict) -> AcceptedImage:
    return AcceptedImage(
        image_id=entry["image_id"],
        coco_image_id=entry["coco_image_id"],
        image_path=entry["image_path"],
        annotation_path=entry["annotation_path"],
        width=entry["width"],
        height=entry["height"],
        source_lot=entry["source_lot"],
        source_session=entry["source_session"],
        near_duplicate_group_id=entry["near_duplicate_group_id"],
    )


def _rejected(entry: dict) -> RejectedImage:
    return RejectedImage(
        image_path=entry["image_path"],
        image_id=entry["image_id"],
        reasons=tuple(
            RejectionReason(category=reason["category"], detail=reason["detail"])
            for reason in entry["reasons"]
        ),
    )


def _write_placeholder(
    root: Path,
    image_path: str,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    destination = resolve_dataset_path(root, image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(_solid_png(width, height, color))


def _solid_png(width: int, height: int, rgb: tuple[int, int, int]) -> bytes:
    def chunk(tag: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(tag + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", crc)

    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    row = b"\x00" + bytes(rgb) * width
    compressed = zlib.compress(row * height, 9)
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", compressed)
        + chunk(b"IEND", b"")
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
