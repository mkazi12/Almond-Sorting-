# DINOS labeled dataset import (SCRUM-42 / SCRUM-43)

This is a read-only **COCO object-detection** importer. It validates actual
images and annotations before constructing the **existing** `DatasetInput`
from `app/dataset/records.py`. No GUI, model, database writes, or source-file
modification are performed.

## Entry point

```python
from app.dataset.importer import import_coco_dataset
from app.dataset.records import SourcePolicy
from app.dataset.summary import summarize
from app.dataset.manifest import build_manifest

result = import_coco_dataset(
    "path/to/unpacked_dataset/train",  # one split/COCO collection per call
    annotation_file="_annotations.coco.json",
    image_directory=".",
    coco_filename_base=".",  # COCO file_name relative to this dataset-root folder
    class_mapping={1: "A", 2: "B"},  # test-only example; use approved real mapping
    dataset_namespace="approved-source-dataset-name",  # stable across versions
    allow_negative_images=False,  # only True when labels are confirmed complete
    source_policy=SourcePolicy(require_lot=True, require_session=True),
)

print(len(result.accepted), len(result.rejected))
for record in result.rejected:
    for reason in record.reasons:
        print(record.image_path, reason.category, reason.detail)

report = summarize(result)
manifest = build_manifest(result, "v1")
```

If COCO `file_name` is just `example.jpg` but images live under `images/`,
set `coco_filename_base="images"` and `image_directory="images"`.
For root-relative `file_name` such as `images/example.png`, keep
`coco_filename_base="."`. Both paths and `annotation_file` are relative to
`dataset_root`. Multiple COCO annotation JSON files should be imported as
separate collections; joining across files requires a higher-level adapter.

## Provenance input

Optionally pass `provenance_by_image`, keyed by normalized dataset-relative
image paths. For example:

```python
provenance_by_image={
    "images/example.png": {
        "source_lot": "lot-1",
        "source_session": "run-1",
        "near_duplicate_group_id": None,
    }
}
```

Unprovided provenance is stored as `None`, **not guessed from a filename**.
Images with valid labels remain accepted if provenance is missing, but
Zakari's `eligible_for_split()` marks them ineligible if `SourcePolicy`
requires the missing lot or session. Unknown near-duplicate IDs remain `None`.

## Stable IDs and repeated imports

The importer uses UUIDv5 of `dataset_namespace + normalized relative image path`.
For the same namespace, path and data, repeated imports yield identical records;
there is **no persistent database insert**, so re-importing cannot append
duplicate records itself. A future database consumer must upsert on `image_id`.
The ID is unchanged by dataset-root relocation or a new version label, but it
**does change on rename** and **does not change if the bytes at the same path
are replaced**. This is a **proposed** identity policy; confirm with Zakari.
Never use a dataset version or an absolute machine path as the namespace.

## Rejection rules

- Unsupported image extension: `unsupported_image`.
- Missing, unreadable, corrupt, or non-JPG/PNG image contents: `unreadable_image`.
- No matching COCO `images` entry: `missing_annotation`.
- Invalid COCO image metadata / annotation fields: `malformed_annotation`.
- Category absent from source categories or approved class mapping: `unknown_class_id`.
- Nonfinite, nonnumeric, negative, zero-sized or out-of-bounds box: `invalid_bbox`.
- COCO image dimensions differ from decoded image dimensions: `dimension_mismatch`.
- Image listed in COCO with no objects and policy disabled: `negative_image_not_allowed`.
- Unsafe path (including symlink escape): `unsafe_path`.

If **one** annotation fails, its entire image/annotation pair is rejected.
Only accepted entries and their objects are placed in `DatasetInput.annotations`.
An image with no objects becomes an accepted negative **only** if it has a
valid COCO image entry and `allow_negative_images=True`.

For missing/broken shared JSON, duplicate IDs, orphan references, unsafe COCO
filenames, conflicting class names, and other ambiguous *document-wide* errors,
`DatasetValidationError` is raised with `.category` and `.detail`. These
failures cannot safely be attributed to a single image. The caller should
report the JSON filename and the exception, not manufacture a rejected image.

The import scans supported JPG/PNG filenames plus common other image suffixes
(e.g. BMP, WebP) so unsupported images are reported. Arbitrary files such as
README.txt and JSON are not treated as candidate images unless COCO lists them.

## Testing

From the project root with the full installed requirements (including PySide6
and Pillow):

```bash
./dinos/Scripts/python.exe -m unittest discover -s tests -p "test_dataset_validation.py" -v
./dinos/Scripts/python.exe -m unittest discover -s tests -p "test_dataset_importer.py" -v
```

The tests construct tiny synthetic images and test-only class names; they do
not download almond data and do not claim RF-DETR model accuracy. Check the
full repository suite separately after focused tests pass. The supported
project Python version and class mapping should be agreed by the team.

## Open team decisions

1. Approve the stable ID namespace and rename/replacement semantics.
2. Agree on actual almond/foreign-object classes, including exact COCO names.
3. Confirm the negative-image labeling-completion policy.
4. Supply real lot/session provenance where available; no fabricated IDs.
5. Agree on handling multiple COCO files/splits and whether to combine them.
6. Confirm rejection category naming with Zakari before UI/report integration.
