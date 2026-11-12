# RF-DETR dataset export

`app.dataset.rfdetr_export` converts an approved versioned manifest and its
COCO labels into separate RF-DETR `train` and `valid` inputs. It validates all
records before publishing the output directory.

```powershell
python -m app.dataset.rfdetr_export `
  DATASET_ROOT `
  manifests/dataset-v1.json `
  DATASET_ROOT/split_manifest.json `
  manifests/class-map.json `
  rfdetr-dataset-v1 `
  --frozen-test-lock manifests/test_lock/test-v1.json
```

The class-map file can be a JSON object such as:

```json
{
  "1": "almond",
  "2": "foreign_material"
}
```

It may instead contain a COCO `categories` list. The split manifest uses its
existing `images` rows. Each row must contain `split` and `group_id`, and is
matched to the versioned manifest by `image_id`, or by an identical `source`
or `output` path.

The refactored `app.dataset.partitions.build_partition_manifest` handoff is
also accepted as the split-manifest argument. For documents with
`kind: labeled_dataset_partitions`, the exporter rebuilds the checked
partition manifest and verifies its complete metadata, labels, class map,
version, hashes and derived source groups before exporting. A changed or
stale handoff fails with `PARTITION_MISMATCH`. No split is chosen or shuffled
by the exporter.

Frozen-test locks use the current validated schema from
`app.dataset.frozen_test_set`. Invalid locks fail before loading image pixels;
frozen stable IDs, groups and copied file content are all excluded.

The exporter creates:

```text
rfdetr-dataset-v1/
  dataset_provenance.json
  train/
    _annotations.coco.json
    <manifest image paths>
  valid/
    _annotations.coco.json
    <manifest image paths>
```

COCO image IDs, category IDs, annotation IDs, paths, box coordinates and
image dimensions are preserved. Accepted images with no objects remain in
the `images` list with no annotation. COCO image records and the provenance
file include the stable image ID and source-group fields. Dataset version and
content hash are recorded in both split files and the provenance file.

Before writing, the exporter checks that:

- the versioned manifest still matches the source records and labels;
- each manifest image has exactly one train, valid or test assignment;
- source groups do not cross splits;
- source files exist, decode successfully and have the recorded dimensions;
- annotation image paths and dimensions agree with the manifest;
- class IDs and names match the approved class map;
- boxes contain finite coordinates, positive dimensions and stay inside the image;
- COCO image and annotation IDs remain unambiguous in the exported files;
- train and valid image bytes and source groups do not match the supplied frozen-test lock.

Test assignments are validated but never exported. Errors contain a stable
code, for example `[MISSING_IMAGE]`, `[UNKNOWN_CLASS]`, `[INVALID_BOX]` or
`[FROZEN_TEST_LEAK]`. A validation failure leaves no output dataset.

Run the focused acceptance tests with:

```powershell
.\.venv\Scripts\python.exe -m pytest tests\test_rfdetr_export.py -q
```
