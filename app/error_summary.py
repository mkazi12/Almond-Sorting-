"""Group validation error records using the originating dataset manifest."""

from collections import Counter

from app.checkpoint import sha256_file
from app.dataset.paths import resolve_dataset_path
from app.dataset.records import source_label


def summarize_validation_errors(error_export, manifest, dataset_root, *, examples_per_group=3):
    """Count every error once per dimension; unavailable images retain their counts."""
    if type(examples_per_group) is not int or examples_per_group < 1:
        raise ValueError("examples_per_group must be a positive integer.")
    if error_export["schema_version"] != 1 or error_export["settings"]["split"] != "valid":
        raise ValueError("Summary requires validation-only error-export schema 1.")
    identity = error_export["dataset"]
    if (identity["version"] != manifest.dataset_version
            or identity["content_hash"] != manifest.content_hash):
        raise ValueError("Error export and manifest identify different datasets.")
    images = {image.image_id: image for image in manifest.images}
    valid_ids = identity["validation_image_ids"]
    if not valid_ids or len(set(valid_ids)) != len(valid_ids) or not set(valid_ids) <= images.keys():
        raise ValueError("Validation membership does not resolve to the manifest.")
    valid_ids = set(valid_ids)
    errors = error_export["errors"]
    if len({error["error_id"] for error in errors}) != len(errors):
        raise ValueError("Duplicate error_id values in error export.")
    expected = error_export["summary"]["by_error_type"]
    actual = Counter(error["error_type"] for error in errors)
    if (error_export["summary"]["error_count"] != len(errors)
            or set(actual) - set(expected)
            or any(actual[kind] != count for kind, count in expected.items())):
        raise ValueError("Error-export totals disagree with its records.")
    classes = {row["class_id"]: row["class_name"] for row in error_export["classes"]}
    dimensions = {name: {} for name in ("class", "source", "lighting", "almond_type")}

    def bucket(dimension, key, label=None):
        return dimensions[dimension].setdefault(key, {
            "group": key,
            "label": key if label is None else label,
            "error_count": 0,
            "by_error_type": dict.fromkeys(expected, 0),
            "examples": [],
            "unavailable_example_count": 0,
            "_image_ids": set(),
        })

    # Include configured classes and observed validation metadata even at zero errors.
    for category_id, name in sorted(classes.items()):
        bucket("class", str(category_id), name)
    for dimension in dimensions:
        bucket(dimension, "unknown")
    for image_id in sorted(valid_ids):
        image = images[image_id]
        for dimension, value in (
            ("source", image.source_lot),
            ("lighting", image.lighting),
            ("almond_type", image.almond_type),
        ):
            bucket(dimension, source_label(value))

    availability = {}
    for error in sorted(errors, key=lambda item: item["error_id"]):
        image_id = error["image_id"]
        if image_id not in valid_ids:
            raise ValueError(f"Error {error['error_id']} references a non-validation image.")
        image = images[image_id]
        if error["image_path"] != image.image_path:
            raise ValueError(f"Error {error['error_id']} has a different image path from its manifest.")
        subject = error["label"] if error["label"] is not None else error["prediction"]
        if subject is None or subject["category_id"] not in classes:
            raise ValueError(f"Error {error['error_id']} has no configured subject class.")
        category_id = subject["category_id"]
        group_keys = {
            "class": str(category_id),
            "source": source_label(image.source_lot),
            "lighting": source_label(image.lighting),
            "almond_type": source_label(image.almond_type),
        }
        # One digest check per image/digest, shared by all dimensions and errors.
        fingerprint = (image_id, error["image_sha256"])
        if fingerprint not in availability:
            path = resolve_dataset_path(dataset_root, image.image_path)
            try:
                availability[fingerprint] = path.is_file() and sha256_file(path) == error["image_sha256"]
            except OSError:
                availability[fingerprint] = False
        for dimension, key in group_keys.items():
            group = dimensions[dimension][key]
            group["error_count"] += 1
            group["by_error_type"][error["error_type"]] += 1
            group["_image_ids"].add(image_id)
            if not availability[fingerprint]:
                group["unavailable_example_count"] += 1
            elif len(group["examples"]) < examples_per_group:
                group["examples"].append({
                    "error_id": error["error_id"],
                    "image_id": image_id,
                    "image_path": image.image_path,
                    "image_sha256": error["image_sha256"],
                    "annotation_id": None if error["label"] is None else error["label"]["annotation_id"],
                    "prediction_index": None if error["prediction"] is None else error["prediction"]["prediction_index"],
                })

    grouped = {}
    for dimension, groups in dimensions.items():
        if sum(group["error_count"] for group in groups.values()) != len(errors):
            raise ValueError(f"Error totals do not reconcile for {dimension}.")
        for kind, count in expected.items():
            if sum(group["by_error_type"][kind] for group in groups.values()) != count:
                raise ValueError(f"Error-type totals do not reconcile for {dimension}/{kind}.")
        for group in groups.values():
            group["image_count"] = len(group.pop("_image_ids"))
            group["examples_status"] = (
                "empty" if group["error_count"] == 0
                else "available" if group["examples"] else "unavailable"
            )
        grouped[dimension] = [groups[key] for key in sorted(groups)]
    return {
        "schema_version": 1,
        "evaluation_id": error_export["evaluation_id"],
        "evaluation_sha256": error_export["evaluation_sha256"],
        "checkpoint": error_export["checkpoint"],
        "dataset": identity,
        "settings": error_export["settings"],
        "grouping_rules": {
            "class": "label_class_else_prediction_class",
            "source": "source_lot",
            "missing_metadata": "unknown",
            "examples_per_group": examples_per_group,
        },
        "error_count": len(errors),
        "groups": grouped,
    }
