"""Audit an existing raw grouped train/valid/test split.

Image and label files are read-only. The CLI writes audit_report.json.
Run from the project root:

    python -m app.dataset.audit dataset_demo

Reads split_manifest.json (and each split's _annotations.coco.json if it
exists) and writes audit_report.json next to them.

A few notes:
  - Splits with no COCO file are reported as PENDING, not as all-negative.
  - DCT similarity only flags pairs for a human to look at. It doesn't prove
    anything is a duplicate.
  - The defaults (distance 6 out of 64 bits, 10 images per rare class) are
    review settings, not calibrated quality thresholds.

Exit codes: 0 = clean, 1 = findings or pending work, 2 = bad input.
"""
import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.neighbors import NearestNeighbors

from app.dataset.image_io import read_image_dimensions
from app.dataset.paths import normalize_relative_path, resolve_dataset_path

SPLITS = ("train", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_of(path):
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fingerprint(image):
    #Shrink to 32x32 grayscale, take the top-left 8x8 DCT block, and threshold on the median. Gives a 64-bit vector of 0s and 1s.
    gray = ImageOps.grayscale(ImageOps.exif_transpose(image))
    small = gray.resize((32, 32), Image.Resampling.LANCZOS)
    coeffs = dctn(np.asarray(small, dtype=float), type=2, norm="ortho")[:8, :8]
    return (coeffs > np.median(coeffs)).astype(float).flatten()



def _partition_path(root, split, relative):
    """Resolve a portable dataset path without leaving its declared split."""
    path = resolve_dataset_path(root, relative)
    partition = resolve_dataset_path(root, split)
    if path == partition or not path.is_relative_to(partition):
        raise ValueError(f"Image path is outside its recorded partition: {relative}")
    return path


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError):
        return False


def read_coco_split(root, split, rows, class_names, errors):
    """Validate a split's COCO document and count its valid object records.

    No approved application class map is inferred here. Categories must agree
    across partitions, and image identity must agree with the raw split
    manifest. Invalid geometry is a blocking finding and never counted.
    """
    path = resolve_dataset_path(root, f"{split}/_annotations.coco.json")
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or any(
        not isinstance(data.get(key), list) for key in ("categories", "images", "annotations")
    ):
        raise ValueError(f"{split}: COCO needs categories, images and annotations arrays.")

    local_names = {}
    for category in data["categories"]:
        if (not isinstance(category, dict) or type(category.get("id")) is not int
                or not isinstance(category.get("name"), str) or not category["name"].strip()):
            errors.append(f"{split}: categories need integer IDs and nonempty names.")
            continue
        class_id, name = category["id"], category["name"]
        if class_id in local_names or name in local_names.values():
            errors.append(f"{split}: duplicate class IDs or names.")
            continue
        local_names[class_id] = name
        if class_id in class_names and class_names[class_id] != name:
            errors.append(f"Class ID {class_id} has different names across partitions.")
        if name in class_names.values() and class_names.get(class_id) != name:
            errors.append(f"Class {name!r} uses different IDs across partitions.")
        class_names[class_id] = name

    image_paths, dimensions = {}, {}
    for image in data["images"]:
        if (not isinstance(image, dict) or type(image.get("id")) is not int
                or not isinstance(image.get("file_name"), str)):
            errors.append(f"{split}: COCO images need integer IDs and filenames.")
            continue
        try:
            relative = normalize_relative_path(image["file_name"])
            file = _partition_path(root, split, f"{split}/{relative}")
        except ValueError as error:
            errors.append(f"{split}: unsafe COCO image path: {error}")
            continue
        image_id = image["id"]
        if image_id in image_paths or file in image_paths.values():
            errors.append(f"{split}: duplicate COCO image ID or filename.")
            continue
        image_paths[image_id] = file
        width, height = image.get("width"), image.get("height")
        if type(width) is not int or type(height) is not int or width <= 0 or height <= 0:
            errors.append(f"{split}: {relative}: image dimensions must be positive integers.")
            continue
        try:
            actual_width, actual_height = read_image_dimensions(file)
            if (actual_width, actual_height) != (width, height):
                raise ValueError(
                    f"COCO dimensions {width}x{height} differ from image "
                    f"{actual_width}x{actual_height}."
                )
        except (OSError, ValueError) as error:
            errors.append(f"{split}: {relative}: {error}")
            continue
        dimensions[image_id] = (width, height)

    in_manifest = {_partition_path(root, split, r["output"]) for r in rows if r["split"] == split}
    if set(image_paths.values()) != in_manifest:
        errors.append(f"{split}: COCO image records do not match the split manifest.")

    counts = {image_id: Counter() for image_id in image_paths}
    seen_ids = set()
    for ann in data["annotations"]:
        if (not isinstance(ann, dict) or any(type(ann.get(key)) is not int
                                           for key in ("id", "image_id", "category_id"))):
            errors.append(f"{split}: annotation IDs, image IDs and class IDs must be integers.")
            continue
        if ann["id"] in seen_ids:
            errors.append(f"{split}: repeated annotation ID {ann['id']}.")
            continue
        seen_ids.add(ann["id"])
        if ann["image_id"] not in counts or ann["category_id"] not in local_names:
            errors.append(f"{split}: annotation has an unknown image/class.")
            continue
        bbox = ann.get("bbox")
        if (not isinstance(bbox, (list, tuple)) or len(bbox) != 4
                or not all(_finite_number(value) for value in bbox)):
            errors.append(f"{split}: annotation {ann['id']}: bbox needs four finite numeric values.")
            continue
        x, y, width, height = bbox
        size = dimensions.get(ann["image_id"])
        if (size is None or x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > size[0] or y + height > size[1]):
            errors.append(f"{split}: annotation {ann['id']}: invalid bbox or image dimensions.")
            continue
        counts[ann["image_id"]][local_names[ann["category_id"]]] += 1
    return counts


def label_counts(root, rows, minimum, errors, warnings):
    """Count objects per class for every split that has a COCO file."""
    missing = []
    loaded = {}
    class_names = {}

    for split in SPLITS:
        if not (root / split / "_annotations.coco.json").is_file():
            missing.append(split)
            continue
        loaded[split] = read_coco_split(root, split, rows, class_names, errors)

    results = {}
    for split in SPLITS:
        if split not in loaded:
            results[split] = {"status": "PENDING_LABELS", "classes": None, "empty_annotation_images": None}
            continue

        per_image = loaded[split]
        classes = {}
        for name in sorted(set(class_names.values())):
            objects = sum(c[name] for c in per_image.values())
            images = sum(c[name] > 0 for c in per_image.values())
            classes[name] = {"objects": objects, "images_with_class": images}
            if images < minimum:
                warnings.append(f"{split}: class {name!r} occurs in {images} images (review threshold: {minimum}).")
        results[split] = {"status": "COUNTED", "classes": classes, "empty_annotation_images": sum(not c for c in per_image.values()),}

    has_empty = any(r["empty_annotation_images"] for r in results.values() if r["empty_annotation_images"] is not None)
    if has_empty:
        warnings.append("Verify that zero-annotation images are fully reviewed negatives, rather than unfinished labels.")

    return {
        "missing_partitions": missing,
        "counts": results,
        "note": "Empty annotation records are negative candidates only after label completion is verified.",
    }


def find_near_duplicates(rows, hashes, vectors, distance, max_pairs):
    """Find image pairs in different splits whose fingerprints are close.

    Returns (pairs, hit_limit). hit_limit is True if we stopped early
    because we reached max_pairs.
    """
    features = np.asarray(vectors)

    index = NearestNeighbors(radius=(distance + 0.01) / 64, metric="hamming", algorithm="ball_tree").fit(features)

    pairs = []
    for i, row in enumerate(rows):
        dists, neighbors = index.radius_neighbors(features[i:i + 1], sort_results=True)
        for d, j in zip(dists[0], neighbors[0]):
            j = int(j)
            if j <= i or rows[j]["split"] == row["split"] or hashes[j] == hashes[i]:
                continue
            if len(pairs) >= max_pairs:
                return pairs, True
            pairs.append({"image_a": row["output"], "image_b": rows[j]["output"], "distance": int(round(d * 64)),})
    return pairs, False


def audit_dataset(dataset, distance=6, rare_min=10, max_pairs=10000):
    if (type(distance) is not int or not 0 <= distance <= 64
            or type(rare_min) is not int or rare_min < 1
            or type(max_pairs) is not int or max_pairs < 1):
        raise ValueError("Distance must be 0–64; rare-min-images and max-pairs must be positive integers.")
    root = Path(dataset).resolve()
    manifest_path = resolve_dataset_path(root, "split_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if not isinstance(manifest, dict) or not isinstance(manifest.get("images"), list) or not manifest["images"]:
        raise ValueError("The split manifest needs a nonempty images array.")
    rows = manifest["images"]
    for row in rows:
        if (not isinstance(row, dict) or row.get("split") not in SPLITS
                or not isinstance(row.get("group_id"), str) or not row["group_id"].strip()
                or not isinstance(row.get("output"), str)):
            raise ValueError("Each image needs a valid split, nonempty group_id and output path.")
        _partition_path(root, row["split"], row["output"])
    if not isinstance(manifest.get("groups", {}), dict):
        raise ValueError("The split manifest groups field must be an object.")

    errors, warnings = [], []

    group_splits = defaultdict(set)
    for split, groups in manifest.get("groups", {}).items():
        if split not in SPLITS:
            raise ValueError(f"Unknown partition: {split}")
        if not isinstance(groups, list) or any(not isinstance(group, str) or not group.strip() for group in groups):
            raise ValueError(f"{split}: groups must be an array of nonempty strings.")
        for group in groups:
            group_splits[group].add(split)

    seen_paths = set()
    good_rows, hashes, vectors = [], [], []
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        path = _partition_path(root, row["split"], row["output"])
        if path in seen_paths:
            errors.append(f"Repeated output record: {row['output']}")
            continue
        seen_paths.add(path)

        try:
            digest = sha256_of(path)
            with Image.open(path) as image:
                image.load()
                if image.format not in {"JPEG", "PNG"}:
                    raise ValueError("Image contents must be JPEG or PNG.")
                bits = fingerprint(image)
        except (OSError, ValueError) as error:
            errors.append(f"Unreadable image {row['output']}: {error}")
            continue
        good_rows.append(row)
        hashes.append(digest)
        vectors.append(bits)

    # Every split needs images, and every image on disk should be in the manifest
    for split in SPLITS:
        expected = {_partition_path(root, split, r["output"]) for r in rows if r["split"] == split}
        if not expected:
            errors.append(f"Partition has no images: {split}")
        on_disk = set()
        for file in (root / split).rglob("*"):
            if file.is_file() and file.suffix.lower() in IMAGE_SUFFIXES:
                try:
                    on_disk.add(_partition_path(root, split, file.relative_to(root).as_posix()))
                except ValueError as error:
                    errors.append(str(error))
        for extra in sorted(on_disk - expected):
            errors.append(f"Image is absent from the manifest: {extra.relative_to(root).as_posix()}")

    # Groups that leak across splits
    overlaps = {g: sorted(s) for g, s in group_splits.items() if len(s) > 1}
    if overlaps:
        errors.append("Some source groups appear in multiple partitions.")

    # Byte-identical files in more than one split
    by_hash = defaultdict(list)
    for row, digest in zip(good_rows, hashes):
        by_hash[digest].append(row)
    exact = [{"sha256": digest, "images": matches} for digest, matches in by_hash.items() if len({r["split"] for r in matches}) > 1]
    if exact:
        errors.append(f"Found {len(exact)} identical-file groups across partitions.")

    # Visually similar images across splits
    near = []
    scan_complete = len(good_rows) == len(rows)
    if vectors:
        near, hit_limit = find_near_duplicates(good_rows, hashes, vectors, distance, max_pairs)
        if hit_limit:
            scan_complete = False
            errors.append("Candidate limit reached; similarity scan is incomplete. Increase --max-pairs.")
    if near:
        warnings.append("Visually review near-duplicate candidates; similar appearance alone does not establish duplication.")

    labels = label_counts(root, rows, rare_min, errors, warnings)

    if errors:
        status = "BLOCKED"
    elif labels["missing_partitions"]:
        status = "PENDING_LABELS"
    elif warnings:
        status = "NEEDS_REVIEW"
    else:
        status = "AUTOMATIC_CHECKS_PASSED"

    return {
        "status": status,
        "image_counts": dict(Counter(r["split"] for r in rows)),
        "group_overlaps": overlaps,
        "exact_cross_split_duplicates": exact,
        "near_duplicate_candidates": near,
        "similarity_scan_complete": scan_complete,
        "settings": {"distance": distance, "rare_min_images": rare_min, "max_pairs": max_pairs},
        "labels": labels,
        "errors": errors,
        "warnings": warnings,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--distance", type=int, default=6, help="max differing bits (of 64) for a near-duplicate candidate")
    parser.add_argument("--rare-min-images", type=int, default=10, help="warn when a class appears in fewer images than this")
    parser.add_argument("--max-pairs", type=int, default=10000, help="stop the similarity scan after this many candidate pairs")
    args = parser.parse_args(argv)

    try:
        if not 0 <= args.distance <= 64 or args.rare_min_images < 1 or args.max_pairs < 1:
            raise ValueError("Distance must be 0–64; rare-min-images and max-pairs must be positive.")

        report = audit_dataset(args.dataset, args.distance, args.rare_min_images, args.max_pairs)
        target = args.dataset / "audit_report.json"
        target.write_text(json.dumps(report, indent=2), encoding="utf-8")

        print(f"SCRUM-52: {report['status']}")
        print(f"Group overlaps: {len(report['group_overlaps'])}")
        print(f"Exact cross-split duplicate groups: {len(report['exact_cross_split_duplicates'])}")
        print(f"Near-duplicate candidates: {len(report['near_duplicate_candidates'])}")
        print(f"Missing COCO labels: {', '.join(report['labels']['missing_partitions']) or 'none'}")
        for message in report["errors"] + report["warnings"]:
            print(f"  {message}")
        print(f"Report: {target.resolve()}")

        return 0 if report["status"] == "AUTOMATIC_CHECKS_PASSED" else 1
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    raise SystemExit(main())
