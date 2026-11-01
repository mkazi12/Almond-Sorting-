""" audit an existing grouped train/valid/test split.

Nothing in the dataset is modified. Run from the project root:

    python -m app.audit_dataset dataset_demo

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
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.fft import dctn
from sklearn.neighbors import NearestNeighbors

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



def read_coco_split(root, split, rows, class_names, errors):

    path = root / split / "_annotations.coco.json"
    data = json.loads(path.read_text(encoding="utf-8-sig"))

    # categories
    local_names = {c["id"]: c["name"] for c in data["categories"]}
    if len(local_names) != len(data["categories"]) or len(set(local_names.values())) != len(local_names):
        errors.append(f"{split}: duplicate class IDs or names.")
    for class_id, name in local_names.items():
        if class_id in class_names and class_names[class_id] != name:
            errors.append(f"Class ID {class_id} has different names across partitions.")
        if name in class_names.values() and class_names.get(class_id) != name:
            errors.append(f"Class {name!r} uses different IDs across partitions.")
        class_names[class_id] = name

    # images
    image_paths = {}
    for image in data["images"]:
        file = (root / split / image["file_name"].replace("\\", "/")).resolve()
        if not file.is_relative_to(root / split):
            errors.append(f"{split}: COCO image path leaves its partition.")
        if image["id"] in image_paths or file in image_paths.values():
            errors.append(f"{split}: duplicate COCO image ID or filename.")
        image_paths[image["id"]] = file

    in_manifest = {(root / r["output"]).resolve() for r in rows if r["split"] == split}
    if set(image_paths.values()) != in_manifest:
        errors.append(f"{split}: COCO image records do not match the split manifest.")

    # annotations
    counts = {image_id: Counter() for image_id in image_paths}
    seen_ids = set()
    for ann in data["annotations"]:
        if ann["id"] in seen_ids:
            errors.append(f"{split}: repeated annotation ID {ann['id']}.")
        seen_ids.add(ann["id"])

        if ann["image_id"] not in counts or ann["category_id"] not in local_names or "bbox" not in ann:
            errors.append(f"{split}: annotation has an unknown image/class or missing box.")
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
    root = Path(dataset).resolve()
    manifest = json.loads((root / "split_manifest.json").read_text(encoding="utf-8-sig"))
    rows = manifest["images"]
    if not rows:
        raise ValueError("The split manifest contains no images.")

    errors, warnings = [], []

    group_splits = defaultdict(set)
    for split, groups in manifest.get("groups", {}).items():
        if split not in SPLITS:
            raise ValueError(f"Unknown partition: {split}")
        for group in groups:
            group_splits[group].add(split)

    seen_paths = set()
    good_rows, hashes, vectors = [], [], []
    for row in rows:
        if row["split"] not in SPLITS or not row["group_id"]:
            raise ValueError("Each image needs a valid split and nonempty group_id.")
        group_splits[row["group_id"]].add(row["split"])

        path = (root / row["output"]).resolve()
        if not path.is_relative_to(root / row["split"]):
            raise ValueError(f"Image path is outside its recorded partition: {row['output']}")
        if path in seen_paths:
            errors.append(f"Repeated output record: {row['output']}")
            continue
        seen_paths.add(path)

        try:
            digest = sha256_of(path)
            with Image.open(path) as image:
                image.load()
                bits = fingerprint(image)
        except (OSError, ValueError) as error:
            errors.append(f"Unreadable image {row['output']}: {error}")
            continue
        good_rows.append(row)
        hashes.append(digest)
        vectors.append(bits)

    # Every split needs images, and every image on disk should be in the manifest
    for split in SPLITS:
        expected = {(root / r["output"]).resolve() for r in rows if r["split"] == split}
        if not expected:
            errors.append(f"Partition has no images: {split}")
        on_disk = {p.resolve() for p in (root / split).rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES}
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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--distance", type=int, default=6, help="max differing bits (of 64) for a near-duplicate candidate")
    parser.add_argument("--rare-min-images", type=int, default=10, help="warn when a class appears in fewer images than this")
    parser.add_argument("--max-pairs", type=int, default=10000, help="stop the similarity scan after this many candidate pairs")
    args = parser.parse_args()

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