"""Compatibility entry point for grouped dataset audits.

should import ``app.dataset.audit``. Existing
``python -m app.audit_dataset DATASET`` options remain supported.
"""

from app.dataset.audit import (
    IMAGE_SUFFIXES,
    SPLITS,
    audit_dataset,
    find_near_duplicates,
    fingerprint,
    label_counts,
    main,
    read_coco_split,
    sha256_of,
)

__all__ = [
    "IMAGE_SUFFIXES", "SPLITS", "audit_dataset", "find_near_duplicates",
    "fingerprint", "label_counts", "main", "read_coco_split", "sha256_of",
]

if __name__ == "__main__":
    raise SystemExit(main())
