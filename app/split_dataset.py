"""Compatibility entry point for raw image splitting.

should import ``app.dataset.splitting``. This command still accepts
``python -m app.split_dataset SOURCE OUTPUT [--seed N]``.
"""

from app.dataset.splitting import IMAGE_EXTENSIONS, main, split_dataset

__all__ = ["IMAGE_EXTENSIONS", "main", "split_dataset"]

if __name__ == "__main__":
    raise SystemExit(main())
