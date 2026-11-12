"""Create reproducible synthetic images for exercising the raw splitter.

Run explicitly from the project root. This module never downloads data or writes
files on import. Synthetic group names do not establish real source independence,
and these unlabeled images are not RF-DETR training data.
"""

import argparse
import random
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw


def create_demo_data(output: str | Path, *, groups: int = 10,
                     images_per_group: int = 5, seed: int = 1) -> Path:
    """Write a new raw-image tree; refuse to overwrite an existing location."""
    if type(groups) is not int or groups < 3:
        raise ValueError("groups must be an integer of at least 3.")
    if type(images_per_group) is not int or images_per_group < 1:
        raise ValueError("images_per_group must be a positive integer.")
    if type(seed) is not int:
        raise ValueError("seed must be an integer.")
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise ValueError(f"Output already exists: {destination}. Choose a new folder.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    with tempfile.TemporaryDirectory(prefix="dinos_demo_", dir=destination.parent) as temp:
        staged = Path(temp) / "images"
        staged.mkdir()
        for group in range(groups):
            folder = staged / f"synthetic_group_{group + 1:02d}"
            folder.mkdir()
            for index in range(images_per_group):
                image = Image.new("RGB", (96, 64), tuple(rng.randrange(256) for _ in range(3)))
                draw = ImageDraw.Draw(image)
                for _ in range(12):
                    x, y = rng.randrange(80), rng.randrange(48)
                    color = tuple(rng.randrange(256) for _ in range(3))
                    draw.ellipse((x, y, x + 12, y + 12), fill=color)
                image.save(folder / f"image_{index:03d}.png")
        staged.rename(destination)
    return destination


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--groups", type=int, default=10)
    parser.add_argument("--images-per-group", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    try:
        path = create_demo_data(args.output, groups=args.groups,
                                images_per_group=args.images_per_group, seed=args.seed)
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
    print(f"Saved {args.groups * args.images_per_group} synthetic, unlabeled images to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
