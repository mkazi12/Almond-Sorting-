"""Split raw images by source group. Run: python -m app.split_dataset SOURCE OUTPUT."""

import argparse
import json
import shutil
import tempfile
from  pathlib import Path
from sklearn.model_selection import train_test_split

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}

def split_dataset(source, dest, seed=1):
    source, dest, = Path(source).resolve(), Path(dest).resolve()
    if not source.is_dir():
        raise ValueError(f"Source folder does not exist: {source}")
    if dest.exists() or dest.is_relative_to(source):
        raise ValueError(f"Choose new output folder outside of source folder")
    groups = sorted(path for path in source.iterdir() if path.is_dir())
    
    if len(groups) < 3:
        raise ValueError("Source must contain at least three independant batch/recording folders")
    if any(path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS for path in source.iterdir()):
            raise ValueError("Put every source image inside its batch/recording folder first.")
        
    images = {}
    
    for group in groups:
        paths = sorted(path for path in group.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
        if not paths:
            raise ValueError(f"No JPG/PNG images found in group: {group.name}")
        if group.is_symlink() or any(not path.resolve().is_relative_to(group) for path in paths):
            raise ValueError(f"Group contains a link to an external path: {group.name}")
        images[group] = paths
        
    train_count = min(max(1, round(len(groups)*.8)), len(groups) - 2)
    valid_count = min(max(1, round(len(groups)*.1)), len(groups) - 1)
    train, remaining = train_test_split(groups, train_size=train_count, random_state=seed)
    valid, test = train_test_split(remaining, train_size=valid_count, random_state=seed)
    
    partitions = {"train": train, "valid": valid, "test": test}

    manifest = {"seed": seed, "source_root": str(source), "groups": {split:[group.name for group in selected] for split, selected in partitions.items()}, "images": []}
    
    dest.parent.mkdir(parents=True, exist_ok = True)
    
    with tempfile.TemporaryDirectory(prefix="image_split_", dir = dest.parent) as temp:
        staged = Path(temp)/"dataset"
        for split, selected in partitions.items():
            (staged/split).mkdir(parents=True)
            for group in selected:
                for image in images[group]:
                    name = f"{len(manifest['images']) + 1:08d}_{image.name}"
                    shutil.copy2(image, staged/split/name)
                    manifest["images"].append({"source": image.relative_to(source).as_posix(), "group_id": group.name, "split": split, "output": f"{split}/{name}"})
        (staged / "split_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        staged.rename(dest)
    return manifest
    

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing one subfolder per batch/recording.")
    parser.add_argument("dest", type=Path, help="New folder for train/valid/test images.")
    parser.add_argument("--seed", type=int, default=1)
    
    args = parser.parse_args()
    
    try:
        manifest = split_dataset(args.source, args.dest, args.seed)
        total = len(manifest["images"])
        for split, groups in manifest["groups"].items():
            count = sum(image["split"] == split for image in manifest["images"])
            print(f"{split}: {len(groups)} groups, {count} images ({count / total:.1%})")
        print(f"Saved images and split_manifest.json to {args.dest.resolve()}")
    except (OSError, ValueError) as error:
        parser.exit(2, f"Error: {error}\n")
        
        
if __name__ == "__main__":
    main()