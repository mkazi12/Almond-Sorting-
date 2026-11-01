from pathlib import Path
from datasets import load_dataset

dataset = load_dataset(
    "Project-AgML/almond_harvest_2021",
    split="train",
)

root = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "source_data"
)
root.mkdir(parents=True, exist_ok=False)

# Create 10 artificial groups of 5 images for testing the splitter.
for i, row in enumerate(dataset):
    group = root / f"demo_group_{i // 5 + 1:02d}"
    group.mkdir(exist_ok=True)

    row["image"].convert("RGB").save(
        group / f"image_{i:03d}.png"
    )

print(f"Saved {len(dataset)} images to {root}")