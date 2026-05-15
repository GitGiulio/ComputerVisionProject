"""
save_split_indices.py
---------------------
Reads the existing split structure:

    /Data/Cats_Dogs_Splitted/
        train/
            Cat/   *.jpg
            Dog/   *.jpg
        val/
            Cat/   *.jpg
            Dog/   *.jpg
        test/
            Cat/   *.jpg
            Dog/   *.jpg

and saves 4 index files (one per subset x class) that list the bare
image numbers (without extension) found in val/ and test/.

Output files (written next to this script, or to --output-dir):
    val_cats.txt
    val_dogs.txt
    test_cats.txt
    test_dogs.txt
"""

import argparse
import os


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}


def collect_numbers(folder: str) -> list[str]:
    """Return sorted list of stem numbers for every image file in *folder*."""
    if not os.path.isdir(folder):
        print(f"  [WARN] folder not found, skipping: {folder}")
        return []
    numbers = []
    for fname in os.listdir(folder):
        stem, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            numbers.append(stem)
    # Sort numerically when possible, otherwise lexicographically
    try:
        numbers.sort(key=lambda x: int(x))
    except ValueError:
        numbers.sort()
    return numbers


def save_index(numbers: list[str], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(numbers))
    print(f"  Saved {len(numbers):>6} entries → {path}")


def main():
    parser = argparse.ArgumentParser(description="Save val/test image indices to text files.")
    parser.add_argument(
        "--root",
        default="/Data/Cats_Dogs_Splitted",
        help="Root directory that contains train/, val/, test/ subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        default=".",
        help="Directory where the 4 index text files will be written.",
    )
    args = parser.parse_args()

    root = args.root
    out = args.output_dir

    for split in ("val", "test"):
        for animal in ("Cat", "Dog"):
            folder = os.path.join(root, split, animal)
            numbers = collect_numbers(folder)
            filename = f"{split}_{animal.lower()}s.txt"
            save_index(numbers, os.path.join(out, filename))

    print("\nDone. The train/ split is implicitly everything NOT listed above.")


if __name__ == "__main__":
    main()