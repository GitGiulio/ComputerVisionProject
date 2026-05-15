import argparse
import os
import shutil


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}

def load_index(path: str) -> set[str]:
    """Load a set of number-strings from a plain-text index file."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Index file not found: {path}")
    with open(path) as f:
        return {line.strip() for line in f if line.strip()}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def transfer(src: str, dst: str, move: bool) -> None:
    ensure_dir(os.path.dirname(dst))
    if move:
        shutil.move(src, dst)
    else:
        shutil.copy2(src, dst)

def restore_split(
    source_root: str,
    dest_root: str,
    index_dir: str,
    move: bool,
) -> None:
    # Load the 4 index sets
    indices: dict[str, dict[str, set[str]]] = {}
    for split in ("val", "test"):
        indices[split] = {}
        for animal in ("Cat", "Dog"):
            fname = os.path.join(index_dir, f"{split}_{animal.lower()}s.txt")
            indices[split][animal] = load_index(fname)
            print(f"  Loaded {len(indices[split][animal]):>6} {split}/{animal} indices from {fname}")

    action = "Moving" if move else "Copying"
    print(f"\n{action} images …\n")

    counters: dict[str, int] = {"train": 0, "val": 0, "test": 0}

    for animal in ("Cat", "Dog"):
        src_folder = os.path.join(source_root, animal)
        if not os.path.isdir(src_folder):
            print(f"[WARN] Source folder not found, skipping: {src_folder}")
            continue

        val_set  = indices["val"][animal]
        test_set = indices["test"][animal]

        for fname in os.listdir(src_folder):
            stem, ext = os.path.splitext(fname)
            if ext.lower() not in IMAGE_EXTENSIONS:
                continue

            src_path = os.path.join(src_folder, fname)

            if stem in test_set:
                split = "test"
            elif stem in val_set:
                split = "val"
            else:
                split = "train"

            dst_path = os.path.join(dest_root, split, animal, fname)
            transfer(src_path, dst_path, move)
            counters[split] += 1

    print("\nSummary:")
    for split, count in counters.items():
        print(f"  {split:>5}: {count} images")
    print(f"\nDone. Output written to: {dest_root}")


def main():
    parser = argparse.ArgumentParser(
        description="Restore train/val/test split from flat Cat+Dog folders using index files."
    )
    parser.add_argument(
        "--source",
        default="/Data/Cats_Dogs",
    )
    parser.add_argument(
        "--dest",
        default="/Data/Cats_Dogs_Restored",
    )
    parser.add_argument(
        "--index-dir",
        default=".",
    )
    parser.add_argument(
        "--move",
        action="store_true",
    )
    args = parser.parse_args()

    restore_split(
        source_root=args.source,
        dest_root=args.dest,
        index_dir=args.index_dir,
        move=args.move,
    )


if __name__ == "__main__":
    main()