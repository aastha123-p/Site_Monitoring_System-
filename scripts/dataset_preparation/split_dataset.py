
from pathlib import Path
import argparse
import random
import shutil
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset_utils import (
    CLASS_NAMES,
    list_images,
    label_for_image,
    image_info,
    sha256_file,
    dhash,
    hamming_distance,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Split YOLO dataset into train, val and test"
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT / "data/raw/images"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data/raw/labels"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "data/processed"
    )
    parser.add_argument("--train", type=float, default=0.70)
    parser.add_argument("--val", type=float, default=0.20)
    parser.add_argument("--test", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dhash-distance", type=int, default=4)
    parser.add_argument(
        "--clean", action="store_true",
        help="Delete existing output images and labels before splitting"
    )
    args = parser.parse_args()

    ratios = [args.train, args.val, args.test]

    if any(r < 0 for r in ratios) or abs(sum(ratios) - 1.0) > 1e-6:
        parser.error("Split ratios must sum to 1.0")

    if not 0 <= args.dhash_distance <= 64:
        parser.error("--dhash-distance must be between 0 and 64")

    images = list_images(args.images)

    if not images:
        print(f"No images found in {args.images}")
        raise SystemExit(1)

    # Union-find keeps duplicate images in the same group.
    parent = list(range(len(images)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        root_a = find(a)
        root_b = find(b)

        if root_a != root_b:
            parent[root_b] = root_a

    exact_hashes = {}
    perceptual_hashes = []

    for i, image in enumerate(images):
        info = image_info(image)

        digest = sha256_file(image) if info["ok"] else ""

        if digest:
            if digest in exact_hashes:
                union(i, exact_hashes[digest])
            else:
                exact_hashes[digest] = i

        perceptual_hashes.append(
            dhash(image) if args.dhash_distance > 0 else None
        )

    # Compare perceptual hashes to find likely near-duplicates.
    # This pairwise comparison is suitable for small/medium datasets.
    if args.dhash_distance > 0:
        for i in range(len(images)):
            if perceptual_hashes[i] is None:
                continue

            for j in range(i + 1, len(images)):
                if perceptual_hashes[j] is None:
                    continue

                distance = hamming_distance(
                    perceptual_hashes[i],
                    perceptual_hashes[j],
                )

                if distance <= args.dhash_distance:
                    union(i, j)

    groups = defaultdict(list)

    for i, image in enumerate(images):
        groups[find(i)].append(image)

    group_list = list(groups.values())
    random.Random(args.seed).shuffle(group_list)

    split_names = ["train", "val", "test"]
    targets = {
        "train": len(images) * args.train,
        "val": len(images) * args.val,
        "test": len(images) * args.test,
    }

    assigned = {name: [] for name in split_names}
    counts = {name: 0 for name in split_names}

    # Largest groups first; every group stays in one split.
    group_list.sort(key=len, reverse=True)

    for group in group_list:
        available_splits = [
            name for name in split_names
            if targets[name] > 0
        ]

        if not available_splits:
            available_splits = split_names

        chosen = min(
            available_splits,
            key=lambda name: (
                counts[name] / targets[name]
                if targets[name] else float("inf")
            ),
        )

        assigned[chosen].extend(group)
        counts[chosen] += len(group)

    # Prevent accidental mixing with a previous split.
    output_images = args.output / "images"
    output_labels = args.output / "labels"

    if args.clean:
        for folder in (output_images, output_labels):
            if folder.exists():
                shutil.rmtree(folder)

    for split in split_names:
        (output_images / split).mkdir(parents=True, exist_ok=True)
        (output_labels / split).mkdir(parents=True, exist_ok=True)

    manifest = []

    for split, split_images in assigned.items():
        for image in split_images:
            relative = image.relative_to(args.images)

            label = label_for_image(
                image, args.images, args.labels
            )

            image_dest = output_images / split / relative
            label_dest = (
                output_labels / split / relative
            ).with_suffix(".txt")

            image_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, image_dest)

            if label.exists():
                label_dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(label, label_dest)
                label_status = "COPIED"
            else:
                label_status = "MISSING"

            manifest.append({
                "image": relative.as_posix(),
                "split": split,
                "label_status": label_status,
            })

    # Create the Ultralytics dataset configuration.
    data_yaml = [
        f"path: {args.output.resolve().as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "",
        "names:",
    ]

    for class_id, name in CLASS_NAMES.items():
        data_yaml.append(f"  {class_id}: {name}")

    (args.output / "data.yaml").write_text(
        "\n".join(data_yaml) + "\n",
        encoding="utf-8",
    )

    write_csv(ROOT / "reports/split_manifest.csv", manifest)

    print("\n===== DATASET SPLIT =====")
    print(f"Total images: {len(images)}")

    for split in split_names:
        print(
            f"{split}: {len(assigned[split])} images "
            f"({len(assigned[split]) / len(images):.1%})"
        )

    print(f"Output directory: {args.output}")

    missing_labels_count = sum(1 for m in manifest if m["label_status"] == "MISSING")
    if missing_labels_count > 0:
        print(f"\nWARNING: {missing_labels_count} images have MISSING label files:")
        for m in manifest:
            if m["label_status"] == "MISSING":
                print(f"  - {m['image']} (in {m['split']} split)")
    else:
        print("All images have corresponding label files.")

    print("\nNext: validate each split and check split leakage.")


if __name__ == "__main__":
    main()
