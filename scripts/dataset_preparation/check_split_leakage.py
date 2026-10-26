
from pathlib import Path
import argparse
import sys
from collections import defaultdict

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset_utils import (
    IMAGE_EXTENSIONS,
    sha256_file,
    dhash,
    hamming_distance,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Check dataset split leakage"
    )
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/processed"
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/leakage_report.csv"
    )
    parser.add_argument("--dhash-distance", type=int, default=4)
    args = parser.parse_args()

    if not 0 <= args.dhash_distance <= 64:
        parser.error("--dhash-distance must be between 0 and 64")

    images = []

    for split in ("train", "val", "test"):
        folder = args.dataset / "images" / split

        if folder.exists():
            for image in sorted(folder.rglob("*")):
                if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS:
                    images.append((split, image))

    if not images:
        print(f"No split images found in {args.dataset / 'images'}")
        raise SystemExit(1)

    report = []
    exact_hashes = defaultdict(list)
    image_hashes = []

    for split, image in images:
        digest = sha256_file(image)
        exact_hashes[digest].append((split, image))

        perceptual_hash = (
            dhash(image) if args.dhash_distance > 0 else None
        )

        image_hashes.append((split, image, perceptual_hash))

    # Exact duplicates across different splits.
    for entries in exact_hashes.values():
        distinct_splits = {split for split, _ in entries}

        if len(entries) > 1 and len(distinct_splits) > 1:
            for split, image in entries:
                matches = [
                    f"{other_split}:{other_image}"
                    for other_split, other_image in entries
                    if other_split != split
                ]

                report.append({
                    "issue": "EXACT_DUPLICATE_CROSS_SPLIT",
                    "image": str(image),
                    "split": split,
                    "match": " | ".join(matches),
                    "distance": 0,
                })

    # Approximate near-duplicate checks across splits.
    if args.dhash_distance > 0:
        for i in range(len(image_hashes)):
            split_a, image_a, hash_a = image_hashes[i]

            if hash_a is None:
                continue

            for j in range(i + 1, len(image_hashes)):
                split_b, image_b, hash_b = image_hashes[j]

                if split_a == split_b or hash_b is None:
                    continue

                distance = hamming_distance(hash_a, hash_b)

                if distance <= args.dhash_distance:
                    report.append({
                        "issue": "POSSIBLE_NEAR_DUPLICATE_CROSS_SPLIT",
                        "image": str(image_a),
                        "split": split_a,
                        "match": f"{split_b}:{image_b}",
                        "distance": distance,
                    })

    write_csv(
        args.report,
        report,
        ["issue", "image", "split", "match", "distance"],
    )

    exact_count = sum(
        row["issue"] == "EXACT_DUPLICATE_CROSS_SPLIT"
        for row in report
    )

    near_count = sum(
        row["issue"] == "POSSIBLE_NEAR_DUPLICATE_CROSS_SPLIT"
        for row in report
    )

    print("\n===== SPLIT LEAKAGE CHECK =====")
    print(f"Images checked: {len(images)}")
    print(f"Exact duplicate records: {exact_count}")
    print(f"Possible near-duplicate pairs: {near_count}")
    print(f"Report saved: {args.report}")

    if report:
        print("WARNING: Review the report before model training.")
        raise SystemExit(1)

    print("No cross-split duplicate candidates found.")


if __name__ == "__main__":
    main()
