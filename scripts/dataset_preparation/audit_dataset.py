
from pathlib import Path
import argparse
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset_utils import (
    list_images,
    label_for_image,
    read_yolo_label,
    image_info,
    sha256_file,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Audit construction-site dataset"
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT / "data/raw/images"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data/raw/labels"
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/audit_report.csv"
    )
    args = parser.parse_args()

    images = list_images(args.images)

    if not images:
        print(f"No images found in {args.images}")
        raise SystemExit(1)

    exact_hashes = {}
    report = []
    image_stems = set()

    for image_path in images:
        relative = image_path.relative_to(args.images)
        image_stems.add(relative.with_suffix("").as_posix())

        label_path = label_for_image(
            image_path, args.images, args.labels
        )

        info = image_info(image_path)
        boxes, errors = read_yolo_label(label_path)

        digest = sha256_file(image_path) if info["ok"] else ""

        duplicate = ""
        if digest:
            if digest in exact_hashes:
                duplicate = exact_hashes[digest]
            else:
                exact_hashes[digest] = relative.as_posix()

        flags = []

        if not info["ok"]:
            flags.append("CORRUPT_IMAGE")

        if not label_path.exists():
            flags.append("MISSING_LABEL")

        if errors:
            flags.append("INVALID_ANNOTATION")

        if label_path.exists() and not boxes and not errors:
            flags.append("EMPTY_LABEL_FILE")

        if duplicate:
            flags.append("EXACT_DUPLICATE")

        if info["ok"] and min(info["width"], info["height"]) < 128:
            flags.append("VERY_SMALL_IMAGE")

        report.append({
            "image": relative.as_posix(),
            "width": info["width"],
            "height": info["height"],
            "size_bytes": image_path.stat().st_size,
            "label_exists": label_path.exists(),
            "object_count": len(boxes),
            "annotation_errors": " | ".join(errors),
            "duplicate_of": duplicate,
            "audit_flags": "; ".join(flags) if flags else "OK",
        })

    label_stems = set()

    if args.labels.exists():
        for label in args.labels.rglob("*.txt"):
            label_stems.add(
                label.relative_to(args.labels)
                .with_suffix("").as_posix()
            )

    orphan_labels = sorted(label_stems - image_stems)

    for orphan in orphan_labels:
        report.append({
            "image": "",
            "width": "",
            "height": "",
            "size_bytes": "",
            "label_exists": "",
            "object_count": "",
            "annotation_errors": "",
            "duplicate_of": "",
            "audit_flags": f"ORPHAN_LABEL: {orphan}",
        })

    write_csv(args.report, report)

    empty_labels = [r["image"] for r in report if "EMPTY_LABEL_FILE" in r["audit_flags"]]

    print("\n===== DATASET AUDIT =====")
    print(f"Images audited: {len(images)}")
    print(f"Exact duplicate images: {sum(bool(r['duplicate_of']) for r in report)}")
    print(f"Missing labels: {sum('MISSING_LABEL' in r['audit_flags'] for r in report)}")
    print(f"Empty label files (0 annotations): {len(empty_labels)}")
    print(f"Invalid annotations: {sum(bool(r['annotation_errors']) for r in report)}")
    print(f"Orphan labels: {len(orphan_labels)}")
    print(f"Report saved: {args.report}")

    if empty_labels:
        print("\nEmpty label files (backgrounds):")
        for img in empty_labels[:10]:
            print(f"  - {img}")
        if len(empty_labels) > 10:
            print(f"  ... and {len(empty_labels) - 10} more")


if __name__ == "__main__":
    main()
