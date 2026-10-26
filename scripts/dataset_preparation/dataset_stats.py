
from pathlib import Path
import argparse
import json
import sys
from collections import Counter

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.dataset_utils import (
    CLASS_NAMES,
    list_images,
    label_for_image,
    read_yolo_label,
    image_info,
    write_csv,
)


def main():
    parser = argparse.ArgumentParser(
        description="Generate construction dataset statistics"
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT / "data/raw/images"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data/raw/labels"
    )
    parser.add_argument(
        "--out-dir", type=Path, default=ROOT / "reports"
    )
    args = parser.parse_args()

    images = list_images(args.images)

    if not images:
        print(f"No images found in {args.images}")
        raise SystemExit(1)

    object_counts = Counter()
    image_counts = Counter()
    dimensions = Counter()

    total_objects = 0
    missing_labels = 0
    invalid_labels = 0
    unreadable_images = 0
    image_rows = []

    for image_path in images:
        info = image_info(image_path)

        if not info["ok"]:
            unreadable_images += 1
        else:
            dimensions[f"{info['width']}x{info['height']}"] += 1

        label_path = label_for_image(
            image_path, args.images, args.labels
        )
        boxes, errors = read_yolo_label(label_path)

        if not label_path.exists():
            missing_labels += 1

        if errors:
            invalid_labels += 1

        per_image = Counter(box["class_id"] for box in boxes)

        for class_id, count in per_image.items():
            object_counts[class_id] += count
            image_counts[class_id] += 1

        total_objects += len(boxes)

        image_rows.append({
            "image": image_path.relative_to(args.images).as_posix(),
            "width": info["width"],
            "height": info["height"],
            "object_count": len(boxes),
            "classes_present": ", ".join(
                CLASS_NAMES[c] for c in sorted(per_image)
            ),
            "annotation_errors": " | ".join(errors),
        })

    class_rows = []

    for class_id, class_name in CLASS_NAMES.items():
        count = object_counts[class_id]

        class_rows.append({
            "class_id": class_id,
            "class_name": class_name,
            "object_instances": count,
            "images_containing_class": image_counts[class_id],
            "object_percentage": round(
                count / total_objects * 100, 2
            ) if total_objects else 0,
        })

    summary = {
        "total_images": len(images),
        "total_valid_objects": total_objects,
        "missing_label_files": missing_labels,
        "labels_with_errors": invalid_labels,
        "unreadable_images": unreadable_images,
        "image_dimensions": dict(dimensions),
        "class_distribution": {
            CLASS_NAMES[c]: object_counts[c]
            for c in CLASS_NAMES
        },
        "images_containing_class": {
            CLASS_NAMES[c]: image_counts[c]
            for c in CLASS_NAMES
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)

    (args.out_dir / "dataset_stats.json").write_text(
        json.dumps(summary, indent=4),
        encoding="utf-8",
    )

    write_csv(args.out_dir / "class_distribution.csv", class_rows)
    write_csv(args.out_dir / "per_image_stats.csv", image_rows)

    print("\n===== DATASET STATISTICS =====")
    print(json.dumps(summary, indent=4))
    print(f"\nReports saved in: {args.out_dir}")


if __name__ == "__main__":
    main()
