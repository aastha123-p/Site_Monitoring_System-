
from pathlib import Path
import argparse
import sys

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
        description="Validate construction-site YOLO dataset"
    )
    parser.add_argument(
        "--images", type=Path, default=ROOT / "data/raw/images"
    )
    parser.add_argument(
        "--labels", type=Path, default=ROOT / "data/raw/labels"
    )
    parser.add_argument(
        "--report", type=Path,
        default=ROOT / "reports/validation_report.csv"
    )
    args = parser.parse_args()

    images = list_images(args.images)

    if not images:
        print(f"ERROR: No images found in {args.images}")
        raise SystemExit(1)

    report = []

    for image_path in images:
        label_path = label_for_image(
            image_path, args.images, args.labels
        )

        info = image_info(image_path)
        valid_boxes, errors = read_yolo_label(label_path)

        status = "PASS"

        if not info["ok"]:
            status = "FAIL"

        if not label_path.exists():
            status = "FAIL"

        if errors:
            status = "FAIL"

        report.append({
            "image": str(image_path.relative_to(args.images)),
            "label": str(label_path),
            "image_valid": info["ok"],
            "width": info["width"],
            "height": info["height"],
            "label_exists": label_path.exists(),
            "valid_objects": len(valid_boxes),
            "errors": " | ".join(errors),
            "image_error": info["error"],
            "status": status,
        })

    # Check for label files that have no matching image
    label_files = sorted(
        path for path in args.labels.rglob("*.txt")
        if path.is_file()
    )
    image_rel_stems = {
        img.relative_to(args.images).with_suffix("").as_posix()
        for img in images
    }

    labels_without_image = []
    for lbl in label_files:
        lbl_rel_stem = lbl.relative_to(args.labels).with_suffix("").as_posix()
        if lbl_rel_stem not in image_rel_stems:
            labels_without_image.append(lbl)
            valid_boxes, errors = read_yolo_label(lbl)
            err_msg = " | ".join(errors) if errors else "No matching image found"
            report.append({
                "image": f"[NO IMAGE] {lbl_rel_stem}.jpg",
                "label": str(lbl),
                "image_valid": False,
                "width": 0,
                "height": 0,
                "label_exists": True,
                "valid_objects": len(valid_boxes),
                "errors": err_msg,
                "image_error": f"No image file found for label: {lbl.name}",
                "status": "FAIL",
            })

    write_csv(args.report, report)

    try:
        dataset_path = args.images.parent.relative_to(ROOT).as_posix()
    except ValueError:
        dataset_path = args.images.parent.as_posix()

    try:
        report_display = args.report.relative_to(ROOT).as_posix()
    except ValueError:
        report_display = args.report.as_posix()

    total_images = len(images)
    image_reports = [r for r in report if not r["image"].startswith("[NO IMAGE]")]
    valid_images = sum(1 for r in image_reports if r["image_valid"])
    corrupted_images = total_images - valid_images
    missing_labels = sum(1 for r in image_reports if not r["label_exists"])
    labels_with_no_image = len(labels_without_image)
    files_with_objects = sum(
        1 for r in image_reports
        if r["label_exists"] and r["valid_objects"] > 0 and not r["errors"]
    )
    empty_labels = sum(
        1 for r in image_reports
        if r["label_exists"] and r["valid_objects"] == 0 and not r["errors"]
    )
    invalid_annotations = sum(1 for r in image_reports if r["label_exists"] and r["errors"])

    class_ids_str = (
        f"{min(CLASS_NAMES.keys())}-{max(CLASS_NAMES.keys())}"
        if CLASS_NAMES else "0-9"
    )

    print("========== DATASET VALIDATION ==========\n")
    print(f"Dataset path: {dataset_path}")
    print(f"Total images found: {total_images}")
    print(f"Valid images: {valid_images}")
    print(f"Corrupted images: {corrupted_images}")
    print(f"Images missing a .txt label file: {missing_labels}")
    print(f"Images with bounding box annotations: {files_with_objects}")
    print(f"Empty label files (0 objects / backgrounds): {empty_labels}")
    print(f"Invalid annotation files: {invalid_annotations}")
    print(f"Orphan label files (label exists, but NO image found): {labels_with_no_image}\n")
    print(f"Class IDs checked: {class_ids_str}\n")

    # If any label files are missing, list them explicitly
    missing_list = [r["image"] for r in image_reports if not r["label_exists"]]
    if missing_list:
        print("Missing label files:")
        for name in missing_list[:10]:
            print(f"  - {name} (no corresponding .txt)")
        if len(missing_list) > 10:
            print(f"  ... and {len(missing_list) - 10} more")
        print()

    # If any label files are empty, list them explicitly
    empty_list = [r["image"] for r in image_reports if r["label_exists"] and r["valid_objects"] == 0 and not r["errors"]]
    if empty_list:
        print(f"Empty label files (0 objects / background): {len(empty_list)}")
        for name in empty_list[:10]:
            print(f"  - {name}")
        if len(empty_list) > 10:
            print(f"  ... and {len(empty_list) - 10} more")
        print()

    # If any label files have no matching image, list them
    if labels_without_image:
        print("Labels with no matching image file:")
        for lbl in labels_without_image[:10]:
            print(f"  - {lbl.name}")
        if len(labels_without_image) > 10:
            print(f"  ... and {len(labels_without_image) - 10} more")
        print()

    print("Validation report saved:")
    print(f"{report_display}\n")
    print("========================================")


if __name__ == "__main__":
    main()
