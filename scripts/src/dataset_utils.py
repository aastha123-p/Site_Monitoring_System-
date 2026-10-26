
from pathlib import Path
import csv
import hashlib
import math

from PIL import Image


# ============================================================
# PROJECT CONFIGURATION
# ============================================================

CLASS_NAMES = {
    0: "person",
    1: "hardhat",
    2: "no_hardhat",
    3: "safety_vest",
    4: "no_safety_vest",
    5: "machinery",
    6: "crane",
    7: "excavator",
    8: "truck",
    9: "safety_cone",
}

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
    ".tif",
    ".tiff",
    ".jfif",
}


# ============================================================
# 1. FIND IMAGE FILES
# ============================================================

def list_images(images_dir):
    """
    Recursively find all supported image files in a directory.

    Returns:
        A sorted list of image paths.
    """
    images_dir = Path(images_dir)

    if not images_dir.exists():
        return []

    return sorted(
        path
        for path in images_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ============================================================
# 2. FIND THE MATCHING YOLO LABEL FILE
# ============================================================

def label_for_image(image_path, images_dir, labels_dir):
    """
    Find the matching YOLO label file for an image.

    Example:
        images/site_001.jpg
        labels/site_001.txt

    Relative subdirectories are preserved.
    """
    image_path = Path(image_path)
    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)

    relative_path = image_path.relative_to(images_dir)

    return (labels_dir / relative_path).with_suffix(".txt")


# ============================================================
# 3. READ AND VALIDATE YOLO ANNOTATIONS
# ============================================================

def read_yolo_label(label_path):
    """
    Read a YOLO-format annotation file.

    Each non-empty line must contain:
        class_id x_center y_center width height

    Coordinates must be normalized between 0 and 1.

    Returns:
        boxes: List of valid annotation dictionaries.
        errors: List of annotation error messages.
    """
    label_path = Path(label_path)

    boxes = []
    errors = []

    if not label_path.exists():
        return boxes, [f"Missing label file: {label_path.name}"]

    try:
        with label_path.open("r", encoding="utf-8-sig") as file:
            lines = file.readlines()
    except (OSError, UnicodeError) as exc:
        return boxes, [f"Cannot read label file: {exc}"]

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()

        # Ignore blank lines.
        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(
                f"Line {line_number}: expected 5 values, "
                f"found {len(parts)}"
            )
            continue

        try:
            class_value = float(parts[0])
            x_center, y_center, width, height = map(
                float, parts[1:]
            )
        except ValueError:
            errors.append(
                f"Line {line_number}: non-numeric annotation value"
            )
            continue

        values = [
            class_value,
            x_center,
            y_center,
            width,
            height,
        ]

        if not all(math.isfinite(value) for value in values):
            errors.append(
                f"Line {line_number}: NaN or infinite value"
            )
            continue

        # Class IDs must be integer values from 0 to 9.
        if not class_value.is_integer():
            errors.append(
                f"Line {line_number}: class ID must be an integer"
            )
            continue

        class_id = int(class_value)

        if class_id not in CLASS_NAMES:
            errors.append(
                f"Line {line_number}: invalid class ID {class_id}"
            )
            continue

        # Center coordinates must be normalized.
        if not (0 <= x_center <= 1 and 0 <= y_center <= 1):
            errors.append(
                f"Line {line_number}: center coordinates "
                f"must be between 0 and 1"
            )
            continue

        # Width and height must be greater than zero.
        if not (0 < width <= 1 and 0 < height <= 1):
            errors.append(
                f"Line {line_number}: width and height "
                f"must be greater than 0 and at most 1"
            )
            continue

        # The bounding box must fit inside the image.
        if (
            x_center - width / 2 < -1e-4
            or x_center + width / 2 > 1 + 1e-4
            or y_center - height / 2 < -1e-4
            or y_center + height / 2 > 1 + 1e-4
        ):
            errors.append(
                f"Line {line_number}: bounding box extends "
                f"outside the image"
            )
            continue

        boxes.append({
            "class_id": class_id,
            "class_name": CLASS_NAMES[class_id],
            "x_center": x_center,
            "y_center": y_center,
            "width": width,
            "height": height,
            "line_number": line_number,
        })

    return boxes, errors


# ============================================================
# 4. GET IMAGE INFORMATION
# ============================================================

def image_info(image_path):
    """
    Return image dimensions and basic image information.

    Returns:
        A dictionary containing ok, width, height, format, and error.
    """
    image_path = Path(image_path)

    try:
        with Image.open(image_path) as image:
            image.verify()

        # Reopen after verify(), because verify() invalidates
        # normal image decoding for that opened object.
        with Image.open(image_path) as image:
            width, height = image.size
            image_format = image.format

        return {
            "ok": True,
            "width": width,
            "height": height,
            "format": image_format,
            "error": None,
        }
    except Exception as exc:
        return {
            "ok": False,
            "width": 0,
            "height": 0,
            "format": None,
            "error": str(exc),
        }


# ============================================================
# 5. CALCULATE SHA-256 IMAGE HASH
# ============================================================

def sha256_file(file_path):
    """
    Calculate a SHA-256 hash of a file.

    Useful for detecting exact duplicate files.
    """
    file_path = Path(file_path)
    sha256 = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# 6. CALCULATE PERCEPTUAL DIFFERENCE HASH
# ============================================================

def dhash(image_path, hash_size=8):
    """
    Calculate a difference hash (dHash) for an image.

    Similar images often produce similar hashes, even when
    their file contents differ.

    Returns:
        An integer representing the perceptual hash.
    """
    image_path = Path(image_path)

    with Image.open(image_path) as image:
        image = image.convert("L")
        image = image.resize(
            (hash_size + 1, hash_size)
        )

        pixels = list(image.getdata())

    differences = []

    for row in range(hash_size):
        row_start = row * (hash_size + 1)

        for column in range(hash_size):
            left_pixel = pixels[row_start + column]
            right_pixel = pixels[row_start + column + 1]

            differences.append(left_pixel > right_pixel)

    result = 0

    for difference in differences:
        result = (result << 1) | int(difference)

    return result


# ============================================================
# 7. COMPARE TWO PERCEPTUAL HASHES
# ============================================================

def hamming_distance(hash1, hash2):
    """
    Count the number of different bits between two hashes.

    Smaller distance generally means greater visual similarity.
    """
    return (int(hash1) ^ int(hash2)).bit_count()


# ============================================================
# 8. WRITE A CSV REPORT
# ============================================================

def write_csv(output_path, rows, fieldnames=None):
    """
    Write dictionaries to a CSV file.

    Args:
        output_path: Destination CSV file path.
        rows: List of dictionaries.
        fieldnames: Optional column order.

    Creates the output directory if it does not exist.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = list(rows)

    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8-sig"
    ) as file:
        if not fieldnames:
            return

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(rows)
