"""
Convert 9-value polygon / oriented bounding box labels to standard 5-value YOLO bounding boxes.

Standard YOLO object detection format:
    class_id x_center y_center width height (5 values)

Roboflow / polygon format (4 corners):
    class_id x1 y1 x2 y2 x3 y3 x4 y4 (9 values)
"""

from pathlib import Path
import argparse


def convert_line_to_bbox(line: str) -> str:
    """
    Convert a single 9-value polygon line to a 5-value bounding box line.
    If the line already has 5 values, returns the line unchanged (with coordinates clamped).
    """
    parts = line.strip().split()
    if not parts:
        return ""

    class_id = parts[0]

    # Standard YOLO format: 5 values (class, xc, yc, w, h)
    if len(parts) == 5:
        xc, yc, w, h = map(float, parts[1:])
        # Clamp slightly to ensure values stay within [0.0, 1.0]
        xc = max(0.0, min(1.0, xc))
        yc = max(0.0, min(1.0, yc))
        w = max(0.0, min(1.0, w))
        h = max(0.0, min(1.0, h))
        # Ensure box doesn't overflow [0, 1] boundaries due to float precision
        half_w = w / 2.0
        half_h = h / 2.0
        if xc - half_w < 0.0:
            xc = half_w
        if xc + half_w > 1.0:
            xc = 1.0 - half_w
        if yc - half_h < 0.0:
            yc = half_h
        if yc + half_h > 1.0:
            yc = 1.0 - half_h
        return f"{int(float(class_id))} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    # Polygon format: 9 values (class, x1, y1, x2, y2, x3, y3, x4, y4)
    if len(parts) == 9:
        coords = list(map(float, parts[1:]))
        xs = coords[0::2]
        ys = coords[1::2]

        # Calculate bounding rectangle
        xmin = max(0.0, min(1.0, min(xs)))
        xmax = max(0.0, min(1.0, max(xs)))
        ymin = max(0.0, min(1.0, min(ys)))
        ymax = max(0.0, min(1.0, max(ys)))

        w = xmax - xmin
        h = ymax - ymin
        xc = (xmin + xmax) / 2.0
        yc = (ymin + ymax) / 2.0

        return f"{int(float(class_id))} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}"

    return line


def convert_label_file(file_path: Path) -> int:
    """
    Convert a single label file in place.
    Returns the count of converted 9-value lines.
    """
    text = file_path.read_text(encoding="utf-8")
    lines = text.strip().splitlines()
    if not lines:
        return 0

    converted_lines = []
    converted_count = 0

    for line in lines:
        line_clean = line.strip()
        if not line_clean:
            continue
        parts = line_clean.split()
        if len(parts) == 9:
            converted_count += 1
        converted_line = convert_line_to_bbox(line_clean)
        if converted_line:
            converted_lines.append(converted_line)

    if converted_lines:
        file_path.write_text("\n".join(converted_lines) + "\n", encoding="utf-8")
    else:
        file_path.write_text("", encoding="utf-8")

    return converted_count


def convert_all_labels(labels_dir: Path) -> dict:
    """
    Convert all label files in a directory.
    """
    files = sorted(labels_dir.glob("*.txt"))
    total_files = len(files)
    affected_files = 0
    total_converted_lines = 0

    for file_path in files:
        lines_converted = convert_label_file(file_path)
        if lines_converted > 0:
            affected_files += 1
            total_converted_lines += lines_converted

    return {
        "total_files": total_files,
        "affected_files": affected_files,
        "converted_lines": total_converted_lines,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert 9-value polygon labels to standard 5-value YOLO bboxes"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "data" / "raw" / "labels",
        help="Path to labels directory",
    )
    args = parser.parse_args()

    print(f"Converting label files in: {args.labels}")
    stats = convert_all_labels(args.labels)
    print(f"Scanned {stats['total_files']} files.")
    print(f"Converted {stats['converted_lines']} polygon boxes across {stats['affected_files']} files.")
    print("Done!")
