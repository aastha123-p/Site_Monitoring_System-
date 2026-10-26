from pathlib import Path
import argparse
import sys

from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_MODEL = ROOT / "models" / "construction_site_best.pt"
DEFAULT_OUTPUT = ROOT / "runs" / "testing"


# ============================================================
# 2. TESTING CONFIGURATION
# ============================================================

IMAGE_SIZE = 640
CONFIDENCE = 0.35
IOU_THRESHOLD = 0.50
DEVICE = "cpu"

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
    ".jpg", ".jpeg", ".png", ".bmp",
    ".webp", ".tif", ".tiff"
}

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv"
}


# ============================================================
# 3. CHECK INPUT PATH
# ============================================================

def validate_input(input_path):
    """Check whether the input is a supported image, video, or folder."""

    input_path = Path(input_path)

    if not input_path.exists():
        raise FileNotFoundError(f"Input does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() not in (
            IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
        ):
            raise ValueError(
                f"Unsupported file type: {input_path.suffix}"
            )

    elif input_path.is_dir():
        supported = [
            path for path in input_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in (
                IMAGE_EXTENSIONS | VIDEO_EXTENSIONS
            )
        ]

        if not supported:
            raise ValueError(
                f"No supported images or videos found in: {input_path}"
            )

    return input_path


# ============================================================
# 4. RUN MODEL TESTING
# ============================================================

def test_model(model_path, input_path, output_dir):
    """Run YOLO predictions and save annotated results."""

    model_path = Path(model_path)
    input_path = validate_input(input_path)
    output_dir = Path(output_dir)

    if not model_path.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {model_path}\n"
            "Run train_baseline.py first."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n========== MODEL TESTING ==========")
    print(f"Model: {model_path}")
    print(f"Input: {input_path}")
    print(f"Output directory: {output_dir}")
    print(f"Confidence threshold: {CONFIDENCE}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Device: {DEVICE}")
    print("===================================\n")

    model = YOLO(str(model_path))

    results = model.predict(
        source=str(input_path),
        imgsz=IMAGE_SIZE,
        conf=CONFIDENCE,
        iou=IOU_THRESHOLD,
        device=DEVICE,
        save=True,
        save_txt=True,
        save_conf=True,
        project=str(output_dir),
        name="predictions",
        exist_ok=True,
        verbose=False,
    )

    total_images = 0
    total_detections = 0
    class_counts = {name: 0 for name in CLASS_NAMES.values()}

    print("========== DETECTION RESULTS ==========")

    for result in results:
        total_images += 1
        boxes = result.boxes
        detection_count = len(boxes) if boxes is not None else 0
        total_detections += detection_count

        print(f"\nFile: {Path(result.path).name}")
        print(f"Objects detected: {detection_count}")

        if boxes is None:
            continue

        for box in boxes:
            class_id = int(box.cls[0].item())
            confidence = float(box.conf[0].item())
            class_name = CLASS_NAMES.get(
                class_id, f"class_{class_id}"
            )

            class_counts[class_name] = (
                class_counts.get(class_name, 0) + 1
            )

            print(
                f"  {class_name}: "
                f"confidence={confidence:.3f}"
            )

    print("\n========== TEST SUMMARY ==========")
    print(f"Files processed: {total_images}")
    print(f"Total detections: {total_detections}")

    print("\nDetections by class:")

    for class_name, count in class_counts.items():
        print(f"  {class_name}: {count}")

    print(f"\nAnnotated results saved under: {output_dir}")
    print("==================================\n")


# ============================================================
# 5. COMMAND-LINE ARGUMENTS
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Test the trained construction-site YOLO model."
    )

    parser.add_argument(
        "--source",
        required=True,
        help="Path to an image, video, or folder of images/videos."
    )

    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL),
        help="Path to the trained .pt model."
    )

    parser.add_argument(
        "--output",
        default=str(DEFAULT_OUTPUT),
        help="Directory for test outputs."
    )

    args = parser.parse_args()

    try:
        test_model(
            model_path=args.model,
            input_path=args.source,
            output_dir=args.output,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
