# pyrefly: ignore[missing-import]

from pathlib import Path
import sys

from ultralytics import YOLO


EXPECTED_CLASSES = {
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


class Detector:
    def __init__(
        self,
        model_path,
        confidence=0.35,
        iou=0.45,
        image_size=640,
        device="cpu",
        use_tracking=True,
        tracker_config="bytetrack.yaml",
    ):
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"YOLO weights not found: {self.model_path.resolve()}"
            )

        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = int(image_size)
        self.device = device
        self.use_tracking = bool(use_tracking)
        self.tracker_config = tracker_config

        # Load weights only once.
        self.model = YOLO(str(self.model_path))

        self.names = self.model.names
        print(f"Loaded model: {self.model_path.resolve()}")
        print(f"Model class names: {self.names}")

        self._validate_class_names()

    def _validate_class_names(self):
        """Warn if the model's class mapping differs from the config."""
        actual = {
            int(class_id): str(name)
            for class_id, name in self.names.items()
        }

        if actual != EXPECTED_CLASSES:
            print("\nWARNING: Model class mapping differs from expected mapping.")
            print("Expected:", EXPECTED_CLASSES)
            print("Actual:  ", actual)
            print(
                "Check the training data and YAML before trusting "
                "the PPE or equipment labels.\n"
            )

    def detect(self, frame):
        """
        Run YOLO inference on a single frame.

        Returns:
            List of dictionaries:
            {
                "bbox": (x1, y1, x2, y2),
                "confidence": float,
                "class_id": int,
                "class_name": str,
                "track_id": int | None
            }
        """
        results = self.model.predict(
            source=frame,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        return self._convert_results(results[0], include_track_id=False)

    def track(self, frame):
        """Run persistent ByteTrack tracking on a video frame."""
        if not self.use_tracking:
            detections = self.detect(frame)

            # Detection-only mode does not generate track IDs.
            return detections

        results = self.model.track(
            source=frame,
            persist=True,
            tracker=self.tracker_config,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            verbose=False,
        )

        return self._convert_results(results[0], include_track_id=True)

    def _convert_results(self, result, include_track_id):
        detections = []

        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return detections

        xyxy_values = boxes.xyxy.cpu().tolist()
        confidence_values = boxes.conf.cpu().tolist()
        class_values = boxes.cls.cpu().tolist()

        track_values = None
        if include_track_id and boxes.id is not None:
            track_values = boxes.id.int().cpu().tolist()

        for index, bbox in enumerate(xyxy_values):
            class_id = int(class_values[index])
            class_name = str(
                self.names.get(class_id, f"class_{class_id}")
            )

            track_id = None
            if track_values is not None:
                track_id = int(track_values[index])

            x1, y1, x2, y2 = map(int, bbox)

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "confidence": float(confidence_values[index]),
                "class_id": class_id,
                "class_name": class_name,
                "track_id": track_id,
            })

        return detections


if __name__ == "__main__":
    import cv2
    import numpy as np

    root = Path(__file__).resolve().parents[2]
    candidate_models = [
        root / "models" / "construction_site_best.pt",
        root / "models" / "best.pt",
        root / "yolov8n.pt",
    ]
    model_path = next((p for p in candidate_models if p.exists()), candidate_models[0])

    print("Initializing Detector...")
    detector = Detector(model_path=model_path, confidence=0.35, use_tracking=False)

    import tkinter as tk
    from tkinter import filedialog

    # Open system native file dialog to select an image
    root_tk = tk.Tk()
    root_tk.withdraw()
    root_tk.attributes("-topmost", True)

    initial_dir = str(root / "data" / "raw" / "images")
    if not Path(initial_dir).exists():
        initial_dir = str(root)

    print("Please select an image from the file dialog window...")
    selected_file = filedialog.askopenfilename(
        title="Select an image for site monitoring detection",
        initialdir=initial_dir,
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*"),
        ],
    )
    root_tk.destroy()

    if not selected_file:
        print("No file selected. Exiting.")
        sys.exit(0)

    test_image_path = Path(selected_file)
    print(f"Loaded image: {test_image_path.resolve()}")
    frame = cv2.imread(str(test_image_path))
    if frame is None:
        raise ValueError(f"Unable to read image from {test_image_path}")

    detections = detector.detect(frame)
    print(f"\nExecution finished successfully. Found {len(detections)} detection(s):")
    for i, det in enumerate(detections, 1):
        print(f"  [{i}] Class: {det['class_name']} ({det['class_id']}), Conf: {det['confidence']:.2f}, BBox: {det['bbox']}")

    # Distinct colors for classes in BGR format
    class_colors = {
        0: (255, 128, 0),    # person: Sky/Dodger blue
        1: (0, 200, 0),      # hardhat: Green (compliant)
        2: (0, 0, 255),      # no_hardhat: Bright Red (violation)
        3: (0, 215, 255),    # safety_vest: Gold/Yellow (compliant)
        4: (0, 69, 255),     # no_safety_vest: Orange Red (violation)
        5: (200, 100, 200),  # machinery: Purple
        6: (255, 0, 255),    # crane: Magenta
        7: (0, 140, 255),    # excavator: Deep Orange
        8: (255, 204, 0),    # truck: Cyan-blue
        9: (0, 165, 255),    # safety_cone: Safety Orange
    }

    # Draw detections with class-specific colors
    annotated = frame.copy()
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        class_id = det["class_id"]
        color = class_colors.get(class_id, (0, 255, 0))
        label = f"{det['class_name']} {det['confidence']:.2f}"

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # Draw filled background badge for text readability
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        badge_y1 = max(0, y1 - text_h - 6)
        badge_y2 = y1
        badge_x2 = min(annotated.shape[1] - 1, x1 + text_w + 6)
        cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

        # Draw label text in contrasting color
        cv2.putText(
            annotated,
            label,
            (x1 + 3, badge_y2 - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    output_dir = root / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = f"{test_image_path.stem}_detected.jpg"
    output_path = output_dir / output_filename
    cv2.imwrite(str(output_path), annotated)
    print(f"\nAnnotated image saved to: {output_path.resolve()}")

    # Display image preview in an interactive window
    try:
        cv2.imshow(f"Detection Results - {test_image_path.name}", annotated)
        print("Press any key in the image window to close...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    except Exception:
        pass


