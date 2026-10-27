
"""Day 1: Stub detector test."""

import time
import cv2
import numpy as np


def detect(frame):
    """Generate a synthetic bounding box for pipeline testing."""

    height, width = frame.shape[:2]

    box_width = max(40, int(width * 0.10))
    box_height = max(60, int(height * 0.25))

    max_x = max(1, width - box_width)
    x1 = int((time.monotonic() * 80) % max_x)

    y1 = int(height * 0.35)
    x2 = min(width - 1, x1 + box_width)
    y2 = min(height - 1, y1 + box_height)

    return [
        {
            "bbox": (x1, y1, x2, y2),
            "class_name": "stub_person",
            "confidence": 0.99,
        }
    ]


if __name__ == "__main__":
    print("Testing stub detector...")

    # Create a blank test frame: height 720, width 1280
    test_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    detections = detect(test_frame)

    print("Detector executed successfully.")
    print("Number of detections:", len(detections))
    print("Detection results:", detections)

    # Draw the fake bounding box
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]

        cv2.rectangle(
            test_frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2,
        )

        cv2.putText(
            test_frame,
            detection["class_name"],
            (x1, max(20, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    cv2.imwrite("detector_test.jpg", test_frame)
    print("Test image saved as detector_test.jpg")
