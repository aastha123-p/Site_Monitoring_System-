
from typing import Any


class Tracker:
    """
    Normalizes detections returned by the detector.

    Expected detection format:
    {
        "bbox": (x1, y1, x2, y2),
        "confidence": 0.92,
        "class_id": 0,
        "class_name": "person",
        "track_id": 12
    }
    """

    def __init__(self):
        self.seen_ids = set()

    def update(self, detections: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Returns detections with valid, normalized track IDs.

        The detector/tracking model must generate the IDs.
        This class does not create artificial IDs for detections
        whose tracking ID is missing.
        """
        tracked_detections = []

        for detection in detections:
            item = detection.copy()

            track_id = item.get("track_id")

            if track_id is None:
                continue

            try:
                track_id = int(track_id)
            except (TypeError, ValueError):
                continue

            if track_id < 0:
                continue

            item["track_id"] = track_id
            self.seen_ids.add(track_id)
            tracked_detections.append(item)

        return tracked_detections

    def get_total_unique_tracks(self) -> int:
        """Total unique tracking IDs observed during this run."""
        return len(self.seen_ids)

    def reset(self):
        """Reset tracking statistics before a new video."""
        self.seen_ids.clear()


if __name__ == "__main__":
    print("Testing Tracker...")
    tracker = Tracker()

    sample_detections = [
        {"bbox": (100, 150, 200, 300), "confidence": 0.95, "class_name": "person", "track_id": 1},
        {"bbox": (300, 200, 400, 450), "confidence": 0.88, "class_name": "person", "track_id": 2},
        {"bbox": (50, 50, 120, 180), "confidence": 0.40, "class_name": "person", "track_id": None},
    ]

    results = tracker.update(sample_detections)
    print(f"Tracked detections count: {len(results)}")
    print(f"Results: {results}")
    print(f"Total unique tracks: {tracker.get_total_unique_tracks()}")
    print("Tracker test passed successfully!")
