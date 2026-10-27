
from collections import defaultdict


class Tracker:
    def __init__(self):
        self.seen_ids = set()
        self.class_unique_tracks = defaultdict(set)
        self.class_detection_counts = defaultdict(int)

    def update(self, detections):
        tracked_detections = []

        for detection in detections:
            item = detection.copy()
            track_id = item.get("track_id")
            class_name = item.get("class_name", "unknown")

            # Track total detection instances seen across frames
            self.class_detection_counts[class_name] += 1

            if track_id is not None:
                try:
                    track_id = int(track_id)
                    if track_id < 0:
                        track_id = None
                except (TypeError, ValueError):
                    track_id = None

            item["track_id"] = track_id

            if track_id is not None:
                self.seen_ids.add(track_id)
                self.class_unique_tracks[class_name].add(track_id)

            # Keep detections even if they have no track ID.
            # PPE and machinery detections are still useful for alerts.
            tracked_detections.append(item)

        return tracked_detections

    def get_total_unique_tracks(self):
        return len(self.seen_ids)

    def get_class_counts(self):
        """Returns dict mapping class_name -> total detections across video."""
        return dict(self.class_detection_counts)

    def get_class_unique_tracks(self):
        """Returns dict mapping class_name -> number of unique tracked instances."""
        return {cls: len(ids) for cls, ids in self.class_unique_tracks.items()}

    def reset(self):
        self.seen_ids.clear()
        self.class_unique_tracks.clear()
        self.class_detection_counts.clear()


if __name__ == "__main__":
    import sys
    import time
    import cv2
    from pathlib import Path
    import tkinter as tk
    from tkinter import filedialog

    # Locate project root and detector
    root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(root / "scripts" / "src"))

    try:
        from detector import Detector
    except ImportError:
        from scripts.src.detector import Detector

    # File manager dialog to choose a video
    root_tk = tk.Tk()
    root_tk.withdraw()
    root_tk.attributes("-topmost", True)

    initial_dir = str(root / "videos" / "input")
    if not Path(initial_dir).exists():
        initial_dir = str(root)

    print("Please select a video file from the file dialog window...")
    selected_video = filedialog.askopenfilename(
        title="Select a video for real-time tracking",
        initialdir=initial_dir,
        filetypes=[
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
            ("All files", "*.*"),
        ],
    )
    root_tk.destroy()

    if not selected_video:
        print("No video selected. Exiting.")
        sys.exit(0)

    video_path = Path(selected_video)
    print(f"Loaded video: {video_path.resolve()}")

    # Candidate weights
    candidate_models = [
        root / "models" / "construction_site_best.pt",
        root / "models" / "best.pt",
        root / "yolov8n.pt",
    ]
    model_path = next((p for p in candidate_models if p.exists()), candidate_models[0])

    print("Initializing Detector with ByteTrack tracking...")
    detector = Detector(
        model_path=model_path,
        confidence=0.35,
        iou=0.45,
        use_tracking=True,
        tracker_config="bytetrack.yaml",
    )
    tracker = Tracker()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    output_dir = root / "videos" / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video_path = output_dir / f"{video_path.stem}_tracked.mp4"

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    class_colors = {
        0: (255, 128, 0),    # person: Dodger blue
        1: (0, 200, 0),      # hardhat: Green
        2: (0, 0, 255),      # no_hardhat: Red
        3: (0, 215, 255),    # safety_vest: Yellow/Gold
        4: (0, 69, 255),     # no_safety_vest: Orange Red
        5: (200, 100, 200),  # machinery: Purple
        6: (255, 0, 255),    # crane: Magenta
        7: (0, 140, 255),    # excavator: Deep Orange
        8: (255, 204, 0),    # truck: Cyan-blue
        9: (0, 165, 255),    # safety_cone: Safety Orange
    }

    print(f"\nProcessing: {width}x{height} @ {fps:.1f} fps ({total_frames} frames)")
    print("Press 'q' in the preview window to stop processing.\n")

    frame_idx = 0
    prev_time = time.time()
    current_fps = fps

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            now = time.time()
            time_diff = now - prev_time
            prev_time = now
            if time_diff > 0:
                current_fps = 0.9 * current_fps + 0.1 * (1.0 / time_diff)
            # Run tracking detection
            raw_detections = detector.track(frame)
            tracked_detections = tracker.update(raw_detections)

            # Draw detections
            for det in tracked_detections:
                x1, y1, x2, y2 = det["bbox"]
                class_id = det.get("class_id", -1)
                track_id = det.get("track_id")
                color = class_colors.get(class_id, (0, 255, 0))

                track_label = f"ID:{track_id} " if track_id is not None else ""
                label = f"{track_label}{det['class_name']} {det['confidence']:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                badge_y1 = max(0, y1 - text_h - 6)
                badge_y2 = y1
                badge_x2 = min(width - 1, x1 + text_w + 6)
                cv2.rectangle(frame, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

                cv2.putText(
                    frame,
                    label,
                    (x1 + 3, badge_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Draw HUD overlay with tracking stats and FPS
            hud_text = f"FPS: {current_fps:.1f} | Frame: {frame_idx}/{total_frames} | Unique Tracks: {tracker.get_total_unique_tracks()}"
            cv2.rectangle(frame, (10, 10), (560, 45), (30, 30, 30), -1)
            cv2.putText(frame, hud_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            writer.write(frame)

            # Show live preview
            cv2.imshow("Real-Time Tracking (Press 'q' to Quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped by user.")
                break

            if frame_idx % 25 == 0 or frame_idx == total_frames:
                progress = (frame_idx / total_frames * 100) if total_frames > 0 else 0
                print(f"Frame {frame_idx}/{total_frames} ({progress:.1f}%) - Tracks seen: {tracker.get_total_unique_tracks()}")

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    print(f"\nProcessing complete!")
    print(f"Total processed frames: {frame_idx}")
    print(f"Total unique objects tracked: {tracker.get_total_unique_tracks()}")

    # Print comprehensive site monitoring report
    class_detections = tracker.get_class_counts()
    class_unique = tracker.get_class_unique_tracks()

    # Categorize classes
    entities = ["person", "truck", "crane", "excavator", "machinery", "safety_cone"]
    compliance_gear = ["hardhat", "safety_vest"]
    violations = ["no_hardhat", "no_safety_vest"]

    video_duration_sec = frame_idx / fps if fps > 0 else 0

    print("\n" + "=" * 80)
    print("                    AI SITE SAFETY & MONITORING REPORT")
    print("=" * 80)
    print(f"Video File        : {video_path.name}")
    print(f"Processed Frames  : {frame_idx} / {total_frames} ({video_duration_sec:.1f}s)")
    print(f"Total Unique IDs  : {tracker.get_total_unique_tracks()}")
    print("-" * 80)
    print(f"{'Category / Class':<22} | {'Unique Count':<14} | {'Total Detections':<16} | {'Est. Presence/Duration'}")
    print("=" * 80)

    # 1. Personnel & Equipment
    print("  [ PERSONNEL & SITE EQUIPMENT ]")
    for cls in entities:
        if cls in class_detections or cls in class_unique:
            uniq = class_unique.get(cls, 0)
            detections = class_detections.get(cls, 0)
            sec = detections / fps if fps > 0 else 0
            print(f"  • {cls:<18} | {uniq:<14} | {detections:<16} | {sec:.1f}s")

    # 2. PPE Compliance
    print("\n  [ PPE COMPLIANCE GEAR ]")
    for cls in compliance_gear:
        if cls in class_detections or cls in class_unique:
            uniq = class_unique.get(cls, 0)
            detections = class_detections.get(cls, 0)
            sec = detections / fps if fps > 0 else 0
            print(f"  • {cls:<18} | {uniq:<14} | {detections:<16} | {sec:.1f}s (Compliant)")

    # 3. PPE Violations
    print("\n  [ SAFETY HAZARDS & VIOLATIONS ]")
    violation_count = 0
    for cls in violations:
        if cls in class_detections or cls in class_unique:
            uniq = class_unique.get(cls, 0)
            detections = class_detections.get(cls, 0)
            sec = detections / fps if fps > 0 else 0
            violation_count += uniq
            print(f"  ! {cls:<18} | {uniq:<14} | {detections:<16} | {sec:.1f}s (VIOLATION ⚠️)")

    if violation_count == 0:
        print("  • No PPE violations detected!")

    print("=" * 80)
    print(f"Tracked video saved to: {output_video_path.resolve()}\n")


