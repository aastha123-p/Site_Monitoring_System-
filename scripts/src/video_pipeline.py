
import argparse
import csv
import json
import time
import numpy as np

from collections import Counter
from pathlib import Path

import cv2
import yaml

from detector import Detector
from tracker import Tracker
from analytics import Analytics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_value):
    path = Path(path_value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(config_path):
    config_path = resolve_path(config_path)

    if not config_path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    return config


def draw_detections(frame, detections):
    colors = {
        "person": (255, 200, 0),
        "hardhat": (0, 200, 0),
        "no_hardhat": (0, 0, 255),
        "safety_vest": (0, 255, 0),
        "no_safety_vest": (0, 0, 255),
        "machinery": (255, 0, 255),
        "crane": (255, 0, 255),
        "excavator": (255, 0, 255),
        "truck": (255, 150, 0),
        "safety_cone": (0, 165, 255),
    }

    for detection in detections:
        bbox = detection.get("bbox")
        if bbox is None:
            continue

        x1, y1, x2, y2 = map(int, bbox)
        class_name = detection.get("class_name", "unknown")
        confidence = float(detection.get("confidence", 0.0))
        track_id = detection.get("track_id")

        color = colors.get(class_name, (255, 255, 255))

        label = f"{class_name} {confidence:.2f}"
        if track_id is not None:
            label += f" ID:{track_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text_y = max(20, y1 - 8)
        cv2.putText(
            frame,
            label,
            (x1, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )


def detect_red_barrier_straps(frame):
    """
    Detects linear red caution tape/straps on the ground using HSV thresholding
    and Hough line transform.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red hue wraps around 0 and 180 in HSV
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([165, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(mask1, mask2)

    # Clean up noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)

    edges = cv2.Canny(red_mask, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=60, minLineLength=80, maxLineGap=25)

    detected_straps = []
    if lines is not None:
        for line in lines:
            line_pts = line[0] if len(line.shape) > 1 else line
            x1, y1, x2, y2 = map(int, line_pts[:4])
            detected_straps.append(((x1, y1), (x2, y2)))

    return detected_straps


def draw_dynamic_safety_hazards(frame, metrics, red_straps):
    """
    Draws dynamic machinery buffer circles, cone barricades, red barrier straps,
    and flashes hazard banners for proximity violations.
    """
    overlay = frame.copy()

    # 1. Draw detected red barrier straps
    for p1, p2 in red_straps:
        cv2.line(frame, p1, p2, (0, 0, 255), 3, cv2.LINE_AA)
        mid_x, mid_y = int((p1[0] + p2[0]) / 2), int((p1[1] + p2[1]) / 2)
        cv2.putText(
            frame,
            "RED BARRIER STRAP",
            (mid_x, mid_y - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 255),
            1,
            cv2.LINE_AA,
        )

    # 2. Draw cone barricade if formed
    dynamic_zones = metrics.get("dynamic_zones", {})
    if "cone_barricade" in dynamic_zones:
        cone_poly = dynamic_zones["cone_barricade"]
        pts_array = np.array(cone_poly, dtype=np.int32)
        cv2.fillPoly(overlay, [pts_array], (0, 140, 255))
        cv2.polylines(frame, [pts_array], isClosed=True, color=(0, 165, 255), thickness=2)
        if len(cone_poly) > 0:
            cx, cy = cone_poly[0]
            cv2.putText(
                frame,
                "CONE BARRICADE ZONE",
                (cx, max(20, cy - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 165, 255),
                2,
                cv2.LINE_AA,
            )

    # 3. Draw Heavy Machinery Buffer Halos (Dynamic moving circles)
    violations = metrics.get("proximity_violations", [])
    active_mach_violations = {v["mach_name"] for v in violations}

    for mach in metrics.get("machinery_entities", []):
        center = mach["center"]
        radius = mach["radius"]
        mach_name = mach["class_name"]

        # If a worker is inside buffer -> red alert halo; else semi-transparent orange/yellow
        if mach_name in active_mach_violations:
            halo_color = (0, 0, 255)  # Bright Red
            cv2.circle(overlay, center, radius, (0, 0, 200), -1)
        else:
            halo_color = (0, 215, 255)  # Amber / Yellow
            cv2.circle(overlay, center, radius, (0, 180, 200), -1)

        cv2.circle(frame, center, radius, halo_color, 2, cv2.LINE_AA)
        cv2.putText(
            frame,
            f"{mach_name.upper()} DANGER BUFFER",
            (center[0] - 60, center[1] - radius - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            halo_color,
            2,
            cv2.LINE_AA,
        )

    # 4. Draw warning lines connecting encroaching worker to machine
    for v in violations:
        p_pt = v["person_center"]
        m_pt = v["mach_center"]
        cv2.line(frame, p_pt, m_pt, (0, 0, 255), 2, cv2.LINE_AA)
        mid_pt = (int((p_pt[0] + m_pt[0]) / 2), int((p_pt[1] + m_pt[1]) / 2))
        cv2.putText(
            frame,
            f"PROXIMITY ALERT! ({v['distance_px']}px)",
            (mid_pt[0] - 40, mid_pt[1]),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # Blend translucent overlays
    cv2.addWeighted(overlay, 0.25, frame, 0.75, 0, frame)

    # 5. Flash warning banner at top for Proximity Hazard or Red Barrier Crowd Alert
    red_crowd = metrics.get("red_barrier_crowd_alert", False)
    red_people = metrics.get("red_barrier_people_count", 0)

    if violations and red_crowd:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 200), -1)
        warning_msg = f"DANGER: {len(violations)} Worker(s) Near Machinery | {red_people} Workers Inside Red Barrier (>4 Limit)!"
        cv2.putText(frame, warning_msg, (30, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    elif violations:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 200), -1)
        warning_msg = f"DANGER: {len(violations)} Worker(s) Too Close to Heavy Machinery!"
        cv2.putText(frame, warning_msg, (int(frame.shape[1] / 2) - 260, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    elif red_crowd:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 42), (0, 0, 200), -1)
        warning_msg = f"DANGER: {red_people} Workers Inside Red Barrier Strap Zone! (Limit: 4 max)"
        cv2.putText(frame, warning_msg, (int(frame.shape[1] / 2) - 280, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)


def draw_zones(frame, zones):
    for zone_name, polygon in zones.items():
        if zone_name == "cone_barricade":
            continue  # Handled dynamically

        points = [tuple(map(int, point)) for point in polygon]
        if len(points) < 3:
            continue

        for index in range(len(points)):
            cv2.line(
                frame,
                points[index],
                points[(index + 1) % len(points)],
                (0, 165, 255),
                2,
            )

        label_x, label_y = points[0]
        cv2.putText(
            frame,
            zone_name,
            (label_x, max(20, label_y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )


def draw_overlay(frame, metrics, processing_fps):
    lines = [
        f"Visible people: {metrics['visible_person_detections']}",
        f"Tracked people: {metrics['tracked_people']}",
        f"Entries: {metrics['entry_count']}",
        f"Exits: {metrics['exit_count']}",
        f"Proximity Alerts: {len(metrics.get('proximity_violations', []))}",
        f"Processing FPS: {processing_fps:.2f}",
    ]

    y = 28
    for line in lines:
        cv2.putText(
            frame,
            line,
            (15, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        y += 25


def write_csv(path, rows):
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as file:
            file.write("no_data\n")
        return

    fieldnames = list(dict.fromkeys(
        key for row in rows for key in row.keys()
    ))

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            serializable = {}
            for key, value in row.items():
                if isinstance(value, (dict, list, tuple)):
                    serializable[key] = json.dumps(value)
                else:
                    serializable[key] = value
            writer.writerow(serializable)


def process_video(config_path="config/video_config.yaml", input_video_path=None, max_frames=None):
    config = load_config(config_path)

    model_path = resolve_path(config.get("model_path", "models/construction_site_best.pt"))
    if not model_path.is_file():
        # Try fallback to project's best trained checkpoint
        candidate_models = [
            PROJECT_ROOT / "models" / "construction_site_best.pt",
            PROJECT_ROOT / "models" / "best.pt",
            PROJECT_ROOT / "yolov8n.pt",
        ]
        model_path = next((p for p in candidate_models if p.is_file()), model_path)

    if input_video_path:
        input_video = resolve_path(input_video_path)
    else:
        # Open file dialog window to choose video dynamically
        import tkinter as tk
        from tkinter import filedialog
        import sys

        root_tk = tk.Tk()
        root_tk.withdraw()
        root_tk.attributes("-topmost", True)

        initial_dir = str(PROJECT_ROOT / "videos" / "input")
        if not Path(initial_dir).exists():
            initial_dir = str(PROJECT_ROOT)

        print("Please select a video file from the file dialog window...")
        selected_file = filedialog.askopenfilename(
            title="Select a video file for the monitoring pipeline",
            initialdir=initial_dir,
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv"),
                ("All files", "*.*"),
            ],
        )
        root_tk.destroy()

        if selected_file:
            input_video = Path(selected_file).resolve()
        else:
            # Fallback to configured default if dialog is closed/canceled
            input_video = resolve_path(config.get("input_video", "videos/input/construction.mp4"))

    output_dir = resolve_path(config.get("output_dir", "videos/output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_video.is_file():
        raise FileNotFoundError(
            f"Input video not found: {input_video}\n"
            "Please select a valid video file."
        )

    line = config["entry_exit_line"]
    line_start = tuple(line["start"])
    line_end = tuple(line["end"])

    zones = {
        name: [tuple(point) for point in polygon]
        for name, polygon in config.get("zones", {}).items()
    }

    detector = Detector(
        model_path=model_path,
        confidence=config.get("confidence_threshold", 0.35),
        iou=config.get("iou_threshold", 0.45),
        image_size=config.get("image_size", 640),
        device=config.get("device", "cpu"),
        use_tracking=config.get("use_tracking", True),
        tracker_config=config.get("tracker_config", "bytetrack.yaml"),
    )

    tracker = Tracker()

    analytics = Analytics(
        line_start=line_start,
        line_end=line_end,
        zones=zones,
        crowd_limit=config.get("restricted_zone_person_limit", 3),
        machinery_proximity_pixels=config.get(
            "machinery_proximity_pixels", 150
        ),
        enable_ppe_alerts=config.get("enable_ppe_alerts", True),
    )

    capture = cv2.VideoCapture(str(input_video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_video}")

    video_fps = capture.get(cv2.CAP_PROP_FPS)
    if not video_fps or video_fps <= 0:
        video_fps = 25.0

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames_in_video = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    output_video = output_dir / f"{input_video.stem}_day3_annotated.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        video_fps,
        (width, height),
    )

    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create output video: {output_video}")

    detection_rows = []
    frame_index = 0
    class_counts = Counter()
    start_time = time.perf_counter()

    print(f"\nInput: {input_video}")
    print(f"Resolution: {width}x{height}")
    print(f"Video FPS: {video_fps:.2f}")
    print(f"Frames reported by video: {total_frames_in_video}")
    print("Processing started. Press Q to stop early.\n")

    try:
        while True:
            if max_frames is not None and frame_index >= max_frames:
                break

            success, frame = capture.read()
            if not success:
                break

            # Use actual YOLO tracking where enabled.
            if config.get("use_tracking", True):
                raw_detections = detector.track(frame)
            else:
                raw_detections = detector.detect(frame)

            detections = tracker.update(raw_detections)

            elapsed_video_time = frame_index / video_fps

            # Detect red caution tape/straps on the ground
            red_straps = detect_red_barrier_straps(frame)

            metrics = analytics.update(
                detections,
                timestamp=elapsed_video_time,
                frame_index=frame_index,
                red_straps=red_straps,
            )

            for detection in detections:
                class_name = detection.get("class_name", "unknown")
                class_counts[class_name] += 1

                x1, y1, x2, y2 = detection["bbox"]
                detection_rows.append({
                    "frame_index": frame_index,
                    "video_timestamp_seconds": round(
                        elapsed_video_time, 3
                    ),
                    "track_id": detection.get("track_id"),
                    "class_id": detection.get("class_id"),
                    "class_name": class_name,
                    "confidence": round(
                        float(detection.get("confidence", 0.0)), 4
                    ),
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                })

            annotated = frame.copy()

            draw_zones(annotated, zones)
            draw_dynamic_safety_hazards(annotated, metrics, red_straps)

            # Draw entry/exit line and label
            cv2.line(
                annotated,
                line_start,
                line_end,
                (0, 255, 255),
                3,
            )
            mid_x = int((line_start[0] + line_end[0]) / 2)
            mid_y = int((line_start[1] + line_end[1]) / 2)
            cv2.putText(
                annotated,
                "ENTRY / EXIT LINE",
                (mid_x + 8, mid_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            draw_detections(annotated, detections)

            elapsed_processing = time.perf_counter() - start_time
            processing_fps = (
                (frame_index + 1) / elapsed_processing
                if elapsed_processing > 0 else 0.0
            )

            draw_overlay(annotated, metrics, processing_fps)
            writer.write(annotated)

            frame_index += 1

            if frame_index % 50 == 0:
                print(
                    f"Processed {frame_index} frames | "
                    f"FPS: {processing_fps:.2f} | "
                    f"Detections in latest frame: {len(detections)}"
                )

            cv2.imshow("Construction Site Monitoring - Day 3", annotated)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("Stopped early by user.")
                break

    finally:
        capture.release()
        writer.release()
        cv2.destroyAllWindows()

    elapsed_processing = time.perf_counter() - start_time
    average_fps = (
        frame_index / elapsed_processing
        if elapsed_processing > 0 else 0.0
    )

    summary = analytics.get_summary(
        total_frames=frame_index,
        fps=average_fps,
    )
    summary.update({
        "input_video": str(input_video),
        "output_video": str(output_video),
        "video_fps": round(video_fps, 2),
        "video_width": width,
        "video_height": height,
        "elapsed_processing_seconds": round(elapsed_processing, 2),
        "detections_by_class": dict(class_counts),
        "unique_tracker_ids_seen": tracker.get_total_unique_tracks(),
        "frames_reported_by_input": total_frames_in_video,
        "frames_processed": frame_index,
    })

    write_csv(output_dir / "detections.csv", detection_rows)
    write_csv(output_dir / "events.csv", analytics.events)
    write_csv(output_dir / "alerts.csv", analytics.alerts)
    write_csv(output_dir / "occupancy_timeseries.csv",
              analytics.occupancy_history)

    with (output_dir / "day3_summary.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(summary, file, indent=2)

    print("\n--- DAY 3 RESULTS ---")
    print(json.dumps(summary, indent=2))
    print(f"\nAnnotated video saved to: {output_video}")
    print(f"Reports saved to: {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run real YOLO construction-site video analytics."
    )
    parser.add_argument(
        "--config",
        default="config/video_config.yaml",
        help="Path to video_config.yaml, relative to the project root.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a quick test.",
    )
    parser.add_argument(
        "--video",
        default=None,
        help="Optional path to input video file (bypasses file dialog if supplied).",
    )
    args = parser.parse_args()

    process_video(
        config_path=args.config,
        input_video_path=args.video,
        max_frames=args.max_frames,
    )
