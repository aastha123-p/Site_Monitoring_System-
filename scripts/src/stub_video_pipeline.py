
import cv2
import json
import time
from pathlib import Path

# Import the stub detector from stub_detector.py
from stub_detector import detect


def draw_detections(frame, detections):
    """Draw bounding boxes, class names and confidence scores."""

    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        label = f"{class_name} {confidence:.2f}"

        # Draw bounding box
        cv2.rectangle(
            frame,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            2
        )

        # Draw label
        cv2.putText(
            frame,
            label,
            (int(x1), max(25, int(y1) - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    return frame


def process_video(
    input_video,
    output_video,
    summary_path,
    display=True,
    max_frames=None
):
    """Process video frames, detect objects and save annotated video."""

    input_video = Path(input_video).resolve()
    output_video = Path(output_video).resolve()
    summary_path = Path(summary_path).resolve()

    print("\nStarting Construction Site Monitoring System")
    print(f"Input video: {input_video}")

    if not input_video.exists():
        raise FileNotFoundError(
            f"Video not found: {input_video}\n"
            "Please check your video path."
        )

    output_video.parent.mkdir(parents=True, exist_ok=True)
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    # Open input video
    cap = cv2.VideoCapture(str(input_video))

    if not cap.isOpened():
        raise RuntimeError("Could not open the input video.")

    source_fps = cap.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video resolution: {width} x {height}")
    print(f"Source video FPS: {source_fps:.2f}")

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    writer = cv2.VideoWriter(
        str(output_video),
        fourcc,
        source_fps,
        (width, height)
    )

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(
            "Could not create output video. "
            "Try another output path or codec."
        )

    frame_count = 0
    total_detections = 0
    processing_start = time.perf_counter()
    average_fps = 0.0

    try:
        while True:
            # Read the next frame
            success, frame = cap.read()

            # Stop when the video ends
            if not success:
                print("\nEnd of video reached.")
                break

            frame_start = time.perf_counter()

            # Run stub detector
            detections = detect(frame)
            total_detections += len(detections)

            # Draw detections
            frame = draw_detections(frame, detections)

            # Calculate processing FPS for this frame
            elapsed = max(
                time.perf_counter() - frame_start,
                1e-9
            )
            current_fps = 1.0 / elapsed

            if frame_count == 0:
                average_fps = current_fps
            else:
                average_fps = (
                    0.9 * average_fps
                    + 0.1 * current_fps
                )

            # Draw FPS and frame number
            cv2.putText(
                frame,
                f"FPS: {average_fps:.1f}",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"Frame: {frame_count + 1}",
                (15, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )

            # Save annotated frame
            writer.write(frame)
            frame_count += 1

            # Display live preview
            if display:
                cv2.imshow(
                    "Construction Site Monitoring - Day 1",
                    frame
                )

                # Press Q to stop
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    print("Stopped by user.")
                    break

            # Optional quick test
            if (
                max_frames is not None
                and frame_count >= max_frames
            ):
                print("Maximum frame limit reached.")
                break

            # Print progress every 100 frames
            if frame_count % 100 == 0:
                print(
                    f"Processed {frame_count} frames | "
                    f"Detections: {total_detections} | "
                    f"FPS: {average_fps:.2f}"
                )

    finally:
        cap.release()
        writer.release()

        if display:
            cv2.destroyAllWindows()

    total_seconds = time.perf_counter() - processing_start

    # Create processing summary
    summary = {
        "input_video": str(input_video),
        "output_video": str(output_video),
        "frames_processed": frame_count,
        "total_stub_detections": total_detections,
        "processing_seconds": round(total_seconds, 3),
        "average_processing_fps": round(
            frame_count / total_seconds, 2
        ) if total_seconds > 0 else 0,
        "detector_mode": "stub"
    }

    # Save summary JSON
    with open(summary_path, "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n========== PROCESSING SUMMARY ==========")

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nAnnotated video saved to:")
    print(output_video)

    print("\nSummary saved to:")
    print(summary_path)

    return summary


if __name__ == "__main__":

    # Your terminal is currently in scripts/src.
    # These paths assume videos/ is in the project root.
    PROJECT_ROOT = Path(__file__).resolve().parents[2]

    INPUT_VIDEO = (
        PROJECT_ROOT / "videos" / "input" / "construction.mp4"
    )

    OUTPUT_VIDEO = (
        PROJECT_ROOT / "videos" / "output" / "day1_annotated.mp4"
    )

    SUMMARY_PATH = (
        PROJECT_ROOT / "videos" / "output" / "day1_summary.json"
    )

    try:
        process_video(
            input_video=INPUT_VIDEO,
            output_video=OUTPUT_VIDEO,
            summary_path=SUMMARY_PATH,
            display=True,
            max_frames=None
        )

    except Exception as error:
        print(f"\nERROR: {error}")
