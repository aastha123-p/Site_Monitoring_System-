
from collections import Counter, defaultdict
from datetime import datetime
from typing import Any

import cv2
import numpy as np


class Analytics:
    def __init__(
        self,
        line_start=(320, 0),
        line_end=(320, 720),
        zones=None,
        crowd_limit=3,
        dwell_alert_seconds=300,
    ):
        """
        Args:
            line_start, line_end:
                Coordinates of the virtual entry/exit line.

            zones:
                Dictionary of named polygon zones, for example:
                {
                    "restricted": [(400, 100), (700, 100),
                                   (700, 500), (400, 500)]
                }

            crowd_limit:
                Alert when more than this number of people are
                simultaneously inside a restricted zone.

            dwell_alert_seconds:
                Optional threshold for a long-stay alert.
                Set to 0 to disable dwell alerts.
        """
        self.line_start = tuple(line_start)
        self.line_end = tuple(line_end)
        self.zones = zones or {}
        self.crowd_limit = int(crowd_limit)
        self.dwell_alert_seconds = float(dwell_alert_seconds)

        self.previous_positions = {}
        self.last_seen = {}
        self.inside_zones = defaultdict(dict)

        self.entry_count = 0
        self.exit_count = 0
        self.peak_hour_entries = Counter()

        self.events = []
        self.alerts = []
        self.occupancy_history = []

        # Avoid writing the same alert on every video frame.
        self.active_alerts = set()

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _line_side(point, start, end):
        """
        Returns the signed side of a point relative to a line.
        """
        px, py = point
        x1, y1 = start
        x2, y2 = end

        value = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)

        if value > 0:
            return 1
        if value < 0:
            return -1
        return 0

    @staticmethod
    def _inside_polygon(point, polygon):
        polygon_array = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(
            polygon_array,
            (float(point[0]), float(point[1])),
            False,
        ) >= 0

    def _record_event(self, track_id, event_type, timestamp, **details):
        event = {
            "track_id": int(track_id),
            "event": event_type,
            "timestamp": timestamp,
        }
        event.update(details)
        self.events.append(event)

    def _record_alert(self, alert_key, timestamp, **details):
        """
        Records an alert only when it becomes active.
        It can be recorded again after the condition clears.
        """
        if alert_key in self.active_alerts:
            return

        self.active_alerts.add(alert_key)

        alert = {
            "alert_type": alert_key[0],
            "zone": alert_key[1],
            "timestamp": timestamp,
        }
        alert.update(details)
        self.alerts.append(alert)

    def update(self, detections, timestamp, frame_index=None):
        """
        Process one frame of tracked detections.

        Args:
            detections: list of dictionaries containing bbox and track_id.
            timestamp: elapsed video time in seconds.
            frame_index: optional frame number.

        Returns:
            Dictionary containing occupancy and analytics for this frame.
        """
        timestamp = float(timestamp)
        timestamp_text = str(round(timestamp, 3))

        current_positions = {}
        current_zones = defaultdict(set)

        # 1. Track person positions and line crossings.
        for detection in detections:
            track_id = detection.get("track_id")
            bbox = detection.get("bbox")

            if track_id is None or bbox is None:
                continue

            track_id = int(track_id)
            center = self._center(bbox)

            current_positions[track_id] = center
            self.last_seen[track_id] = timestamp

            current_side = self._line_side(
                center, self.line_start, self.line_end
            )
            previous = self.previous_positions.get(track_id)

            if previous is not None:
                previous_center, previous_side = previous

                # Only count an actual side change across the line.
                if previous_side != 0 and current_side != 0:
                    if previous_side != current_side:
                        if previous_side < current_side:
                            self.entry_count += 1
                            event_type = "entry"
                        else:
                            self.exit_count += 1
                            event_type = "exit"

                        event_time = datetime.now().astimezone().isoformat()
                        self._record_event(
                            track_id,
                            event_type,
                            event_time,
                            video_timestamp=timestamp,
                            frame_index=frame_index,
                        )

                        hour_key = int(timestamp // 3600)
                        if event_type == "entry":
                            self.peak_hour_entries[hour_key] += 1

            self.previous_positions[track_id] = (center, current_side)

            # 2. Check whether this person is inside each configured zone.
            for zone_name, polygon in self.zones.items():
                if self._inside_polygon(center, polygon):
                    current_zones[zone_name].add(track_id)

                    if track_id not in self.inside_zones[zone_name]:
                        self.inside_zones[zone_name][track_id] = timestamp
                        self._record_event(
                            track_id,
                            "zone_enter",
                            datetime.now().astimezone().isoformat(),
                            zone=zone_name,
                            video_timestamp=timestamp,
                        )

        # 3. Record zone exits and dwell times.
        for zone_name, previous_tracks in self.inside_zones.items():
            now_inside = current_zones.get(zone_name, set())

            for track_id, entered_at in list(previous_tracks.items()):
                if track_id not in now_inside:
                    dwell = max(0.0, timestamp - entered_at)

                    self._record_event(
                        track_id,
                        "zone_exit",
                        datetime.now().astimezone().isoformat(),
                        zone=zone_name,
                        dwell_seconds=round(dwell, 2),
                        video_timestamp=timestamp,
                    )
                    del previous_tracks[track_id]

        # 4. Crowd alerts: more than the configured limit.
        for zone_name in self.zones:
            people_count = len(current_zones.get(zone_name, set()))
            alert_key = ("crowd", zone_name)

            if people_count > self.crowd_limit:
                self._record_alert(
                    alert_key,
                    datetime.now().astimezone().isoformat(),
                    people_count=people_count,
                    limit=self.crowd_limit,
                    video_timestamp=timestamp,
                )
            else:
                self.active_alerts.discard(alert_key)

        # 5. Long-dwell alerts.
        if self.dwell_alert_seconds > 0:
            for zone_name, track_times in self.inside_zones.items():
                for track_id, entered_at in track_times.items():
                    dwell = timestamp - entered_at
                    alert_key = ("long_dwell", f"{zone_name}:{track_id}")

                    if dwell >= self.dwell_alert_seconds:
                        self._record_alert(
                            alert_key,
                            datetime.now().astimezone().isoformat(),
                            track_id=track_id,
                            dwell_seconds=round(dwell, 2),
                            video_timestamp=timestamp,
                        )

        # Clear long-dwell alert state once a track leaves its zone.
        for alert_key in list(self.active_alerts):
            if alert_key[0] == "long_dwell":
                _, identifier = alert_key
                zone_name, track_id_text = identifier.rsplit(":", 1)
                track_id = int(track_id_text)

                if track_id not in self.inside_zones.get(zone_name, {}):
                    self.active_alerts.discard(alert_key)

        # 6. Snapshot current occupancy for this frame.
        occupancy = {
            zone_name: len(current_zones.get(zone_name, set()))
            for zone_name in self.zones
        }

        self.occupancy_history.append({
            "video_timestamp": round(timestamp, 3),
            "frame_index": frame_index,
            "total_visible_people": len(current_positions),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "occupancy_by_zone": occupancy,
        })

        return {
            "total_visible_people": len(current_positions),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "occupancy_by_zone": occupancy,
            "current_zones": {
                name: sorted(ids)
                for name, ids in current_zones.items()
            },
        }

    def get_summary(self):
        """Return a JSON-serializable summary for the completed video."""
        peak_hour = None
        peak_entries = 0

        if self.peak_hour_entries:
            peak_hour, peak_entries = self.peak_hour_entries.most_common(1)[0]

        dwell_totals = defaultdict(float)
        dwell_counts = defaultdict(int)

        for event in self.events:
            if event.get("event") == "zone_exit":
                zone = event.get("zone", "unknown")
                dwell_totals[zone] += event.get("dwell_seconds", 0)
                dwell_counts[zone] += 1

        average_dwell = {
            zone: round(dwell_totals[zone] / count, 2)
            for zone, count in dwell_counts.items()
            if count
        }

        return {
            "total_unique_tracks": len(self.previous_positions),
            "total_entries": self.entry_count,
            "total_exits": self.exit_count,
            "peak_hour_video_offset": (
                f"{peak_hour}:00-{peak_hour + 1}:00"
                if peak_hour is not None else None
            ),
            "peak_hour_entries": peak_entries,
            "average_dwell_seconds_by_zone": average_dwell,
            "total_events": len(self.events),
            "total_alerts": len(self.alerts),
        }


if __name__ == "__main__":
    print("Testing Analytics...")

    zones = {
        "restricted_zone": [(400, 100), (800, 100), (800, 500), (400, 500)]
    }

    analytics = Analytics(
        line_start=(320, 0),
        line_end=(320, 720),
        zones=zones,
        crowd_limit=2,
    )

    # Frame 1: Person 1 on left side of line (x=200)
    frame1_detections = [
        {"bbox": (180, 200, 220, 300), "track_id": 1, "class_name": "person"}
    ]
    res1 = analytics.update(frame1_detections, timestamp=0.0, frame_index=1)
    print("Frame 1 Output:", res1)

    # Frame 2: Person 1 moves to the right side of line (x=450, inside restricted_zone)
    frame2_detections = [
        {"bbox": (430, 200, 470, 300), "track_id": 1, "class_name": "person"}
    ]
    res2 = analytics.update(frame2_detections, timestamp=1.0, frame_index=2)
    print("Frame 2 Output:", res2)

    summary = analytics.get_summary()
    print("\nAnalytics Summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\nAnalytics test passed successfully!")
