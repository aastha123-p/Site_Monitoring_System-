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
        machinery_proximity_pixels=150,
        enable_ppe_alerts=True,
    ):
        """
        Args:
            line_start, line_end:
                Coordinates of the virtual entry/exit line.
            zones:
                Dictionary of named polygon zones.
            crowd_limit:
                Alert when more than this number of people are simultaneously inside a restricted zone.
            dwell_alert_seconds:
                Threshold for a long-stay alert (seconds). 0 to disable.
            machinery_proximity_pixels:
                Alert if a person is closer than this distance to heavy machinery.
            enable_ppe_alerts:
                Enable detection-based PPE alerts.
        """
        self.line_start = tuple(line_start)
        self.line_end = tuple(line_end)
        self.zones = zones or {}
        self.crowd_limit = int(crowd_limit)
        self.dwell_alert_seconds = float(dwell_alert_seconds)
        self.machinery_proximity_pixels = float(machinery_proximity_pixels)
        self.enable_ppe_alerts = bool(enable_ppe_alerts)

        self.previous_positions = {}
        self.last_seen = {}
        self.inside_zones = defaultdict(dict)

        self.entry_count = 0
        self.exit_count = 0
        self.peak_hour_entries = Counter()

        self.events = []
        self.alerts = []
        self.occupancy_history = []

        # Avoid spamming duplicate alerts on consecutive frames
        self.active_alerts = set()

    @staticmethod
    def _center(bbox):
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)

    @staticmethod
    def _ccw(a, b, c):
        return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

    @classmethod
    def _segments_intersect(cls, p1, p2, p3, p4):
        """Returns True if line segment p1-p2 intersects segment p3-p4."""
        return cls._ccw(p1, p3, p4) != cls._ccw(p2, p3, p4) and cls._ccw(p1, p2, p3) != cls._ccw(p1, p2, p4)

    @staticmethod
    def _line_side(point, start, end):
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

    def update(self, detections, timestamp, frame_index=None, red_straps=None):
        timestamp = float(timestamp)

        current_positions = {}
        current_zones = defaultdict(set)
        visible_person_count = 0
        machinery_entities = []
        cone_points = []
        proximity_violations = []

        # 1. Classify detections for dynamic spatial awareness
        for detection in detections:
            class_name = detection.get("class_name", "")
            bbox = detection.get("bbox")
            track_id = detection.get("track_id")

            if bbox is None:
                continue

            center = self._center(bbox)

            if class_name in ("machinery", "crane", "excavator", "truck"):
                x1, y1, x2, y2 = bbox
                radius = max(int(np.hypot(x2 - x1, y2 - y1) / 2), 40) + int(self.machinery_proximity_pixels * 0.4)
                machinery_entities.append({
                    "class_name": class_name,
                    "center": center,
                    "radius": radius,
                    "track_id": track_id,
                })

            elif class_name == "safety_cone":
                # Use bottom contact point of cone on ground
                x1, y1, x2, y2 = bbox
                cone_points.append((int((x1 + x2) / 2), int(y2)))

        # 2. Form dynamic red barrier strap zone polygon if straps are detected
        if red_straps and len(red_straps) >= 2:
            strap_pts = []
            for p1, p2 in red_straps:
                strap_pts.extend([p1, p2])
            hull_strap = cv2.convexHull(np.array(strap_pts, dtype=np.int32))
            self.zones["red_barrier_zone"] = [tuple(pt[0]) for pt in hull_strap]
        else:
            self.zones.pop("red_barrier_zone", None)

        # Compute dynamic cone perimeter zone if >= 3 cones are visible
        dynamic_cone_polygon = None
        if len(cone_points) >= 3:
            hull = cv2.convexHull(np.array(cone_points, dtype=np.int32))
            dynamic_cone_polygon = [tuple(pt[0]) for pt in hull]
            self.zones["cone_barricade"] = dynamic_cone_polygon
        else:
            self.zones.pop("cone_barricade", None)

        # 3. Process workers: movement, line crossings, zone entries & machinery proximity
        for detection in detections:
            class_name = detection.get("class_name", "")
            bbox = detection.get("bbox")
            track_id = detection.get("track_id")

            if bbox is None or class_name != "person":
                continue

            visible_person_count += 1
            center = self._center(bbox)
            feet = (center[0], int(bbox[3]))  # Ground contact point for zone tests

            if track_id is not None:
                track_id = int(track_id)
                current_positions[track_id] = center
                self.last_seen[track_id] = timestamp

                current_side = self._line_side(center, self.line_start, self.line_end)
                previous = self.previous_positions.get(track_id)

                if previous is not None:
                    previous_center, previous_side = previous
                    crossed = self._segments_intersect(previous_center, center, self.line_start, self.line_end)
                    if not crossed and previous_side != 0 and current_side != 0:
                        crossed = (previous_side != current_side)

                    if crossed:
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

                # Check all zones (static + dynamic cone barricade)
                for zone_name, polygon in self.zones.items():
                    if self._inside_polygon(feet, polygon) or self._inside_polygon(center, polygon):
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

            # Check Proximity to Heavy Machinery (Dynamic Buffer Zones)
            for mach in machinery_entities:
                dist = np.hypot(center[0] - mach["center"][0], center[1] - mach["center"][1])
                if dist < mach["radius"]:
                    proximity_violations.append({
                        "person_center": center,
                        "person_id": track_id,
                        "mach_center": mach["center"],
                        "mach_name": mach["class_name"],
                        "distance_px": round(dist, 1),
                    })
                    alert_key = ("proximity_hazard", f"{mach['class_name']}:{track_id}")
                    self._record_alert(
                        alert_key,
                        datetime.now().astimezone().isoformat(),
                        person_id=track_id,
                        machinery=mach["class_name"],
                        distance_px=round(dist, 1),
                        video_timestamp=timestamp,
                    )

        # 4. PPE Violation alerts
        if self.enable_ppe_alerts:
            for detection in detections:
                class_name = detection.get("class_name", "")
                if class_name in ("no_hardhat", "no_safety_vest"):
                    alert_key = ("ppe_violation", f"{class_name}:{frame_index}")
                    self._record_alert(
                        alert_key,
                        datetime.now().astimezone().isoformat(),
                        violation=class_name,
                        confidence=detection.get("confidence", 0.0),
                        video_timestamp=timestamp,
                    )

        # Record zone exits and dwell times
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

        # Crowd alerts in zones (> 4 people for red_barrier_zone, crowd_limit for others)
        red_barrier_crowd_alert = False
        red_barrier_people_count = 0

        for zone_name in list(self.zones.keys()):
            people_count = len(current_zones.get(zone_name, set()))
            zone_limit = 4 if zone_name == "red_barrier_zone" else self.crowd_limit
            alert_key = ("crowd", zone_name)

            if zone_name == "red_barrier_zone" and people_count > 4:
                red_barrier_crowd_alert = True
                red_barrier_people_count = people_count

            if people_count > zone_limit:
                self._record_alert(
                    alert_key,
                    datetime.now().astimezone().isoformat(),
                    people_count=people_count,
                    limit=zone_limit,
                    video_timestamp=timestamp,
                )
            else:
                self.active_alerts.discard(alert_key)

        # Long-dwell alerts
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

        # Clear long-dwell alert state once a track leaves its zone
        for alert_key in list(self.active_alerts):
            if alert_key[0] == "long_dwell":
                _, identifier = alert_key
                zone_name, track_id_text = identifier.rsplit(":", 1)
                track_id = int(track_id_text)
                if track_id not in self.inside_zones.get(zone_name, {}):
                    self.active_alerts.discard(alert_key)

        occupancy = {
            zone_name: len(current_zones.get(zone_name, set()))
            for zone_name in self.zones
        }

        self.occupancy_history.append({
            "video_timestamp": round(timestamp, 3),
            "frame_index": frame_index,
            "total_visible_people": visible_person_count,
            "tracked_people": len(current_positions),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "occupancy_by_zone": occupancy,
        })

        return {
            "visible_person_detections": visible_person_count,
            "tracked_people": len(current_positions),
            "entry_count": self.entry_count,
            "exit_count": self.exit_count,
            "occupancy_by_zone": occupancy,
            "machinery_entities": machinery_entities,
            "proximity_violations": proximity_violations,
            "dynamic_zones": dict(self.zones),
            "red_barrier_crowd_alert": red_barrier_crowd_alert,
            "red_barrier_people_count": red_barrier_people_count,
            "current_zones": {
                name: sorted(ids) for name, ids in current_zones.items()
            },
        }

    def get_summary(self, total_frames=None, fps=None, **kwargs):
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
                f"{peak_hour}:00-{peak_hour + 1}:00" if peak_hour is not None else None
            ),
            "peak_hour_entries": peak_entries,
            "average_dwell_seconds_by_zone": average_dwell,
            "total_events": len(self.events),
            "total_alerts": len(self.alerts),
        }


if __name__ == "__main__":
    print("Testing Analytics Engine...")

    zones = {
        "restricted_zone": [(400, 100), (800, 100), (800, 500), (400, 500)]
    }

    analytics = Analytics(
        line_start=(320, 0),
        line_end=(320, 720),
        zones=zones,
        crowd_limit=2,
        dwell_alert_seconds=5,
        machinery_proximity_pixels=150,
        enable_ppe_alerts=True,
    )

    # Frame 1: Person 1 on left side of line (x=200) + excavator
    frame1 = [
        {"bbox": (180, 200, 220, 300), "track_id": 1, "class_name": "person", "confidence": 0.92},
        {"bbox": (210, 210, 230, 230), "track_id": None, "class_name": "no_hardhat", "confidence": 0.85},
        {"bbox": (700, 300, 950, 550), "track_id": 2, "class_name": "excavator", "confidence": 0.89},
    ]
    res1 = analytics.update(frame1, timestamp=0.0, frame_index=1)
    print("\n--- Frame 1 Processed ---")
    print(f"  Visible people: {res1['visible_person_detections']} | Tracked: {res1['tracked_people']}")
    print(f"  Entries: {res1['entry_count']} | Exits: {res1['exit_count']}")

    # Frame 2: Person 1 crosses line to right side (x=450, inside restricted_zone)
    frame2 = [
        {"bbox": (430, 200, 470, 300), "track_id": 1, "class_name": "person", "confidence": 0.94},
        {"bbox": (440, 205, 460, 225), "track_id": None, "class_name": "hardhat", "confidence": 0.88},
    ]
    res2 = analytics.update(frame2, timestamp=1.0, frame_index=2)
    print("\n--- Frame 2 Processed ---")
    print(f"  Visible people: {res2['visible_person_detections']} | Tracked: {res2['tracked_people']}")
    print(f"  Entries: {res2['entry_count']} | Exits: {res2['exit_count']}")
    print(f"  Zone occupancy: {res2['occupancy_by_zone']}")

    summary = analytics.get_summary()
    print("\n" + "=" * 50)
    print("          ANALYTICS SUMMARY TEST RESULTS")
    print("=" * 50)
    for k, v in summary.items():
        print(f"  {k:<32}: {v}")
    print("=" * 50)
    print(f"Total Events Logged: {len(analytics.events)}")
    print(f"Total Alerts Fired : {len(analytics.alerts)}")
    print("Analytics engine test completed successfully.")

