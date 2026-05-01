"""Player movement analyzer for distance metrics."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from tracker import Tracker, open_video_at_start


class Analyzer:
    """Accumulate per-player running distance."""

    SCALE_METERS_PER_PIXEL = 0.01
    MAX_STEP_PIXELS = 33
    SMOOTHING_WINDOW = 5

    def __init__(self):
        self.previous_points = {}
        self.distances_m = {}
        self.point_history = defaultdict(lambda: deque(maxlen=self.SMOOTHING_WINDOW))
        self.smoothed_points = {}
        self.current_rally_dist = {1: 0.0, 2: 0.0}
        self.rally_history = []

    def analyze(self, tracking_data):
        """Update distance metrics from stable player court points."""
        smoothed_tracking_data = self._smooth_tracking_data(tracking_data)

        if not smoothed_tracking_data:
            return self.get_metrics()

        for track_id, court_point in smoothed_tracking_data.items():
            self.distances_m.setdefault(track_id, 0.0)
            self.current_rally_dist.setdefault(track_id, 0.0)

            previous_point = self.previous_points.get(track_id)
            if previous_point is not None:
                step_pixels = self._distance_pixels(previous_point, court_point)
                if step_pixels > self.MAX_STEP_PIXELS:
                    self.previous_points[track_id] = court_point
                    continue
                step_meters = step_pixels * self.SCALE_METERS_PER_PIXEL
                self.distances_m[track_id] += step_meters
                self.current_rally_dist[track_id] += step_meters

            self.previous_points[track_id] = court_point

        return self.get_metrics()

    def get_smoothed_tracking_data(self):
        """Return the latest smoothed court points."""
        return dict(self.smoothed_points)

    def get_metrics(self):
        """Return Dict[track_id, {distance_m}] plus rally summaries."""
        metrics = {}
        for track_id in sorted(self.distances_m):
            metrics[track_id] = {
                "distance_m": self.distances_m.get(track_id, 0.0),
            }
        metrics["rally_history"] = list(self.rally_history)
        metrics["current_rally_dist"] = dict(self.current_rally_dist)
        return metrics

    def export_metrics(self, output_path):
        """Placeholder for exporting metrics later."""
        raise NotImplementedError("Metric export is not implemented yet.")

    def _distance_pixels(self, point_a, point_b):
        """Compute Euclidean distance between two court-coordinate points."""
        dx = point_b[0] - point_a[0]
        dy = point_b[1] - point_a[1]
        return (dx * dx + dy * dy) ** 0.5

    def close_current_rally(self):
        """Record the just-finished rally and reset per-rally distances."""
        if not self.previous_points and not any(distance > 0.0 for distance in self.current_rally_dist.values()):
            self.current_rally_dist = {1: 0.0, 2: 0.0}
            return

        self.rally_history.append(
            {
                "rally": len(self.rally_history) + 1,
                "p1": self.current_rally_dist.get(1, 0.0),
                "p2": self.current_rally_dist.get(2, 0.0),
            }
        )
        self.current_rally_dist = {1: 0.0, 2: 0.0}
        self.previous_points = {}
        self.point_history.clear()

    def _smooth_tracking_data(self, tracking_data):
        """Smooth court points with a 5-frame moving average per track."""
        smoothed = {}
        for track_id, court_point in tracking_data.items():
            self.point_history[track_id].append(court_point)
            points = np.array(self.point_history[track_id], dtype=np.float32)
            averaged = points.mean(axis=0)
            smoothed[track_id] = (float(averaged[0]), float(averaged[1]))
        self.smoothed_points = smoothed
        return smoothed


def track_frame_with_bbox(tracker, frame):
    """Track one frame and return valid court points plus bbox height values."""
    return tracker.track_with_bbox(frame)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Analyze tracked badminton players")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--calibration", required=True, help="Path to H.npy calibration matrix")
    return parser.parse_args()


def main():
    """Analyze 100 frames from 65 seconds and print metrics every 10 frames."""
    args = parse_args()
    tracker = Tracker(args.calibration)
    analyzer = Analyzer()
    capture = open_video_at_start(Path(args.video), start_seconds=65)

    try:
        for frame_index in range(100):
            success, frame = capture.read()
            if not success or frame is None:
                print(f"Frame {frame_index}: failed to read frame")
                break

            tracking_data, bbox_height_by_track_id, _ = track_frame_with_bbox(tracker, frame)
            metrics = analyzer.analyze(
                tracking_data,
            )

            if frame_index % 10 == 0:
                print(f"Frame {frame_index}: {metrics}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
