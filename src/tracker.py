"""ByteTrack-based player tracker with court-coordinate projection."""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


class PlayerIDMapper:
    """Map raw tracks to player ids by fixed court half."""

    def __init__(self):
        self.total_dist = {1: 0.0, 2: 0.0}
        self.last_pos = {1: None, 2: None}
        self.side_flipped = False

    def get_player_id(self, court_pos, side_flipped=False):
        """Return stable player id from court y coordinate."""
        y = court_pos[1]
        if not side_flipped:
            return 1 if y < 670 else 2
        return 2 if y < 670 else 1

    def update(self, raw_tracks):
        """Return Dict[player_id, court_pos] from Dict[raw_id, court_pos]."""
        result = {}
        for raw_id, pos in sorted(raw_tracks.items()):
            player_id = self.get_player_id(pos, self.side_flipped)
            result[player_id] = pos
            self.last_pos[player_id] = pos

        return result


class Tracker:
    """Track people and return valid player positions in court coordinates."""

    TRACKER_CONFIG = Path(r"F:\Fun-Activities\badminton_analysis\data\logs\bytetrack.yaml")
    YOLO_IMAGE_SIZE = 416

    def __init__(self, calibration_path, model_name="yolov8s.pt", confidence_threshold=0.5):
        self.h_matrix = np.load(calibration_path)
        self.model = YOLO(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.use_half = self.device == "cuda"
        self.confidence_threshold = confidence_threshold
        self.tracker_config = self._ensure_tracker_config()
        self.id_mapper = PlayerIDMapper()
        self.mapper = self.id_mapper

    def _ensure_tracker_config(self):
        """Create a ByteTrack config that preserves tracks through short rally gaps."""
        self.TRACKER_CONFIG.parent.mkdir(parents=True, exist_ok=True)
        self.TRACKER_CONFIG.write_text(
            "\n".join(
                [
                    "tracker_type: bytetrack",
                    "track_high_thresh: 0.25",
                    "track_low_thresh: 0.1",
                    "new_track_thresh: 0.25",
                    "track_buffer: 90",
                    "match_thresh: 0.8",
                    "fuse_score: True",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return str(self.TRACKER_CONFIG)

    def image_to_court(self, foot_point):
        """Project an image-space foot point into court coordinates."""
        point = np.float32([[foot_point]]).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(point, self.h_matrix).reshape(-1, 2)[0]
        return float(transformed[0]), float(transformed[1])

    def is_valid_court_point(self, court_point):
        """Return True when a court point is inside the badminton court bounds."""
        x, y = court_point
        return 0 <= x <= 610 and 0 <= y <= 1340

    def track(self, frame):
        """Track players in one frame and return Dict[track_id, court_point]."""
        raw_players, _ = self._track_raw_players(frame)
        return self.id_mapper.update(raw_players)

    def track_with_bbox(self, frame):
        """Track one frame and return stable player ids with bbox metadata."""
        raw_players, players = self._track_raw_players(frame)
        tracking_data = self.id_mapper.update(raw_players)
        bbox_height_by_track_id = {}
        bbox_by_track_id = {}

        for raw_id, player in sorted(players.items()):
            player_id = self.id_mapper.get_player_id(
                player["court_point"],
                self.id_mapper.side_flipped,
            )
            if player_id is None or player_id not in tracking_data:
                continue
            if tracking_data[player_id] != player["court_point"]:
                continue

            bbox_height_by_track_id[player_id] = player["bbox_height"]
            bbox_by_track_id[player_id] = player["bbox"]

        return tracking_data, bbox_height_by_track_id, bbox_by_track_id

    def _track_raw_players(self, frame):
        """Return valid raw ByteTrack ids plus bbox metadata before stable remapping."""
        results = self.model.track(
            frame,
            classes=[0],
            conf=self.confidence_threshold,
            persist=True,
            tracker=self.tracker_config,
            device=self.device,
            half=self.use_half,
            imgsz=self.YOLO_IMAGE_SIZE,
            verbose=False,
        )
        raw_players = {}
        players = {}

        for result in results:
            if result.boxes is None or result.boxes.id is None:
                continue

            for box in result.boxes:
                track_id = int(box.id.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                foot_point = ((x1 + x2) / 2.0, y2)
                court_point = self.image_to_court(foot_point)

                if self.is_valid_court_point(court_point):
                    raw_players[track_id] = court_point
                    players[track_id] = {
                        "court_point": court_point,
                        "bbox_height": y2 - y1,
                        "bbox": (x1, y1, x2, y2),
                    }

        return raw_players, players


def open_video_at_start(video_path, start_seconds):
    """Open a video capture and seek to the requested start time."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
    return capture


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Track badminton players with ByteTrack")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--calibration", required=True, help="Path to H.npy calibration matrix")
    return parser.parse_args()


def main():
    """Track 100 frames from 65 seconds and print positions every 10 frames."""
    args = parse_args()
    tracker = Tracker(args.calibration)
    capture = open_video_at_start(Path(args.video), start_seconds=65)

    try:
        for frame_index in range(100):
            success, frame = capture.read()
            if not success or frame is None:
                print(f"Frame {frame_index}: failed to read frame")
                break

            tracked_players = tracker.track(frame)
            if frame_index % 10 == 0:
                print(f"Frame {frame_index}: {tracked_players}")
    finally:
        capture.release()


if __name__ == "__main__":
    main()
