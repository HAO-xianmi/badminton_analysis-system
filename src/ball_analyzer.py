"""Shuttlecock speed and shot attribution analysis."""

import csv
from pathlib import Path

import cv2
import numpy as np


class BallAnalyzer:
    """Calculate shuttlecock speed and record per-shot events."""

    def __init__(self, H, fps, scale=0.01, csv_path=None):
        self.H = H
        self.fps = fps
        self.scale = scale
        self.prev_pos = None
        self.prev_court_pos = None
        self.current_speed = 0.0
        self.shots = []
        self.rally_shots = []
        self.direction_history = []
        self.csv_path = Path(csv_path) if csv_path is not None else None
        self._written_shots = 0
        if self.csv_path is not None:
            self.csv_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames())
                writer.writeheader()

    def warp(self, pt):
        """Project an image-space point into court coordinates."""
        p = np.float32([[pt]])
        return cv2.perspectiveTransform(p, self.H)[0][0]

    def update(self, ball_pos, player_positions, frame_idx, rally_id):
        """Update speed state and return the current speed in km/h."""
        if ball_pos is None or self.prev_pos is None:
            self.prev_pos = ball_pos
            if ball_pos is not None:
                self.prev_court_pos = self.warp(ball_pos)
            return None

        court_pos = self.warp(ball_pos)

        if self.prev_court_pos is not None:
            dx = court_pos[0] - self.prev_court_pos[0]
            dy = court_pos[1] - self.prev_court_pos[1]
            pixel_dist = np.sqrt(dx**2 + dy**2)
            real_dist = pixel_dist * self.scale
            speed_ms = real_dist * self.fps
            speed_kmh = speed_ms * 3.6

            if speed_kmh < 493:
                self.current_speed = speed_kmh

            self.direction_history.append((dx, dy))
            if len(self.direction_history) > 3:
                self.direction_history.pop(0)
                if self._detect_shot():
                    player_id = self._assign_player(ball_pos, player_positions)
                    shot = {
                        "frame": frame_idx,
                        "rally_id": rally_id,
                        "player_id": player_id,
                        "speed_kmh": float(round(float(self.current_speed), 1)),
                        "ball_pos_pixel": ball_pos,
                        "ball_pos_court": tuple(float(v) for v in court_pos),
                        "shot_type": "",
                    }
                    self.shots.append(shot)
                    self.rally_shots.append(shot)
                    self._append_new_shots()

        self.prev_pos = ball_pos
        self.prev_court_pos = court_pos
        return self.current_speed

    def _detect_shot(self):
        """Return True when recent ball direction changes sharply."""
        if len(self.direction_history) < 3:
            return False
        v1 = np.array(self.direction_history[-3])
        v2 = np.array(self.direction_history[-1])
        if np.linalg.norm(v1) < 1 or np.linalg.norm(v2) < 1:
            return False
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        return cos_angle < -0.3

    def _assign_player(self, ball_pos, player_positions):
        """Assign the shot to the player nearest the ball."""
        min_dist = float("inf")
        player_id = 1
        for pid, pos in player_positions.items():
            d = np.linalg.norm(np.array(ball_pos) - np.array(pos))
            if d < min_dist:
                min_dist = d
                player_id = pid
        return player_id

    def reset_rally(self):
        """Clear per-rally shot records."""
        self.rally_shots = []

    def reset_position(self):
        """Clear previous ball position so speed restarts cleanly."""
        self.prev_pos = None
        self.prev_court_pos = None
        self.current_speed = 0.0
        self.direction_history = []

    def get_rally_max_speed(self):
        """Return current-rally max speed by player id."""
        if not self.rally_shots:
            return {1: 0.0, 2: 0.0}
        p1 = max((s["speed_kmh"] for s in self.rally_shots if s["player_id"] == 1), default=0.0)
        p2 = max((s["speed_kmh"] for s in self.rally_shots if s["player_id"] == 2), default=0.0)
        return {1: p1, 2: p2}

    def get_state(self):
        """Return visualization-friendly ball speed state."""
        return {
            "current_speed_kmh": float(self.current_speed),
            "rally_max_speed": self.get_rally_max_speed(),
            "shots": list(self.shots),
        }

    def _append_new_shots(self):
        if self.csv_path is None or self._written_shots >= len(self.shots):
            return
        with open(self.csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self._csv_fieldnames())
            for shot in self.shots[self._written_shots :]:
                row = {
                    "frame": shot["frame"],
                    "rally_id": shot["rally_id"],
                    "player_id": shot["player_id"],
                    "speed_kmh": shot["speed_kmh"],
                    "ball_x": shot["ball_pos_pixel"][0],
                    "ball_y": shot["ball_pos_pixel"][1],
                    "court_x": shot["ball_pos_court"][0],
                    "court_y": shot["ball_pos_court"][1],
                    "shot_type": shot["shot_type"],
                }
                writer.writerow(row)
        self._written_shots = len(self.shots)

    def _csv_fieldnames(self):
        return [
            "frame",
            "rally_id",
            "player_id",
            "speed_kmh",
            "ball_x",
            "ball_y",
            "court_x",
            "court_y",
            "shot_type",
        ]
