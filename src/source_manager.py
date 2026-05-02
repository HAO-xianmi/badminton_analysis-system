"""Configuration management for multiple badminton video sources."""

import json
import os
from pathlib import Path

import cv2


CONFIG_DIR = Path("F:/Fun-Activities/badminton_analysis/data/calibration")


class SourceManager:
    """Create, load, update, and list per-source analysis configuration files."""

    def __init__(self):
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        self.config_path = None
        self.config = {}

    def create_source(self, video_path, source_name):
        """Create a config for a new source."""
        video_path = os.fspath(video_path)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        self.config = {
            "source_name": source_name,
            "video_path": video_path,
            "fps": fps,
            "resolution": [width, height],
            "total_frames": total_frames,
            "h_matrix_path": str(CONFIG_DIR / f"{source_name}_H.npy"),
            "score_roi": {
                "y1": 50,
                "y2": 135,
                "x1": 630,
                "x2": 705,
                "calibrated": False,
            },
            "court_h_threshold": 670,
            "side_flipped": False,
            "ball_detection": {
                "enabled": False,
                "model_path": "",
                "min_confidence": 0.5,
                "max_speed_kmh": 500,
                "trail_frames": 8,
            },
            "players": {
                "player1": {"name": "Player 1", "color": [0, 255, 0]},
                "player2": {"name": "Player 2", "color": [255, 0, 0]},
            },
        }

        config_file = CONFIG_DIR / f"{source_name}_config.json"
        with open(config_file, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)

        self.config_path = config_file
        return self.config

    def load_source(self, source_name):
        """Load an existing source config."""
        config_file = CONFIG_DIR / f"{source_name}_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Source config not found: {source_name}")
        with open(config_file, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.config_path = config_file
        return self.config

    def save_score_roi(self, y1, y2, x1, x2):
        """Save calibrated scoreboard digits coordinates."""
        self.config["score_roi"] = {
            "y1": int(y1),
            "y2": int(y2),
            "x1": int(x1),
            "x2": int(x2),
            "calibrated": True,
        }
        self._save()

    def list_sources(self):
        """List all configured sources."""
        configs = list(CONFIG_DIR.glob("*_config.json"))
        sources = []
        for config_path in configs:
            with open(config_path, encoding="utf-8") as f:
                cfg = json.load(f)
            h_exists = Path(cfg["h_matrix_path"]).exists()
            score_roi = cfg.get("score_roi", {})
            sources.append(
                {
                    "name": cfg["source_name"],
                    "video": cfg["video_path"],
                    "calibrated": h_exists,
                    "score_roi_calibrated": score_roi.get("calibrated", False),
                }
            )
        return sources

    def _save(self):
        if self.config_path is None:
            raise RuntimeError("No source config has been loaded or created.")
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
