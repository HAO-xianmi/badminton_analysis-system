"""Kalman-filter smoothing for shuttlecock positions."""

import cv2
import numpy as np


MAX_SPEED_PX = 250
MIN_SPEED_PX = 0
MIN_CONFIDENCE = 0.5


class BallTracker:
    """Smooth detected shuttlecock positions and bridge short missing spans."""

    def __init__(self, max_missing=5):
        self.positions = []
        self.missing = 0
        self.max_missing = max_missing

        self.kf = cv2.KalmanFilter(4, 2)
        self.kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kf.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            np.float32,
        )
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.initialized = False

    def update(self, detection):
        """Return a smoothed (x, y) position or None when tracking is lost."""
        if detection is not None:
            pos_tuple = detection["pos"]
            if self.positions:
                last = self.positions[-1]
                dist = np.sqrt((pos_tuple[0] - last[0]) ** 2 + (pos_tuple[1] - last[1]) ** 2)
                if dist > MAX_SPEED_PX or dist < MIN_SPEED_PX:
                    detection = None

            if detection is not None and detection["confidence"] < MIN_CONFIDENCE:
                detection = None

        if detection is not None:
            pos = np.array([[detection["pos"][0]], [detection["pos"][1]]], dtype=np.float32)
            if not self.initialized:
                self.kf.statePre = np.array(
                    [pos[0, 0], pos[1, 0], 0, 0],
                    dtype=np.float32,
                ).reshape(-1, 1)
                self.kf.statePost = self.kf.statePre.copy()
                self.initialized = True
            corrected = self.kf.correct(pos)
            self.missing = 0
            smoothed = (int(corrected[0, 0]), int(corrected[1, 0]))
        else:
            self.missing += 1
            if self.missing > self.max_missing or not self.initialized:
                return None
            predicted = self.kf.predict()
            smoothed = (int(predicted[0, 0]), int(predicted[1, 0]))
            self.positions.append(smoothed)
            if len(self.positions) > 30:
                self.positions.pop(0)
            return smoothed

        self.kf.predict()
        self.positions.append(smoothed)
        if len(self.positions) > 30:
            self.positions.pop(0)

        return smoothed

    def reset(self):
        """Clear Kalman state and recent shuttlecock positions."""
        self.positions = []
        self.missing = 0
        self.initialized = False
