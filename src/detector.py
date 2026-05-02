"""YOLOv8 person detector and MediaPipe pose renderer for badminton frames."""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import torch
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from ultralytics import YOLO

from paths import pose_model_path, yolo_model_path


class Detector:
    """Detect people in a frame and draw pose skeletons for tracked players."""

    POSE_IMAGE_WIDTH = 640

    def __init__(self, model_name="yolov8s.pt", confidence_threshold=0.5):
        self.model_name = yolo_model_path(model_name)
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.use_half = self.device == "cuda"
        self.cached_pose_points_by_track_id = {}
        self.cached_pose_landmarks_by_track_id = {}
        self.cached_head_y_by_track_id = {}
        self.pose_model_path = pose_model_path(required=True)
        self.pose_delegate = "GPU"
        self.pose_detector = self._create_pose_detector()

    def _create_pose_detector(self):
        """Create a MediaPipe pose landmarker, falling back to CPU when needed."""
        base_options = mp_python.BaseOptions(
            model_asset_path=str(self.pose_model_path),
            delegate=mp_python.BaseOptions.Delegate.GPU,
        )
        options = mp_vision.PoseLandmarkerOptions(
            base_options=base_options,
            num_poses=2,
        )
        try:
            return mp_vision.PoseLandmarker.create_from_options(options)
        except NotImplementedError as error:
            self.pose_delegate = "CPU"
            print(f"MediaPipe GPU delegate unavailable, falling back to CPU: {error}")
            base_options = mp_python.BaseOptions(model_asset_path=str(self.pose_model_path))
            options = mp_vision.PoseLandmarkerOptions(
                base_options=base_options,
                num_poses=2,
            )
            return mp_vision.PoseLandmarker.create_from_options(options)

    def load_model(self):
        """Load the YOLO model."""
        if self.model is None:
            self.model = YOLO(self.model_name)
            self.model.to(self.device)
        return self.model

    def detect(self, frame):
        """Detect person objects from the input frame."""
        model = self.load_model()
        results = model(frame, verbose=False, device=self.device, half=self.use_half)
        detections = []

        for result in results:
            if result.boxes is None:
                continue

            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                if class_id != 0 or confidence < self.confidence_threshold:
                    continue

                x1, y1, x2, y2 = box.xyxy[0].tolist()
                bbox = (x1, y1, x2, y2)
                foot_point = ((x1 + x2) / 2.0, y2)
                detections.append(
                    {
                        "bbox": bbox,
                        "foot_point": foot_point,
                        "confidence": confidence,
                    }
                )

        return detections

    def draw_skeleton(self, frame, tracked_bboxes, run_pose_detection=True):
        """Draw MediaPipe pose skeletons for tracked player bounding boxes."""
        colors = {
            1: (0, 255, 0),
            2: (255, 0, 0),
        }
        connections = [
            (11, 12),
            (11, 13),
            (13, 15),
            (12, 14),
            (14, 16),
            (11, 23),
            (12, 24),
            (23, 24),
            (23, 25),
            (25, 27),
            (24, 26),
            (26, 28),
        ]

        if run_pose_detection and tracked_bboxes:
            (
                self.cached_pose_points_by_track_id,
                self.cached_pose_landmarks_by_track_id,
                self.cached_head_y_by_track_id,
            ) = self._detect_frame_pose_points(frame, tracked_bboxes)

        for track_id, bbox in tracked_bboxes.items():
            _ = bbox
            color = colors.get(track_id, (0, 255, 255))
            cached_points = self.cached_pose_points_by_track_id.get(track_id)
            if cached_points:
                self._draw_pose_points(frame, cached_points, connections, color)

        return (
            frame,
            dict(self.cached_pose_landmarks_by_track_id),
            dict(self.cached_head_y_by_track_id),
        )

    def _detect_frame_pose_points(self, frame, tracked_bboxes):
        """Run MediaPipe once on the full frame and assign poses to tracked boxes."""
        height, width = frame.shape[:2]
        scale = min(1.0, self.POSE_IMAGE_WIDTH / float(width))
        if scale < 1.0:
            pose_width = int(round(width * scale))
            pose_height = int(round(height * scale))
            pose_frame = cv2.resize(frame, (pose_width, pose_height), interpolation=cv2.INTER_AREA)
        else:
            pose_frame = frame
            pose_height, pose_width = height, width

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=cv2.cvtColor(pose_frame, cv2.COLOR_BGR2RGB),
        )
        result = self.pose_detector.detect(mp_image)
        if not result.pose_landmarks:
            return {}, {}, {}

        inverse_scale = 1.0 / scale
        candidate_points = []
        candidate_landmarks = []
        for landmarks in result.pose_landmarks:
            points = {
                index: (
                    int(landmark.x * pose_width * inverse_scale),
                    int(landmark.y * pose_height * inverse_scale),
                )
                for index, landmark in enumerate(landmarks)
            }
            candidate_points.append(points)
            candidate_landmarks.append(landmarks)

        return self._assign_pose_points_to_tracks(candidate_points, candidate_landmarks, tracked_bboxes)

    def _assign_pose_points_to_tracks(self, candidate_points, candidate_landmarks, tracked_bboxes):
        """Match detected full-frame poses back to tracker ids by bbox overlap."""
        point_assignments = {}
        landmark_assignments = {}
        head_y_assignments = {}
        used_pose_indices = set()

        for track_id, bbox in sorted(tracked_bboxes.items()):
            best_pose_index = None
            best_score = -1
            x1, y1, x2, y2 = bbox

            for pose_index, points in enumerate(candidate_points):
                if pose_index in used_pose_indices:
                    continue

                inside_count = sum(
                    1
                    for px, py in points.values()
                    if x1 <= px <= x2 and y1 <= py <= y2
                )
                if inside_count > best_score:
                    best_score = inside_count
                    best_pose_index = pose_index

            if best_pose_index is not None and best_score > 0:
                point_assignments[track_id] = candidate_points[best_pose_index]
                landmark_assignments[track_id] = candidate_landmarks[best_pose_index]
                head_y_assignments[track_id] = candidate_landmarks[best_pose_index][0].y
                used_pose_indices.add(best_pose_index)

        return point_assignments, landmark_assignments, head_y_assignments

    def _draw_pose_points(self, frame, points, connections, color):
        """Draw cached or freshly detected pose points on a frame."""
        for point in points.values():
            cv2.circle(frame, point, 3, color, -1)

        for start, end in connections:
            if start in points and end in points:
                cv2.line(frame, points[start], points[end], color, 2)

    def draw_pose_on_players(self, frame, bbox_by_track_id):
        """Backward-compatible wrapper for skeleton drawing."""
        pose_frame, _, _ = self.draw_skeleton(frame, bbox_by_track_id)
        return pose_frame


def load_frame(video_path, start_seconds=65):
    """Load one frame from the requested second in the video."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
    success, frame = capture.read()
    capture.release()

    if not success or frame is None:
        raise RuntimeError(f"Cannot read frame at {start_seconds} seconds: {video_path}")

    return frame


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="YOLOv8 person detector test")
    parser.add_argument("--video", required=True, help="Path to the input video")
    return parser.parse_args()


def main():
    """Load a frame at 65 seconds, run detection, and print foot points."""
    args = parse_args()
    video_path = Path(args.video)
    frame = load_frame(video_path, start_seconds=65)

    detector = Detector()
    detections = detector.detect(frame)

    print(f"Detected {len(detections)} people")
    for index, detection in enumerate(detections, start=1):
        print(f"Person {index} foot_point: {detection['foot_point']}")


if __name__ == "__main__":
    main()
