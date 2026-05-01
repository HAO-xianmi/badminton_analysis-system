"""End-to-end badminton analysis pipeline for video processing and export."""

import argparse
import re
import threading
import time
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import easyocr
import torch

from analyzer import Analyzer, track_frame_with_bbox
from detector import Detector
from tracker import Tracker, open_video_at_start
from visualizer import Visualizer


class BadmintonAnalysisApp:
    """Run tracking, analysis, and visualization over a video segment."""

    TRACKING_INTERVAL = 2
    POSE_INTERVAL = 3
    SCOREBOARD_ROI_WIDTH = 200
    SCOREBOARD_ROI_HEIGHT = 100
    SCOREBOARD_WHITE_THRESHOLD = 0.05
    SCOREBOARD_SCAN_STEP = 50
    SCORE_OCR_INTERVAL = 15
    SCORE_ROI = (60, 120, 380, 700)
    SCORE_DIGITS_ROI = (50, 135, 630, 705)
    SCORE_CHANGE_CONFIRMATIONS = 2

    def __init__(self, video_path, calibration_path, output_path, start_seconds=0):
        self.video_path = Path(video_path)
        self.calibration_path = Path(calibration_path)
        self.output_path = Path(output_path)
        self.start_seconds = start_seconds

        self.detector = Detector()
        self.tracker = Tracker(str(self.calibration_path))
        self.analyzer = Analyzer()
        self.visualizer = Visualizer()
        self.score_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        self.prev_score = None
        self.score_confirm_count = 0

    def run(self, max_frames=None, duration_seconds=None):
        """Process the video, save visualization output, and return final metrics."""
        capture = open_video_at_start(self.video_path, self.start_seconds)
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        if duration_seconds is not None:
            duration_frames = max(0, int(round(duration_seconds * fps)))
            max_frames = duration_frames if max_frames is None else min(max_frames, duration_frames)
        writer = None
        frame_index = 0
        final_metrics = {}
        total_processing_time = 0.0
        stage_times = {
            "tracking": 0.0,
            "analysis": 0.0,
            "pose": 0.0,
            "visualization": 0.0,
            "write": 0.0,
        }
        cached_tracking_data = {}
        cached_bbox_height_by_track_id = {}
        cached_bbox_by_track_id = {}
        has_cached_tracking = False
        frame_queue = Queue(maxsize=30)
        write_queue = Queue(maxsize=30)
        stop_reader = threading.Event()
        stop_writer = threading.Event()
        writer_error = []

        def read_frames():
            while not stop_reader.is_set():
                success, frame = capture.read()
                if not success or frame is None:
                    while not stop_reader.is_set():
                        try:
                            frame_queue.put(None, timeout=0.1)
                            break
                        except Full:
                            continue
                    break
                while not stop_reader.is_set():
                    try:
                        frame_queue.put(frame, timeout=0.1)
                        break
                    except Full:
                        continue

        reader_thread = threading.Thread(target=read_frames, daemon=True)
        reader_thread.start()

        def write_frames():
            while not stop_writer.is_set() or not write_queue.empty():
                try:
                    output_frame = write_queue.get(timeout=0.1)
                except Empty:
                    continue

                if output_frame is None:
                    write_queue.task_done()
                    break

                try:
                    writer.write(output_frame)
                except Exception as error:
                    writer_error.append(error)
                    break
                finally:
                    write_queue.task_done()

        writer_thread = None

        try:
            while True:
                if max_frames is not None and frame_index >= max_frames:
                    break

                frame = frame_queue.get()
                if frame is None:
                    break

                frame_start_time = time.perf_counter()

                if frame_index % self.TRACKING_INTERVAL == 0 or not has_cached_tracking:
                    stage_start_time = time.perf_counter()
                    (
                        cached_tracking_data,
                        cached_bbox_height_by_track_id,
                        cached_bbox_by_track_id,
                    ) = track_frame_with_bbox(
                        self.tracker,
                        frame,
                    )
                    stage_times["tracking"] += time.perf_counter() - stage_start_time
                    has_cached_tracking = True

                stage_start_time = time.perf_counter()
                (
                    pose_frame,
                    _,
                    _,
                ) = self.detector.draw_skeleton(
                    frame,
                    cached_bbox_by_track_id,
                    run_pose_detection=(frame_index % self.POSE_INTERVAL == 0),
                )
                stage_times["pose"] += time.perf_counter() - stage_start_time

                stage_start_time = time.perf_counter()
                is_live = self._is_live_frame(frame)
                if is_live:
                    score = self._read_score(frame, frame_index)
                    if score is not None:
                        if self.prev_score is None:
                            self.prev_score = score
                            self.score_confirm_count = 0
                        if score != self.prev_score:
                            self.score_confirm_count += 1
                            if self.score_confirm_count >= self.SCORE_CHANGE_CONFIRMATIONS:
                                self.analyzer.close_current_rally()
                                self.prev_score = score
                                self.score_confirm_count = 0
                        else:
                            self.score_confirm_count = 0

                    final_metrics = self.analyzer.analyze(cached_tracking_data)
                else:
                    final_metrics = self.analyzer.get_metrics()
                smoothed_tracking_data = self.analyzer.get_smoothed_tracking_data()
                stage_times["analysis"] += time.perf_counter() - stage_start_time

                stage_start_time = time.perf_counter()
                output_frame = self.visualizer.draw(
                    frame_index,
                    pose_frame,
                    smoothed_tracking_data,
                    final_metrics,
                )
                stage_times["visualization"] += time.perf_counter() - stage_start_time

                if writer is None:
                    writer = self._create_writer(output_frame, fps)
                    writer_thread = threading.Thread(target=write_frames, daemon=True)
                    writer_thread.start()

                stage_start_time = time.perf_counter()
                write_queue.put(output_frame)
                stage_times["write"] += time.perf_counter() - stage_start_time
                total_processing_time += time.perf_counter() - frame_start_time

                if writer_error:
                    raise RuntimeError("Video writer failed") from writer_error[0]

                if (frame_index + 1) % 100 == 0:
                    player1 = final_metrics.get(1, {}).get("distance_m", 0.0)
                    player2 = final_metrics.get(2, {}).get("distance_m", 0.0)
                    print(f"Frame {frame_index + 1}, Player1: {player1:.2f}m, Player2: {player2:.2f}m")

                frame_index += 1
        finally:
            stop_reader.set()
            reader_thread.join(timeout=1.0)
            if writer_thread is not None:
                write_queue.put(None)
                write_queue.join()
                stop_writer.set()
                writer_thread.join(timeout=5.0)
            capture.release()
            if writer is not None:
                writer.release()

        average_frame_time_ms = (total_processing_time / frame_index * 1000.0) if frame_index else 0.0
        average_stage_times_ms = {
            name: (elapsed / frame_index * 1000.0) if frame_index else 0.0
            for name, elapsed in stage_times.items()
        }
        return frame_index, final_metrics, average_frame_time_ms, average_stage_times_ms

    def _is_live_frame(self, frame):
        """Return True when the top-left scoreboard suggests this is live play."""
        roi_height = min(self.SCOREBOARD_ROI_HEIGHT, frame.shape[0])
        roi_width = min(self.SCOREBOARD_ROI_WIDTH, frame.shape[1])
        if roi_height == 0 or roi_width == 0:
            return True

        roi = frame[0:roi_height, 0:roi_width]
        if self._white_ratio(roi) > self.SCOREBOARD_WHITE_THRESHOLD:
            return True

        scan_width = min(frame.shape[1], self.SCOREBOARD_ROI_WIDTH * 4)
        for x in range(0, max(1, scan_width - roi_width + 1), self.SCOREBOARD_SCAN_STEP):
            roi = frame[0:roi_height, x : x + roi_width]
            if self._white_ratio(roi) > self.SCOREBOARD_WHITE_THRESHOLD:
                return True
        return False

    def _white_ratio(self, roi):
        """Return the share of bright white pixels in a scoreboard candidate ROI."""
        white_mask = cv2.inRange(roi, (200, 200, 200), (255, 255, 255))
        return cv2.countNonZero(white_mask) / float(roi.shape[0] * roi.shape[1])

    def _read_score(self, frame, frame_index):
        """Read the two scoreboard numbers with OCR at a throttled interval."""
        if frame_index % self.SCORE_OCR_INTERVAL != 0:
            return None

        y1, y2, x1, x2 = self.SCORE_DIGITS_ROI
        y2 = min(y2, frame.shape[0])
        x2 = min(x2, frame.shape[1])
        if y1 >= y2 or x1 >= x2:
            return None

        roi = frame[y1:y2, x1:x2]
        big = cv2.resize(roi, None, fx=4, fy=4)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        result = self.score_reader.readtext(
            threshold,
            allowlist="0123456789",
            detail=0,
            paragraph=False,
        )
        digits = []
        for text in result:
            digits.extend(int(match) for match in re.findall(r"\d+", str(text)))

        if len(digits) >= 2:
            return digits[0], digits[1]
        return None

    def _create_writer(self, output_frame, fps):
        """Create a VideoWriter for the visualized output."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        height, width = output_frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(self.output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {self.output_path}")
        return writer


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full badminton analysis pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--calibration", required=True, help="Path to H.npy calibration matrix")
    parser.add_argument("--output", required=True, help="Path to the output mp4 video")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    return parser.parse_args()


def main():
    """Process the video and print progress plus final player distances."""
    args = parse_args()
    app = BadmintonAnalysisApp(
        video_path=args.video,
        calibration_path=args.calibration,
        output_path=args.output,
        start_seconds=args.start,
    )
    (
        processed_frames,
        final_metrics,
        average_frame_time_ms,
        average_stage_times_ms,
    ) = app.run(max_frames=args.max_frames, duration_seconds=args.duration)

    player1 = final_metrics.get(1, {}).get("distance_m", 0.0)
    player2 = final_metrics.get(2, {}).get("distance_m", 0.0)
    print(f"Finished processing {processed_frames} frames")
    print(f"Player1 total distance: {player1:.2f}m")
    print(f"Player2 total distance: {player2:.2f}m")
    print(f"Average frame time: {average_frame_time_ms:.2f} ms")
    for name, elapsed_ms in average_stage_times_ms.items():
        print(f"Average {name} time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
