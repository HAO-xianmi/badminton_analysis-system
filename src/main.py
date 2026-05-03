"""End-to-end badminton analysis pipeline for video processing and export."""

import argparse
import os
import re
import shutil
import subprocess
import threading
import time
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue

import cv2
import easyocr
import numpy as np
import torch

from analyzer import Analyzer, track_frame_with_bbox
from ball_analyzer import BallAnalyzer
from ball_detector import BallDetector
from ball_tracker import BallTracker
from detector import Detector
from paths import ball_speed_csv_path, tracknet_weights_path
from source_manager import SourceManager
from tracker import Tracker
from visualizer import Visualizer


def open_video_at_start(video_path, start_seconds):
    """Open a video with FFMPEG hardware acceleration and seek to start time."""
    capture = cv2.VideoCapture(str(video_path), cv2.CAP_FFMPEG)
    if hasattr(cv2, "CAP_PROP_HW_ACCELERATION") and hasattr(cv2, "VIDEO_ACCELERATION_ANY"):
        capture.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Cannot open video: {video_path}")

    capture.set(cv2.CAP_PROP_POS_MSEC, start_seconds * 1000)
    return capture


def detect_scene_cut(prev_frame, curr_frame, threshold=20):
    """Return True when adjacent frames differ enough to be a scene cut."""
    mean_brightness = curr_frame.mean()
    if mean_brightness < 30:
        return True
    if prev_frame is None:
        return False
    diff = cv2.absdiff(prev_frame, curr_frame)
    mean_diff = diff.mean()
    return mean_diff > threshold


def is_valid_court_view(frame, H):
    """Return True when the calibrated court is visible in the current frame."""
    court_corners = np.float32([[0, 0], [610, 0], [610, 1340], [0, 1340]])
    H_inv = np.linalg.inv(H)
    img_corners = cv2.perspectiveTransform(
        court_corners.reshape(-1, 1, 2),
        H_inv,
    ).reshape(-1, 2)
    h, w = frame.shape[:2]
    in_frame_count = sum(1 for p in img_corners if 0 <= p[0] <= w and 0 <= p[1] <= h)
    if in_frame_count < 2:
        return False

    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [img_corners.astype(np.int32)], 255)
    court_area = cv2.countNonZero(mask)
    if court_area == 0:
        return False

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, (35, 45, 35), (95, 255, 255))
    green_ratio = cv2.countNonZero(cv2.bitwise_and(green_mask, green_mask, mask=mask)) / court_area
    return green_ratio >= 0.08


def find_ffmpeg_executable():
    """Return ffmpeg from PATH or the WinGet install location."""
    executable = shutil.which("ffmpeg")
    if executable:
        return executable

    package_root = Path.home() / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if package_root.exists():
        matches = sorted(package_root.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe"))
        if matches:
            return str(matches[-1])

    common_paths = [
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/Program Files/Gyan/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
    ]
    for candidate in common_paths:
        if candidate.exists():
            return str(candidate)

    raise RuntimeError("ffmpeg executable not found. Install it with: winget install --id Gyan.FFmpeg -e")


def has_ffmpeg_encoder(ffmpeg_path, encoder):
    """Return True when the local ffmpeg build advertises an encoder."""
    try:
        result = subprocess.run(
            [ffmpeg_path, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return encoder in result.stdout


def preferred_h264_encoder(ffmpeg_path):
    """Choose a portable H.264 encoder, using NVENC only when it is likely usable."""
    forced_codec = os.environ.get("BADMINTON_FFMPEG_CODEC")
    if forced_codec:
        return forced_codec

    if torch.cuda.is_available() and has_ffmpeg_encoder(ffmpeg_path, "h264_nvenc"):
        return "h264_nvenc"
    return "libx264"


def create_writer(output_path, fps, width, height):
    """Create an ffmpeg process that accepts raw BGR frames on stdin."""
    width = width if width % 2 == 0 else width + 1
    height = height if height % 2 == 0 else height + 1
    ffmpeg_path = find_ffmpeg_executable()
    codec = preferred_h264_encoder(ffmpeg_path)
    preset = "fast" if codec == "h264_nvenc" else "veryfast"
    cmd = [
        ffmpeg_path,
        "-y",
        "-f",
        "rawvideo",
        "-vcodec",
        "rawvideo",
        "-s",
        f"{width}x{height}",
        "-pix_fmt",
        "bgr24",
        "-r",
        str(int(fps)),
        "-i",
        "pipe:0",
        "-vcodec",
        codec,
        "-preset",
        preset,
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    return subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)


class FFmpegVideoWriter:
    """Write raw BGR frames to an ffmpeg NVENC subprocess."""

    def __init__(self, output_path, fps, frame_size):
        self.output_path = Path(output_path)
        self.fps = fps
        source_width, source_height = frame_size
        self.width = source_width if source_width % 2 == 0 else source_width + 1
        self.height = source_height if source_height % 2 == 0 else source_height + 1
        self.stderr_lines = deque(maxlen=50)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        self.process = create_writer(self.output_path, self.fps, self.width, self.height)
        self.stderr_thread = threading.Thread(target=self._drain_stderr, daemon=True)
        self.stderr_thread.start()

    def _drain_stderr(self):
        if self.process.stderr is None:
            return
        for line in self.process.stderr:
            self.stderr_lines.append(line.decode("utf-8", errors="replace").rstrip())

    def write(self, frame):
        """Write one BGR frame to ffmpeg stdin."""
        if self.process.stdin is None:
            raise RuntimeError("ffmpeg stdin is not available")

        return_code = self.process.poll()
        if return_code is not None:
            raise RuntimeError(self._format_error(return_code))

        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.copyMakeBorder(
                frame,
                0,
                self.height - frame.shape[0],
                0,
                self.width - frame.shape[1],
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

        try:
            self.process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            return_code = self.process.poll()
            raise RuntimeError(self._format_error(return_code)) from error

    def release(self):
        """Finish encoding and wait for ffmpeg to exit."""
        if self.process.stdin is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass

        return_code = self.process.wait()
        if self.stderr_thread is not None:
            self.stderr_thread.join(timeout=1.0)

        if return_code != 0:
            raise RuntimeError(self._format_error(return_code))

    def _format_error(self, return_code):
        message = f"ffmpeg exited with code {return_code}"
        if self.stderr_lines:
            message += ":\n" + "\n".join(self.stderr_lines)
        return message


def check_side_flip(p1, p2, set_number, decided_sets, side_flipped):
    """Return whether the set ended, next set number, and side-flip state."""

    def set_over(a, b):
        if max(a, b) >= 30:
            return True
        if max(a, b) >= 21 and abs(a - b) >= 2:
            return True
        return False

    if set_over(p1, p2) and decided_sets < set_number:
        return True, min(set_number + 1, 3), not side_flipped

    if set_number == 3 and max(p1, p2) == 11 and not side_flipped:
        return False, set_number, not side_flipped

    return False, set_number, side_flipped


class BadmintonAnalysisApp:
    """Run tracking, analysis, and visualization over a video segment."""

    TRACKING_INTERVAL = 2
    POSE_INTERVAL = 3
    SCOREBOARD_ROI_WIDTH = 200
    SCOREBOARD_ROI_HEIGHT = 100
    SCOREBOARD_WHITE_THRESHOLD = 0.05
    SCOREBOARD_SCAN_STEP = 50
    SCORE_OCR_INTERVAL = 15
    SCORE_CHANGE_CONFIRMATIONS = 3
    MIN_RALLY_FRAME_GAP = 200
    MISSING_PLAYER_CONFIRM_FRAMES = 100
    FALLBACK_RALLY_FRAME_GAP = 300

    def __init__(
        self,
        video_path,
        calibration_path,
        output_path,
        start_seconds=0,
        fps=None,
        score_roi=None,
        court_h_threshold=None,
        side_flipped=False,
        ball_detection_config=None,
    ):
        self.video_path = Path(video_path)
        self.calibration_path = Path(calibration_path)
        self.output_path = Path(output_path)
        self.start_seconds = start_seconds
        self.config_fps = fps
        self.score_roi = self._normalize_score_roi(score_roi)
        if court_h_threshold is None:
            raise ValueError("Source config is missing court_h_threshold.")
        self.court_h_threshold = court_h_threshold
        self.ball_detection_config = ball_detection_config or {}

        self.detector = Detector()
        self.tracker = Tracker(
            str(self.calibration_path),
            court_h_threshold=self.court_h_threshold,
        )
        self.h_matrix = self.tracker.h_matrix
        self.analyzer = Analyzer()
        self.visualizer = Visualizer()
        self.ball_detector = None
        self.ball_tracker = None
        self.ball_analyzer = None
        self.ball_h_matrix = None
        self.ball_detected_frames = 0
        self._init_ball_pipeline()
        self.score_reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())
        self.prev_score = None
        self.score_confirm_count = 0
        self.last_rally_frame = -max(self.MIN_RALLY_FRAME_GAP, self.FALLBACK_RALLY_FRAME_GAP)
        self.missing_frames = 0
        self.set_number = 1
        self.side_flipped = side_flipped
        self.set_scores = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        self.decided_sets = 0
        self.side_flip_count = 0

    def _init_ball_pipeline(self):
        """Initialize shuttlecock speed analysis when TrackNet weights exist."""
        if self.ball_detection_config and not self.ball_detection_config.get("enabled", True):
            print("Ball speed disabled: source config ball_detection.enabled is false")
            return

        weights_path = tracknet_weights_path(required=False)
        if not weights_path.exists():
            print(f"Ball speed disabled: TrackNetV2 weights not found at {weights_path}")
            return

        try:
            h_matrix = np.load(str(self.calibration_path))
            self.ball_detector = BallDetector(
                weights_path,
                device="cuda" if torch.cuda.is_available() else "cpu",
                min_confidence=self.ball_detection_config.get("min_confidence", 0.5),
            )
            self.ball_tracker = BallTracker()
            self.ball_analyzer = BallAnalyzer(
                h_matrix,
                fps=self.config_fps or 50.0,
                csv_path=ball_speed_csv_path(),
            )
            self.ball_h_matrix = h_matrix
            print(f"Ball speed enabled: {weights_path}")
        except Exception as error:
            self.ball_detector = None
            self.ball_tracker = None
            self.ball_analyzer = None
            self.ball_h_matrix = None
            print(f"Ball speed disabled: failed to initialize TrackNetV2 ({error})")

    def run(self, max_frames=None, duration_seconds=None):
        """Process the video, save visualization output, and return final metrics."""
        capture = open_video_at_start(self.video_path, self.start_seconds)
        fps = self.config_fps or capture.get(cv2.CAP_PROP_FPS) or 30.0
        source_total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if duration_seconds is not None:
            duration_frames = max(0, int(round(duration_seconds * fps)))
            max_frames = duration_frames if max_frames is None else min(max_frames, duration_frames)
        total_frames = max_frames or source_total_frames
        writer = None
        frame_index = 0
        final_metrics = {}
        total_processing_time = 0.0
        run_start_time = time.perf_counter()
        last_progress_time = run_start_time
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
        prev_frame = None
        scene_cut_cooldown = 0

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
                    stop_writer.set()
                    while True:
                        try:
                            write_queue.get_nowait()
                            write_queue.task_done()
                        except Empty:
                            break
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
                valid_court_view = self._is_valid_ball_view(frame)
                if detect_scene_cut(prev_frame, frame) or not valid_court_view:
                    scene_cut_cooldown = 3
                scene_cut_active = scene_cut_cooldown > 0
                if scene_cut_active:
                    self._reset_ball_motion_state()
                    scene_cut_cooldown -= 1
                prev_frame = frame.copy()
                self.tracker.mapper.side_flipped = self.side_flipped

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
                rally_closed_this_frame = False
                if is_live:
                    score = self._read_score(frame, frame_index)
                    if score is not None:
                        if self.prev_score is None:
                            self.prev_score = score
                            self.score_confirm_count = 0
                        if score != self.prev_score:
                            self.score_confirm_count += 1
                            if (
                                self.score_confirm_count >= self.SCORE_CHANGE_CONFIRMATIONS
                                and frame_index - self.last_rally_frame >= self.MIN_RALLY_FRAME_GAP
                            ):
                                self.analyzer.close_current_rally()
                                if self.ball_analyzer is not None:
                                    self.ball_analyzer.reset_rally()
                                self.visualizer.reset_rally_chart()
                                self.prev_score = score
                                self._update_side_flip_state(score)
                                self.score_confirm_count = 0
                                self.last_rally_frame = frame_index
                                rally_closed_this_frame = True
                        else:
                            self.score_confirm_count = 0

                current_players = cached_tracking_data if is_live else {}
                if len(current_players) == 0:
                    self.missing_frames += 1
                else:
                    self.missing_frames = 0

                if (
                    not rally_closed_this_frame
                    and self.missing_frames >= self.MISSING_PLAYER_CONFIRM_FRAMES
                    and frame_index - self.last_rally_frame >= self.FALLBACK_RALLY_FRAME_GAP
                ):
                    self.analyzer.close_current_rally()
                    if self.ball_analyzer is not None:
                        self.ball_analyzer.reset_rally()
                    self.visualizer.reset_rally_chart()
                    self.last_rally_frame = frame_index
                    self.missing_frames = 0

                if is_live:
                    final_metrics = self.analyzer.analyze(cached_tracking_data)
                else:
                    final_metrics = self.analyzer.get_metrics()
                smoothed_tracking_data = self.analyzer.get_smoothed_tracking_data()
                if scene_cut_active:
                    ball_state = self._scene_cut_ball_state()
                else:
                    ball_state = self._update_ball_state(
                        frame,
                        smoothed_tracking_data,
                        frame_index,
                        len(final_metrics.get("rally_history", [])) + 1,
                        fps,
                    )
                stage_times["analysis"] += time.perf_counter() - stage_start_time

                stage_start_time = time.perf_counter()
                output_frame = self.visualizer.draw(
                    frame_index,
                    pose_frame,
                    smoothed_tracking_data,
                    final_metrics,
                    ball_state=ball_state,
                    scene_cut=scene_cut_active,
                    H=self.h_matrix,
                    fps=fps,
                )
                stage_times["visualization"] += time.perf_counter() - stage_start_time

                if writer is None:
                    writer = self._create_writer(output_frame, fps)
                    writer_thread = threading.Thread(target=write_frames, daemon=True)
                    writer_thread.start()

                stage_start_time = time.perf_counter()
                while True:
                    if writer_error:
                        raise RuntimeError("Video writer failed") from writer_error[0]
                    try:
                        write_queue.put(output_frame, timeout=0.1)
                        break
                    except Full:
                        continue
                stage_times["write"] += time.perf_counter() - stage_start_time
                total_processing_time += time.perf_counter() - frame_start_time

                if writer_error:
                    raise RuntimeError("Video writer failed") from writer_error[0]

                self.tracker.mapper.side_flipped = self.side_flipped

                current_time = time.perf_counter()
                should_print_progress = (
                    current_time - last_progress_time >= 300.0
                    or (total_frames and frame_index + 1 >= total_frames)
                )
                if should_print_progress:
                    player1 = final_metrics.get(1, {}).get("distance_m", 0.0)
                    player2 = final_metrics.get(2, {}).get("distance_m", 0.0)
                    average_frame_time_ms = total_processing_time / (frame_index + 1) * 1000.0
                    progress_percent = (
                        (frame_index + 1) / total_frames * 100.0
                        if total_frames
                        else 0.0
                    )
                    remaining_frames = max(total_frames - frame_index - 1, 0) if total_frames else 0
                    eta_minutes = remaining_frames * average_frame_time_ms / 1000.0 / 60.0
                    print(
                        f"Frame {frame_index + 1}/{total_frames or '?'} | "
                        f"进度{progress_percent:.1f}% | "
                        f"P1: {player1:.2f}m P2: {player2:.2f}m | "
                        f"avg:{average_frame_time_ms:.2f}ms | "
                        f"预计剩余:{eta_minutes:.1f}分钟",
                        flush=True,
                    )
                    last_progress_time = current_time

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
        total_elapsed_minutes = (time.perf_counter() - run_start_time) / 60.0
        return (
            frame_index,
            final_metrics,
            average_frame_time_ms,
            average_stage_times_ms,
            total_elapsed_minutes,
        )

    def _reset_ball_motion_state(self):
        if self.ball_detector is not None:
            self.ball_detector.frame_buffer = []
        if self.ball_tracker is not None:
            self.ball_tracker.reset()
        if self.ball_analyzer is not None:
            self.ball_analyzer.reset_position()
        self.visualizer.clear_ball_trail()

    def _is_valid_ball_view(self, frame):
        if self.ball_detector is None or self.ball_h_matrix is None:
            return True
        return is_valid_court_view(frame, self.ball_h_matrix)

    def _scene_cut_ball_state(self):
        if self.ball_analyzer is None:
            return {
                "current_speed_kmh": None,
                "rally_max_speed": {1: 0.0, 2: 0.0},
                "shots": [],
                "pos": None,
                "scene_cut": True,
            }
        state = self.ball_analyzer.get_state()
        state["current_speed_kmh"] = None
        state["pos"] = None
        state["detection"] = None
        state["speed_kmh"] = None
        state["detected_frames"] = self.ball_detected_frames
        state["scene_cut"] = True
        return state

    def _update_ball_state(self, frame, player_positions, frame_index, rally_id, fps):
        if self.ball_detector is None or self.ball_tracker is None or self.ball_analyzer is None:
            return None

        if self.ball_analyzer.fps != fps:
            self.ball_analyzer.fps = fps

        detection = self.ball_detector.detect(frame)
        if detection is not None:
            self.ball_detected_frames += 1

        ball_pos = self.ball_tracker.update(detection)
        speed = self.ball_analyzer.update(ball_pos, player_positions, frame_index, rally_id)
        state = self.ball_analyzer.get_state()
        state["pos"] = ball_pos
        state["detection"] = detection
        state["speed_kmh"] = speed
        state["detected_frames"] = self.ball_detected_frames
        return state

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

        y1, y2, x1, x2 = self.score_roi
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

    def _update_side_flip_state(self, score):
        """Update set score bookkeeping and flip court-side mapping when needed."""
        p1, p2 = score
        self.set_scores[self.set_number] = [p1, p2]
        set_finished, next_set_number, next_side_flipped = check_side_flip(
            p1,
            p2,
            self.set_number,
            self.decided_sets,
            self.side_flipped,
        )

        side_flipped_changed = next_side_flipped != self.side_flipped
        if set_finished:
            self.decided_sets += 1

        self.set_number = next_set_number
        self.side_flipped = next_side_flipped

        if side_flipped_changed:
            self.side_flip_count += 1
            print(f"Side flip | set: {self.set_number} | side_flipped: {self.side_flipped}")

    def _create_writer(self, output_frame, fps):
        """Create an ffmpeg NVENC writer for the visualized output."""
        height, width = output_frame.shape[:2]
        return FFmpegVideoWriter(self.output_path, fps, (width, height))

    def _normalize_score_roi(self, score_roi):
        """Return score ROI as (y1, y2, x1, x2)."""
        if score_roi is None:
            raise ValueError("Source config is missing score_roi.")
        return (
            int(score_roi["y1"]),
            int(score_roi["y2"]),
            int(score_roi["x1"]),
            int(score_roi["x2"]),
        )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run the full badminton analysis pipeline")
    parser.add_argument("--source", required=True, help="Configured source name")
    parser.add_argument("--video", default=None, help="Override video path from source config")
    parser.add_argument("--output", required=True, help="Path to the output mp4 video")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--duration", type=float, default=None, help="Duration to process in seconds")
    parser.add_argument("--max-frames", type=int, default=None, help="Maximum frames to process")
    return parser.parse_args()


def main():
    """Process the video and print progress plus final player distances."""
    args = parse_args()
    manager = SourceManager()
    config = manager.load_source(args.source)
    video_path = args.video or config["video_path"]
    app = BadmintonAnalysisApp(
        video_path=video_path,
        calibration_path=config["h_matrix_path"],
        output_path=args.output,
        start_seconds=args.start,
        fps=config.get("fps"),
        score_roi=config.get("score_roi"),
        court_h_threshold=config["court_h_threshold"],
        side_flipped=config.get("side_flipped", False),
        ball_detection_config=config.get("ball_detection"),
    )
    (
        processed_frames,
        final_metrics,
        average_frame_time_ms,
        average_stage_times_ms,
        total_elapsed_minutes,
    ) = app.run(max_frames=args.max_frames, duration_seconds=args.duration)

    player1 = final_metrics.get(1, {}).get("distance_m", 0.0)
    player2 = final_metrics.get(2, {}).get("distance_m", 0.0)
    rally_count = len(final_metrics.get("rally_history", []))
    completion_titles = {
        "momota_vs_ishiyuki_2018": "第一个视频完成",
        "limeijia_vs_axelsen_2021": "第二个视频完成",
    }
    title = completion_titles.get(args.source, "视频完成")
    print(f"=== {title} ===")
    print(f"总帧数：{processed_frames}")
    print(f"P1总距离：{player1:.2f}m")
    print(f"P2总距离：{player2:.2f}m")
    print(f"Rally数量：{rally_count}")
    print(f"总耗时：{total_elapsed_minutes:.2f}分钟")
    print(f"Finished processing {processed_frames} frames")
    print(f"Player1 total distance: {player1:.2f}m")
    print(f"Player2 total distance: {player2:.2f}m")
    print(f"Rally count: {rally_count}")
    print(f"Side flip count: {app.side_flip_count}")
    print(f"Ball detected frames: {app.ball_detected_frames}")
    print(f"Average frame time: {average_frame_time_ms:.2f} ms")
    for name, elapsed_ms in average_stage_times_ms.items():
        print(f"Average {name} time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
