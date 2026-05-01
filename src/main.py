"""End-to-end badminton analysis pipeline for video processing and export."""

import argparse
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
import ffmpeg
import torch

from analyzer import Analyzer, track_frame_with_bbox
from detector import Detector
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

    raise RuntimeError("ffmpeg executable not found. Install it with: winget install ffmpeg")


class FFmpegVideoWriter:
    """Write raw BGR frames to an ffmpeg NVENC subprocess."""

    def __init__(self, output_path, fps, frame_size):
        self.output_path = Path(output_path)
        self.fps = fps
        self.width, self.height = frame_size
        self.stderr_lines = deque(maxlen=50)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        stream = ffmpeg.input(
            "pipe:0",
            format="rawvideo",
            vcodec="rawvideo",
            s=f"{self.width}x{self.height}",
            pix_fmt="bgr24",
            r=str(self.fps),
        ).output(
            str(self.output_path),
            vcodec="h264_nvenc",
            preset="fast",
            crf="18",
        )
        self.command = ffmpeg.compile(stream.overwrite_output(), cmd=find_ffmpeg_executable())
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
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
    SCORE_ROI = (60, 120, 380, 700)
    SCORE_DIGITS_ROI = (50, 135, 630, 705)
    SCORE_CHANGE_CONFIRMATIONS = 3
    MIN_RALLY_FRAME_GAP = 200
    MISSING_PLAYER_CONFIRM_FRAMES = 100
    FALLBACK_RALLY_FRAME_GAP = 300

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
        self.last_rally_frame = -max(self.MIN_RALLY_FRAME_GAP, self.FALLBACK_RALLY_FRAME_GAP)
        self.missing_frames = 0
        self.set_number = 1
        self.side_flipped = False
        self.set_scores = {1: [0, 0], 2: [0, 0], 3: [0, 0]}
        self.decided_sets = 0

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
                    self.visualizer.reset_rally_chart()
                    self.last_rally_frame = frame_index
                    self.missing_frames = 0

                if is_live:
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

                if (frame_index + 1) % 1000 == 0:
                    player1 = final_metrics.get(1, {}).get("distance_m", 0.0)
                    player2 = final_metrics.get(2, {}).get("distance_m", 0.0)
                    average_frame_time_ms = total_processing_time / (frame_index + 1) * 1000.0
                    print(
                        f"Frame {frame_index + 1}, "
                        f"Player1: {player1:.2f}m, "
                        f"Player2: {player2:.2f}m, "
                        f"Average frame time: {average_frame_time_ms:.2f} ms"
                    )

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
            print(f"换边！当前局：{self.set_number}，side_flipped：{self.side_flipped}")

    def _create_writer(self, output_frame, fps):
        """Create an ffmpeg NVENC writer for the visualized output."""
        height, width = output_frame.shape[:2]
        return FFmpegVideoWriter(self.output_path, fps, (width, height))


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
    rally_count = len(final_metrics.get("rally_history", []))
    print(f"Finished processing {processed_frames} frames")
    print(f"Player1 total distance: {player1:.2f}m")
    print(f"Player2 total distance: {player2:.2f}m")
    print(f"Rally count: {rally_count}")
    print(f"Average frame time: {average_frame_time_ms:.2f} ms")
    for name, elapsed_ms in average_stage_times_ms.items():
        print(f"Average {name} time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()
