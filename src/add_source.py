"""Interactive workflow for adding a badminton video source."""

import argparse
import re
from pathlib import Path

import cv2
import easyocr
import torch

from calibrate import Calibrator
from source_manager import SourceManager


class ScoreRoiCalibrator:
    """Select and OCR-test the scoreboard digits ROI."""

    WINDOW_NAME = "Scoreboard ROI"

    def __init__(self, video_path, frame_seconds):
        self.video_path = Path(video_path)
        self.frame_seconds = frame_seconds
        self.reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available())

    def run(self):
        """Loop until the selected ROI can be read as two score numbers."""
        frame = self._load_frame()
        print("请用鼠标框选记分牌数字区域。")
        print("按 Enter/Space 确认，按 Esc 取消。")

        while True:
            roi = cv2.selectROI(self.WINDOW_NAME, frame, showCrosshair=True, fromCenter=False)
            cv2.destroyWindow(self.WINDOW_NAME)
            x, y, w, h = [int(value) for value in roi]
            if w <= 0 or h <= 0:
                print("Scoreboard ROI calibration was cancelled.")
                return None

            score = self.read_score(frame, y, y + h, x, x + w)
            if score is not None:
                print(f"✓ 识别成功：{score[0]} - {score[1]}")
                return y, y + h, x, x + w

            print("未能识别出两个数字，请重新框选。")

    def _load_frame(self):
        capture = cv2.VideoCapture(str(self.video_path))
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"无法打开视频：{self.video_path}")

        capture.set(cv2.CAP_PROP_POS_MSEC, self.frame_seconds * 1000)
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            raise RuntimeError(f"无法读取 {self.frame_seconds} 秒处的视频帧。")
        return frame

    def read_score(self, frame, y1, y2, x1, x2):
        """Read two numbers from a selected scoreboard ROI."""
        y1 = max(0, min(int(y1), frame.shape[0]))
        y2 = max(0, min(int(y2), frame.shape[0]))
        x1 = max(0, min(int(x1), frame.shape[1]))
        x2 = max(0, min(int(x2), frame.shape[1]))
        if y1 >= y2 or x1 >= x2:
            return None

        roi = frame[y1:y2, x1:x2]
        big = cv2.resize(roi, None, fx=4, fy=4)
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, threshold = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        result = self.reader.readtext(
            threshold,
            allowlist="0123456789",
            detail=1,
            paragraph=False,
        )

        numbers = []
        for box, text, _ in sorted(result, key=lambda item: _box_left(item[0])):
            _ = box
            numbers.extend(re.findall(r"\d+", str(text)))

        if len(numbers) >= 2:
            return int(numbers[0]), int(numbers[1])
        if len(numbers) == 1 and len(numbers[0]) == 2:
            return int(numbers[0][0]), int(numbers[0][1])
        return None


def _box_left(box):
    return min(point[0] for point in box)


def prompt_text(message, default=None):
    """Prompt for one input value, with an optional default."""
    suffix = f" [{default}]" if default not in (None, "") else ""
    value = input(f"{message}{suffix}: ").strip()
    return value or default


def prompt_float(message, default=None):
    """Prompt for one float value."""
    while True:
        value = prompt_text(message, default)
        try:
            return float(value)
        except (TypeError, ValueError):
            print("请输入数字。")


def print_sources(sources):
    """Print configured sources in a compact table."""
    if not sources:
        print("还没有已配置的片源。")
        return

    print("已配置片源：")
    for source in sources:
        h_status = "已标定" if source["calibrated"] else "未标定"
        roi_status = "ROI已校准" if source["score_roi_calibrated"] else "ROI未校准"
        print(f"- {source['name']} | {h_status} | {roi_status} | {source['video']}")


def validate_source_name(source_name):
    """Keep source names safe for config and H-matrix filenames."""
    if not source_name:
        raise ValueError("片源名称不能为空。")
    if any(char in source_name for char in '\\/:*?"<>|'):
        raise ValueError('片源名称不能包含：\\ / : * ? " < > |')


def add_source_interactive():
    """Guide the user through creating and calibrating a source."""
    manager = SourceManager()

    source_name = prompt_text("请输入片源名称（如 match_2026_uberCup）")
    validate_source_name(source_name)
    video_path = prompt_text("请输入视频完整路径")
    if not Path(video_path).exists():
        raise FileNotFoundError(f"视频不存在：{video_path}")

    config = manager.create_source(video_path, source_name)
    width, height = config["resolution"]
    print(
        f"视频信息：{width}x{height} | "
        f"{config['fps']:.2f} fps | {config['total_frames']} frames"
    )

    calibration_seconds = prompt_float("请输入标定开始时间（秒，找到有完整球场的帧）", 0)
    print("请依次点击球场四个角点：左上、右上、右下、左下。")
    h_matrix = Calibrator(
        video_path,
        start_seconds=calibration_seconds,
        output_path=config["h_matrix_path"],
    ).run()
    if h_matrix is None:
        raise RuntimeError("标定未完成。")

    score_seconds = prompt_float("请找到有记分牌的帧时间（秒）", calibration_seconds)
    selected_roi = ScoreRoiCalibrator(video_path, score_seconds).run()
    if selected_roi is None:
        raise RuntimeError("记分牌ROI校准未完成。")

    y1, y2, x1, x2 = selected_roi
    manager.save_score_roi(y1, y2, x1, x2)
    print(f"配置已保存：{manager.config_path}")


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Add or list badminton video sources")
    parser.add_argument("--list", action="store_true", help="List configured sources")
    return parser.parse_args()


def main():
    """Program entry point."""
    args = parse_args()
    manager = SourceManager()
    if args.list:
        print_sources(manager.list_sources())
        return

    add_source_interactive()


if __name__ == "__main__":
    main()
