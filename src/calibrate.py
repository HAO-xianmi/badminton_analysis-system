"""Interactive video-frame calibration tool for badminton court homography."""

import argparse
from pathlib import Path

import cv2
import numpy as np


class Calibrator:
    """Collect four court points from a chosen video timestamp and save homography."""

    WINDOW_NAME = "Calibration"
    OUTPUT_PATH = Path(r"F:\Fun-Activities\badminton_analysis\data\calibration\H.npy")
    VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv"}

    def __init__(self, video_path, start_seconds=0, output_path=None):
        self.video_path = Path(video_path)
        self.start_seconds = start_seconds
        self.output_path = Path(output_path) if output_path is not None else self.OUTPUT_PATH
        self.frame = None
        self.display_frame = None
        self.clicked_points = []

    def resolve_video_path(self):
        """Return a usable video file path from a file or directory input."""
        if self.video_path.is_file():
            return self.video_path

        if self.video_path.is_dir():
            for candidate in sorted(self.video_path.iterdir()):
                if candidate.is_file() and candidate.suffix.lower() in self.VIDEO_EXTENSIONS:
                    return candidate
            return None

        return None

    def load_frame(self):
        """Load one frame after seeking to the requested start time."""
        video_file = self.resolve_video_path()
        if video_file is None:
            return False

        capture = cv2.VideoCapture(str(video_file))
        if not capture.isOpened():
            capture.release()
            return False

        capture.set(cv2.CAP_PROP_POS_MSEC, self.start_seconds * 1000)
        success, frame = capture.read()
        capture.release()
        if not success or frame is None:
            return False

        self.frame = frame
        self.display_frame = frame.copy()
        self.video_path = video_file
        return True

    def draw_point(self, x, y):
        """Draw a red point on the display frame."""
        cv2.circle(self.display_frame, (x, y), 6, (0, 0, 255), -1)

    def on_mouse_click(self, event, x, y, flags, param):
        """Handle left-click events and collect up to four points."""
        _ = flags, param
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if len(self.clicked_points) >= 4:
            return

        self.clicked_points.append((x, y))
        print(f"Point {len(self.clicked_points)}: ({x}, {y})", flush=True)
        self.draw_point(x, y)
        cv2.imshow(self.WINDOW_NAME, self.display_frame)

    def compute_homography(self):
        """Compute the homography matrix from clicked points to court coordinates."""
        src_points = np.array(self.clicked_points, dtype=np.float32)
        dst_points = np.array(
            [(0, 0), (610, 0), (610, 1340), (0, 1340)],
            dtype=np.float32,
        )
        h_matrix, _ = cv2.findHomography(src_points, dst_points)
        return h_matrix

    def save_results(self, h_matrix):
        """Save the homography matrix to disk."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(self.output_path, h_matrix)

    def run(self):
        """Run the interactive calibration workflow."""
        if not self.load_frame():
            print("Cannot read a usable video frame. Check the video path and start time.")
            return None

        cv2.namedWindow(self.WINDOW_NAME)
        cv2.setMouseCallback(self.WINDOW_NAME, self.on_mouse_click)
        cv2.imshow(self.WINDOW_NAME, self.display_frame)

        while len(self.clicked_points) < 4:
            key = cv2.waitKey(20) & 0xFF
            if key == 27:
                break

        if len(self.clicked_points) == 4:
            h_matrix = self.compute_homography()
            self.save_results(h_matrix)
            print(f"Calibration complete. H matrix saved to: {self.output_path}")
            cv2.destroyAllWindows()
            return h_matrix

        cv2.destroyAllWindows()
        return None


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Badminton court calibration tool")
    parser.add_argument("--video", required=True, help="Path to a video file or folder")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--output", default=None, help="Path to save H.npy calibration matrix")
    return parser.parse_args()


def main():
    """Program entry point."""
    args = parse_args()
    calibrator = Calibrator(args.video, start_seconds=args.start, output_path=args.output)
    calibrator.run()


if __name__ == "__main__":
    main()
