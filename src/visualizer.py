"""Realtime visualization for badminton tracking and analysis."""

import argparse
from collections import defaultdict, deque
from pathlib import Path

import cv2
import numpy as np

from analyzer import Analyzer, track_frame_with_bbox
from tracker import Tracker, open_video_at_start

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:
    Image = None
    ImageDraw = None
    ImageFont = None


class Visualizer:
    """Render a left metrics panel plus the original video frame."""

    PANEL_WIDTH = 300
    PANEL_HEIGHT = 1080
    COURT_HEIGHT = 630
    SPEED_PANEL_HEIGHT = 180
    CHART_HEIGHT = PANEL_HEIGHT - COURT_HEIGHT - SPEED_PANEL_HEIGHT
    RALLY_TEXT_HEIGHT = 62
    SOURCE_WIDTH = 610
    SOURCE_HEIGHT = 1340
    HISTORY_LENGTH = 80
    BALL_TRAIL_LENGTH = 20
    SPEED_HISTORY_LENGTH = 50
    PANEL_RENDER_INTERVAL = 3
    TOTAL_DISTANCE_TEXT_HEIGHT = 30
    COURT_HORIZONTAL_LINES = (0, 72, 460, 670, 880, 1268, 1340)
    COURT_FULL_VERTICAL_LINES = (0, 46, 564, 610)
    COURT_CENTER_X = 305
    COURT_CENTER_LINE_Y1 = 460
    COURT_CENTER_LINE_Y2 = 880
    PLAYER_COLORS = {
        1: (0, 255, 0),
        2: (255, 0, 0),
    }
    GRID_COLOR = (210, 210, 210)
    TEXT_COLOR = (30, 30, 30)

    def __init__(self):
        self.position_history = defaultdict(lambda: deque(maxlen=self.HISTORY_LENGTH))
        self.stay_counts = defaultdict(int)
        self.frames = []
        self.p1_dist_history = []
        self.p2_dist_history = []
        self.distance_history = {
            1: self.p1_dist_history,
            2: self.p2_dist_history,
        }
        self.total_distances = {1: 0.0, 2: 0.0}
        self.rally_history = []
        self.ball_history = deque(maxlen=self.BALL_TRAIL_LENGTH)
        self.ball_speed_frames = deque(maxlen=self.SPEED_HISTORY_LENGTH)
        self.ball_speed_history = deque(maxlen=self.SPEED_HISTORY_LENGTH)
        self.current_ball_speed = 0.0
        self.rally_max_speed = {1: 0.0, 2: 0.0}
        self.rally_font = self._load_rally_font()
        self.court_background = self._create_court_background()
        self.cached_panel = None
        self.cached_padded_panel = None

    def draw(self, frame_index, frame_bgr, tracking_data, metrics, ball_state=None, scene_cut=False):
        """Return one combined BGR visualization frame."""
        if scene_cut:
            self.ball_history.clear()
        self._update_history(frame_index, tracking_data, metrics, ball_state, scene_cut=scene_cut)
        if self.cached_panel is None or frame_index % self.PANEL_RENDER_INTERVAL == 0:
            court_view = self._draw_court_view()
            distance_chart = self._draw_line_chart(
                title="Distance (m)",
                history_by_player=self.distance_history,
                value_min=0.0,
                value_max=self._distance_max(),
                footer_lines=self._recent_rally_lines(),
                total_distances=self.total_distances,
            )
            speed_panel = self._draw_speed_panel()
            self.cached_panel = cv2.vconcat([court_view, distance_chart, speed_panel])
            self.cached_padded_panel = None
        panel = self._panel_for_frame_height(frame_bgr.shape[0])
        video_frame = self._draw_ball_overlay(frame_bgr.copy(), scene_cut=scene_cut)
        return cv2.hconcat([panel, video_frame])

    def save_video(self, output_path):
        """Placeholder for future video export."""
        raise NotImplementedError("Video export is not implemented yet.")

    def draw_court(self):
        """Return a clean top-down court view for standalone inspection."""
        return self.court_background.copy()

    def reset_rally_chart(self):
        """Clear per-rally distance chart data after a confirmed score change."""
        self.frames = []
        self.p1_dist_history = []
        self.p2_dist_history = []
        self.distance_history = {
            1: self.p1_dist_history,
            2: self.p2_dist_history,
        }
        self.ball_speed_frames.clear()
        self.ball_speed_history.clear()
        self.ball_history.clear()
        self.current_ball_speed = 0.0
        self.rally_max_speed = {1: 0.0, 2: 0.0}
        self.cached_panel = None
        self.cached_padded_panel = None

    def clear_ball_trail(self):
        """Clear only the shuttlecock trail and live speed display."""
        self.ball_history.clear()
        self.current_ball_speed = None
        self.cached_panel = None
        self.cached_padded_panel = None

    def _update_history(self, frame_index, tracking_data, metrics, ball_state=None, scene_cut=False):
        self.frames.append(frame_index)

        for track_id, court_point in tracking_data.items():
            self.position_history[track_id].append(court_point)
            cell = self._court_cell(court_point)
            if cell is not None:
                self.stay_counts[(track_id, cell[0], cell[1])] += 1

        for player_id in (1, 2):
            player_metrics = metrics.get(player_id, {})
            self.total_distances[player_id] = player_metrics.get("distance_m", 0.0)

        current_rally_dist = metrics.get("current_rally_dist", {})
        for player_id in (1, 2):
            self.distance_history[player_id].append(current_rally_dist.get(player_id, np.nan))

        self.rally_history = metrics.get("rally_history", [])
        self._update_ball_history(frame_index, ball_state, scene_cut=scene_cut)

    def _update_ball_history(self, frame_index, ball_state, scene_cut=False):
        if scene_cut:
            self.current_ball_speed = None
            self.ball_speed_frames.append(frame_index)
            self.ball_speed_history.append(np.nan)
            return

        if ball_state is None:
            self.ball_speed_frames.append(frame_index)
            self.ball_speed_history.append(np.nan)
            return

        self.current_ball_speed = float(ball_state.get("current_speed_kmh", 0.0) or 0.0)
        self.rally_max_speed = ball_state.get("rally_max_speed", {1: 0.0, 2: 0.0})
        ball_pos = ball_state.get("pos")
        if ball_pos is not None:
            self.ball_history.append((tuple(map(int, ball_pos)), self.current_ball_speed))

        self.ball_speed_frames.append(frame_index)
        self.ball_speed_history.append(self.current_ball_speed)

    def _draw_court_view(self):
        court = self.court_background.copy()

        for track_id, history in self.position_history.items():
            color = self.PLAYER_COLORS.get(track_id, (0, 255, 255))
            points = [self._court_to_view(point) for point in history]
            self._draw_fading_trail(court, points, color)

            for court_point, point in zip(history, points):
                cell = self._court_cell(court_point)
                stay_count = self.stay_counts.get((track_id, cell[0], cell[1]), 0) if cell else 0
                if stay_count > 3:
                    radius = min(8, 4 + stay_count // 3)
                    cv2.circle(court, point, radius, self._darken_color(color), -1)

            if points:
                cv2.circle(court, points[-1], 6, color, -1)
                cv2.circle(court, points[-1], 7, (255, 255, 255), 1)

        return court

    def _create_court_background(self):
        court = np.full((self.COURT_HEIGHT, self.PANEL_WIDTH, 3), (38, 128, 54), dtype=np.uint8)
        white = (255, 255, 255)
        left = self._scale_x(0)
        right = self._scale_x(self.SOURCE_WIDTH)
        top = self._scale_y(0)
        bottom = self._scale_y(self.SOURCE_HEIGHT)

        cv2.rectangle(court, (left, top), (right, bottom), white, 2)

        for x in self.COURT_FULL_VERTICAL_LINES:
            view_x = self._scale_x(x)
            cv2.line(court, (view_x, top), (view_x, bottom), white, 1)

        center_x = self._scale_x(self.COURT_CENTER_X)
        center_y1 = self._scale_y(self.COURT_CENTER_LINE_Y1)
        center_y2 = self._scale_y(self.COURT_CENTER_LINE_Y2)
        cv2.line(court, (center_x, center_y1), (center_x, center_y2), white, 1)

        for y in self.COURT_HORIZONTAL_LINES:
            view_y = self._scale_y(y)
            cv2.line(court, (left, view_y), (right, view_y), white, 1)

        return court

    def _draw_fading_trail(self, image, points, color):
        if len(points) < 2:
            return

        overlay = image.copy()
        for index in range(1, len(points)):
            alpha = index / max(1, len(points) - 1)
            faded_color = tuple(int(channel * alpha + 255 * (1 - alpha)) for channel in color)
            cv2.line(overlay, points[index - 1], points[index], faded_color, 2)
        cv2.addWeighted(overlay, 0.9, image, 0.1, 0, image)

    def _draw_ball_overlay(self, frame, scene_cut=False):
        if self.ball_history:
            overlay = frame.copy()
            history = list(self.ball_history)
            for index in range(1, len(history)):
                pos_a, _ = history[index - 1]
                pos_b, speed = history[index]
                alpha = index / max(1, len(history) - 1)
                color = self._speed_color(speed)
                blended = tuple(int(channel * alpha + 255 * (1 - alpha)) for channel in color)
                cv2.line(overlay, pos_a, pos_b, blended, 3, cv2.LINE_AA)
            cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
            current_pos, current_speed = history[-1]
            cv2.circle(frame, current_pos, 7, self._speed_color(current_speed), -1, cv2.LINE_AA)
            cv2.circle(frame, current_pos, 9, (255, 255, 255), 2, cv2.LINE_AA)

        speed_text = "--- km/h" if scene_cut or self.current_ball_speed is None else f"{self.current_ball_speed:.0f} km/h"
        x = max(16, frame.shape[1] - 300)
        y = 58
        cv2.rectangle(frame, (x - 18, y - 44), (frame.shape[1] - 20, y + 18), (0, 0, 0), -1)
        cv2.putText(
            frame,
            speed_text,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.35,
            (210, 210, 210) if scene_cut or self.current_ball_speed is None else self._speed_color(self.current_ball_speed),
            3,
            cv2.LINE_AA,
        )
        return frame

    def _speed_color(self, speed_kmh):
        if speed_kmh < 100:
            return (255, 0, 0)
        if speed_kmh <= 200:
            return (0, 255, 255)
        return (0, 0, 255)

    def _draw_speed_panel(self):
        panel = np.full((self.SPEED_PANEL_HEIGHT, self.PANEL_WIDTH, 3), 246, dtype=np.uint8)
        self._draw_text(panel, "球速", (10, 8), 22, self.TEXT_COLOR)
        self._draw_text(
            panel,
            f"本分P1最快: {self.rally_max_speed.get(1, 0.0):.1f} km/h",
            (10, 36),
            17,
            self.TEXT_COLOR,
        )
        self._draw_text(
            panel,
            f"本分P2最快: {self.rally_max_speed.get(2, 0.0):.1f} km/h",
            (10, 60),
            17,
            self.TEXT_COLOR,
        )

        left = 36
        right = self.PANEL_WIDTH - 12
        top = 92
        bottom = self.SPEED_PANEL_HEIGHT - 18
        cv2.rectangle(panel, (left, top), (right, bottom), (180, 180, 180), 1)
        for fraction in (0.0, 0.5, 1.0):
            y = int(round(bottom - fraction * (bottom - top)))
            cv2.line(panel, (left, y), (right, y), self.GRID_COLOR, 1)
            value = int(round(fraction * self._speed_chart_max()))
            cv2.putText(
                panel,
                str(value),
                (4, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        points = self._speed_chart_points(left, right, top, bottom)
        if len(points) >= 2:
            cv2.polylines(panel, [np.array(points, dtype=np.int32)], False, (30, 30, 30), 2, cv2.LINE_AA)
        elif len(points) == 1:
            cv2.circle(panel, points[0], 2, (30, 30, 30), -1)
        return panel

    def _draw_text(self, image, text, origin, size, color):
        if Image is None or ImageDraw is None:
            cv2.putText(
                image,
                text,
                origin,
                cv2.FONT_HERSHEY_SIMPLEX,
                size / 30.0,
                color,
                1,
                cv2.LINE_AA,
            )
            return

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        draw.text(origin, text, font=self._load_font(size), fill=color[::-1])
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _load_font(self, size):
        if ImageFont is None:
            return None

        font_paths = [
            Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\ARIALUNI.ttf"),
        ]
        for font_path in font_paths:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), size)
        return ImageFont.load_default()

    def _speed_chart_points(self, left, right, top, bottom):
        frames = list(self.ball_speed_frames)
        values = list(self.ball_speed_history)
        if not frames or not values:
            return []

        frame_min = frames[0]
        frame_max = max(frames[-1], frame_min + 1)
        value_max = self._speed_chart_max()
        points = []
        for frame_index, value in zip(frames, values):
            if value is None or np.isnan(value):
                continue
            x = left + int(round((frame_index - frame_min) / (frame_max - frame_min) * (right - left)))
            normalized = min(max(float(value) / value_max, 0.0), 1.0)
            y = bottom - int(round(normalized * (bottom - top)))
            points.append((x, y))
        return points

    def _speed_chart_max(self):
        values = [float(v) for v in self.ball_speed_history if v is not None and not np.isnan(v)]
        return max(250.0, max(values, default=0.0) * 1.15)

    def _draw_line_chart(
        self,
        title,
        history_by_player,
        value_min,
        value_max,
        footer_lines=None,
        total_distances=None,
    ):
        chart = np.full((self.CHART_HEIGHT, self.PANEL_WIDTH, 3), 250, dtype=np.uint8)
        left = 36
        right = self.PANEL_WIDTH - 12
        top = 42
        footer_lines = footer_lines or []
        footer_height = self.RALLY_TEXT_HEIGHT if footer_lines else 0
        bottom = self.CHART_HEIGHT - 24 - footer_height - self.TOTAL_DISTANCE_TEXT_HEIGHT
        axis_label_y = bottom + 18

        cv2.putText(chart, title, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.TEXT_COLOR, 1, cv2.LINE_AA)
        cv2.rectangle(chart, (left, top), (right, bottom), (180, 180, 180), 1)

        for fraction in (0.0, 0.5, 1.0):
            y = int(round(bottom - fraction * (bottom - top)))
            cv2.line(chart, (left, y), (right, y), self.GRID_COLOR, 1)
            label_value = value_min + fraction * (value_max - value_min)
            cv2.putText(
                chart,
                f"{label_value:.1f}",
                (2, y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        if self.frames:
            cv2.putText(
                chart,
                str(self.frames[0]),
                (left - 6, axis_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                chart,
                str(self.frames[-1]),
                (right - 42, axis_label_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )

        for player_id in (1, 2):
            points = self._chart_points(
                self.frames,
                history_by_player[player_id],
                left,
                right,
                top,
                bottom,
                value_min,
                value_max,
            )
            if len(points) >= 2:
                cv2.polylines(
                    chart,
                    [np.array(points, dtype=np.int32)],
                    False,
                    self.PLAYER_COLORS[player_id],
                    2,
                    cv2.LINE_AA,
                )
            elif len(points) == 1:
                cv2.circle(chart, points[0], 2, self.PLAYER_COLORS[player_id], -1)

        if footer_lines:
            self._draw_footer_lines(chart, footer_lines)

        self._draw_total_distance_line(chart, total_distances or {})

        return chart

    def _chart_points(self, frames, values, left, right, top, bottom, value_min, value_max):
        if not frames or not values:
            return []

        points = []
        frame_min = frames[0]
        frame_max = max(frames[-1], frame_min + 1)
        value_span = max(value_max - value_min, 1e-6)

        for frame_index, value in zip(frames, values):
            if value is None or np.isnan(value):
                continue
            x = left + int(round((frame_index - frame_min) / (frame_max - frame_min) * (right - left)))
            normalized = (float(value) - value_min) / value_span
            normalized = min(max(normalized, 0.0), 1.0)
            y = bottom - int(round(normalized * (bottom - top)))
            points.append((x, y))

        return points

    def _distance_max(self):
        values = []
        for player_id in (1, 2):
            for value in self.distance_history[player_id]:
                if value is not None and not np.isnan(value):
                    values.append(float(value))
        return max(values, default=1.0) * 1.1

    def _recent_rally_lines(self):
        lines = []
        for rally in reversed(self.rally_history[-3:]):
            lines.append(f"Rally {rally['rally']}: P1 {rally['p1']:.1f}m  P2 {rally['p2']:.1f}m")
        return lines

    def _draw_footer_lines(self, image, lines):
        if Image is not None and self.rally_font is not None:
            self._draw_footer_lines_with_pillow(image, lines)
            return

        y = self.CHART_HEIGHT - self.RALLY_TEXT_HEIGHT + 18
        for line in lines:
            cv2.putText(
                image,
                line,
                (10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                self.TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            y += 20

    def _draw_footer_lines_with_pillow(self, image, lines):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        y = self.CHART_HEIGHT - self.RALLY_TEXT_HEIGHT + 2
        for line in lines:
            draw.text((10, y), line, font=self.rally_font, fill=self.TEXT_COLOR[::-1])
            y += 20
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _draw_total_distance_line(self, image, total_distances):
        line = (
            f"总距离  P1: {total_distances.get(1, 0.0):.1f}m  "
            f"P2: {total_distances.get(2, 0.0):.1f}m"
        )
        y1 = self.CHART_HEIGHT - self.RALLY_TEXT_HEIGHT - self.TOTAL_DISTANCE_TEXT_HEIGHT
        if not self.rally_history:
            y1 = self.CHART_HEIGHT - self.TOTAL_DISTANCE_TEXT_HEIGHT
        y1 = max(0, y1)
        y2 = min(self.CHART_HEIGHT, y1 + self.TOTAL_DISTANCE_TEXT_HEIGHT)
        cv2.rectangle(image, (0, y1), (self.PANEL_WIDTH - 1, y2), (45, 45, 45), -1)

        if Image is not None and self.rally_font is not None:
            self._draw_total_distance_line_with_pillow(image, line, y1 + 4)
            return

        cv2.putText(
            image,
            line,
            (8, y1 + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    def _draw_total_distance_line_with_pillow(self, image, line, y):
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        draw = ImageDraw.Draw(pil_image)
        draw.text((8, y), line, font=self.total_distance_font, fill=(255, 255, 255))
        image[:] = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    def _load_rally_font(self):
        if ImageFont is None:
            return None

        font_paths = [
            Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\ARIALUNI.ttf"),
        ]
        for font_path in font_paths:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), 18)
        return None

    @property
    def total_distance_font(self):
        if ImageFont is None:
            return None

        font_paths = [
            Path(r"C:\Windows\Fonts\NotoSansSC-VF.ttf"),
            Path(r"C:\Windows\Fonts\msyh.ttc"),
            Path(r"C:\Windows\Fonts\simhei.ttf"),
            Path(r"C:\Windows\Fonts\ARIALUNI.ttf"),
        ]
        for font_path in font_paths:
            if font_path.exists():
                return ImageFont.truetype(str(font_path), 20)
        return self.rally_font

    def _panel_for_frame_height(self, frame_height):
        if self.cached_panel.shape[0] == frame_height:
            return self.cached_panel

        if self.cached_padded_panel is not None and self.cached_padded_panel.shape[0] == frame_height:
            return self.cached_padded_panel

        if self.cached_panel.shape[0] > frame_height:
            self.cached_padded_panel = cv2.resize(
                self.cached_panel,
                (self.PANEL_WIDTH, frame_height),
                interpolation=cv2.INTER_AREA,
            )
            return self.cached_padded_panel

        pad_height = frame_height - self.cached_panel.shape[0]
        padding = np.full((pad_height, self.PANEL_WIDTH, 3), 250, dtype=np.uint8)
        self.cached_padded_panel = cv2.vconcat([self.cached_panel, padding])
        return self.cached_padded_panel

    def _court_cell(self, court_point):
        x, y = court_point
        x = int(round(float(x)))
        y = int(round(float(y)))
        if 0 <= x < self.SOURCE_WIDTH and 0 <= y < self.SOURCE_HEIGHT:
            return x, y
        return None

    def _darken_color(self, color):
        return tuple(max(0, int(channel * 0.55)) for channel in color)

    def _court_to_view(self, court_point):
        x, y = court_point
        return self._scale_x(x), self._scale_y(y)

    def _scale_x(self, x):
        display_width = int(round(self.SOURCE_WIDTH * self._court_scale()))
        left = (self.PANEL_WIDTH - display_width) // 2
        scaled_x = int(round(float(x) * self._court_scale()))
        scaled_x = min(max(scaled_x, 0), display_width - 1)
        return left + scaled_x

    def _scale_y(self, y):
        scaled_y = int(round(float(y) * self._court_scale()))
        return min(max(scaled_y, 0), self.COURT_HEIGHT - 1)

    def _court_scale(self):
        return self.COURT_HEIGHT / self.SOURCE_HEIGHT


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Visualize badminton tracking metrics")
    parser.add_argument("--video", required=True, help="Path to the input video")
    parser.add_argument("--calibration", required=True, help="Path to H.npy calibration matrix")
    return parser.parse_args()


def main():
    """Process 300 frames from 65 seconds and show the realtime visualization."""
    args = parse_args()
    tracker = Tracker(args.calibration)
    analyzer = Analyzer()
    visualizer = Visualizer()
    capture = open_video_at_start(Path(args.video), start_seconds=65)

    try:
        for frame_index in range(300):
            success, frame = capture.read()
            if not success or frame is None:
                print(f"Frame {frame_index}: failed to read frame")
                break

            tracking_data, _, _ = track_frame_with_bbox(tracker, frame)
            metrics = analyzer.analyze(tracking_data)
            output_frame = visualizer.draw(
                frame_index,
                frame,
                analyzer.get_smoothed_tracking_data(),
                metrics,
            )

            cv2.imshow("Badminton Analysis", output_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
