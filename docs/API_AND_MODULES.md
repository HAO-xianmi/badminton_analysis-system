# Modules

## `src/main.py`

End-to-end pipeline entry point. It loads a configured source, opens the input video, runs player tracking, pose drawing, score OCR, rally segmentation, optional shuttlecock speed analysis, visualization, and MP4 export.

Main command:

```bash
python src/main.py --source <source> --output data/output/result.mp4
```

## `src/add_source.py`

Interactive source setup. It creates `data/calibration/<source>_config.json`, calls the court calibrator, and stores the scoreboard ROI.

## `src/calibrate.py`

Interactive homography calibration. Click four court corners in order: top-left, top-right, bottom-right, bottom-left. The output is a NumPy matrix at `data/calibration/<source>_H.npy`.

## `src/tracker.py`

YOLOv8 + ByteTrack player tracker. It projects player foot points into calibrated court coordinates and maps raw track IDs into Player 1/Player 2 based on court side.

## `src/detector.py`

YOLO person detector and MediaPipe pose skeleton renderer.

## `src/analyzer.py`

Distance analysis and rally distance bookkeeping.

## `src/visualizer.py`

Rendering layer for the output video. It builds the left metrics panel and draws player/shuttle trails over the source frame.

## `src/ball_detector.py`, `src/ball_tracker.py`, `src/ball_analyzer.py`

Optional shuttlecock pipeline. It is enabled only when `models/tracknet_weights.pt` or `BADMINTON_TRACKNET_WEIGHTS` exists.

## `src/paths.py`

Centralized project paths and environment-variable overrides. New code should use this module instead of hard-coded absolute paths.
