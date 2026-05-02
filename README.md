# Badminton Analysis System

[中文文档](docs/README.zh-CN.md)

An offline badminton match analysis pipeline. It tracks both players with YOLOv8 + ByteTrack, maps image positions onto court coordinates through homography calibration, measures running distance and rally distance, and exports an annotated analysis video. Optional TrackNetV2 weights enable shuttlecock detection and speed estimation.

## Features

- Player detection and tracking with YOLOv8 and ByteTrack.
- Interactive court calibration from a video frame.
- Total distance, current-rally distance, and rally history metrics.
- Scoreboard OCR with EasyOCR for rally segmentation.
- Exported MP4 with a metrics panel, court view, player trails, and shuttlecock overlay.
- Optional shuttlecock speed CSV when TrackNetV2 weights are provided.

## Requirements

- Python 3.10-3.12
- Git
- FFmpeg
- Windows, macOS, or Linux
- NVIDIA GPU + CUDA is recommended. CPU works, but processing is slower.

## One-Command Setup

Windows PowerShell:

```powershell
git clone https://github.com/HAO-xianmi/badminton_analysis-system.git
cd badminton_analysis-system
.\scripts\setup.ps1
```

macOS/Linux:

```bash
git clone https://github.com/HAO-xianmi/badminton_analysis-system.git
cd badminton_analysis-system
bash scripts/setup.sh
```

Then verify the install:

```bash
python scripts/check_install.py
```

If FFmpeg is missing:

- Windows: `winget install --id Gyan.FFmpeg -e`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`

## Quick Start

1. Put match videos in `data/raw/`.
2. Create and calibrate a video source:

```bash
python src/add_source.py
```

The calibration corner order is top-left, top-right, bottom-right, bottom-left.

3. Run a short analysis:

```bash
python src/main.py --source your_source_name --output data/output/result.mp4 --duration 30
```

Run a full video by omitting `--duration`:

```bash
python src/main.py --source your_source_name --output data/output/result.mp4
```

## Model Assets

The setup scripts download default public assets into `models/`:

- `models/pose_landmarker.task`
- `models/yolov8s.pt`

YOLO weights can also be downloaded automatically by Ultralytics during the first online run.

TrackNetV2 shuttlecock weights are optional and are not bundled. To enable ball speed analysis, place the file here:

```text
models/tracknet_weights.pt
```

Or set:

```bash
BADMINTON_TRACKNET_WEIGHTS=/path/to/tracknet_weights.pt
```

## Useful Commands

List configured sources:

```bash
python src/add_source.py --list
```

Run player tracking only:

```bash
python src/tracker.py --video data/raw/example.mp4 --calibration data/calibration/example_H.npy
```

Preview visualization:

```bash
python src/visualizer.py --video data/raw/example.mp4 --calibration data/calibration/example_H.npy
```

Run tests:

```bash
pytest
```

## Project Layout

```text
.
├── data/
│   ├── calibration/   # Source configs and H.npy homography matrices
│   ├── logs/          # Runtime logs, speed CSV, generated tracker config
│   ├── output/        # Generated videos
│   └── raw/           # Input videos
├── docs/              # Additional documentation
├── models/            # Local model files, ignored by Git
├── scripts/           # Setup, download, and health-check scripts
├── src/               # Application code
└── tests/             # Tests
```

## Configuration

Each source is stored at `data/calibration/<source>_config.json`.

Important fields:

- `video_path`: input video path. Project-local files are stored as relative paths.
- `h_matrix_path`: court homography matrix path.
- `score_roi`: scoreboard digit ROI.
- `court_h_threshold`: court Y threshold for mapping players to sides.
- `side_flipped`: initial side mapping when players have switched sides.

See `.env.example` for supported environment overrides.

## Large Files

Raw videos, generated videos, model weights, `.npy` calibration matrices, and runtime logs are ignored by Git. This keeps the repository small and easy to clone. After deployment, place customer videos in `data/raw/` and calibrate them locally.

## Troubleshooting

- `ffmpeg executable not found`: install FFmpeg and make sure `ffmpeg` is in PATH.
- `MediaPipe pose model not found`: run `python scripts/download_assets.py`, or place the model at `models/pose_landmarker.task`.
- Slow processing: CPU mode is expected to be slow. Use CUDA-capable PyTorch and an NVIDIA GPU for production runs.
- Video encoding failure: the app auto-selects `h264_nvenc` when available and falls back to `libx264`. Force software encoding with `BADMINTON_FFMPEG_CODEC=libx264`.
- OCR misses scores: recalibrate the source and select a tighter scoreboard digit ROI.
