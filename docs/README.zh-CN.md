# 羽毛球视频分析系统

这是一个面向羽毛球比赛视频的离线分析管线。系统会基于 YOLOv8/ByteTrack 跟踪双方球员，使用球场单应性矩阵把画面坐标映射到标准球场坐标，统计跑动距离、回合距离，并输出带可视化面板的视频。可选的 TrackNetV2 权重存在时，还会尝试检测羽毛球并估算球速。

## 功能

- 球员检测和跟踪：YOLOv8 + ByteTrack。
- 球场标定：交互式点击四个球场角点，生成 `H.npy`。
- 跑动分析：总跑动距离、当前回合距离、回合历史。
- 记分牌 OCR：通过 EasyOCR 识别比分变化，用于切分回合。
- 输出视频：左侧数据面板 + 原视频叠加球员轨迹/羽毛球轨迹。
- 可选球速：放入 TrackNetV2 权重后启用羽毛球检测和速度 CSV。

## 环境要求

- Python 3.10-3.12
- Git
- FFmpeg
- Windows、macOS 或 Linux
- 推荐 NVIDIA GPU + CUDA；没有 GPU 也可以用 CPU 运行，只是速度较慢。

## 一键部署

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

安装检查:

```bash
python scripts/check_install.py
```

如果 FFmpeg 未安装:

- Windows: `winget install --id Gyan.FFmpeg -e`
- macOS: `brew install ffmpeg`
- Ubuntu/Debian: `sudo apt-get install ffmpeg`

## 快速使用

1. 把比赛视频放到 `data/raw/`。
2. 创建并标定视频源:

```bash
python src/add_source.py
```

脚本会依次让你输入视频源名称、视频路径、球场标定时间点和记分牌 ROI。球场四角点击顺序是：左上、右上、右下、左下。

3. 运行分析:

```bash
python src/main.py --source your_source_name --output data/output/result.mp4 --duration 30
```

处理完整视频时去掉 `--duration`:

```bash
python src/main.py --source your_source_name --output data/output/result.mp4
```

## 模型文件

`scripts/setup.ps1` 和 `scripts/setup.sh` 会自动下载默认公开模型到 `models/`:

- `models/pose_landmarker.task`
- `models/yolov8s.pt`

YOLO 权重也可以由 Ultralytics 在首次联网运行时自动下载。

TrackNetV2 羽毛球检测权重不随仓库发布。要启用球速分析，请把文件放到:

```text
models/tracknet_weights.pt
```

也可以通过环境变量覆盖:

```bash
BADMINTON_TRACKNET_WEIGHTS=/path/to/tracknet_weights.pt
```

## 常用命令

列出已配置的视频源:

```bash
python src/add_source.py --list
```

单独运行球员跟踪测试:

```bash
python src/tracker.py --video data/raw/example.mp4 --calibration data/calibration/example_H.npy
```

单独运行可视化窗口:

```bash
python src/visualizer.py --video data/raw/example.mp4 --calibration data/calibration/example_H.npy
```

运行测试:

```bash
pytest
```

## 项目结构

```text
.
├── data/
│   ├── calibration/   # 视频源配置和 H.npy 标定矩阵
│   ├── logs/          # 运行日志、球速 CSV、ByteTrack 临时配置
│   ├── output/        # 生成的视频
│   └── raw/           # 原始比赛视频
├── docs/              # 额外文档
├── models/            # 本地模型文件，不提交到 Git
├── scripts/           # 部署、下载、检查脚本
├── src/               # 核心代码
└── tests/             # 测试
```

## 配置说明

每个视频源会在 `data/calibration/<source>_config.json` 生成配置。核心字段:

- `video_path`: 输入视频路径，项目内文件会保存为相对路径。
- `h_matrix_path`: 球场单应性矩阵路径。
- `score_roi`: 记分牌数字区域。
- `court_h_threshold`: 用球场 Y 坐标区分双方半场的阈值。
- `side_flipped`: 换边后的初始球员映射。

支持的环境变量见 `.env.example`。

## GitHub 大文件说明

原始视频、输出视频、模型权重、`.npy` 标定矩阵和运行日志默认不提交到 Git。这样仓库更轻，客户可以快速克隆。部署后只需要把自己的视频放到 `data/raw/`，再按上面的步骤标定即可。

## 故障排查

- `ffmpeg executable not found`: 安装 FFmpeg，并确认 `ffmpeg` 在 PATH 中。
- `MediaPipe pose model not found`: 运行 `python scripts/download_assets.py`，或手动把 pose 模型放到 `models/pose_landmarker.task`。
- 处理速度慢：CPU 可以运行但会慢很多，建议安装 CUDA 版 PyTorch 并使用 NVIDIA GPU。
- 输出编码失败：默认会自动选择 `h264_nvenc` 或 `libx264`。如需强制软件编码，设置 `BADMINTON_FFMPEG_CODEC=libx264`。
- OCR 不准：重新运行 `python src/add_source.py`，选择更紧贴比分数字的 ROI。
