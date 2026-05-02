"""Quick environment and project-asset health check."""

import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))


def check_import(module_name):
    """Import a dependency and return an error string on failure."""
    try:
        __import__(module_name)
    except Exception as error:
        return f"{module_name}: {error}"
    return None


def main():
    """Print whether the installation is ready for analysis commands."""
    errors = []
    for module_name in ("cv2", "numpy", "torch", "ultralytics", "mediapipe", "easyocr"):
        error = check_import(module_name)
        if error:
            errors.append(error)

    from paths import MODEL_DIR, pose_model_path, yolo_model_path

    print(f"Project root: {ROOT}")
    print(f"Model dir: {MODEL_DIR}")
    print(f"YOLO model: {yolo_model_path()}")

    try:
        print(f"Pose model: {pose_model_path(required=True)}")
    except FileNotFoundError as error:
        errors.append(str(error))

    ffmpeg = shutil.which("ffmpeg") or find_common_windows_ffmpeg()
    if ffmpeg:
        print(f"ffmpeg: {ffmpeg}")
    else:
        errors.append("ffmpeg was not found in PATH.")

    if errors:
        print("")
        print("Install check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("")
    print("Install check passed.")
    return 0


def find_common_windows_ffmpeg():
    """Return ffmpeg from common Windows install locations."""
    home = Path.home()
    candidates = []
    package_root = home / "AppData" / "Local" / "Microsoft" / "WinGet" / "Packages"
    if package_root.exists():
        candidates.extend(sorted(package_root.glob("Gyan.FFmpeg_*/ffmpeg-*/bin/ffmpeg.exe")))

    candidates.extend(
        [
            Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/Program Files/Gyan/ffmpeg/bin/ffmpeg.exe"),
            Path("C:/ffmpeg/bin/ffmpeg.exe"),
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
