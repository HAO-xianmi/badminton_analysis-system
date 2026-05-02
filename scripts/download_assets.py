"""Download public runtime model assets used by the analysis pipeline."""

from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import urlretrieve


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"

ASSETS = [
    {
        "name": "MediaPipe pose landmarker",
        "url": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
        "path": MODEL_DIR / "pose_landmarker.task",
        "required": True,
    },
    {
        "name": "YOLOv8s person detector",
        "url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8s.pt",
        "path": MODEL_DIR / "yolov8s.pt",
        "required": False,
    },
]


def download_asset(asset):
    """Download one asset if it is missing."""
    path = asset["path"]
    if path.exists() and path.stat().st_size > 0:
        print(f"OK: {asset['name']} already exists at {path}")
        return True

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {asset['name']}...")
    try:
        urlretrieve(asset["url"], path)
    except (HTTPError, URLError, OSError) as error:
        print(f"WARNING: failed to download {asset['name']}: {error}")
        if asset["required"]:
            print(f"Place the file manually at: {path}")
        return False

    print(f"Saved: {path}")
    return True


def main():
    """Download all configured assets and report optional next steps."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    results = [download_asset(asset) for asset in ASSETS]

    if not all(results):
        print("")
        print("Some assets were not downloaded. You can still run setup again later.")
        print("YOLO weights are also downloaded automatically by Ultralytics when online.")

    print("")
    print("Optional TrackNetV2 shuttlecock weights are not bundled.")
    print("To enable ball speed analysis, place tracknet_weights.pt under models/")


if __name__ == "__main__":
    main()
