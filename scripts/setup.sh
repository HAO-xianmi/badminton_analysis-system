#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

SKIP_MODEL_DOWNLOAD="${SKIP_MODEL_DOWNLOAD:-0}"

echo "== Badminton Analysis setup =="

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 was not found. Install Python 3.10 or newer, then run this script again." >&2
  exit 1
fi

echo "Python: $(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python3 -m venv .venv
fi

PYTHON="$ROOT/.venv/bin/python"
PIP="$ROOT/.venv/bin/pip"

"$PYTHON" -m pip install --upgrade pip
"$PIP" install -r requirements.txt

mkdir -p models data/raw data/output data/logs data/calibration

if [ "$SKIP_MODEL_DOWNLOAD" != "1" ]; then
  echo "Downloading default public model assets..."
  "$PYTHON" scripts/download_assets.py
fi

cat <<'MSG'

Setup complete.
Activate the environment with:
  source .venv/bin/activate

Next steps:
  1. Put your input videos under data/raw
  2. Run: python src/add_source.py
  3. Run: python src/main.py --source <source_name> --output data/output/result.mp4 --duration 30
MSG
