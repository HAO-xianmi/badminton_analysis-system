# Deployment Notes

This project is designed for local/offline video processing rather than a web server. A successful deployment means the Python environment, FFmpeg, model assets, and source calibration files are available on the target machine.

## Minimal Production Checklist

1. Clone the repository.
2. Run the platform setup script.
3. Run `python scripts/check_install.py`.
4. Put customer videos under `data/raw/` or configure absolute paths.
5. Run `python src/add_source.py` once for every camera angle/video source.
6. Run `python src/main.py --source <source> --output data/output/<name>.mp4`.

## Windows Deployment

```powershell
git clone https://github.com/HAO-xianmi/badminton_analysis-system.git
cd badminton_analysis-system
winget install --id Gyan.FFmpeg -e
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
python scripts\check_install.py
```

## Linux Deployment

```bash
git clone https://github.com/HAO-xianmi/badminton_analysis-system.git
cd badminton_analysis-system
sudo apt-get update
sudo apt-get install -y ffmpeg python3-venv
bash scripts/setup.sh
source .venv/bin/activate
python scripts/check_install.py
```

## GPU Notes

The default `requirements.txt` installs PyTorch from PyPI. For best GPU performance, install the CUDA build recommended for the target machine from the official PyTorch selector, then run:

```bash
pip install -r requirements.txt
```

The app automatically uses CUDA when `torch.cuda.is_available()` is true.

## Moving Calibrated Sources Between Machines

Source configs use project-relative paths when videos and calibration files are inside the repository folder. For easiest handoff:

1. Put the video in `data/raw/`.
2. Keep the source config in `data/calibration/`.
3. Keep the matching `<source>_H.npy` next to the config.

`.npy` files are ignored by Git because they are generated assets. Transfer them separately if you want to reuse an existing calibration without clicking corners again.
