# Contributing

Thanks for improving this project.

## Local Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_assets.py
python scripts/check_install.py
```

On Windows, use:

```powershell
.\scripts\setup.ps1
.\.venv\Scripts\Activate.ps1
```

## Development Rules

- Keep raw videos, generated videos, model weights, logs, and `.npy` matrices out of Git.
- Prefer project-relative paths for files under the repository.
- Run `pytest` before submitting changes.
- Keep calibration configs in `data/calibration/`.
- Document new command-line flags in both `README.md` and `docs/README.zh-CN.md`.
