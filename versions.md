# Permitted versions

This file records the exact versions the team develops and demos against, so a fresh
checkout reproduces the same setup. If you change a version, update this file and
`requirements.txt` in the same commit.

Status key: **Pinned** = decided and in `requirements.txt`. **TBD** = not chosen yet.

## Runtime

| Item | Version | Status | Notes |
|---|---|---|---|
| Python | 3.12.x (`.python-version` = `3.12`) | Pinned | Supported by PySide6 6.12.0 (its wheels target CPython 3.10+). Python 3.12 is supported upstream until October 2028. |

## Python packages

| Package | Version | Status | Why |
|---|---|---|---|
| PySide6_Essentials | 6.12.0 | Pinned | Desktop shell. Provides `QtGui` and `QtWidgets`, the only modules the app imports. |
| shiboken6 | 6.12.0 | Pinned | Required by PySide6. Must match the PySide6 version. |
| Image decoder / array library | TBD | TBD (SCRUM-32) | Needed to turn a JPG/PNG into an array for the model. Candidates: Pillow, OpenCV, numpy. |
| Inference runtime | TBD | TBD (SCRUM-34) | Candidates: onnxruntime, torch. Depends on the model format below. |

Not used, so not installed: PySide6_Addons, PySide6_Pdf, PySide6_WebEngine. The full
`PySide6` meta-package installs these; use `PySide6_Essentials` instead. If a later
feature needs one of them, add it here with a reason.

## Model

| Field | Value |
|---|---|
| Model name / architecture | TBD |
| Version | TBD |
| File name and format (e.g. `.onnx`, `.pt`) | TBD |
| Source / download link | TBD |
| License | TBD |
| SHA-256 checksum | TBD |
| Input size and preprocessing | TBD |
| Classes / labels | TBD |

Model files are not committed unless the team decides they should be. If they are
downloaded instead, document the download step here and verify the checksum after
downloading.

## Camera, sample media, and workstation

These are also shown in the app under **Recorded setup**. Change
`app/recorded_setup.py` and this table together.

| Field | Recorded value | Status |
|---|---|---|
| Camera protocol / interface | USB UVC | Recorded development camera (not the industrial camera yet) |
| Resolution | 1280x720 | Recorded |
| Frame rate | 30 FPS | Recorded |
| Sample media | `media/Almond sample.jpg` (still frame) | Recorded |
| Target OS | Windows 11 | Recorded from the current team workstation |
| Target GPU | AMD Radeon(TM) Graphics | Recorded from the current team workstation |

## Platform assumptions

- Developed and run as a desktop app (PySide6) on the Windows 11 workstation above.
  The same PC has an AMD Radeon(TM) Graphics GPU; CPU fallback is still allowed.
- Verified so far: the pinned packages install on Python 3.12.3 and the app's imports
  work (Linux, headless/offscreen). Confirm the Windows 11 demo machine before the
  sprint demo.

## Setting up from a fresh checkout

```bash
python3.12 -m venv .venv
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# macOS/Linux:         source .venv/bin/activate
pip install -r requirements.txt
<run command, documented in SCRUM-29>
```

## Updating versions

1. Create a clean virtual environment and install the new version.
2. Launch the app and open a JPG, a PNG and a malformed file.
3. Update `requirements.txt`, `.python-version` and this file together.
4. Save `requirements.txt` as UTF-8. (Redirecting `pip freeze` in Windows PowerShell
   produces UTF-16; use `pip freeze | Out-File -Encoding utf8 requirements.txt`.)
