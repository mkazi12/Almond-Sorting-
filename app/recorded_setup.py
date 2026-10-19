"""Recorded camera, sample media, and target workstation.

Update these constants when the camera or demo machine changes.
"""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_MEDIA_PATH = REPO_ROOT / "media" / "Almond sample.jpg"

CAMERA_PROTOCOL = "USB UVC"
CAMERA_RESOLUTION = "1280x720"
CAMERA_FPS = 30
SAMPLE_MEDIA = "media/Almond sample.jpg"
WORKSTATION_OS = "Windows 11"
WORKSTATION_GPU = "AMD Radeon(TM) Graphics"


def setup_text() -> str:
    return "\n".join(
        [
            f"Camera protocol: {CAMERA_PROTOCOL}",
            f"Resolution: {CAMERA_RESOLUTION}",
            f"Frame rate: {CAMERA_FPS} FPS",
            f"Sample media: {SAMPLE_MEDIA}",
            f"Workstation OS: {WORKSTATION_OS}",
            f"Workstation GPU: {WORKSTATION_GPU}",
        ]
    )
