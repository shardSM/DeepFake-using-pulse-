"""Configuration constants for PulseCheck rPPG deepfake detection pipeline."""

# Video sampling and windowing parameters
TARGET_FPS: float = 30.0  # All videos resampled to this uniform frame rate
WINDOW: int = 128  # Frames per PPG map (~4.27 s at 30 FPS)
STRIDE: int = 64  # Stride between consecutive windows (50% overlap)

# Physiological frequency boundaries (in Hz)
LOW_HZ: float = 0.7  # 42 BPM
HIGH_HZ: float = 3.3  # 198 BPM

# POS algorithm window size (seconds)
POS_WIN_SEC: float = 1.6

# Dataset extraction parameters
MAX_SECONDS: int = 20  # Max seconds per video during dataset caching
MIN_FACE_RATIO: float = 0.8  # Minimum fraction of tracked frames required inside a window
PATCH_FRAC: float = 0.06  # Patch half-size as fraction of face width

# MediaPipe landmark indices
FACE_WIDTH_LANDMARKS: tuple[int, int] = (234, 454)  # Left ear to right ear cheekbone

ROI_LANDMARKS: list[int] = [
    10, 67, 109, 338, 297, 151, 108, 337,   # forehead    rows 0–7
    50, 101, 118, 117, 123, 187, 205, 36,   # left cheek  rows 8–15
    280, 330, 347, 346, 352, 411, 425, 266, # right cheek rows 16–23
    4, 195, 5,                              # nose        rows 24–26
    152, 175, 199, 377, 148,                # chin / jaw  rows 27–31
]

ROI_GROUPS: dict[str, range] = {
    "forehead": range(0, 8),
    "left_cheek": range(8, 16),
    "right_cheek": range(16, 24),
    "nose": range(24, 27),
    "chin": range(27, 32),
}

N_ROI: int = len(ROI_LANDMARKS)  # Total number of ROIs (asserts to 32)
assert N_ROI == 32, f"Expected 32 ROIs, found {N_ROI}"
