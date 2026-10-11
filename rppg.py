"""rPPG signal extraction, preprocessing, POS algorithm, and PPG map generation."""

import logging
from typing import Optional, Tuple
import cv2
import mediapipe as mp
import numpy as np
from scipy.signal import butter, detrend, filtfilt

import config

logger = logging.getLogger(__name__)

# Precompute 3rd-order Butterworth bandpass filter coefficients at module level
_NYQ: float = 0.5 * config.TARGET_FPS
_LOW_NORM: float = config.LOW_HZ / _NYQ
_HIGH_NORM: float = config.HIGH_HZ / _NYQ
_B, _A = butter(3, [_LOW_NORM, _HIGH_NORM], btype="bandpass")


class FaceTracker:
    """Wraps MediaPipe FaceMesh for single-face landmark tracking."""

    def __init__(
        self,
        static_image_mode: bool = False,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        """Initialize MediaPipe FaceMesh."""
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    def __call__(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Track face landmarks in a BGR frame.

        Args:
            frame_bgr: Input image of shape (H, W, 3) in BGR order.

        Returns:
            np.ndarray of shape (468, 2) with float32 pixel coordinates,
            or None if no face was detected.
        """
        if frame_bgr is None or frame_bgr.size == 0:
            return None

        h, w = frame_bgr.shape[:2]
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        results = self._mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None

        face_landmarks = results.multi_face_landmarks[0]
        # MediaPipe provides at least 468 landmarks
        coords = np.empty((468, 2), dtype=np.float32)
        for i in range(468):
            lm = face_landmarks.landmark[i]
            coords[i, 0] = lm.x * w
            coords[i, 1] = lm.y * h

        return coords

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._mesh is not None:
            self._mesh.close()

    def __enter__(self) -> "FaceTracker":
        """Enter context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Exit context manager and release resources."""
        self.close()


def roi_means(frame_bgr: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """Extract mean RGB colors from 32 adaptive skin patches.

    Args:
        frame_bgr: Image of shape (H, W, 3) in BGR format.
        pts: Landmark coordinates of shape (468, 2) in pixels.

    Returns:
        np.ndarray of shape (32, 3) containing mean RGB color for each ROI.
        Empty or out-of-bounds patches contain NaNs.
    """
    h, w = frame_bgr.shape[:2]
    face_w = float(np.linalg.norm(pts[config.FACE_WIDTH_LANDMARKS[0]] - pts[config.FACE_WIDTH_LANDMARKS[1]]))
    r = max(2, int(face_w * config.PATCH_FRAC))

    means = np.empty((config.N_ROI, 3), dtype=np.float64)

    for idx, lm_idx in enumerate(config.ROI_LANDMARKS):
        cx, cy = pts[lm_idx]
        if np.isnan(cx) or np.isnan(cy):
            means[idx] = np.nan
            continue

        ix = int(round(cx))
        iy = int(round(cy))

        y1 = max(0, iy - r)
        y2 = min(h, iy + r + 1)
        x1 = max(0, ix - r)
        x2 = min(w, ix + r + 1)

        if y2 <= y1 or x2 <= x1:
            means[idx] = np.nan
            continue

        patch = frame_bgr[y1:y2, x1:x2]
        if patch.size == 0:
            means[idx] = np.nan
            continue

        # Mean BGR converted to RGB
        mean_bgr = patch.mean(axis=(0, 1))
        means[idx, 0] = mean_bgr[2]  # R
        means[idx, 1] = mean_bgr[1]  # G
        means[idx, 2] = mean_bgr[0]  # B

    return means


def resample(
    traces: np.ndarray, valid: np.ndarray, fps: float
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Resample raw ROI RGB traces and validity mask to uniform TARGET_FPS.

    Args:
        traces: Array of shape (T, 32, 3) containing raw RGB values.
        valid: Boolean array of shape (T,) indicating frame tracking status.
        fps: Native source video frames per second.

    Returns:
        Tuple of (resampled_traces, resampled_valid):
            resampled_traces: (T_new, 32, 3) float64 array.
            resampled_valid: (T_new,) boolean array.
        Returns (None, None) if fewer than 2 usable frames exist.
    """
    t_len = len(traces)
    if t_len == 0 or fps <= 0:
        return None, None

    # Source time vector
    t_source = np.arange(t_len, dtype=np.float64) / fps
    last_time = t_source[-1]

    # Target uniform time vector
    t_target = np.arange(0.0, last_time, 1.0 / config.TARGET_FPS, dtype=np.float64)
    if len(t_target) == 0:
        return None, None

    # Frames usable for interpolation: valid AND completely NaN-free
    nan_free_mask = ~np.isnan(traces).any(axis=(1, 2))
    usable_source_mask = valid & nan_free_mask
    usable_indices = np.where(usable_source_mask)[0]

    if len(usable_indices) < 2:
        return None, None

    t_usable = t_source[usable_indices]

    # Resample all 32*3 = 96 columns via linear interpolation
    reshaped_source = traces[usable_indices].reshape(len(usable_indices), -1)
    resampled_reshaped = np.empty((len(t_target), reshaped_source.shape[1]), dtype=np.float64)

    for col in range(reshaped_source.shape[1]):
        resampled_reshaped[:, col] = np.interp(
            t_target, t_usable, reshaped_source[:, col]
        )

    resampled_traces = resampled_reshaped.reshape(len(t_target), config.N_ROI, 3)

    # Resample validity: linear interp of float mask thresholded at 0.5
    valid_float = valid.astype(np.float64)
    resampled_valid_float = np.interp(t_target, t_source, valid_float)
    resampled_valid = resampled_valid_float > 0.5

    return resampled_traces, resampled_valid


def read_traces(
    path: str, max_seconds: Optional[float] = None
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Read video file, track face, extract ROI means, and resample to TARGET_FPS.

    Args:
        path: Path to input video file.
        max_seconds: Optional maximum video duration in seconds to process.

    Returns:
        Tuple of (traces, valid) resampled to TARGET_FPS, or (None, None) on failure.
    """
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        logger.warning("Could not open video file: %s", path)
        return None, None

    try:
        native_fps = cap.get(cv2.CAP_PROP_FPS)
        if native_fps is None or np.isnan(native_fps) or native_fps < 5.0:
            logger.info("Invalid or missing FPS (%s) for %s; defaulting to 30.0", native_fps, path)
            fps = 30.0
        else:
            fps = float(native_fps)

        max_frames = int(max_seconds * fps) if max_seconds is not None else None

        raw_traces: list[np.ndarray] = []
        raw_valid: list[bool] = []

        with FaceTracker() as tracker:
            frame_count = 0
            while True:
                if max_frames is not None and frame_count >= max_frames:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                pts = tracker(frame)
                if pts is not None:
                    means = roi_means(frame, pts)
                    raw_traces.append(means)
                    raw_valid.append(True)
                else:
                    nan_means = np.full((config.N_ROI, 3), np.nan, dtype=np.float64)
                    raw_traces.append(nan_means)
                    raw_valid.append(False)

                frame_count += 1

        if len(raw_traces) == 0:
            return None, None

        traces_arr = np.array(raw_traces, dtype=np.float64)
        valid_arr = np.array(raw_valid, dtype=bool)

        return resample(traces_arr, valid_arr, fps)

    finally:
        cap.release()


def pos(rgb: np.ndarray, fps: float = config.TARGET_FPS) -> np.ndarray:
    """Plane-Orthogonal-to-Skin (POS) rPPG algorithm (Wang et al., 2017).

    Args:
        rgb: Input RGB array of shape (T, 3).
        fps: Sampling frame rate in Hz.

    Returns:
        Extracted pulse signal of shape (T,).
    """
    t_len = len(rgb)
    l_win = int(config.POS_WIN_SEC * fps)
    h = np.zeros(t_len, dtype=np.float64)
    p_mat = np.array([[0.0, 1.0, -1.0], [-2.0, 1.0, 1.0]], dtype=np.float64)

    for n in range(t_len - l_win + 1):
        c = rgb[n : n + l_win].T  # shape (3, l_win)
        cn = c / (c.mean(axis=1, keepdims=True) + 1e-8)
        s = p_mat @ cn  # shape (2, l_win)
        p = s[0] + (s[0].std() / (s[1].std() + 1e-8)) * s[1]
        h[n : n + l_win] += p - p.mean()

    return h


def bandpass(x: np.ndarray) -> np.ndarray:
    """Detrend and apply 3rd-order Butterworth bandpass filter along the last axis.

    Args:
        x: Input signal array of shape (..., T).

    Returns:
        Filtered signal with identical shape.
    """
    d = detrend(x, axis=-1)
    return filtfilt(_B, _A, d, axis=-1)


def build_map(window: np.ndarray) -> np.ndarray:
    """Construct a 3-channel PPG map from a 128-frame window of ROI RGB traces.

    Channels:
        ch0: Bandpassed POS signal per ROI.
        ch1: Bandpassed normalized Green channel per ROI.
        ch2: Interpolated FFT spectrum of ch0 within [LOW_HZ, HIGH_HZ].

    Args:
        window: Raw RGB window of shape (WINDOW, 32, 3).

    Returns:
        Normalized PPG map of shape (3, 32, WINDOW) as float32 in [0, 1].
    """
    # Transpose to (32, WINDOW, 3) float64
    rgb = np.transpose(window, (1, 0, 2)).astype(np.float64)
    w_len = rgb.shape[1]  # config.WINDOW

    # Channel 0: POS rPPG
    pos_signals = np.empty((config.N_ROI, w_len), dtype=np.float64)
    for i in range(config.N_ROI):
        pos_signals[i] = pos(rgb[i], fps=config.TARGET_FPS)
    ch0 = bandpass(pos_signals)

    # Channel 1: Normalized Green channel
    green = rgb[:, :, 1]
    norm_green = green / (green.mean(axis=1, keepdims=True) + 1e-8)
    ch1 = bandpass(norm_green)

    # Channel 2: Single-sided FFT magnitude of Hanning-windowed ch0 in cardiac band
    hanning_win = np.hanning(w_len)
    windowed_ch0 = ch0 * hanning_win
    fft_spec = np.abs(np.fft.rfft(windowed_ch0, axis=1))
    freqs = np.fft.rfftfreq(w_len, 1.0 / config.TARGET_FPS)

    band_indices = np.where((freqs >= config.LOW_HZ) & (freqs <= config.HIGH_HZ))[0]
    if len(band_indices) == 0:
        band_spec = fft_spec
        sub_freqs = freqs
    else:
        band_spec = fft_spec[:, band_indices]
        sub_freqs = freqs[band_indices]

    # Interpolate each ROI's spectrum across WINDOW columns
    col_x = np.linspace(sub_freqs[0], sub_freqs[-1], w_len)
    ch2 = np.empty((config.N_ROI, w_len), dtype=np.float64)
    for i in range(config.N_ROI):
        ch2[i] = np.interp(col_x, sub_freqs, band_spec[i])

    out_map = np.stack([ch0, ch1, ch2], axis=0)  # (3, 32, WINDOW)

    # Min-max normalize EACH ROW of EACH CHANNEL to [0, 1]
    eps = 1e-8
    row_mins = out_map.min(axis=2, keepdims=True)
    row_maxs = out_map.max(axis=2, keepdims=True)
    denom = row_maxs - row_mins
    denom[denom < eps] = eps
    norm_map = (out_map - row_mins) / denom

    # Clean any potential NaNs or out of bound values
    norm_map = np.nan_to_num(norm_map, nan=0.0, posinf=1.0, neginf=0.0)
    norm_map = np.clip(norm_map, 0.0, 1.0).astype(np.float32)

    return norm_map


def video_to_maps(
    path: str, max_seconds: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a video file into sliding-window PPG maps and start timestamps.

    Args:
        path: Path to video file.
        max_seconds: Optional maximum seconds of video to process.

    Returns:
        Tuple of (maps, starts):
            maps: (N, 3, 32, WINDOW) float32 array.
            starts: (N,) float64 array of start times in seconds.
        Returns empty arrays ((0, 3, 32, WINDOW), (0,)) if unreadable or invalid.
    """
    empty_maps = np.empty((0, 3, config.N_ROI, config.WINDOW), dtype=np.float32)
    empty_starts = np.empty((0,), dtype=np.float64)

    try:
        traces, valid = read_traces(path, max_seconds=max_seconds)
        if traces is None or valid is None:
            return empty_maps, empty_starts

        t_len = len(traces)
        if t_len < config.WINDOW:
            return empty_maps, empty_starts

        maps_list: list[np.ndarray] = []
        starts_list: list[float] = []

        for start_idx in range(0, t_len - config.WINDOW + 1, config.STRIDE):
            end_idx = start_idx + config.WINDOW
            win_valid = valid[start_idx:end_idx]
            valid_ratio = win_valid.sum() / float(config.WINDOW)

            if valid_ratio < config.MIN_FACE_RATIO:
                continue

            win_traces = traces[start_idx:end_idx]
            ppg_map = build_map(win_traces)
            maps_list.append(ppg_map)
            starts_list.append(float(start_idx) / config.TARGET_FPS)

        if not maps_list:
            return empty_maps, empty_starts

        maps_arr = np.stack(maps_list, axis=0).astype(np.float32)
        starts_arr = np.array(starts_list, dtype=np.float64)
        return maps_arr, starts_arr

    except Exception as e:
        logger.warning("Failed processing %s: %s", path, e)
        return empty_maps, empty_starts


def estimate_heart_rate(maps: np.ndarray) -> Optional[float]:
    """Estimate heart rate in BPM from PPG maps.

    Args:
        maps: Array of shape (N, 3, 32, WINDOW).

    Returns:
        Median estimated heart rate in BPM across windows, or None if empty.
    """
    if maps is None or len(maps) == 0:
        return None

    w_len = maps.shape[3]
    forehead_indices = list(config.ROI_GROUPS["forehead"])
    freqs = np.fft.rfftfreq(w_len, 1.0 / config.TARGET_FPS)
    in_band = (freqs >= config.LOW_HZ) & (freqs <= config.HIGH_HZ)
    sub_freqs = freqs[in_band]

    if len(sub_freqs) == 0:
        return None

    bpms: list[float] = []
    for m in maps:
        # Mean of forehead rows of channel 0
        forehead_signal = m[0, forehead_indices, :].mean(axis=0)
        hanning_signal = forehead_signal * np.hanning(w_len)
        spec = np.abs(np.fft.rfft(hanning_signal))[in_band]
        peak_idx = int(np.argmax(spec))
        dom_freq = sub_freqs[peak_idx]
        bpms.append(float(dom_freq * 60.0))

    if not bpms:
        return None

    return float(np.median(bpms))


def pulse_coherence(maps: np.ndarray) -> float:
    """Compute spatial pulse coherence across ROIs.

    Args:
        maps: Array of shape (N, 3, 32, WINDOW).

    Returns:
        Average correlation coefficient between ROI signals in Channel 0.
        Returns 0.0 if empty or non-computable.
    """
    if maps is None or len(maps) == 0:
        return 0.0

    coherences: list[float] = []
    for m in maps:
        ch0 = m[0]  # shape (32, WINDOW)
        corr = np.corrcoef(ch0)  # shape (32, 32)
        triu_indices = np.triu_indices(config.N_ROI, k=1)
        off_diags = corr[triu_indices]
        val = np.nanmean(off_diags)
        if not np.isnan(val):
            coherences.append(float(val))

    if not coherences:
        return 0.0

    return float(np.mean(coherences))
