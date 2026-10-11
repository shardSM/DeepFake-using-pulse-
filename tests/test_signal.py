"""Unit tests for signal processing, POS algorithm, and PPG map generation."""

import numpy as np
import pytest
import cv2

import config
import rppg


def test_synthetic_signal_and_resampling(tmp_path):
    """Test synthetic pulse extraction, resampling, map construction, and frequency estimation."""
    np.random.seed(42)
    source_fps = 25.0
    duration_sec = 10.0
    t_source = np.arange(0, duration_sec, 1.0 / source_fps)
    t_len = len(t_source)  # 250 frames

    # Base skin color [R=180, G=120, B=100]
    base_color = np.array([180.0, 120.0, 100.0])
    pulse_freq = 1.2  # 1.2 Hz = 72 BPM
    pulse = 0.5 * np.sin(2.0 * np.pi * pulse_freq * t_source)

    # Channel weights [0.3, 1.0, 0.5]
    weights = np.array([0.3, 1.0, 0.5])
    raw_rgb = base_color[None, :] + pulse[:, None] * weights[None, :]
    noise = np.random.normal(0, 0.05, size=raw_rgb.shape)
    raw_rgb += noise

    # Replicate across 32 ROIs: shape (T, 32, 3)
    traces = np.tile(raw_rgb[:, None, :], (1, config.N_ROI, 1))
    valid = np.ones(t_len, dtype=bool)

    # Corrupt 5 frames with NaNs and valid=False
    corrupt_indices = [10, 25, 50, 120, 200]
    for idx in corrupt_indices:
        traces[idx] = np.nan
        valid[idx] = False

    # 1. Resample to TARGET_FPS (30.0)
    res_traces, res_valid = rppg.resample(traces, valid, source_fps)
    assert res_traces is not None
    assert res_valid is not None
    # Expected length: 10 s * 30 fps = 300 (±2)
    assert abs(len(res_traces) - 300) <= 2
    assert res_traces.shape[1:] == (config.N_ROI, 3)
    assert not np.isnan(res_traces).any()

    # 2. Test build_map on first 128 frames
    window = res_traces[: config.WINDOW]
    ppg_map = rppg.build_map(window)
    assert ppg_map.shape == (3, config.N_ROI, config.WINDOW)
    assert ppg_map.dtype == np.float32
    assert not np.isnan(ppg_map).any()
    assert (ppg_map >= 0.0).all() and (ppg_map <= 1.0).all()

    # 3. Test dominant frequency of bandpass(pos(ROI 0))
    pos_roi0 = rppg.pos(window[:, 0, :], fps=config.TARGET_FPS)
    bp_roi0 = rppg.bandpass(pos_roi0)
    freqs = np.fft.rfftfreq(len(bp_roi0), 1.0 / config.TARGET_FPS)
    fft_mag = np.abs(np.fft.rfft(bp_roi0))
    in_band = (freqs >= config.LOW_HZ) & (freqs <= config.HIGH_HZ)
    dom_freq = freqs[in_band][np.argmax(fft_mag[in_band])]
    assert abs(dom_freq - pulse_freq) <= 0.15

    # 4. Test estimate_heart_rate
    maps_batch = ppg_map[None, ...]  # (1, 3, 32, WINDOW)
    est_hr = rppg.estimate_heart_rate(maps_batch)
    assert est_hr is not None
    expected_hr = pulse_freq * 60.0  # 72 BPM
    assert abs(est_hr - expected_hr) <= 9.0


def test_video_to_maps_no_face_and_missing(tmp_path):
    """Test video_to_maps on synthetic face-less video and non-existent file."""
    # Write a 3-s synthetic video (moving gradient, no face)
    video_file = tmp_path / "synthetic_noface.mp4"
    fps = 30
    width, height = 120, 120
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_file), fourcc, fps, (width, height))

    for frame_i in range(90):  # 3 seconds
        frame = np.full((height, width, 3), (frame_i * 2) % 255, dtype=np.uint8)
        out.write(frame)
    out.release()

    # Process face-less video -> should return correctly shaped empty arrays without crashing
    maps, starts = rppg.video_to_maps(str(video_file))
    assert maps.shape == (0, 3, config.N_ROI, config.WINDOW)
    assert starts.shape == (0,)
    assert maps.dtype == np.float32

    # Process non-existent video -> empty arrays, no exception
    non_existent = str(tmp_path / "does_not_exist.mp4")
    maps_ne, starts_ne = rppg.video_to_maps(non_existent)
    assert maps_ne.shape == (0, 3, config.N_ROI, config.WINDOW)
    assert starts_ne.shape == (0,)
