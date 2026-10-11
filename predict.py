"""Inference API and CLI for video deepfake detection using rPPG signals."""

import argparse
from functools import lru_cache
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2

import config
from model import PPGNet
import rppg
from train import DEVICE

logger = logging.getLogger("predict")


@lru_cache(maxsize=4)
def load_model(ckpt_path: str) -> Tuple[PPGNet, float]:
    """Load model checkpoint and optimal decision threshold with LRU caching.

    Args:
        ckpt_path: Path string to checkpoint file.

    Returns:
        Tuple of (model.eval(), threshold).

    Raises:
        FileNotFoundError: If checkpoint file is missing.
    """
    path = Path(ckpt_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Checkpoint file '{ckpt_path}' does not exist. Please train a model first using train.py."
        )

    checkpoint = torch.load(path, map_location=DEVICE)
    threshold = float(checkpoint.get("threshold", 0.5))

    model = PPGNet().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    return model, threshold


def analyse(
    video_path: str,
    ckpt: str = "checkpoints/best.pt",
    max_seconds: float = 60.0,
) -> Dict[str, Any]:
    """Analyze a single video file for rPPG deepfake artifacts.

    Args:
        video_path: Path to video file.
        ckpt: Checkpoint path to load.
        max_seconds: Maximum duration in seconds to process.

    Returns:
        Dict containing verdict (REAL, FAKE, NO_SIGNAL), scores, heart rate,
        coherence, and timeline.
    """
    model, threshold = load_model(ckpt)

    maps, starts = rppg.video_to_maps(video_path, max_seconds=max_seconds)
    if len(maps) == 0:
        return {
            "verdict": "NO_SIGNAL",
            "fake_score": None,
            "threshold": threshold,
            "windows": 0,
            "heart_rate_bpm": None,
            "pulse_coherence": 0.0,
            "timeline": [],
            "reason": "Needs ~4.3 s continuous front-facing face to extract rPPG physiological signals.",
        }

    # Infer window probabilities
    with torch.no_grad():
        x = torch.from_numpy(maps).to(DEVICE)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu().numpy().tolist()

    fake_score = float(np.mean(probs))
    verdict = "FAKE" if fake_score >= threshold else "REAL"

    # Physiological metrics
    hr_bpm = rppg.estimate_heart_rate(maps)
    coherence = rppg.pulse_coherence(maps)

    timeline: List[Dict[str, float]] = [
        {"t": float(starts[i]), "p_fake": float(probs[i])}
        for i in range(len(probs))
    ]

    return {
        "verdict": verdict,
        "fake_score": fake_score,
        "threshold": threshold,
        "windows": len(maps),
        "heart_rate_bpm": hr_bpm,
        "pulse_coherence": coherence,
        "timeline": timeline,
    }


def record_webcam(duration_sec: float = 10.0) -> Optional[str]:
    """Record camera 0 to a temporary .mp4 file.

    Args:
        duration_sec: Duration in seconds to record.

    Returns:
        Path to temporary video file, or None if camera could not be opened.
    """
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("Could not open webcam index 0.")
        return None

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
        fps = 30.0

        tmp_file = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        tmp_path = tmp_file.name
        tmp_file.close()

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(tmp_path, fourcc, fps, (width, height))

        print(f"Recording webcam for {duration_sec:.1f} seconds... Look directly into the camera.")
        start_time = time.time()
        while time.time() - start_time < duration_sec:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            writer.write(frame)

        writer.release()
        return tmp_path

    finally:
        cap.release()


def main() -> None:
    """CLI entry point for video prediction."""
    parser = argparse.ArgumentParser(
        description="Predict real vs fake for a video using rPPG."
    )
    parser.add_argument(
        "video",
        type=str,
        nargs="?",
        default=None,
        help="Path to video file to analyse.",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate timeline.png plotting p_fake over time.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print results formatted as JSON.",
    )
    parser.add_argument(
        "--webcam",
        action="store_true",
        help="Record a stream from default webcam to analyze.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Webcam recording duration in seconds.",
    )

    args = parser.parse_args()

    video_to_process = args.video
    is_temp_file = False

    if args.webcam:
        video_to_process = record_webcam(args.seconds)
        if not video_to_process:
            print("Failed to record from webcam.")
            sys.exit(1)
        is_temp_file = True
    elif not video_to_process:
        parser.print_help()
        sys.exit(1)

    try:
        results = analyse(video_to_process, ckpt=args.ckpt)

        if args.json:
            print(json.dumps(results, indent=2))
        else:
            print("\n" + "=" * 50)
            print("PulseCheck Deepfake Analysis:")
            print(f"  Verdict:         {results['verdict']}")
            if results["verdict"] == "NO_SIGNAL":
                print(f"  Reason:          {results.get('reason')}")
            else:
                print(f"  Fake Score:      {results['fake_score']:.4f}")
                print(f"  Threshold:       {results['threshold']:.4f}")
                print(f"  Windows:         {results['windows']}")
                hr_str = f"{results['heart_rate_bpm']:.1f} BPM" if results['heart_rate_bpm'] else "N/A"
                print(f"  Heart Rate:      {hr_str}")
                print(f"  Pulse Coherence: {results['pulse_coherence']:.4f}")
            print("=" * 50)

        if args.plot and results["timeline"]:
            t_vals = [p["t"] for p in results["timeline"]]
            p_vals = [p["p_fake"] for p in results["timeline"]]

            plt.figure(figsize=(8, 4))
            plt.plot(t_vals, p_vals, marker="o", color="blue", label="Window Fake Probability")
            plt.axhline(
                y=results["threshold"],
                color="red",
                linestyle="--",
                label=f"Decision Threshold ({results['threshold']:.2f})",
            )
            plt.ylim([0.0, 1.0])
            plt.xlabel("Video Timestamp (seconds)")
            plt.ylabel("Fake Probability")
            plt.title(f"Prediction Timeline — Verdict: {results['verdict']}")
            plt.legend()
            plt.grid(True, linestyle=":", alpha=0.6)
            plt.tight_layout()
            plot_file = Path("timeline.png")
            plt.savefig(plot_file, dpi=150)
            plt.close()
            print(f"Saved timeline plot to {plot_file.resolve()}")

    finally:
        if is_temp_file and video_to_process:
            Path(video_to_process).unlink(missing_ok=True)


if __name__ == "__main__":
    main()
