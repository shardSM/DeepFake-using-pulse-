"""Script to create a synthetic dummy cache for testing and pipeline verification."""

import argparse
from pathlib import Path
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


def generate_dummy_cache(out_dir: Path, n_samples: int = 60) -> None:
    """Generate dummy .npz caches for smoke testing pipeline.

    Args:
        out_dir: Target cache directory.
        n_samples: Number of video cache files to generate.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    manipulations = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
    n_groups = 20
    groups = [f"id{i:02d}" for i in range(n_groups)]

    for i in range(n_samples):
        # 50/50 balanced labels
        label = i % 2
        grp = groups[i % n_groups]

        if label == 0:
            source = f"data/real/original/video_{i:03d}.mp4"
            parent = "original"
        else:
            method = manipulations[i % len(manipulations)]
            source = f"data/fake/{method}/video_{i:03d}.mp4"
            parent = method

        # Number of windows k between 2 and 6
        k_windows = np.random.randint(2, 7)

        # Generate maps: shape (k, 3, 32, 128) in [0, 1]
        maps = np.random.uniform(0.1, 0.9, size=(k_windows, 3, config.N_ROI, config.WINDOW)).astype(np.float32)

        # Introduce structured differences so model can easily distinguish real vs fake:
        # Real videos: synchronized phase and high green amplitude
        if label == 0:
            t = np.linspace(0, 4 * np.pi, config.WINDOW)
            sync_wave = 0.5 + 0.4 * np.sin(t)
            maps[:, 0, :, :] = sync_wave[None, None, :] + np.random.normal(0, 0.02, size=(k_windows, config.N_ROI, config.WINDOW))
            maps[:, 1, :, :] = sync_wave[None, None, :] + np.random.normal(0, 0.02, size=(k_windows, config.N_ROI, config.WINDOW))
        else:
            # Fake videos: desynchronized rows and randomized frequencies per ROI
            for roi in range(config.N_ROI):
                freq_mult = (roi + 1) * 0.5
                t = np.linspace(0, freq_mult * np.pi, config.WINDOW)
                maps[:, 0, roi, :] = 0.5 + 0.3 * np.sin(t) + np.random.normal(0, 0.1, size=(k_windows, config.WINDOW))

        maps = np.clip(maps, 0.0, 1.0).astype(np.float32)
        starts = np.array([j * (config.STRIDE / config.TARGET_FPS) for j in range(k_windows)], dtype=np.float64)

        stem = f"video_{i:03d}"
        filename = out_dir / f"{label}_{parent}_{stem}.npz"

        np.savez_compressed(
            filename,
            maps=maps,
            starts=starts,
            label=int(label),
            group=str(grp),
            source=str(source),
        )

    print(f"Successfully generated {n_samples} dummy cache files in {out_dir.resolve()}")


def main() -> None:
    """CLI entry point for generating dummy cache."""
    parser = argparse.ArgumentParser(description="Generate synthetic dummy cache files.")
    parser.add_argument("--out", type=str, default="cache_dummy", help="Output directory for dummy cache.")
    parser.add_argument("--n", type=int, default=60, help="Number of dummy files to generate.")
    args = parser.parse_args()

    generate_dummy_cache(Path(args.out), n_samples=args.n)


if __name__ == "__main__":
    main()
