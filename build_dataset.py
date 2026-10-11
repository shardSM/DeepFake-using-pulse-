"""Dataset builder CLI for extracting and caching rPPG maps from video datasets.

Scans video collections for real and fake videos, extracts sliding-window
PPG maps using FaceMesh and POS rPPG, and saves compressed .npz caches.
"""

import argparse
import logging
import multiprocessing as mp
from pathlib import Path
import sys
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

import config
import rppg

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("build_dataset")

VIDEO_EXTENSIONS: set[str] = {".mp4", ".avi", ".mov", ".mkv"}


def group_id(path: Path) -> str:
    """Extract identity/subject identifier from video filename to prevent split leakage.

    Splits the filename stem at the first underscore:
    - FaceForensics++: '000.mp4' -> '000', '000_003.mp4' -> '000'
    - Celeb-DF: 'id0_0000.mp4' -> 'id0', 'id0_id16_0000.mp4' -> 'id0'

    Args:
        path: Path object to the video file.

    Returns:
        String representing subject/identity group ID.
    """
    return path.stem.split("_")[0]


def process_video(
    args_tuple: Tuple[str, int, str, str, float]
) -> Tuple[str, bool, int, str]:
    """Worker task to process a single video and save its cached .npz map.

    Args:
        args_tuple: Tuple containing (video_path, label, out_dir, group, max_seconds).

    Returns:
        Tuple of (video_path, success, num_windows, message).
    """
    video_path_str, label, out_dir_str, group, max_seconds = args_tuple
    video_path = Path(video_path_str)
    out_dir = Path(out_dir_str)

    parent_folder = video_path.parent.name
    stem = video_path.stem
    target_cache_path = out_dir / f"{label}_{parent_folder}_{stem}.npz"

    # Resumable: skip if output already exists
    if target_cache_path.exists():
        return (video_path_str, True, -1, "already_cached")

    try:
        maps, starts = rppg.video_to_maps(str(video_path), max_seconds=max_seconds)
        if len(maps) == 0:
            return (video_path_str, False, 0, "zero_windows")

        # Save compressed NPZ
        np.savez_compressed(
            target_cache_path,
            maps=maps,
            starts=starts,
            label=int(label),
            group=str(group),
            source=str(video_path),
        )
        return (video_path_str, True, len(maps), "saved")

    except Exception as e:
        return (video_path_str, False, 0, f"error: {e}")


def discover_videos(data_dir: Path) -> list[Tuple[Path, int, str]]:
    """Scan data directory for real (0) and fake (1) videos.

    Layout:
      - data/real/**/<videos> -> label 0
      - data/fake/<method>/**/<videos> -> label 1

    Args:
        data_dir: Root dataset folder path.

    Returns:
        List of tuples: (video_path, label, group_id).
    """
    items: list[Tuple[Path, int, str]] = []

    real_dir = data_dir / "real"
    if real_dir.exists():
        for path in real_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                items.append((path, 0, group_id(path)))

    fake_dir = data_dir / "fake"
    if fake_dir.exists():
        for path in fake_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                items.append((path, 1, group_id(path)))

    return items


def main() -> None:
    """CLI entry point for dataset construction."""
    parser = argparse.ArgumentParser(
        description="Build and cache PPG map dataset from video directory."
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data",
        help="Path to input data directory (containing real/ and fake/ subdirectories).",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="cache",
        help="Output directory to save compressed .npz map caches.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel worker processes.",
    )
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=config.MAX_SECONDS,
        help="Maximum video duration in seconds to process per video.",
    )

    args = parser.parse_args()

    data_path = Path(args.data)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    if not data_path.exists():
        print(f"Dataset directory '{data_path}' does not exist.")
        print("Please create the directory with 'real/' and 'fake/' subfolders.")
        sys.exit(0)

    video_entries = discover_videos(data_path)
    if not video_entries:
        print(f"No videos found in '{data_path}'.")
        print("Expected structure: data/real/**/<video> and data/fake/<method>/**/<video>")
        print(f"Supported video extensions: {', '.join(sorted(VIDEO_EXTENSIONS))}")
        sys.exit(0)

    tasks = [
        (str(p), lbl, str(out_path), grp, args.max_seconds)
        for p, lbl, grp in video_entries
    ]

    total_found = len(tasks)
    cached_count = 0
    skipped_count = 0
    real_count = 0
    fake_count = 0

    print(f"Found {total_found} videos to process with {args.workers} workers.")

    if args.workers <= 1:
        # Sequential execution
        for t in tqdm(tasks, desc="Processing videos", unit="video"):
            v_path, success, n_win, msg = process_video(t)
            lbl = t[1]
            if success:
                if n_win == -1:
                    skipped_count += 1
                else:
                    cached_count += 1
                    if lbl == 0:
                        real_count += 1
                    else:
                        fake_count += 1
            else:
                skipped_count += 1
    else:
        # Multi-process execution
        with mp.Pool(processes=args.workers) as pool:
            for v_path, success, n_win, msg in tqdm(
                pool.imap_unordered(process_video, tasks),
                total=total_found,
                desc="Processing videos",
                unit="video",
            ):
                if success:
                    if n_win == -1:
                        skipped_count += 1
                    else:
                        cached_count += 1
                        # Infer label from filename stem or search
                        if Path(v_path).as_posix().find("/real/") != -1:
                            real_count += 1
                        else:
                            fake_count += 1
                else:
                    skipped_count += 1

    print("\n" + "=" * 50)
    print("Dataset Processing Summary:")
    print(f"  Total videos found:    {total_found}")
    print(f"  Newly cached:          {cached_count} (Real: {real_count}, Fake: {fake_count})")
    print(f"  Skipped / Zero-window: {skipped_count}")
    print(f"  Cache directory:       {out_path.resolve()}")
    print("=" * 50)


if __name__ == "__main__":
    main()
