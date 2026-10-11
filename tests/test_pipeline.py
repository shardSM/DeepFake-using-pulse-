"""End-to-end integration and pipeline smoke test using synthetic cache."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from scripts.make_dummy_cache import generate_dummy_cache


def test_full_pipeline_smoke(tmp_path):
    """Test full pipeline: dummy cache -> training -> splits integrity -> evaluation."""
    cache_dir = tmp_path / "cache_dummy"
    ckpt_dir = tmp_path / "checkpoints"

    # 1. Generate 60 dummy cache files
    generate_dummy_cache(cache_dir, n_samples=60)
    assert len(list(cache_dir.glob("*.npz"))) == 60

    # 2. Run train.py via subprocess using sys.executable
    train_cmd = [
        sys.executable,
        "train.py",
        "--cache",
        str(cache_dir),
        "--out",
        str(ckpt_dir),
        "--epochs",
        "2",
        "--workers",
        "0",
        "--bs",
        "16",
    ]
    res_train = subprocess.run(train_cmd, capture_output=True, text=True)
    assert res_train.returncode == 0, f"train.py failed:\nSTDOUT:\n{res_train.stdout}\nSTDERR:\n{res_train.stderr}"

    # Verify best.pt and splits.json exist
    best_pt = ckpt_dir / "best.pt"
    splits_json = ckpt_dir / "splits.json"
    assert best_pt.exists()
    assert splits_json.exists()

    # 3. Verify no group identity overlaps between splits
    with open(splits_json, "r", encoding="utf-8") as f:
        splits_data = json.load(f)

    train_groups = {item["group"] for item in splits_data["train"]}
    val_groups = {item["group"] for item in splits_data["val"]}
    test_groups = {item["group"] for item in splits_data["test"]}

    assert train_groups.isdisjoint(val_groups)
    assert train_groups.isdisjoint(test_groups)
    assert val_groups.isdisjoint(test_groups)

    # 4. Run evaluate.py via subprocess using sys.executable
    eval_cmd = [
        sys.executable,
        "evaluate.py",
        "--ckpt",
        str(best_pt),
        "--splits",
        str(splits_json),
        "--split",
        "test",
        "--name",
        "test_eval",
    ]
    res_eval = subprocess.run(eval_cmd, capture_output=True, text=True)
    assert res_eval.returncode == 0, f"evaluate.py failed:\nSTDOUT:\n{res_eval.stdout}\nSTDERR:\n{res_eval.stderr}"

    metrics_json = ckpt_dir / "metrics_test_eval.json"
    assert metrics_json.exists()

    with open(metrics_json, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    assert "video_accuracy" in metrics
    assert "confusion_matrix" in metrics
