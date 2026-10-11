"""Training script and dataset utilities for PPGNet deepfake detection."""

import argparse
import json
import logging
from pathlib import Path
import random
import sys
from typing import Any, Dict, List, Tuple
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

import config
from model import PPGNet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("train")

# Device selection: cuda -> mps -> cpu
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def seed_all(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_index(cache_dir: Path) -> List[Dict[str, Any]]:
    """Scan cache directory and load metadata index for each cached video file.

    Args:
        cache_dir: Path containing cached .npz files.

    Returns:
        List of dicts containing {file, label, group, source}.
    """
    index: List[Dict[str, Any]] = []
    npz_files = list(cache_dir.glob("*.npz"))
    for f in npz_files:
        try:
            with np.load(f, allow_pickle=True) as data:
                lbl = int(data["label"])
                grp = str(data["group"])
                src = str(data["source"])
                index.append({"file": str(f.resolve()), "label": lbl, "group": grp, "source": src})
        except Exception as e:
            logger.warning("Could not read %s: %s", f, e)
    return index


def make_splits(
    meta: List[Dict[str, Any]], seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split metadata into train (70%), val (15%), and test (15%) by identity groups.

    Prevents identity leakage by ensuring no group ID appears across multiple splits.

    Args:
        meta: Full dataset index.
        seed: Random seed for group splitting.

    Returns:
        Tuple of (train_meta, val_meta, test_meta).
    """
    groups = np.array([m["group"] for m in meta])
    indices = np.arange(len(meta))

    # 1. Split 70% train / 30% rest
    gss1 = GroupShuffleSplit(n_splits=1, train_size=0.7, random_state=seed)
    train_idx, rest_idx = next(gss1.split(indices, groups=groups))

    rest_meta = [meta[i] for i in rest_idx]
    rest_groups = groups[rest_idx]
    rest_indices = np.arange(len(rest_meta))

    # 2. Split rest 50/50 into val and test
    gss2 = GroupShuffleSplit(n_splits=1, train_size=0.5, random_state=seed)
    val_sub_idx, test_sub_idx = next(gss2.split(rest_indices, groups=rest_groups))

    train_meta = [meta[i] for i in train_idx]
    val_meta = [rest_meta[i] for i in val_sub_idx]
    test_meta = [rest_meta[i] for i in test_sub_idx]

    # Verify no group overlaps
    train_grps = set(m["group"] for m in train_meta)
    val_grps = set(m["group"] for m in val_meta)
    test_grps = set(m["group"] for m in test_meta)

    assert train_grps.isdisjoint(val_grps), "Train and Val splits share identity groups!"
    assert train_grps.isdisjoint(test_grps), "Train and Test splits share identity groups!"
    assert val_grps.isdisjoint(test_grps), "Val and Test splits share identity groups!"

    # Warn if any split lacks real or fake classes
    for split_name, s_meta in [("Train", train_meta), ("Val", val_meta), ("Test", test_meta)]:
        classes = set(m["label"] for m in s_meta)
        if len(classes) < 2:
            logger.warning("Split '%s' only contains class(es) %s!", split_name, classes)

    return train_meta, val_meta, test_meta


class MapDataset(Dataset):
    """In-memory dataset holding window-level PPG maps with augmentations."""

    def __init__(self, meta: List[Dict[str, Any]], augment: bool = False) -> None:
        """Initialize dataset by loading all maps into memory.

        Args:
            meta: List of metadata dicts.
            augment: Whether to apply data augmentations.
        """
        self.augment = augment
        all_maps: List[np.ndarray] = []
        all_labels: List[float] = []
        all_vids: List[int] = []

        for vid_idx, item in enumerate(meta):
            f_path = item["file"]
            with np.load(f_path, allow_pickle=True) as data:
                maps = data["maps"]  # shape (K, 3, 32, 128)
                lbl = float(data["label"])
                for m in maps:
                    all_maps.append(m)
                    all_labels.append(lbl)
                    all_vids.append(vid_idx)

        if all_maps:
            self.X = np.stack(all_maps, axis=0).astype(np.float32)
            self.y = np.array(all_labels, dtype=np.float32)
            self.vid = np.array(all_vids, dtype=np.int64)
        else:
            self.X = np.empty((0, 3, config.N_ROI, config.WINDOW), dtype=np.float32)
            self.y = np.empty((0,), dtype=np.float32)
            self.vid = np.empty((0,), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        x = self.X[idx].copy()
        y = self.y[idx]
        vid = int(self.vid[idx])

        if self.augment:
            # 1. Additive Gaussian noise sigma=0.03
            noise = np.random.normal(0.0, 0.03, size=x.shape).astype(np.float32)
            x += noise

            # 2. Random ROI zeroing with p=0.1 per row
            drop_mask = (np.random.rand(config.N_ROI) > 0.1).astype(np.float32)  # shape (32,)
            x *= drop_mask[None, :, None]

            # 3. Time reversal on channels 0 and 1 only with p=0.5
            if np.random.rand() > 0.5:
                x[0] = x[0, :, ::-1]
                x[1] = x[1, :, ::-1]

            # Ensure values remain clipped to [0, 1]
            x = np.clip(x, 0.0, 1.0)

        # Return contiguous tensors
        x_tensor = torch.from_numpy(np.ascontiguousarray(x))
        y_tensor = torch.tensor(y, dtype=torch.float32)
        return x_tensor, y_tensor, vid


def predict_windows(model: nn.Module, ds: MapDataset, bs: int = 256) -> np.ndarray:
    """Evaluate model and output sigmoid probabilities for all windows in dataset.

    Args:
        model: Trained PyTorch model.
        ds: MapDataset instance.
        bs: Batch size.

    Returns:
        Array of probabilities of shape (N,).
    """
    model.eval()
    if len(ds) == 0:
        return np.empty((0,), dtype=np.float32)

    loader = DataLoader(ds, batch_size=bs, shuffle=False, drop_last=False)
    all_probs: List[float] = []

    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(DEVICE)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs.tolist())

    return np.array(all_probs, dtype=np.float32)


def video_scores(probs: np.ndarray, ds: MapDataset) -> Tuple[np.ndarray, np.ndarray]:
    """Aggregate window probabilities to video-level scores (mean window probability).

    Args:
        probs: Window probabilities of shape (N,).
        ds: MapDataset containing video index mapping.

    Returns:
        Tuple of (scores, labels) per unique video.
    """
    if len(probs) == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.float32)

    unique_vids = np.unique(ds.vid)
    vid_scores: List[float] = []
    vid_labels: List[float] = []

    for v in unique_vids:
        mask = ds.vid == v
        vid_probs = probs[mask]
        vid_scores.append(float(vid_probs.mean()))
        vid_labels.append(float(ds.y[mask][0]))

    return np.array(vid_scores, dtype=np.float32), np.array(vid_labels, dtype=np.float32)


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    """Calculate optimal classification threshold using Youden's J statistic (argmax TPR - FPR).

    Args:
        labels: True binary labels (0 or 1).
        scores: Predicted risk scores / probabilities in [0, 1].

    Returns:
        Optimal decision threshold clipped to <= 1.0.
    """
    if len(np.unique(labels)) < 2:
        return 0.5

    fpr, tpr, thresholds = roc_curve(labels, scores)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))
    threshold = float(thresholds[best_idx])
    return min(1.0, max(0.0, threshold))


def main() -> None:
    """CLI entry point for training PPGNet."""
    parser = argparse.ArgumentParser(description="Train PPGNet deepfake detector.")
    parser.add_argument("--cache", type=str, default="cache", help="Path to cache directory.")
    parser.add_argument("--out", type=str, default="checkpoints", help="Output directory for checkpoints.")
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs.")
    parser.add_argument("--bs", type=int, default=64, help="Batch size for training.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate.")
    parser.add_argument("--patience", type=int, default=8, help="Early stopping patience.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--workers", type=int, default=2, help="Number of dataloader workers.")

    args = parser.parse_args()

    seed_all(args.seed)
    cache_path = Path(args.cache)
    out_path = Path(args.out)
    out_path.mkdir(parents=True, exist_ok=True)

    index = load_index(cache_path)
    if len(index) < 20:
        logger.error(
            "Found only %d cached videos in '%s'. At least 20 videos are required to train with valid splits.",
            len(index),
            cache_path,
        )
        sys.exit(1)

    print(f"Loaded {len(index)} cached videos from {cache_path}. Building grouped splits...")
    train_meta, val_meta, test_meta = make_splits(index, seed=args.seed)

    # Save splits.json
    splits_record = {
        "train": train_meta,
        "val": val_meta,
        "test": test_meta,
    }
    with open(out_path / "splits.json", "w", encoding="utf-8") as f:
        json.dump(splits_record, f, indent=2)
    print(f"Saved splits to {out_path / 'splits.json'}")

    # Build MapDatasets
    train_ds = MapDataset(train_meta, augment=True)
    val_ds = MapDataset(val_meta, augment=False)
    test_ds = MapDataset(test_meta, augment=False)

    print(f"Windows count — Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")

    # WeightedRandomSampler for class balance
    train_labels = train_ds.y
    class_counts = np.bincount(train_labels.astype(int), minlength=2)
    class_weights = 1.0 / np.maximum(class_counts, 1)
    sample_weights = class_weights[train_labels.astype(int)]
    sampler = WeightedRandomSampler(
        weights=torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.bs,
        sampler=sampler,
        num_workers=args.workers,
        drop_last=True if len(train_ds) >= args.bs else False,
    )

    model = PPGNet().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_auc = -1.0
    best_threshold = 0.5
    no_improve_epochs = 0
    history: List[Dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        n_batches = 0

        for x, y, _ in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        scheduler.step()
        epoch_loss = running_loss / max(1, n_batches)

        # Validation evaluation
        val_probs = predict_windows(model, val_ds)
        val_scores, val_labels = video_scores(val_probs, val_ds)

        if len(np.unique(val_labels)) >= 2:
            val_auc = float(roc_auc_score(val_labels, val_scores))
            opt_thresh = youden_threshold(val_labels, val_scores)
        else:
            val_auc = 0.5
            opt_thresh = 0.5

        history.append({
            "epoch": epoch,
            "train_loss": epoch_loss,
            "val_auc": val_auc,
            "threshold": opt_thresh,
        })

        print(f"Epoch {epoch:02d}/{args.epochs:02d} | Loss: {epoch_loss:.4f} | Val Video AUC: {val_auc:.4f} | Thresh: {opt_thresh:.3f}")

        # Checkpoint on improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_threshold = opt_thresh
            no_improve_epochs = 0

            ckpt = {
                "state_dict": model.state_dict(),
                "threshold": best_threshold,
                "val_auc": best_val_auc,
                "config": {
                    "WINDOW": config.WINDOW,
                    "N_ROI": config.N_ROI,
                    "TARGET_FPS": config.TARGET_FPS,
                },
            }
            torch.save(ckpt, out_path / "best.pt")
            logger.info("Saved new best checkpoint to %s (Val AUC: %.4f)", out_path / "best.pt", best_val_auc)
        else:
            no_improve_epochs += 1

        if no_improve_epochs >= args.patience:
            print(f"Early stopping triggered after {epoch} epochs (patience: {args.patience}).")
            break

    # Save history and training curve plot
    with open(out_path / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    epochs_list = [h["epoch"] for h in history]
    losses = [h["train_loss"] for h in history]
    aucs = [h["val_auc"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="tab:blue")
    ax1.plot(epochs_list, losses, color="tab:blue", label="Train Loss", lw=2)
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.set_ylabel("Val AUC", color="tab:orange")
    ax2.plot(epochs_list, aucs, color="tab:orange", label="Val Video AUC", lw=2)
    ax2.tick_params(axis="y", labelcolor="tab:orange")
    ax2.set_ylim(0, 1.05)

    plt.title("PPGNet Training & Validation Curve")
    fig.tight_layout()
    plt.savefig(out_path / "training_curve.png", dpi=150)
    plt.close()
    print(f"Training complete. Artifacts saved in {out_path.resolve()}")


if __name__ == "__main__":
    main()
