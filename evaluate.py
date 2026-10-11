"""Evaluation script for PPGNet deepfake detection.

Computes comprehensive window-level and video-level metrics, ROC curves,
confusion matrices, and breakdowns per manipulation method/source folder.
"""

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
import torch

from model import PPGNet
from train import DEVICE, MapDataset, load_index, predict_windows, video_scores

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("evaluate")


def main() -> None:
    """CLI entry point for model evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate trained PPGNet model.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="checkpoints/best.pt",
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--splits",
        type=str,
        default="checkpoints/splits.json",
        help="Path to splits.json file.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate from splits.json.",
    )
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="Optional: evaluate all .npz files in cache folder instead of splits.json (cross-dataset evaluation).",
    )
    parser.add_argument(
        "--name",
        type=str,
        default="eval",
        help="Output naming suffix for saved metrics and ROC curve.",
    )

    args = parser.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        logger.error("Checkpoint file not found: %s", ckpt_path)
        sys.exit(1)

    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    threshold = float(checkpoint.get("threshold", 0.5))

    model = PPGNet().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # Load metadata
    if args.cache:
        cache_path = Path(args.cache)
        if not cache_path.exists():
            logger.error("Cache path '%s' does not exist.", cache_path)
            sys.exit(1)
        meta = load_index(cache_path)
        eval_source_desc = f"all files in {cache_path}"
    else:
        splits_path = Path(args.splits)
        if not splits_path.exists():
            logger.error("Splits file not found: %s", splits_path)
            sys.exit(1)
        with open(splits_path, "r", encoding="utf-8") as f:
            splits_data = json.load(f)
        if args.split not in splits_data:
            logger.error("Split '%s' not found in %s", args.split, splits_path)
            sys.exit(1)
        meta = splits_data[args.split]
        eval_source_desc = f"{args.split} split from {splits_path}"

    if not meta:
        logger.error("No samples found to evaluate (%s).", eval_source_desc)
        sys.exit(1)

    dataset = MapDataset(meta, augment=False)
    if len(dataset) == 0:
        logger.error("No windows found in evaluated dataset.")
        sys.exit(1)

    print(f"Evaluating {len(meta)} videos ({len(dataset)} windows) from {eval_source_desc}")
    print(f"Decision threshold (from checkpoint): {threshold:.4f}")

    # Compute window-level predictions
    window_probs = predict_windows(model, dataset)
    window_labels = dataset.y
    window_preds = (window_probs >= threshold).astype(int)

    # Compute video-level predictions
    video_probs, video_labels = video_scores(window_probs, dataset)
    video_preds = (video_probs >= threshold).astype(int)

    has_both_classes_window = len(np.unique(window_labels)) >= 2
    has_both_classes_video = len(np.unique(video_labels)) >= 2

    # Window AUC
    if has_both_classes_window:
        window_auc = float(roc_auc_score(window_labels, window_probs))
    else:
        logger.warning("Only one class present in windows. Skipping Window AUC.")
        window_auc = None

    # Video AUC
    if has_both_classes_video:
        video_auc = float(roc_auc_score(video_labels, video_probs))
    else:
        logger.warning("Only one class present in videos. Skipping Video AUC.")
        video_auc = None

    # Video-level classification metrics
    vid_acc = float(accuracy_score(video_labels, video_preds))
    vid_prec = float(precision_score(video_labels, video_preds, zero_division=0))
    vid_rec = float(recall_score(video_labels, video_preds, zero_division=0))
    vid_f1 = float(f1_score(video_labels, video_preds, zero_division=0))
    cm = confusion_matrix(video_labels, video_preds, labels=[0, 1]).tolist()
    clf_report = classification_report(video_labels, video_preds, output_dict=True, zero_division=0)

    # Accuracy per source folder (e.g. Deepfakes / Face2Face / FaceSwap / NeuralTextures / original)
    source_accuracies: Dict[str, Dict[str, Any]] = {}
    unique_vids = np.unique(dataset.vid)
    for v_idx in unique_vids:
        item = meta[v_idx]
        src_path = Path(item["source"])
        src_folder = src_path.parent.name
        v_label = int(item["label"])
        v_pred = int(video_preds[v_idx])
        is_correct = (v_label == v_pred)

        if src_folder not in source_accuracies:
            source_accuracies[src_folder] = {"total": 0, "correct": 0}
        source_accuracies[src_folder]["total"] += 1
        if is_correct:
            source_accuracies[src_folder]["correct"] += 1

    for src_folder, data in source_accuracies.items():
        data["accuracy"] = float(data["correct"] / data["total"])

    # Output report
    print("\n" + "=" * 50)
    print("Evaluation Results:")
    print(f"  Videos:            {len(meta)}")
    print(f"  Windows:           {len(dataset)}")
    print(f"  Threshold:         {threshold:.4f}")
    print(f"  Window AUC:        {f'{window_auc:.4f}' if window_auc is not None else 'N/A'}")
    print(f"  Video AUC:         {f'{video_auc:.4f}' if video_auc is not None else 'N/A'}")
    print(f"  Video Accuracy:    {vid_acc:.4f}")
    print(f"  Video Precision:   {vid_prec:.4f}")
    print(f"  Video Recall:      {vid_rec:.4f}")
    print(f"  Video F1 Score:    {vid_f1:.4f}")
    print(f"  Confusion Matrix:  [[TN: {cm[0][0]}, FP: {cm[0][1]}], [FN: {cm[1][0]}, TP: {cm[1][1]}]]")
    print("\nAccuracy per source manipulation folder:")
    for src_folder, d in source_accuracies.items():
        print(f"  - {src_folder:20s}: {d['accuracy']*100:.1f}% ({d['correct']}/{d['total']})")
    print("=" * 50)

    # Save metrics JSON
    metrics_out = {
        "videos_count": len(meta),
        "windows_count": len(dataset),
        "threshold": threshold,
        "window_auc": window_auc,
        "video_auc": video_auc,
        "video_accuracy": vid_acc,
        "video_precision": vid_prec,
        "video_recall": vid_rec,
        "video_f1": vid_f1,
        "confusion_matrix": cm,
        "classification_report": clf_report,
        "per_source_accuracy": source_accuracies,
    }

    out_dir = ckpt_path.parent
    metrics_file = out_dir / f"metrics_{args.name}.json"
    with open(metrics_file, "w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2)
    print(f"Saved metrics to {metrics_file}")

    # Plot and save ROC Curve
    if has_both_classes_video:
        fpr, tpr, _ = roc_curve(video_labels, video_probs)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {video_auc:.3f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve ({args.name})")
        plt.legend(loc="lower right")
        plt.tight_layout()
        roc_file = out_dir / f"roc_{args.name}.png"
        plt.savefig(roc_file, dpi=150)
        plt.close()
        print(f"Saved ROC plot to {roc_file}")


if __name__ == "__main__":
    main()
