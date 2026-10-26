"""YOLOv8n Baseline Model Evaluation Script.

Evaluates the trained baseline model (models/construction_site_best.pt)
on both validation and test splits, extracts per-class metrics (P, R, F1, mAP50, mAP50-95),
and exports CSV & JSON reports to reports/.

Usage:
    python scripts/model_training/evaluate_baseline.py
    python scripts/model_training/evaluate_baseline.py --split test
"""

from pathlib import Path
import argparse
import csv
import json
import sys
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = ROOT / "models" / "construction_site_best.pt"
DEFAULT_DATA = ROOT / "data" / "processed" / "data.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 baseline model")
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="Path to trained model weights (.pt)",
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA,
        help="Path to dataset YAML file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="both",
        choices=["val", "test", "both"],
        help="Dataset split to evaluate (val, test, or both)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu' or '0')",
    )
    return parser.parse_args()


def evaluate_split(model, data_path: Path, split: str, device: str):
    print(f"\nEvaluating on '{split}' split...")
    results = model.val(
        data=str(data_path),
        split=split,
        device=device,
        verbose=False,
    )

    names = model.names
    metrics_box = results.box

    overall = {
        "split": split,
        "instances": int(metrics_box.nc),
        "precision": float(metrics_box.mp),
        "recall": float(metrics_box.mr),
        "f1": float(2 * metrics_box.mp * metrics_box.mr / (metrics_box.mp + metrics_box.mr + 1e-16)),
        "map50": float(metrics_box.map50),
        "map50_95": float(metrics_box.map),
    }

    per_class = []
    # maps has mAP50-95 per class, maps50 has mAP50
    class_map50_95 = metrics_box.maps
    class_map50 = getattr(metrics_box, "map50_per_class", None)

    for cid, cname in sorted(names.items()):
        # Extract per-class P, R, mAP50, mAP50-95
        p = float(metrics_box.p[cid]) if hasattr(metrics_box, "p") and len(metrics_box.p) > cid else 0.0
        r = float(metrics_box.r[cid]) if hasattr(metrics_box, "r") and len(metrics_box.r) > cid else 0.0
        f1 = (2 * p * r) / (p + r + 1e-16)

        m50_95 = float(class_map50_95[cid]) if cid < len(class_map50_95) else 0.0
        m50 = float(class_map50[cid]) if class_map50 is not None and cid < len(class_map50) else m50_95

        per_class.append({
            "class_id": cid,
            "class_name": cname,
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(f1, 4),
            "map50_95": round(m50_95, 4),
        })

    return overall, per_class, results


def print_table(overall, per_class):
    split_title = overall["split"].upper()
    print("\n" + "=" * 70)
    print(f"EVALUATION RESULTS: {split_title} SPLIT")
    print("=" * 70)
    print(f"Overall Precision:   {overall['precision']:.4f}")
    print(f"Overall Recall:      {overall['recall']:.4f}")
    print(f"Overall F1-Score:    {overall['f1']:.4f}")
    print(f"Overall mAP@0.50:    {overall['map50']:.4f}")
    print(f"Overall mAP@0.50:95: {overall['map50_95']:.4f}")
    print("-" * 70)
    print(f"{'Class ID':<9}{'Class Name':<18}{'Precision':<12}{'Recall':<12}{'F1':<10}{'mAP50-95':<10}")
    print("-" * 70)
    for row in per_class:
        print(
            f"{row['class_id']:<9}{row['class_name']:<18}{row['precision']:<12.4f}"
            f"{row['recall']:<12.4f}{row['f1']:<10.4f}{row['map50_95']:<10.4f}"
        )
    print("=" * 70)


def save_reports(all_data):
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Save summary JSON
    json_path = reports_dir / "baseline_evaluation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2)

    # Save CSV per split
    for split_key, split_info in all_data.items():
        csv_path = reports_dir / f"baseline_evaluation_{split_key}.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["class_id", "class_name", "precision", "recall", "f1", "map50_95"]
            )
            writer.writeheader()
            writer.writerows(split_info["per_class"])

    print(f"\nSaved evaluation reports to: {reports_dir}")


def main():
    args = parse_args()

    if not args.weights.exists():
        # Check runs/detect fallback
        fallback = ROOT / "runs" / "detect" / "construction_site_yolov8n" / "weights" / "best.pt"
        if fallback.exists():
            args.weights = fallback
        else:
            print(f"ERROR: Model weights not found at: {args.weights}")
            sys.exit(1)

    print(f"Loading weights from: {args.weights}")
    model = YOLO(str(args.weights))

    splits_to_eval = ["val", "test"] if args.split == "both" else [args.split]
    summary_data = {}

    for split in splits_to_eval:
        overall, per_class, _ = evaluate_split(model, args.data, split, args.device)
        print_table(overall, per_class)
        summary_data[split] = {
            "overall": overall,
            "per_class": per_class,
        }

    save_reports(summary_data)


if __name__ == "__main__":
    main()
