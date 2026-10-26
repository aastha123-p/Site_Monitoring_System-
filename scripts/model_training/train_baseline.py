from pathlib import Path
import shutil
import sys

from ultralytics import YOLO


# ============================================================
# 1. PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA_YAML = ROOT / "data" / "processed" / "data.yaml"
MODEL_DIR = ROOT / "models"
RUNS_DIR = ROOT / "runs" / "detect"


# ============================================================
# 2. TRAINING CONFIGURATION
# ============================================================

MODEL_NAME = "yolov8n.pt"

EPOCHS = 50
IMAGE_SIZE = 640
BATCH_SIZE = 8
PATIENCE = 15
SEED = 42

# Use "cpu" for CPU training.
# Change to 0 if a compatible NVIDIA GPU is available.
DEVICE = "cpu"

WORKERS = 2

# Loss configuration
CLS_LOSS_GAIN = 1.0       # Increased class loss (cls) from default 0.5 to 1.0

# Strong Data Augmentations
SCALE = 0.7              # Image scale (+/- gain) for multi-scale object detection
FLIPLR = 0.5             # Horizontal flip probability (left-right)
MOSAIC = 1.0             # Mosaic augmentation probability (stitches 4 images)
CLOSE_MOSAIC = 10        # Disable mosaic for the final 10 epochs for stable convergence
MIXUP = 0.15             # Mixup augmentation probability (blends two images)
COPY_PASTE = 0.3         # Copy-paste probability to boost minority class instances
TRANSLATE = 0.1          # Image translation (+/- fraction)
DEGREES = 5.0            # Small rotation angle (+/- degrees)
HSV_H = 0.015            # Hue jitter
HSV_S = 0.7              # Saturation jitter
HSV_V = 0.4              # Value (brightness) jitter
COS_LR = True            # Cosine learning rate decay schedule


# ============================================================
# 3. CHECK DATASET CONFIGURATION
# ============================================================

def check_dataset():
    """Check that the dataset configuration exists."""

    print("\n========== DATASET CHECK ==========")

    if not DATA_YAML.exists():
        print(f"ERROR: Dataset configuration not found:\n{DATA_YAML}")
        print("Run split_dataset.py first.")
        sys.exit(1)

    print(f"Dataset config: {DATA_YAML}")
    print("Dataset configuration file found.")

    try:
        import yaml

        with DATA_YAML.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        if not isinstance(config, dict):
            raise ValueError("data.yaml must contain a YAML mapping.")

        names = config.get("names")

        if isinstance(names, list):
            class_count = len(names)
        elif isinstance(names, dict):
            class_count = len(names)
        else:
            raise ValueError(
                "data.yaml must define class names using 'names'."
            )

        if class_count != 10:
            raise ValueError(
                f"Expected 10 classes, but found {class_count}."
            )

        print(f"Number of classes: {class_count}")
        print("Dataset configuration check passed.")

    except Exception as exc:
        print(f"ERROR: Invalid dataset configuration: {exc}")
        sys.exit(1)

    print("===================================\n")


# ============================================================
# 4. TRAIN YOLO MODEL
# ============================================================

def train_model():
    """Train YOLOv8 on the prepared construction-site dataset."""

    check_dataset()

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    print("========== YOLO TRAINING ==========")
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print(f"Seed: {SEED}")
    print(f"Class Loss Gain (cls): {CLS_LOSS_GAIN}")
    print(f"Augmentations: Scale={SCALE}, FlipLR={FLIPLR}, Mosaic={MOSAIC}, Mixup={MIXUP}, Copy-Paste={COPY_PASTE}, CosineLR={COS_LR}")
    print("===================================\n")

    # Load pretrained YOLOv8 weights.
    model = YOLO(MODEL_NAME)

    # Start training with enhanced class loss and strong augmentations.
    results = model.train(
        data=str(DATA_YAML),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        seed=SEED,
        device=DEVICE,
        workers=WORKERS,
        project=str(RUNS_DIR),
        name="construction_site_yolov8n",
        pretrained=True,
        optimizer="auto",
        cls=CLS_LOSS_GAIN,
        scale=SCALE,
        fliplr=FLIPLR,
        mosaic=MOSAIC,
        close_mosaic=CLOSE_MOSAIC,
        mixup=MIXUP,
        copy_paste=COPY_PASTE,
        translate=TRANSLATE,
        degrees=DEGREES,
        hsv_h=HSV_H,
        hsv_s=HSV_S,
        hsv_v=HSV_V,
        cos_lr=COS_LR,
        plots=True,
        save=True,
        verbose=True,
    )

    # Ultralytics saves best.pt and last.pt in the run directory.
    save_dir = Path(results.save_dir)

    best_weights = save_dir / "weights" / "best.pt"
    last_weights = save_dir / "weights" / "last.pt"

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Training results: {save_dir}")

    if best_weights.exists():
        destination = MODEL_DIR / "construction_site_best.pt"
        shutil.copy2(best_weights, destination)
        print(f"Best model copied to: {destination}")
    else:
        print("WARNING: best.pt was not found.")

    if last_weights.exists():
        print(f"Last checkpoint: {last_weights}")

    print("========================================\n")


# ============================================================
# 5. MAIN
# ============================================================

if __name__ == "__main__":
    train_model()
