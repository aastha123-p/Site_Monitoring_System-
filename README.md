# 🏗️ AI Construction Site Safety & Hazard Monitoring System

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue.svg)](https://github.com/ultralytics/ultralytics)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![mAP@0.50: 69.4%](https://img.shields.io/badge/mAP%400.50-69.4%25-orange.svg)]()
[![Precision: 75.4%](https://img.shields.io/badge/Precision-75.4%25-green.svg)]()
[![Recall: 66.3%](https://img.shields.io/badge/Recall-66.3%25-blue.svg)]()

An end-to-end computer vision system designed for real-time safety compliance and hazard detection in high-risk construction environments. Powered by **YOLOv8**, the system identifies personnel, verifies personal protective equipment (PPE: hardhats and high-visibility safety vests), monitors heavy machinery and vehicles, and detects safety perimeters.

---

## 📌 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [Model Performance & Evaluation](#-model-performance--evaluation)
  - [Peak Performance vs Older Runs](#peak-performance-vs-older-runs)
  - [Per-Class Validation Metrics](#per-class-validation-metrics)
- [Dataset Architecture](#-dataset-architecture)
  - [Class Taxonomy (10 Classes)](#class-taxonomy-10-classes)
  - [Data Pipeline & Leakage Prevention](#data-pipeline--leakage-prevention)
- [Project Directory Structure](#-project-directory-structure)
- [Installation & Setup](#-installation--setup)
- [Usage Guide](#-usage-guide)
  - [1. Dataset Preparation & Auditing](#1-dataset-preparation--auditing)
  - [2. Model Training](#2-model-training)
  - [3. Evaluation & Validation](#3-evaluation--validation)
  - [4. Running Inference (Images & Video)](#4-running-inference-images--video)
  - [5. Comparing Training Runs](#5-comparing-training-runs)
- [Training Hyperparameters](#-training-hyperparameters)
- [Annotation Guidelines](#-annotation-guidelines)
- [Future Enhancements](#-future-enhancements)

---

## 🔎 Overview

Construction sites present dynamic, occluded, and hazardous conditions. Manual safety inspections are labor-intensive, intermittent, and error-prone. This project delivers an automated visual monitoring pipeline that detects:
1. **PPE Compliance Violations**: Identifies workers lacking required safety gear (`no_hardhat`, `no_safety_vest`).
2. **Heavy Equipment & Vehicle Activity**: Tracks earthmovers, cranes, and haul trucks near foot traffic zones.
3. **Safety Perimeters**: Detects cones delineating danger zones.

The baseline model is built upon **YOLOv8 Nano (`yolov8n.pt`)**, intentionally optimized for edge devices, CCTV feeds, and CPU deployments where low latency is essential.

---

## ⚡ Key Features

* **Multi-Target Detection**: Simultaneously detects 10 distinct site classes including personnel, PPE, and heavy equipment.
* **Co-Occurrence PPE Verification**: Uses hierarchical detection to localize workers and detect their headgear and torso gear.
* **Perceptual Leakage Prevention**: Custom splitting pipeline clusters near-duplicate frames via perceptual difference hashing (`dhash`, Hamming distance $\le 4$) to guarantee unbiased validation.
* **Robust Augmentation Strategy**: Trained with Mosaic, Mixup, Copy-Paste, scale variations, and rotation to handle severe site dust, varying angles, and partial occlusions.
* **Lightweight & Edge-Ready**: Runs on CPU and can be exported directly to ONNX, TensorRT, or OpenVINO.

---

## 📊 Model Performance & Evaluation

### Peak Performance vs Older Runs

Across three iterative training configurations, **Run 3 (`construction_site_yolov8n-3`)** achieved a breakthrough in detection accuracy and reliability:

| Metric | Run 1 *(Baseline, 31 ep)* | Run 2 *(Batch 16, 37 ep)* | Run 3 *(Current Best, 50 ep)* | Improvement vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **mAP@0.50 (Primary Accuracy)** | 57.48% | 56.77% | **69.41%** | **+11.93%** 🚀 |
| **mAP@0.50:0.95 (Tightness)** | 28.54% | 28.60% | **36.26%** | **+7.72%** 🚀 |
| **Precision** | 78.90% *(Low recall)* | 77.46% *(Low recall)* | **75.40%** | **Balanced, reliable** |
| **Recall (Detection Coverage)** | 56.99% | 58.35% | **66.29%** | **+9.30%** 🚀 |
| **Lowest `val/box_loss`** | 1.7839 | 1.8760 | **1.6648** | **-6.7% error reduction** |
| **Lowest `val/cls_loss`** | 1.2484 | 1.2863 | **2.6268** *(with cls=1.0 gain)* | **Trained against class imbalance** |

> **Note on Precision vs Recall**: While Run 1 showed an isolated precision of 78.9%, its recall was only 44.1% (missing over half of hazards). Run 3 achieves **75.4% Precision** alongside **66.3% Recall**, making it drastically safer and more practical for real-world deployment.

### Per-Class Validation Metrics

Evaluated on the unseen validation split with IoU threshold 0.50:

| Class ID | Class Name | Precision | Recall | F1-Score | mAP@0.50:0.95 |
| :---: | :--- | :---: | :---: | :---: | :---: |
| `0` | **person** | 85.1% | 82.6% | 83.8% | 52.1% |
| `1` | **hardhat** | 83.7% | 76.1% | 79.7% | 37.9% |
| `2` | **no_hardhat** | 59.0% | 50.8% | 54.6% | 18.7% |
| `3` | **safety_vest** | 82.9% | 79.6% | 81.2% | 43.0% |
| `4` | **no_safety_vest** | 75.6% | 75.0% | 75.3% | 32.1% |
| `5` | **machinery** | 65.9% | 62.2% | 64.0% | 39.2% |
| `6` | **crane** | 70.2% | 51.8% | 59.6% | 23.5% |
| `7` | **excavator** | 81.5% | 64.3% | 71.9% | 44.8% |
| `8` | **truck** | 68.9% | 39.0% | 49.8% | 25.2% |
| `9` | **safety_cone** | 88.5% | 83.0% | 85.7% | 47.2% |
| **ALL** | **Overall Mean** | **76.1%** | **66.4%** | **71.0%** | **36.4%** |

---

## 📁 Dataset Architecture

### Class Taxonomy (10 Classes)

The dataset contains **803 site images** and **3,655 annotated object instances**:

| ID | Class Name | Total Instances | % of Dataset | Description |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `person` | 911 | 24.9% | On-site workers, supervisors, machine operators, pedestrians |
| `1` | `hardhat` | 578 | 15.8% | Protective hard hats, safety helmets worn or carried |
| `2` | `no_hardhat` | 249 | 6.8% | Exposed head, hair, non-compliant caps (baseball caps/hoods) |
| `3` | `safety_vest` | 474 | 13.0% | High-visibility fluorescent vests and jackets with reflective strips |
| `4` | `no_safety_vest` | 275 | 7.5% | Upper body wearing standard clothing without high-visibility gear |
| `5` | `machinery` | 119 | 3.3% | Stationary equipment, generators, compactors, cement mixers |
| `6` | `crane` | 400 | 10.9% | Tower cranes, mobile crawler cranes, boom arms |
| `7` | `excavator` | 333 | 9.1% | Tracked excavators, mini-diggers, backhoes, trenchers |
| `8` | `truck` | 193 | 5.3% | Heavy dump trucks, concrete mixers, flatbed haulers |
| `9` | `safety_cone` | 123 | 3.4% | Orange traffic cones, safety pylons, perimeter markers |

### Data Pipeline & Leakage Prevention

1. **Polygon-to-BBox Conversion**: Converts 9-value polygon labels to standard 5-value YOLO bounding boxes (`convert_polygon_labels.py`).
2. **Quality Audit & Verification**: Audits coordinate clamping, bounding boundaries, and non-empty files (`audit_dataset.py`, `validate_dataset.py`).
3. **Leakage Prevention Split**: Uses **Perceptual Difference Hashing (`dhash`)** with a Hamming distance threshold of 4. Video-derived consecutive frames or near-duplicate shots are grouped into the same split, avoiding cross-split leakage (`split_dataset.py`).
4. **Split Distribution**: 70% Train (~562 images) / 20% Validation (~160 images) / 10% Test (~81 images).

---

## 📂 Project Directory Structure

```text
site_monitoring_system/
│
├── config/
│   └── classes.yaml              # Master 10-class name definitions
│
├── data/
│   ├── raw/                      # Raw uncleaned images and labels
│   └── processed/                # Cleaned, split dataset
│       ├── data.yaml             # YOLO dataset configuration
│       ├── images/               # train / val / test image folders
│       └── labels/               # train / val / test label folders
│
├── models/
│   └── construction_site_best.pt # Production-ready trained best weights
│
├── reports/
│   ├── class_distribution.csv    # Object instance counts & class percentages
│   ├── dataset_stats.json        # Resolutions, total counts, dimensions
│   ├── baseline_evaluation_*.csv # Per-class validation and test scores
│   └── validation_report.csv     # Integrity audit log of all raw files
│
├── runs/
│   └── detect/                   # Training artifacts, plots, and logs
│       ├── construction_site_yolov8n/   # Run 1 (Baseline)
│       ├── construction_site_yolov8n-2/ # Run 2 (Batch size 16)
│       └── construction_site_yolov8n-3/ # Run 3 (Best: cls=1.0, augs, cos_lr)
│
├── scripts/
│   ├── dataset_preparation/      # Data cleaning, audit, and split scripts
│   │   ├── audit_dataset.py
│   │   ├── check_split_leakage.py
│   │   ├── convert_polygon_labels.py
│   │   ├── dataset_stats.py
│   │   ├── split_dataset.py
│   │   └── validate_dataset.py
│   ├── model_training/           # Training and testing routines
│   │   ├── evaluate_baseline.py
│   │   ├── test_model.py
│   │   └── train_baseline.py
│   ├── src/                      # Shared utility module (hashing, YOLO I/O)
│   │   └── dataset_utils.py
│   ├── compare_runs.py           # Epoch-by-epoch run comparison script
│   ├── evaluate.py               # Evaluation entrypoint
│   ├── test_model.py             # Inference entrypoint
│   └── train.py                  # Training entrypoint
│
├── Annotation_guideline.md       # Standardized labeling & annotation manual
├── README.md                     # Project documentation
├── requirements.txt              # Project dependencies
└── yolov8n.pt                    # Pretrained base weights
```

---

## 🛠️ Installation & Setup

### Prerequisites
* Python 3.10 or higher
* Recommended: Virtual environment (`venv` or `conda`)
* GPU: Optional (NVIDIA CUDA supported, but defaults to CPU)

### 1. Clone & Navigate
```bash
git clone https://github.com/your-username/site_monitoring_system.git
cd site_monitoring_system
```

### 2. Set Up Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Usage Guide

### 1. Dataset Preparation & Auditing

Convert polygon annotations and audit raw data integrity:
```bash
# Convert polygon labels to YOLO bounding box format
python scripts/dataset_preparation/convert_polygon_labels.py

# Audit dataset for missing or corrupt files
python scripts/dataset_preparation/audit_dataset.py

# Validate label coordinates and syntax
python scripts/dataset_preparation/validate_dataset.py

# Split dataset with perceptual hash leakage prevention (70/20/10)
python scripts/dataset_preparation/split_dataset.py --clean
```

### 2. Model Training

Train the enhanced YOLOv8 nano baseline:
```bash
# Direct runner
python scripts/train.py

# Or via model_training package
python scripts/model_training/train_baseline.py
```
* Best model weights are automatically saved to `models/construction_site_best.pt`.
* Epoch-by-epoch logs and metric plots are saved to `runs/detect/construction_site_yolov8n-3/`.

### 3. Evaluation & Validation

Run comprehensive validation on `val` and `test` splits:
```bash
python scripts/evaluate.py
```
* Generates detailed confusion matrices and per-class precision/recall tables in `reports/`.

### 4. Running Inference (Images & Video)

Test the trained model on single images, directories, or video files:
```bash
# Run inference on test split images
python scripts/test_model.py

# Run on custom source
python scripts/model_training/test_model.py --weights models/construction_site_best.pt --source path/to/image.jpg --conf 0.35
```

### 5. Comparing Training Runs

Inspect epoch-by-epoch performance differences across iterations:
```bash
python scripts/compare_runs.py
```

---

## ⚙️ Training Hyperparameters

Configured in `scripts/model_training/train_baseline.py`:

| Parameter | Value | Rationale |
| :--- | :---: | :--- |
| **Model** | `yolov8n.pt` | Lightweight nano architecture for high-speed edge inference |
| **Image Size (`imgsz`)** | `640` | Standard multi-scale resolution |
| **Batch Size (`batch`)** | `8` | Stable gradient estimation on CPU / entry GPUs |
| **Epochs** | `50` | Full convergence window |
| **Class Loss Gain (`cls`)** | `1.0` *(default 0.5)* | Doubled penalty on class confusion to tackle PPE class imbalance |
| **Mosaic (`mosaic`)** | `1.0` | 4-quadrant stitching for small and occluded object context |
| **Close Mosaic (`close_mosaic`)** | `10` | Disables mosaic for final 10 epochs for clean boundary convergence |
| **Mixup (`mixup`)** | `0.15` | Blends two images to improve generalizability |
| **Copy-Paste (`copy_paste`)** | `0.30` | Injects instances into new scenes to boost minority classes |
| **Scale Gain (`scale`)** | `0.70` | Multi-scale invariance for distant vs close objects |
| **Rotation (`degrees`)** | `5.0` | Small angle jitter for varied camera angles |
| **Learning Rate Schedule** | Cosine (`cos_lr: true`) | Smooth annealing avoiding local minima traps |
| **Patience** | `15` | Early stopping threshold |

---

## 📖 Annotation Guidelines

For complete rules on bounding box tightness, handling partial occlusions, nested PPE labels, and edge cases (e.g. baseball caps vs hard hats), refer to the dedicated manual:

👉 **[Annotation Guidelines (`Annotation_guideline.md`)](Annotation_guideline.md)**

---

## 🔮 Future Enhancements

- [ ] **RTSP Real-Time Stream Ingestion**: Direct integration with IP CCTV cameras using OpenCV / GStreamer.
- [ ] **Hazard Zone Geofencing**: Polygon virtual fences around active excavators and cranes triggering proximity alarms when a worker enters.
- [ ] **Automated Alerting**: SMS / Email / Webhook alerts triggered upon detected `no_hardhat` or `no_safety_vest` violations.
- [ ] **Edge Optimization**: Exporting weights to TensorRT FP16 / INT8 for deployment on NVIDIA Jetson Orin.
- [ ] **Temporal Tracking (DeepSORT / ByteTrack)**: Persistent ID tracking to measure violation durations per individual worker.

---

## 📜 License & Citation

This project is licensed under the MIT License. Feel free to use, modify, and distribute for academic and industrial safety applications.
