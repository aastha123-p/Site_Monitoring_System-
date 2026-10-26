"""Script to synchronize and rename image and label pairs to standard format:
site_image_001.jpg, site_image_001.txt, etc.

Rules:
1. Matches each image in `data/raw/images` to its corresponding `.txt` in `data/raw/labels`.
2. Renames only when both the image and the label file exist.
3. If an image has no label file, it is left untouched.
4. If a label file has no image file, it is left untouched.
5. Standardizes all images to .jpg (converting PNG/JFIF/JPEG to JPEG format).
6. Outputs a mapping log (mapping.json) for traceability.
"""

import json
import os
import shutil
from pathlib import Path
from PIL import Image

IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.jfif', '.webp', '.bmp'}

def rename_dataset(
    images_dir: str = 'data/raw/images',
    labels_dir: str = 'data/raw/labels',
    prefix: str = 'site_image_',
    digits: int = 3,
    save_mapping: bool = True,
    dry_run: bool = False
):
    images_path = Path(images_dir)
    labels_path = Path(labels_dir)

    if not images_path.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    # Gather images
    all_image_files = [
        f for f in images_path.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    # Gather label files
    all_label_files = [
        f for f in labels_path.iterdir()
        if f.is_file() and f.suffix.lower() == '.txt'
    ]

    # Map stems (filenames without extension) to file paths
    image_stem_map = {f.stem: f for f in all_image_files}
    label_stem_map = {f.stem: f for f in all_label_files}

    image_stems = set(image_stem_map.keys())
    label_stems = set(label_stem_map.keys())

    # Find matching pairs and unmatched files
    common_stems = sorted(image_stems & label_stems)
    unmatched_images = sorted(image_stems - label_stems)
    unmatched_labels = sorted(label_stems - image_stems)

    print("=" * 60)
    print("DATASET SYNCHRONIZATION & RENAMING SUMMARY")
    print("=" * 60)
    print(f"Total image files found:       {len(all_image_files)}")
    print(f"Total label files found:       {len(all_label_files)}")
    print(f"Matching image-label pairs:    {len(common_stems)}")
    print(f"Images without label (skipped): {len(unmatched_images)}")
    for s in unmatched_images:
        print(f"  - [SKIPPED IMAGE]: {image_stem_map[s].name}")
    print(f"Labels without image (skipped): {len(unmatched_labels)}")
    for s in unmatched_labels:
        print(f"  - [SKIPPED LABEL]: {label_stem_map[s].name}")
    print("=" * 60)

    if not common_stems:
        print("No matching pairs found. Exiting.")
        return

    mapping = []

    # Perform rename / conversion
    print(f"\nProcessing {len(common_stems)} matching pairs...")
    for idx, stem in enumerate(common_stems, start=1):
        num_str = str(idx).zfill(digits)
        new_base_name = f"{prefix}{num_str}"
        new_img_name = f"{new_base_name}.jpg"
        new_lbl_name = f"{new_base_name}.txt"

        old_img_file = image_stem_map[stem]
        old_lbl_file = label_stem_map[stem]

        target_img_path = images_path / new_img_name
        target_lbl_path = labels_path / new_lbl_name

        mapping.append({
            "index": idx,
            "new_image": new_img_name,
            "new_label": new_lbl_name,
            "original_image": old_img_file.name,
            "original_label": old_lbl_file.name,
            "original_extension": old_img_file.suffix.lower()
        })

        if dry_run:
            if idx <= 5 or idx >= len(common_stems) - 2:
                print(f"[DRY-RUN] {old_img_file.name} -> {new_img_name} | {old_lbl_file.name} -> {new_lbl_name}")
            continue

        # Rename label file
        if old_lbl_file != target_lbl_path:
            old_lbl_file.rename(target_lbl_path)

        # Handle image file
        if old_img_file.suffix.lower() == '.jpg':
            if old_img_file != target_img_path:
                old_img_file.rename(target_img_path)
        else:
            # Convert non-.jpg (png, jfif, jpeg, etc.) to genuine JPEG
            with Image.open(old_img_file) as img:
                if img.mode in ('RGBA', 'LA', 'P'):
                    img = img.convert('RGB')
                img.save(target_img_path, format="JPEG", quality=95)
            # Remove old file if it had different name/extension
            if old_img_file != target_img_path:
                old_img_file.unlink()

        if idx <= 5 or idx >= len(common_stems) - 2:
            print(f"Renamed: {old_img_file.name} -> {new_img_name} & {old_lbl_file.name} -> {new_lbl_name}")

    if save_mapping and not dry_run:
        mapping_file = Path('data/raw/mapping.json')
        with open(mapping_file, 'w', encoding='utf-8') as f:
            json.dump({
                "total_pairs": len(common_stems),
                "skipped_images": [image_stem_map[s].name for s in unmatched_images],
                "skipped_labels": [label_stem_map[s].name for s in unmatched_labels],
                "pairs": mapping
            }, f, indent=2)
        print(f"\nSaved filename mapping to: {mapping_file}")

    print("\nRenaming completed successfully!")

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Rename dataset images and labels.")
    parser.add_argument('--dry-run', action='store_true', help="Preview changes without renaming")
    args = parser.parse_args()

    rename_dataset(dry_run=args.dry_run)
