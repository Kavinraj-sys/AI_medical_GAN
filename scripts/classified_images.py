import os
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add the project root to the Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from disease_mapping import disease_label_map

def classify_dataset_images():
    base_dir = Path(project_root)
    
    # Paths
    output_dir = base_dir / "classified_images"
    os.makedirs(output_dir, exist_ok=True)

    # Metadata
    metadata_file = base_dir / "preprocessed_data" / "combined_metadata.csv"
    if not metadata_file.exists():
        print("Error: Metadata file not found!")
        return

    print(f"Using metadata: {metadata_file}")

    try:
        df = pd.read_csv(metadata_file)
        print(f"Loaded {len(df)} metadata entries")
    except Exception as e:
        print(f"Error reading metadata: {e}")
        return

    # Columns
    possible_image_cols = ['image', 'image_id', 'image_name', 'filename', 'file_name', 'img_id', 'path']
    possible_label_cols = ['diagnosis', 'disease', 'label', 'class', 'category', 'dx', 'lesion_type']
    
    image_col = next((col for col in possible_image_cols if col in df.columns), None)
    label_col = next((col for col in possible_label_cols if col in df.columns), None)

    if image_col is None or label_col is None:
        print(f"Error: Could not find image or label columns. Columns found: {df.columns}")
        return

    print(f"Detected columns: image = {image_col}, label = {label_col}")

    # Folders where images are stored
    image_directories = [
        base_dir / "preprocessed_data" / "train_images",
        base_dir / "preprocessed_data" / "val_images",
    ]

    image_extensions = ['.jpg', '.jpeg', '.png']

    # Prepare disease folders
    for disease in df[label_col].unique():
        std_disease_name = next((name for code, name in disease_label_map.items()
                                if str(disease).lower() in [str(code).lower(), str(name).lower()]), None)
        if not std_disease_name:
            std_disease_name = disease.lower().replace(' ', '_')
        os.makedirs(output_dir / std_disease_name, exist_ok=True)

    # Start copying
    copied_count = 0
    skipped_count = 0

    print("\nCopying images...")

    for img_entry, label in zip(df[image_col], df[label_col]):
        basename = str(img_entry).strip()
        found = False
        img_path = None

        for img_dir in image_directories:
            if not img_dir.exists():
                continue
            for prefix in ["", "ISIC_", "img_"]:
                for ext in image_extensions:
                    test_path = img_dir / f"{prefix}{basename}{ext}"
                    if test_path.exists():
                        img_path = test_path
                        found = True
                        break
                if found:
                    break
            if found:
                break

        if not found:
            print(f"  Warning: Image not found for {basename}")
            skipped_count += 1
            continue

        # Disease folder
        std_disease_name = next((name for code, name in disease_label_map.items()
                                if str(label).lower() in [str(code).lower(), str(name).lower()]), None)
        if not std_disease_name:
            std_disease_name = label.lower().replace(' ', '_')

        dest_path = output_dir / std_disease_name / img_path.name

        try:
            shutil.copy2(img_path, dest_path)
            copied_count += 1
            if copied_count % 100 == 0:
                print(f"  Copied {copied_count} images...")
        except Exception as e:
            print(f"  Error copying {img_path}: {e}")
            skipped_count += 1

    print("\nClassification complete!")
    print(f"Copied {copied_count} images")
    print(f"Skipped {skipped_count} images")

    print("\nClassified images per folder:")
    for disease_dir in output_dir.iterdir():
        if disease_dir.is_dir():
            count = len(list(disease_dir.glob('*.*')))
            print(f"  {disease_dir.name}: {count} images")

if __name__ == "__main__":
    classify_dataset_images()
