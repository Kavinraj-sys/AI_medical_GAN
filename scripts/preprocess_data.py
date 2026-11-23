import os
import pandas as pd
import numpy as np
from pathlib import Path
import pydicom
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import shutil 

def preprocess_mixed_datasets():
    print("Starting preprocessing of ISIC dataset with separate CSVs...")

    base_dir = Path("D:/GAN_training2")

    isic_2019_dir = base_dir / "ISIC_skin_lesion_dataset" / "isic_2019_training_input"
    isic_2019_metadata = base_dir / "ISIC_skin_lesion_dataset" / "isic_2019_metadata.csv"
    isic_2019_diagnosis = base_dir / "ISIC_skin_lesion_dataset" / "isic_2019_labels.csv"

    processed_dir = base_dir / "preprocessed_data"
    train_dir = processed_dir / "train_images"
    val_dir = processed_dir / "val_images"
    temp_dir = processed_dir / "temp"

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(temp_dir, exist_ok=True)

    isic_2019_df = process_jpg_dataset(isic_2019_dir, isic_2019_metadata, isic_2019_diagnosis, temp_dir)

    if isic_2019_df is None:
        print("No valid data found.")
        return

    isic_2019_df['source'] = 'ISIC_2019'

    if 'image_id' not in isic_2019_df.columns:
        isic_2019_df['image_id'] = [f"img_{i:06d}" for i in range(len(isic_2019_df))]

    isic_2019_df.to_csv(str(processed_dir / 'combined_metadata.csv'), index=False)

    diagnosis_counts = isic_2019_df['diagnosis'].value_counts()
    plt.figure(figsize=(12, 8))
    diagnosis_counts.plot(kind='bar')
    plt.title('Diagnosis Distribution')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(str(processed_dir / 'diagnosis_distribution.png'))

    with open(str(processed_dir / 'class_distribution.json'), 'w') as f:
        json.dump(diagnosis_counts.to_dict(), f, indent=4)

    print("Top 5 diagnoses:")
    print(diagnosis_counts.head(5))

    train_df, val_df = train_test_split(
        isic_2019_df,
        test_size=0.2,
        random_state=42,
        stratify=isic_2019_df['diagnosis']
    )

    train_df.to_csv(str(processed_dir / 'train_metadata.csv'), index=False)
    val_df.to_csv(str(processed_dir / 'val_metadata.csv'), index=False)

    process_final_images(train_df, temp_dir, train_dir)
    process_final_images(val_df, temp_dir, val_dir)

    print("Cleaning up...")
    shutil.rmtree(temp_dir)

    print("Preprocessing complete!")
    print(f"Training set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")

    return processed_dir

def process_jpg_dataset(src_dir, metadata_file, diagnosis_file, output_dir):
    if not Path(src_dir).exists():
        print(f"Warning: Source directory {src_dir} not found. Skipping JPG processing.")
        return None

    metadata_df = pd.read_csv(metadata_file)
    print(f"Loaded patient metadata with {len(metadata_df)} entries")

    diagnosis_df = pd.read_csv(diagnosis_file)
    print(f"Loaded diagnosis metadata with {len(diagnosis_df)} entries")

    diagnosis_df = diagnosis_df.rename(columns={"image": "image_name"})

    label_cols = ['MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC', 'UNK']
    diagnosis_df['diagnosis'] = diagnosis_df[label_cols].idxmax(axis=1)

    metadata_df = metadata_df.merge(diagnosis_df[['image_name', 'diagnosis']], on='image_name', how='inner')
    print(f"Merged dataframe has {len(metadata_df)} entries")

    metadata_df['image_path'] = metadata_df['image_name'].apply(lambda name: str(Path(src_dir) / f"{name}.jpg"))

    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df), desc="Processing JPG files"): #progress bar
        image_path = Path(row['image_path'])
        if not image_path.exists():
            continue

        try:
            img = Image.open(image_path)
            img = img.resize((128, 128), Image.Resampling.LANCZOS)
            output_path = Path(output_dir) / f"{row['image_name']}.png"
            img.save(output_path)
        except Exception as e:
            print(f"Error processing {image_path}: {e}")

    return metadata_df

def process_final_images(df, src_dir, dest_dir):
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing images for {dest_dir.name}"):
        img_name = row['image_name'] if 'image_name' in row else row['image_id']
        src_path = Path(src_dir) / f"{img_name}.png"
        if not src_path.exists():
            print(f"Warning: Image {img_name} not found in temp directory")
            continue
        dest_path = Path(dest_dir) / f"{img_name}.png"
        shutil.copy2(src_path, dest_path)

if __name__ == "__main__":
    preprocess_mixed_datasets()
