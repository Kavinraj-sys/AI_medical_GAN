import pandas as pd

# === Set the correct paths ===
image_metadata_path = 'D:/GAN_training2/preprocessed_data/train_metadata.csv'  # Contains image filenames with .png
labels_path = 'D:/GAN_training2/preprocessed_data/labels.csv'            # Contains labels (image,MEL,NV,BCC,...)
output_path = 'D:/GAN_training2/preprocessed_data/merged_train_metadata.csv'   # Output path

# === Load CSVs ===
image_meta = pd.read_csv(image_metadata_path)   # Expecting column 'image'
labels = pd.read_csv(labels_path)               # Expecting column 'image' and labels

# === Check actual column names (optional debug print) ===
print("Image metadata columns:", image_meta.columns.tolist())
print("Labels columns:", labels.columns.tolist())

# === Strip '.png' from image names ===
image_meta['image'] = image_meta['image'].str.replace('.png', '', regex=False)

# === Merge on 'image' ===
merged = pd.merge(image_meta[['image']], labels, on='image', how='inner')

# === Save the result ===
merged.to_csv(output_path, index=False)

print(f"\n✅ Merged dataset saved to: {output_path}")
print(f"🔢 Total matched images with labels: {len(merged)}")
