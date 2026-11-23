import os
import pandas as pd

image_dir = "../preprocessed_data/train_images"
output_csv = "../preprocessed_data/train_metadata.csv"

# List all .jpg or .png files
image_names = os.listdir(image_dir)
image_names = [name for name in image_names if name.endswith(".jpg") or name.endswith(".png")]

# Save to CSV
df = pd.DataFrame({"image_name": image_names})
df.to_csv(output_csv, index=False)

print(f"Saved {len(df)} entries to {output_csv}")
