import os
import torch
import numpy as np
from PIL import Image
from pathlib import Path
import matplotlib.pyplot as plt
import sys
import time

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from disease_mapping import disease_label_map

# Generator class definition (matching the one used in training)
class Generator(torch.nn.Module):
    def __init__(self, z_dim=100):
        super(Generator, self).__init__()
        self.gen = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(True),

            torch.nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            torch.nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

def find_generator_weights():
    """Search for generator weights in common locations"""
    base_dir = Path(project_root)
    possible_weight_paths = [
        base_dir / "models" / "generator_weights.pth",
        base_dir / "models" / "generator.pth",
        base_dir / "model_weights" / "generator_weights.pth",
        base_dir / "model_weights" / "generator.pth",
        base_dir / "generator_weights.pth",
        base_dir / "generator.pth",
        Path("D:/GAN_training2/models/generator_weights.pth"),
        Path("D:/GAN_training2/models/generator.pth")
    ]
    
    for path in possible_weight_paths:
        if path.exists():
            return path
    
    # Look for checkpoints
    checkpoint_dir = base_dir / "models" / "checkpoints"
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob("gan_checkpoint_epoch_*.pth"))
        if checkpoints:
            # Return the latest checkpoint
            latest_checkpoint = max(checkpoints, key=lambda p: int(p.stem.split('_')[-1]))
            return latest_checkpoint
    
    return None

def generate_medical_images(diagnosis_type, num_images=50, latent_dim=100, image_size=64, device=None):
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Generating {num_images} synthetic {diagnosis_type} images...")
    print(f"Using device: {device}")
    
    base_dir = Path(project_root)
    # Use classified_images directory instead of generated_images
    output_dir = base_dir / "classified_images" / diagnosis_type
    os.makedirs(output_dir, exist_ok=True)
    
    # Find generator weights
    generator_weights_path = find_generator_weights()
    
    if generator_weights_path is None:
        print("Error: No generator weights file found!")
        print("\nPlease specify the path to your trained model weights:")
        custom_path = input("Path to weights file: ")
        if custom_path and Path(custom_path).exists():
            generator_weights_path = Path(custom_path)
        else:
            print("No valid weights file provided. Cannot generate images.")
            return
    else:
        print(f"Found generator weights at: {generator_weights_path}")
        
    try:
        # Initialize the generator
        generator = Generator(z_dim=latent_dim).to(device)
        
        # Handle both direct weights and checkpoints
        weights_data = torch.load(str(generator_weights_path), map_location=device)
        if isinstance(weights_data, dict) and 'generator_state_dict' in weights_data:
            # This is a checkpoint file
            generator.load_state_dict(weights_data['generator_state_dict'])
            print(f"Loaded generator from checkpoint (epoch {weights_data['epoch']})")
        else:
            # This is a direct state dict
            generator.load_state_dict(weights_data)
            print("Generator loaded successfully!")
            
        generator.eval()
    except Exception as e:
        print(f"Error loading generator: {e}")
        return
        
    print("Generating images...")
    start_time = time.time()
    
    # Find existing images in the directory to avoid overwriting
    existing_files = list(output_dir.glob(f"{diagnosis_type}_*.png"))
    existing_numbers = []
    
    for file in existing_files:
        try:
            # Extract the number from filename format "diagnosis_type_0001.png"
            file_num = int(file.stem.split('_')[-1])
            existing_numbers.append(file_num)
        except:
            continue
    
    # Find the next available number
    start_index = 1
    if existing_numbers:
        start_index = max(existing_numbers) + 1
    
    with torch.no_grad():
        for i in range(num_images):
            noise = torch.randn(1, latent_dim, 1, 1, device=device)
            fake = generator(noise).detach().cpu().squeeze(0)
            fake_img = (fake + 1) / 2  # Unnormalize from [-1, 1] to [0, 1]
            fake_img = fake_img.permute(1, 2, 0).numpy()
            fake_img = (fake_img * 255).astype(np.uint8)
            img = Image.fromarray(fake_img)
            img_path = output_dir / f"{diagnosis_type}_{start_index + i:04d}.png"
            img.save(str(img_path))
            if (i + 1) % 10 == 0:
                elapsed = time.time() - start_time
                images_per_sec = (i + 1) / elapsed
                print(f"  Generated {i+1}/{num_images} images ({images_per_sec:.2f} images/sec)")
                
    total_time = time.time() - start_time
    print(f"Successfully generated {num_images} {diagnosis_type} images at {output_dir}")
    print(f"Total generation time: {total_time:.2f} seconds ({num_images/total_time:.2f} images/sec)")
    
    create_image_collage(output_dir, diagnosis_type, samples=16)

def create_image_collage(image_dir, diagnosis_type, samples=16):
    image_files = list(Path(image_dir).glob('*.png'))
    if len(image_files) < samples:
        samples = len(image_files)
    if samples == 0:
        print("No images found to create collage")
        return
        
    selected_files = np.random.choice(image_files, samples, replace=False)
    rows = int(np.sqrt(samples))
    cols = samples // rows
    if rows * cols < samples:
        cols += 1
        
    plt.figure(figsize=(12, 12))
    for i, img_path in enumerate(selected_files):
        if i >= samples:
            break
        img = Image.open(img_path)
        plt.subplot(rows, cols, i+1)
        plt.imshow(np.array(img))
        plt.axis('off')
        
    plt.suptitle(f"Sample Generated {diagnosis_type.replace('_', ' ').title()} Images", fontsize=16)
    plt.tight_layout()
    
    collage_dir = Path(project_root) / "results"
    os.makedirs(collage_dir, exist_ok=True)
    collage_path = collage_dir / f"{diagnosis_type}_samples.png"
    plt.savefig(str(collage_path))
    plt.close()
    print(f"Sample collage saved to {collage_path}")
    
    try:
        # to display the collage 
        Image.open(collage_path).show()
    except:
        pass

def generate_specific_disease(disease_name, num_images=50):
    """Generate images for a specific disease type"""
    if disease_name not in disease_label_map.values():
        available_diseases = ", ".join(disease_label_map.values())
        print(f"Error: '{disease_name}' not found in disease mapping.")
        print(f"Available diseases: {available_diseases}")
        return
    
    generate_medical_images(disease_name, num_images=num_images)

if __name__ == "__main__":
   
    images_per_disease = {
        "melanoma": 30000,
        "nevus": 30000,  
        "basal_cell_carcinoma": 30000,
        "actinic_keratosis": 30000,
        "benign_keratosis": 30000,
        "dermatofibroma": 30000,
        "vascular_lesion": 30000,
        "squamous_cell_carcinoma": 30000  
    }

    if len(sys.argv) > 1:
        disease_name = sys.argv[1]
        num_images = int(sys.argv[2]) if len(sys.argv) > 2 else images_per_disease.get(disease_name, 50)
        generate_specific_disease(disease_name, num_images)
    else:
        for diagnosis in disease_label_map.values():
            num_images = images_per_disease.get(diagnosis, 50)  # Default to 50 if not specified
            generate_medical_images(diagnosis, num_images=num_images)
        print("All image generation complete!")