import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import pandas as pd
from pathlib import Path

# Use CPU or GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
image_size = 64
z_dim = 100
num_epochs = 25
batch_size = 32
learning_rate = 0.0002

# Project paths
project_root = Path("D:/GAN_training2")
image_dir = project_root / "preprocessed_data/train_images"
metadata_path = project_root / "preprocessed_data/merged_train_metadata.csv"
models_dir = project_root / "models"
os.makedirs(models_dir, exist_ok=True)

print(f"Image directory: {image_dir}")
print(f"Metadata path: {metadata_path}")
print(f"Models will be saved to: {models_dir}")

# Custom Dataset
class SkinLesionDataset(Dataset):
    def __init__(self, image_dir, metadata_file, transform=None):
        self.image_dir = image_dir
        self.metadata = pd.read_csv(metadata_file)
        self.transform = transform
        print(f"Dataset initialized with {len(self.metadata)} images")

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        img_name = self.metadata.iloc[idx]['image'] + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, 0  

# Image transforms
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Dataset and DataLoader
try:
    dataset = SkinLesionDataset(image_dir, metadata_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"DataLoader created with {len(dataset)} images in {len(dataloader)} batches")
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit(1)


class Generator(nn.Module):
    def __init__(self, z_dim):
        super(Generator, self).__init__()
        self.gen = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        return self.gen(x)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.disc(x).view(-1)

# Instantiate models
generator = Generator(z_dim).to(device)
discriminator = Discriminator().to(device)
print("Models created and moved to device")

# Loss and optimizers
criterion = nn.BCELoss()
optimizerG = optim.Adam(generator.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerD = optim.Adam(discriminator.parameters(), lr=learning_rate, betas=(0.5, 0.999))


def save_checkpoint(epoch):
    checkpoint_dir = models_dir / "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'generator_state_dict': generator.state_dict(),
        'discriminator_state_dict': discriminator.state_dict(),
        'optimizerG_state_dict': optimizerG.state_dict(),
        'optimizerD_state_dict': optimizerD.state_dict(),
    }
    
    torch.save(checkpoint, checkpoint_dir / f"gan_checkpoint_epoch_{epoch}.pth")
    print(f"Checkpoint saved for epoch {epoch}")

# Training loop
print("Starting Training...")
for epoch in range(num_epochs):
    for i, (real_images, _) in enumerate(dataloader):
        real_images = real_images.to(device)
        batch_size = real_images.size(0)

        # Labels
        real_labels = torch.ones(batch_size, device=device)
        fake_labels = torch.zeros(batch_size, device=device)

        # Train Discriminator
        discriminator.zero_grad()
        output_real = discriminator(real_images)
        lossD_real = criterion(output_real, real_labels)

        noise = torch.randn(batch_size, z_dim, 1, 1, device=device)
        fake_images = generator(noise)
        output_fake = discriminator(fake_images.detach())
        lossD_fake = criterion(output_fake, fake_labels)

        lossD = lossD_real + lossD_fake
        lossD.backward()
        optimizerD.step()

        # Train Generator
        generator.zero_grad()
        output = discriminator(fake_images)
        lossG = criterion(output, real_labels)
        lossG.backward()
        optimizerG.step()

        if i % 50 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i}/{len(dataloader)}], "
                  f"Loss D: {lossD.item():.4f}, Loss G: {lossG.item():.4f}")
    
    # Save checkpoint after each epoch
    save_checkpoint(epoch + 1)

# Save final models
torch.save(generator.state_dict(), models_dir / "generator_weights.pth")
torch.save(discriminator.state_dict(), models_dir / "discriminator_weights.pth")
print(f"Final model weights saved to {models_dir}")
print("Training complete.") 