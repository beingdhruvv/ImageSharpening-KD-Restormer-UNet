import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from student_model_unet import UNet
from vgg_loss import VGGPerceptualLoss
from tqdm import tqdm

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint_path = "student_checkpoint.pth"

# Dataset class that accepts a list of filenames
class KDImageDataset(Dataset):
    def __init__(self, blurry_dir, sharp_dir, teacher_dir, filenames, transform=None):
        self.blurry_dir = blurry_dir
        self.sharp_dir = sharp_dir
        self.teacher_dir = teacher_dir
        self.filenames = filenames
        self.transform = transform or transforms.ToTensor()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        blur = Image.open(os.path.join(self.blurry_dir, fname)).convert("RGB")
        sharp = Image.open(os.path.join(self.sharp_dir, fname)).convert("RGB")
        teacher = Image.open(os.path.join(self.teacher_dir, fname)).convert("RGB")
        return self.transform(blur), self.transform(sharp), self.transform(teacher)

def train_student(
    blurry_dir, sharp_dir, teacher_dir,
    epochs=18, batch_size=8, lr=1e-4,
    lambda_kd=1.0, lambda_vgg=0.1, chunk_size=2500, save_path="student_unet.pth"
):
    # Get all available filenames (intersection of triplets)
    blur = set(os.listdir(blurry_dir))
    sharp = set(os.listdir(sharp_dir))
    teacher = set(os.listdir(teacher_dir))
    full_filenames = sorted(list(blur & sharp & teacher))
    total_images = len(full_filenames)

    model = UNet(base_filters=64, use_dropout=False, depth=4).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.L1Loss()
    perceptual_loss = VGGPerceptualLoss().to(device)  

    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"\nResuming from epoch {start_epoch}")

    print(f"\nStarting training from epoch {start_epoch + 1} to {epochs}")

    for epoch in range(start_epoch, epochs):
        # Wrap-around chunking
        start_idx = (epoch * chunk_size) % total_images
        end_idx = min(start_idx + chunk_size, total_images)

        subset_filenames = full_filenames[start_idx:end_idx]
        if len(subset_filenames) == 0:
            print(f"\nNo samples to train on for epoch {epoch+1}. Skipping.")
            break

        print(f"\nEpoch {epoch+1}: Training on samples {start_idx} to {end_idx} ({len(subset_filenames)})")

        dataset = KDImageDataset(blurry_dir, sharp_dir, teacher_dir, subset_filenames)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

        model.train()
        total_loss = 0.0

        for blurry, sharp, teacher in tqdm(loader, desc=f"Training Epoch {epoch+1}"):
            blurry, sharp, teacher = blurry.to(device), sharp.to(device), teacher.to(device)
            output = model(blurry)

            loss_gt = criterion(output, sharp)
            loss_teacher = criterion(output, teacher)
            loss_vgg = perceptual_loss(output, teacher)

            loss = loss_gt + lambda_kd * loss_teacher + lambda_vgg * loss_vgg  

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        print(f"\nEpoch {epoch+1} - Avg Loss: {avg_loss:.4f}")

        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict()
        }, checkpoint_path)
        print(f"Saved checkpoint at epoch {epoch+1}")

    torch.save(model.state_dict(), save_path)
    print(f"\nTraining complete. Final model saved to: {save_path}")

if __name__ == "__main__":
    train_student(
        blurry_dir="data/blurry/train/train",
        sharp_dir="data/sharp/train/train",
        teacher_dir="outputs/teacher_output/train/train",
        epochs=18,
        batch_size=8,
        lambda_kd=1.0,
        lambda_vgg=0.1,  
        chunk_size=2500,
        save_path="student_unet.pth"
    )
