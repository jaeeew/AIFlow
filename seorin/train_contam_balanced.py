
import os
import re
import csv
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2

# ====== CONFIG ======
BATCH_SIZE = 4
NUM_EPOCHS = 50
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST_MODEL_PATH = "/Users/eumseorin/Desktop/recycle_sj/seorin/train_contam_balanced/best_model.pth"
CSV_LOG_PATH = "/Users/eumseorin/Desktop/recycle_sj/seorin/train_contam_balanced/train_log.csv"

# ====== 유효 쌍 필터링 함수 (1개 디렉토리 대상)
def get_valid_pairs_single(image_dir, mask_dir):
    image_files = [f for f in os.listdir(image_dir) if f.endswith(('.jpg', '.png'))]
    mask_files  = [f for f in os.listdir(mask_dir) if f.endswith(('.jpg', '.png'))]

    image_dict = {os.path.splitext(f)[0]: f for f in image_files}
    mask_dict  = {os.path.splitext(f)[0]: f for f in mask_files}

    common_keys = sorted(set(image_dict.keys()) & set(mask_dict.keys()))
    image_paths = [os.path.join(image_dir, image_dict[k]) for k in common_keys]
    mask_paths  = [os.path.join(mask_dir,  mask_dict[k]) for k in common_keys]

    return image_paths, mask_paths

# ====== 원본 + 증강 각각 split 후 병합 ======
orig_img, orig_mask = get_valid_pairs_single(
    "/Users/eumseorin/Desktop/recycle_sj/images",
    "/Users/eumseorin/Desktop/recycle_sj/masks"
)

aug_img, aug_mask = get_valid_pairs_single(
    "/Users/eumseorin/Desktop/recycle_sj/aug_images",
    "/Users/eumseorin/Desktop/recycle_sj/aug_masks"
)

# 각각 split
train_orig, val_orig = train_test_split(list(zip(orig_img, orig_mask)), test_size=0.2, random_state=42)
train_aug , val_aug  = train_test_split(list(zip(aug_img, aug_mask)),  test_size=0.2, random_state=42)

# 병합
train_img = [x[0] for x in train_orig + train_aug]
train_mask = [x[1] for x in train_orig + train_aug]
val_img   = [x[0] for x in val_orig + val_aug]
val_mask  = [x[1] for x in val_orig + val_aug]

print(f"✅ Train set: {len(train_img)} images")
print(f"✅ Val set:   {len(val_img)} images")

# ====== DATASET 정의 ======
class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB").resize((256, 256))
        mask  = Image.open(self.mask_paths[idx]).convert("L").resize((256, 256))
        image = np.array(image)
        mask = np.array(mask) / 255.0
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask  = augmented["mask"]
        return image, mask

# ====== TRANSFORM ======
train_transform = A.Compose([
    A.Normalize(),
    ToTensorV2()
])

# ====== LOADERS ======
train_dataset = SegmentationDataset(train_img, train_mask, transform=train_transform)
val_dataset   = SegmentationDataset(val_img, val_mask, transform=train_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=1)

# ====== MODEL ======
model = smp.Unet(
    encoder_name="efficientnet-b0",
    encoder_weights="imagenet",
    in_channels=3,
    classes=1
).to(DEVICE)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# ====== TRAIN + EVAL ======
best_psnr = 0
with open(CSV_LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Epoch", "TrainLoss", "Val_PSNR", "Val_Accuracy"])

    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        for x, y in train_loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE).float().unsqueeze(1)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # ====== VALIDATION (PSNR + ACCURACY) ======
        model.eval()
        val_psnr = 0
        val_accuracy = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(DEVICE)
                y = y.to(DEVICE).float().unsqueeze(1)
                out = torch.sigmoid(model(x))

                mse = ((out - y) ** 2).mean().item()
                psnr = 10 * np.log10(1 / (mse + 1e-8))
                val_psnr += psnr

                pred_binary = (out > 0.5).float()
                acc = (pred_binary == y).float().mean().item()
                val_accuracy += acc

        avg_psnr = val_psnr / len(val_loader)
        avg_acc = val_accuracy / len(val_loader)
        writer.writerow([epoch + 1, train_loss, avg_psnr, avg_acc])

        print(f"Epoch {epoch+1:03d} | Loss: {train_loss:.4f} | Val PSNR: {avg_psnr:.2f} | Val Accuracy: {avg_acc:.4f}")

        if avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"📦 Best model saved at epoch {epoch+1} with PSNR {best_psnr:.2f}")
