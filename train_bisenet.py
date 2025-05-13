import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import numpy as np
from tqdm import tqdm
from bisenetv2 import BiSeNetV2

def dice_loss(pred, target, smooth=1.):
    pred = pred.contiguous()
    target = target.contiguous()
    intersection = (pred * target).sum(dim=(2, 3))
    return 1 - ((2. * intersection + smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + smooth)).mean()

class CombinedLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCELoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + dice_loss(pred, target)

class VinylDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.images = sorted(os.listdir(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        filename = os.path.splitext(self.images[idx])[0]
        img_path = os.path.join(self.img_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, f"{filename}_mask.png")

        image = Image.open(img_path).convert("RGB").resize((256, 256))
        mask = Image.open(mask_path).convert("L").resize((256, 256))

        if self.transform:
            image = self.transform(image)

        mask = np.array(mask)
        mask = (mask > 127).astype(np.float32)
        mask = torch.from_numpy(mask).unsqueeze(0)

        return image, mask

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    dataset = VinylDataset("data/vinyldata", "data/vinylmasks_binary", transform=transform)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)

    model = BiSeNetV2().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)
    criterion = CombinedLoss()

    for epoch in range(60):
        model.train()
        total_loss = 0
        for imgs, masks in tqdm(loader):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}: Loss = {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), "bisenet_vinyl.pth")
    print("✅ 모듈 저장 완료")

if __name__ == "__main__":
    train()
