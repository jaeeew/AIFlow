# test_vinyl.py
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_vinyl import get_model
from train_vinyl import SegmentationDataset, DEVICE

# 경로
image_dir = './vinyl_dataset/images'
mask_dir = './vinyl_dataset/masks'

# Dataset & transform
test_transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

full_dataset = SegmentationDataset(image_dir, mask_dir, transform=test_transform)
test_idx = sorted([i for i in range(len(full_dataset))])[::10]  # 샘플만 시각화

test_subset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_subset, batch_size=1)

# 모델 로드 및 평가 모드
model = get_model(DEVICE)
model.load_state_dict(torch.load("./best_vinyl_model.pth", map_location=DEVICE))
model.eval()

# 예측 시각화
with torch.no_grad():
    for i, (img, gt) in enumerate(test_loader):
        img = img.to(DEVICE)
        pred = torch.sigmoid(model(img))
        pred_mask = pred.squeeze().cpu().numpy()
        binary_mask = (pred_mask > 0.3).astype(np.float32)
        gt_mask = gt.squeeze().numpy()

        # 오염도 계산
        vinyl_ratio = binary_mask.mean()
        level = "Detected" if vinyl_ratio > 0.02 else "None"

        print(f"[Sample {i+1}] GT sum: {gt_mask.sum()}, Pred sum: {binary_mask.sum()}")
        print(f"[Sample {i + 1}] Vinyl Ratio: {vinyl_ratio:.4f} → {level}")

        # 시각화용 이미지 역정규화
        img_np = img.cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Test Sample {i+1} - Vinyl: {level}", fontsize=14)

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(binary_mask, cmap='gray')
        plt.title("Predicted")
        plt.axis("off")

        os.makedirs("./test_results_vinyl", exist_ok=True)
        plt.savefig(f"./test_results_vinyl/test_results_vinyl_{i + 1}.png")
        # plt.show()