# test_contamination.py (기본 버전)
import os
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from train_contamination import SegmentationDataset, DEVICE
from model_contamination import get_model

# 경로
image_dir = './contamination_dataset/images'
mask_dir = './contamination_dataset/masks'

# Transform
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# Dataset 구성 및 Subset 샘플링
full_dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
test_idx = sorted([i for i in range(len(full_dataset))])[::10]
test_subset = Subset(full_dataset, test_idx)
test_loader = DataLoader(test_subset, batch_size=1)

# 모델 로드 및 평가 모드
device = DEVICE
model = get_model(device)
model.load_state_dict(torch.load("./best_contamination_model.pth", map_location=device))
model.eval()

# 예측 및 시각화
with torch.no_grad():
    for i, (img, gt) in enumerate(test_loader):
        img = img.to(device)
        pred = torch.sigmoid(model(img))
        pred_mask = (pred > 0.3).float().cpu().squeeze().numpy()
        gt_mask = gt.squeeze().numpy()

        # 오염도 비율 및 등급 판단
        contamination_ratio = pred_mask.mean()
        if contamination_ratio < 0.03:
            level = 'clean'
        elif contamination_ratio < 0.20:
            level = 'slight'
        else:
            level = 'heavy'

        print(f"[Sample {i + 1}] GT sum: {gt_mask.sum()}, Pred sum: {pred_mask.sum()}")
        print(f"[Sample {i + 1}] Contamination Ratio: {contamination_ratio:.4f} → {level}")

        # 시각화용 이미지 복원
        img_np = img.cpu().squeeze().permute(1, 2, 0).numpy()
        img_np = img_np * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
        img_np = np.clip(img_np, 0, 1)

        # 시각화 출력
        plt.figure(figsize=(12, 4))
        plt.suptitle(f"Test Sample {i+1}", fontsize=14)

        plt.subplot(1, 3, 1)
        plt.imshow(img_np)
        plt.title("Input")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(gt_mask, cmap='gray')
        plt.title("Ground Truth")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(pred_mask, cmap='gray')
        plt.title("Predicted")
        plt.axis("off")

        plt.tight_layout()

        os.makedirs("./test_results_ctm", exist_ok=True)

        plt.savefig(f"./test_results_ctm/test_results_ctm_{i+1}.png")
        # plt.show()