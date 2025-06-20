# vinyl_model.py
import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model_vinyl import get_model

# 모델 로딩
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
model.load_state_dict(torch.load("best_vinyl_model.pth", map_location=device))
model.to(device)
model.eval()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 비닐 감지 함수
def detect_vinyl(img: Image.Image) -> str:
    img_np = np.array(img)
    augmented = transform(image=img_np)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred_mask = model(img_tensor)
        mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    vinyl_ratio = (mask > 0.3).mean()

    # 비닐 탐지 결과 반환 (비닐이 일정 이상 있으면 True)
    return 'vinyl' if vinyl_ratio > 0.03 else 'not'