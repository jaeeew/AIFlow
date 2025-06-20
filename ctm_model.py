# ctm_model.py
import torch
import numpy as np
from PIL import Image
from model_contamination import get_model
import albumentations as A
from albumentations.pytorch import ToTensorV2

# 오염도 분류 기준 함수 => 예측에 사용됨
def classify_contamination(contamination: float) -> str:
    if contamination < 0.01:
        return 'clean'
    elif contamination < 0.20:
        return 'slight'
    else:
        return 'heavy'

# 전처리: 학습과 동일한 방식 (Normalize 포함)
transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# 모델 로드
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = get_model(device)
model.load_state_dict(torch.load("best_contamination_model.pth", map_location=device))
model.to(device)
model.eval()

# 오염도 예측 함수
def predict_dirty_level(img: Image.Image) -> str:
    # PIL 이미지 → numpy
    img_np = np.array(img)

    # Albumentations transform
    augmented = transform(image=img_np)
    img_tensor = augmented['image'].unsqueeze(0).to(device)

    # 모델 예측
    with torch.no_grad():
        pred_mask = model(img_tensor)
        mask = torch.sigmoid(pred_mask).squeeze().cpu().numpy()

    contamination = (mask > 0.3).mean()

    # 분류 결과 반환
    return classify_contamination(contamination)