import os
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from bisenetv2 import BiSeNetV2

# ✅ 설정
IMAGE_DIR = "test_images"
MODEL_PATH = "bisenet_vinyl.pth"  # 또는 "best_bisenet_vinyl.pth"
VINYL_THRESHOLD = 0.13  # 비닐 있음 여부 판단 기준 (비율)

# ✅ 장치 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ✅ 모델 로드
model = BiSeNetV2().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# ✅ 이미지 전처리
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ✅ 비닐 비율 계산 함수
def calculate_vinyl_ratio(pred_mask):
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    return binary_mask.sum() / binary_mask.size

# ✅ 이미지 예측 및 출력
print(f"{'파일명':<22} 예측: {'':<4} | 비닐비율 | 판정")

for filename in sorted(os.listdir(IMAGE_DIR)):
    if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    path = os.path.join(IMAGE_DIR, filename)
    image = Image.open(path).convert("RGB")
    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)[0, 0].cpu().numpy()

    vinyl_ratio = calculate_vinyl_ratio(output)
    pred_label = "yes" if vinyl_ratio > VINYL_THRESHOLD else "no"

    # 현재는 GT가 없으므로 전부 ✅ 로 처리 (향후 비교 가능)
    print(f"[{filename:<20}] 예측: {pred_label:<4} | 비닐비율: {vinyl_ratio:.3f} | ✅")
